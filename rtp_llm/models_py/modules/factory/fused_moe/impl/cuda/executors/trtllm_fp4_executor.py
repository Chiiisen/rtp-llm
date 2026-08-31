from typing import Any, Dict, Optional

import torch
from flashinfer import ActivationType, fp4_quantize
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
from flashinfer.utils import device_support_pdl

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.utils.model_weight import W


class TrtllmFp4Executor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.TRTLLM_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        # Check if quantization is enabled and uses FP4 (uint8 dtype)
        # FP4 weights are packed as uint8, so we check for quant_config with uint8 dtype
        checker.check(
            resolver.has_quantization(config)
            and resolver.get_quant_method(config) == "modelopt_fp4"
        )

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        self.w1_scale = weights.get(W.moe_s1, None)
        self.w2_scale = weights.get(W.moe_s2, None)

        w13_input_scale = weights.get(W.moe_w1_i_s, None)
        w13_weight_scale_2 = weights.get(W.moe_w1_s2, None)
        w2_input_scale = weights.get(W.moe_w2_i_s, None)
        w2_weight_scale_2 = weights.get(W.moe_w2_s2, None)

        assert self.w1 is not None
        assert self.w2 is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        assert w13_input_scale is not None
        assert w13_weight_scale_2 is not None
        assert w2_input_scale is not None
        assert w2_weight_scale_2 is not None

        self.expert_x_scale = 1 / w13_input_scale
        self.g1_alphas = w13_input_scale * w13_weight_scale_2
        self.g2_alphas = w2_input_scale * w2_weight_scale_2
        self.g1_scale_c = self.g1_alphas / w2_input_scale
        # sm_120 CUTLASS path (see _execute_sm120_cutlass): activation global
        # scale for gemm2 and per-expert dequant scales.
        self.expert_x2_scale = 1 / w2_input_scale
        self._sm120_w1_i64 = None
        self._sm120_w2_i64 = None
        self._sm120_w1_sf_i32 = None
        self._sm120_w2_sf_i32 = None
        self._sm120_g1_alphas_e = None
        self._sm120_g2_alphas_e = None

        self.global_num_experts = config.expert_num
        self._enable_pdl = device_support_pdl(self.w1.device)

    @property
    def local_num_experts(self) -> int:
        assert self.w1 is not None
        return self.w1.size(0)

    @property
    def intermediate_size(self) -> int:
        assert self.w1 is not None
        return int(self.w1.size(1) / 2)

    @property
    def hidden_size(self) -> int:
        assert self.w2 is not None
        return self.w2.size(-2)

    def _to_per_expert_vector(self, t: torch.Tensor) -> torch.Tensor:
        """Broadcast a scalar/[E,1]/[E] scale to a float32 [num_experts] vector."""
        t = t.detach().reshape(-1).float()
        if t.numel() == 1:
            t = t.expand(self.local_num_experts)
        assert t.numel() == self.local_num_experts, (
            f"per-expert scale: {t.numel()} vs {self.local_num_experts}"
        )
        return t.contiguous()

    def _execute_sm120_cutlass(
        self,
        payload: ExpertForwardPayload,
        activation: str,
    ) -> CombineForwardPayload:
        """sm_120 (consumer Blackwell) path.

        The trtllmgen FP4 MoE cubins shipped in flashinfer-cubin are sm100f-only,
        so route to flashinfer's CUTLASS sm_120 fused MoE instead. Requires the
        FP4_MOE_OP=cutedsl weight prep (w13 halves swapped to [up, gate] and
        tcgen05-128x4-swizzled block scales) — the sm120 kernel contract:
          weights: uint8 [E, N, K/2] viewed as int64 [E, N, K/16]
          weight block scales: swizzled e4m3 viewed as int32
          quant_scales: [fc1_act_global, fc1_sf, fc1_dequant[E],
                         fc2_act_global, fc2_sf, fc2_dequant[E]]
        Numerics verified against a BF16 reference (cos 0.976) on SM120.
        """
        from flashinfer.fused_moe import cutlass_fused_moe

        act_type_map = {
            "silu": ActivationType.Swiglu,
            "swiglu": ActivationType.Swiglu,
            "geglu": ActivationType.Geglu,
            "siglu": ActivationType.Swiglu,
        }
        act = act_type_map[activation.lower()]

        topk_ids = payload.expert_topk_ids.to(torch.int32)
        topk_weights = payload.expert_topk_weights.to(torch.float32)
        # The pure-TP router emits -1 sentinels for experts owned by other
        # ranks (EP-equivalent layout); the trtllmgen kernel understands them
        # but the CUTLASS sm120 path does not. Neutralize: route masked slots
        # to expert 0 with zero weight (adds a harmless no-op compute).
        invalid_mask = topk_ids < 0
        if invalid_mask.any():
            topk_ids = topk_ids.masked_fill(invalid_mask, 0)
            topk_weights = topk_weights.masked_fill(invalid_mask, 0.0)
        # DEBUG (E1): routing contract — ids must be LOCAL expert indices in
        # [0, E_local). An out-of-range positive id indexes past the weight
        # tensor = illegal access inside cutlass_fused_moe. Log + clamp instead
        # of crashing so a single server cycle proves or kills the hypothesis.
        if topk_ids.shape[0] > 128:
            max_id = int(topk_ids.max())
            min_id = int(topk_ids.min())
            import logging
            if not getattr(TrtllmFp4Executor, "_route_dbg", False):
                logging.error(
                    f"SM120ROUTE first: E_loc={self.local_num_experts} "
                    f"E_glob={self.global_num_experts} ids[min={min_id},max={max_id}] "
                    f"shape={tuple(topk_ids.shape)} dtype={topk_ids.dtype}"
                )
                TrtllmFp4Executor._route_dbg = True
            if max_id >= self.local_num_experts or min_id >= self.local_num_experts:
                logging.error(
                    f"SM120ROUTE VIOLATION: max_id={max_id} min_id={min_id} "
                    f"E_loc={self.local_num_experts} shape={tuple(topk_ids.shape)} — "
                    f"router emitted out-of-range expert ids; clamping"
                )
                oob = topk_ids >= self.local_num_experts
                topk_ids = topk_ids.masked_fill(oob, 0)
                topk_weights = topk_weights.masked_fill(oob, 0.0)

        if self._sm120_w1_i64 is None:
            # One-time zero-copy views (weights are static after load).
            self._sm120_w1_i64 = self.w1.view(torch.int64)
            self._sm120_w2_i64 = self.w2.view(torch.int64)
            self._sm120_w1_sf_i32 = self.w1_scale.view(torch.int32)
            self._sm120_w2_sf_i32 = self.w2_scale.view(torch.int32)
            self._sm120_g1_alphas_e = self._to_per_expert_vector(self.g1_alphas)
            self._sm120_g2_alphas_e = self._to_per_expert_vector(self.g2_alphas)
            # Activation global scales. Expert-uniform checkpoints (Qwen3.5-122B,
            # Qwen3.5-397B-V2: one input_scale shared by all experts) degenerate
            # to the scalar path. Checkpoints with PER-EXPERT input_scales
            # (Qwen3-Coder-30B fc2; Qwen3-235B-A22B both gemms, ~9x spread)
            # must keep the [E] vectors: collapsing to element [0] rescales
            # every expert's GEMM output by x[0]/x[e] (verified 2026-08-28 on
            # SM120: 30B MoE cos 0.85->0.99+, 235B 0.865->0.996).
            x1_vec = self._to_per_expert_vector(self.expert_x_scale)
            x2_vec = self._to_per_expert_vector(self.expert_x2_scale)
            self._sm120_x1_uniform = bool(
                torch.allclose(x1_vec, x1_vec[0].expand_as(x1_vec))
            )
            self._sm120_x2_uniform = bool(
                torch.allclose(x2_vec, x2_vec[0].expand_as(x2_vec))
            )
            self._sm120_x_scale = (
                x1_vec[0].contiguous() if self._sm120_x1_uniform else x1_vec
            )
            self._sm120_x2_scale = (
                x2_vec[0].contiguous() if self._sm120_x2_uniform else x2_vec
            )
            # Pre-quantizing applies ONE global fc1 scale; per-expert fc1
            # scales require the kernel's internal per-expert quantization
            # (verified: the kernel ignores perE fc1 act scale for
            # pre-quantized input). Pre-quantize only when both gemms are
            # expert-uniform.
            self._sm120_prequant_input = (
                self._sm120_x1_uniform and self._sm120_x2_uniform
            )

        # SM120 cutlass fix: fp4_quantize(is_sf_swizzled_layout=True) pads the
        # scale-factor tile to 128-row blocks but only WRITES the first
        # num_tokens rows — the rest of the swizzled tile is uninitialized
        # memory. The tcgen05 kernel processes full 128-row tiles, so stale
        # allocator bytes in the padding surface as NaN in the MoE output
        # (non-deterministic, co-batched decode on SM120; verified
        # 2026-08-29: filling the padding with garbage reproduces the NaN
        # offline, padding with zeros fixes it). Pad the input to the tile
        # boundary with zero rows and zero-weight routing pairs so the
        # quantizer writes deterministic zeros, then slice the padding off
        # the result.
        num_tokens = topk_ids.shape[0]
        tile_rows = 128
        is_bf16_input = payload.expert_x.dtype is torch.bfloat16
        pad_rows = (-num_tokens) % tile_rows if is_bf16_input else 0
        if pad_rows:
            topk_ids = torch.cat(
                [topk_ids, topk_ids.new_zeros(pad_rows, topk_ids.shape[1])]
            )
            topk_weights = torch.cat(
                [
                    topk_weights,
                    topk_weights.new_zeros(pad_rows, topk_weights.shape[1]),
                ]
            )
            expert_x = torch.nn.functional.pad(
                payload.expert_x, (0, 0, 0, pad_rows)
            )
        else:
            expert_x = payload.expert_x

        if payload.expert_x.dtype is torch.bfloat16:
            if self._sm120_prequant_input:
                hidden_states, hidden_states_scale = fp4_quantize(
                    expert_x,
                    self._sm120_x_scale.reshape(1),
                    is_sf_swizzled_layout=True,
                )
            else:
                # bf16 input: cutlass_fused_moe quantizes per-expert
                # internally using the [E] activation global scales.
                hidden_states = expert_x.contiguous()
                hidden_states_scale = None
        else:
            hidden_states, hidden_states_scale = (
                expert_x,
                payload.expert_x_scale,
            )

        output = cutlass_fused_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            self._sm120_w1_i64,
            self._sm120_w2_i64,
            payload.expert_x.dtype if payload.expert_x.dtype is torch.bfloat16 else torch.bfloat16,
            [
                self._sm120_x_scale,       # fc1 activation global scale (scalar)
                self._sm120_w1_sf_i32,     # fc1 weight block scales (swizzled)
                self._sm120_g1_alphas_e,   # fc1 dequant scale (per expert)
                self._sm120_x2_scale,      # fc2 activation global scale (scalar)
                self._sm120_w2_sf_i32,     # fc2 weight block scales (swizzled)
                self._sm120_g2_alphas_e,   # fc2 dequant scale (per expert)
            ],
            None,  # fc1_expert_biases
            None,  # fc2_expert_biases
            hidden_states_scale,  # input_sf (swizzled)
            None,  # swiglu_alpha
            None,  # swiglu_beta
            None,  # swiglu_limit
            activation_type=act,
            # Cap the AutoTuner profiling bucket: the tuner allocates its
            # profiling workspace sized by this cap (multi-GB at the default
            # 8192), which faults on memory-tight deployments (KV cache pool
            # leaves ~1 GiB free; observed as illegal memory access inside
            # the MoE call on prefills > 128 tokens). 1024 bounds the workspace
            # to a few hundred MB while still covering decode batches.
            tune_max_num_tokens=1024,
        )
        if isinstance(output, (list, tuple)):
            output = output[0]
        if pad_rows:
            # strip the zero-padding rows before returning to the router
            output = output[:num_tokens]
        # Defensive copy: flashinfer's cutlass_fused_moe allocates its output
        # from a per-call workspace that is freed on return; downstream NCCL
        # all_reduce / shared-expert fusion then touches freed memory at larger
        # token counts (illegal memory access surfacing at the next GEMM).
        output = output.clone()
        return CombineForwardPayload(fused_expert_output=output)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        from rtp_llm.models_py.utils.arch import is_sm12x

        if is_sm12x():
            # sm_120: trtllmgen cubins are sm100f-only; use the CUTLASS sm120 MoE.
            return self._execute_sm120_cutlass(payload, activation)

        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        topk = topk_ids.size(-1)

        act_type_map = {
            "silu": ActivationType.Swiglu.value,
            "swiglu": ActivationType.Swiglu.value,
            "geglu": ActivationType.Geglu.value,
            "siglu": ActivationType.Swiglu.value,
        }
        gated_act_type = act_type_map[activation.lower()]

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
            torch.bfloat16
        ).view(torch.int16)

        if payload.expert_x.dtype is torch.bfloat16:
            hidden_states, hidden_states_scale = fp4_quantize(
                payload.expert_x, self.expert_x_scale, is_sf_swizzled_layout=False
            )
        else:
            hidden_states, hidden_states_scale = (
                payload.expert_x,
                payload.expert_x_scale,
            )
            assert (
                hidden_states.dtype is torch.uint8
            ), f"hidden_states: {hidden_states.dtype}"
            assert (
                hidden_states_scale is not None
            ), f"hidden_states_scale: {hidden_states_scale}"
            assert (
                hidden_states_scale.dtype is torch.uint8
            ), f"hidden_states_scale: {hidden_states_scale.dtype}"
            assert hidden_states.shape[-1] == hidden_states_scale.shape[-1] * 8, (
                f"hidden_states: {hidden_states.shape}"
                f"hidden_states_scale: {hidden_states_scale.shape}"
            )

        output = trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_tensor,  # topk_ids
            routing_bias=None,  # routing_bias
            hidden_states=hidden_states,  # hidden_states
            hidden_states_scale=hidden_states_scale.view(
                torch.float8_e4m3fn
            ),  # hidden_states_scale
            gemm1_weights=self.w1,  # gemm1_weights
            gemm1_weights_scale=self.w1_scale.view(
                torch.float8_e4m3fn
            ),  # gemm1_weights_scale
            gemm1_bias=None,  # gemm1_bias
            gemm1_alpha=None,  # gemm1_alpha
            gemm1_beta=None,  # gemm1_beta
            gemm1_clamp_limit=None,  # gemm1_clamp_limit
            gemm2_weights=self.w2,  # gemm2_weights
            gemm2_weights_scale=self.w2_scale.view(
                torch.float8_e4m3fn
            ),  # gemm2_weights_scale
            gemm2_bias=None,  # gemm2_bias
            output1_scale_scalar=self.g1_scale_c,  # output1_scale_scalar
            output1_scale_gate_scalar=self.g1_alphas,  # output1_scale_gate_scalar
            output2_scale_scalar=self.g2_alphas,  # output2_scale_scalar
            num_experts=self.global_num_experts,  # num_experts
            top_k=topk,  # top_k
            n_group=None,  # n_group
            topk_group=None,  # topk_group
            intermediate_size=self.intermediate_size,  # intermediate_size
            local_expert_offset=0,  # local_expert_offset
            local_num_experts=self.local_num_experts,  # local_num_experts
            routed_scaling_factor=None,  # routed_scaling_factor
            routing_method_type=1,  # routing_method_type: Renormalize
            do_finalize=True,  # do_finalize
            enable_pdl=self._enable_pdl,  # enable_pdl
            activation_type=gated_act_type,  # activation_type
            output=None,  # output (optional inplace)
            # tune_max_num_tokens: int = 8192
        )[
            0
        ]  # Returns list, get first element

        return CombineForwardPayload(fused_expert_output=output)
