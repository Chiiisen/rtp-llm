import logging
import os
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FakeBalanceExpert,
    FMHAImplBase,
    FusedMoeFactory,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    RMSResNorm,
    SelectTopk,
    SigmoidGateScaleAdd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


_NAN_DUMP_DONE = False


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.ffn_tp_size = parallelism_config.get_ffn_tp_size()
        self.ep_size = parallelism_config.ep_size

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, quant_config, hw_kernel_config
        )
        self.select_topk = SelectTopk(config=config)
        if moe_config.fake_balance_expert:
            self.fake_balance_expert = FakeBalanceExpert(
                expert_num=config.expert_num,
                moe_k=config.moe_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=parallelism_config.ep_size,
            )
        else:
            self.fake_balance_expert = None
        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)
        router = self.fused_moe.router
        router_tp_size = router.tp_collective_size

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.add_shared_expert = config.moe_style == 2
        if self.add_shared_expert:
            self.shared_expert = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.shared_expert = None
        if weights.get(W.shared_expert_gate, None) is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights,
                W.shared_expert_gate,
                None,
                None,
                quant_config=quant_config,
                # For ROCm devices shared_expert_gate is not pre-swizzled during weight
                # loading and its single output column does not satisfy the SwizzleA
                # layout. Keep this scalar projection on the no-swizzle backend.
                hw_kernel_config=None,
            )
            self.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
        else:
            self.shared_expert_gate = None
            self.sigmoid_gate_scale_add = None

        self.use_ep_shared_allreduce = (
            self.shared_expert is not None and self.ffn_tp_size > 1 and self.ep_size > 1
        )
        self.use_unified_tp_allreduce = (
            self.shared_expert is not None
            and self.ffn_tp_size > 1
            and self.ep_size == 1
            and self.ffn_tp_size == router_tp_size
            and router.supports_skip_tp_allreduce
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "GenericMoE unified TP all-reduce %s "
                "(router=%s, ffn_tp_size=%d, router_tp_size=%d, ep_size=%d)",
                "enabled" if self.use_unified_tp_allreduce else "disabled",
                type(router).__name__,
                self.ffn_tp_size,
                router_tp_size,
                self.ep_size,
            )

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

    def _merge_shared_expert_output(
        self,
        hidden_states: torch.Tensor,
        experts_output: torch.Tensor,
        shared_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_expert_gate is not None:
            gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
            self.sigmoid_gate_scale_add(
                gate_output, shared_expert_output, experts_output
            )
            return experts_output
        return experts_output + shared_expert_output

    def _gate_shared_expert_output(
        self,
        hidden_states: torch.Tensor,
        shared_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_expert_gate is not None:
            gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
            return torch.sigmoid(gate_output) * shared_expert_output
        return shared_expert_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import os
        _moe_probe = os.environ.get("RTP_LLM_NAN_DEBUG") in ("1", "2") and (hidden_states.shape[0] > 1 or os.environ.get("RTP_LLM_NAN_DEBUG") == "2")
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        # different executor may need different topk_ids dtype
        topk_ids_dtype = self.fused_moe.topk_ids_dtype
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=hidden_states.device,
        )

        if self.correction_bias is not None:
            self.group_topk = GroupTopK()
            self.renormalize = self.config.has_moe_norm
            self.num_expert_group = self.config.moe_n_group

            self.topk_group = self.config.moe_topk_group
            self.n_routed_experts = self.config.expert_num  # config.n_routed_experts
            self.routed_scaling_factor = self.config.routed_scaling_factor
            self.group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits_fp32,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            # Top-K selection using C++ SelectTopkOp
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        # In pure-TP mode both the routed experts and the shared expert produce
        # TP-partial outputs.  Reduce their sum once instead of reducing each
        # path separately.  This is especially important for decode, where the
        # hidden dimension is small enough that collective launch latency
        # dominates the payload transfer.
        experts_output = self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
            skip_tp_allreduce=self.use_unified_tp_allreduce,
        )
        if _moe_probe:
            import logging
            _in_bad = (~torch.isfinite(hidden_states)).any().item()
            _out_bad = (~torch.isfinite(experts_output)).any().item()
            _in_abs = hidden_states.float().abs().max().item()
            _log = logging.getLogger("nan_debug")
            _log.warning(
                f"[moe-probe] batch={num_tokens} in_bad={_in_bad} in_absmax={_in_abs:.2f} "
                f"out_bad={_out_bad} router_finite={torch.isfinite(router_logits_fp32).all().item()} "
                f"topk_ids_finite={torch.isfinite(topk_ids.float()).all().item()}"
            )
            global _NAN_DUMP_DONE
            if not _NAN_DUMP_DONE:
                _NAN_DUMP_DONE = True
                try:
                    ex = self.fused_moe.fused_experts
                    torch.save(
                        {
                            "w1": ex.w1,
                            "w2": ex.w2,
                            "w1_scale": ex.w1_scale,
                            "w2_scale": ex.w2_scale,
                            "g1_alphas": ex.g1_alphas,
                            "g2_alphas": ex.g2_alphas,
                            "expert_x_scale": ex.expert_x_scale,
                            "expert_x2_scale": ex.expert_x2_scale,
                        },
                        f"/data/xuezhichen.xzc/RTP-LLM/tmp_tests/moe_exec_tensors_r{os.getpid()}.pt",
                    )
                    _log.warning("[moe-probe] executor tensors dumped")
                except Exception as _e:
                    _log.warning(f"[moe-probe] executor dump failed: {_e}")
            try:
                ex = self.fused_moe.fused_experts
                _ck_now = (
                    int(ex.w1.view(torch.int64).sum().item()),
                    int(ex.w2.view(torch.int64).sum().item()),
                    int(ex.w1_scale.view(torch.int64).sum().item()),
                    int(ex.w2_scale.view(torch.int64).sum().item()),
                )
                if not hasattr(self, "_nan_ck0"):
                    self._nan_ck0 = _ck_now
                elif self._nan_ck0 != _ck_now:
                    import logging
                    logging.getLogger("nan_debug").warning(
                        f"[moe-probe] WEIGHT STOMP on this layer: was={self._nan_ck0} now={_ck_now}"
                    )
                    self._nan_ck0 = _ck_now
            except Exception:
                pass
            if _out_bad:
                _cap = {
                    "moe_in": hidden_states.detach().clone(),
                    "moe_out": experts_output.detach().clone(),
                    "topk_ids": topk_ids.detach().clone(),
                    "topk_w": topk_weights.detach().clone(),
                }
                torch.save(_cap, f"/data/xuezhichen.xzc/RTP-LLM/tmp_tests/nan_moe_capture_{num_tokens}.pt")
                _log.warning(
                    f"[moe-probe] saved capture: nan={torch.isnan(experts_output).sum().item()} "
                    f"inf={torch.isinf(experts_output).sum().item()} out_absmax_pre={experts_output.float().abs().max().item():.2f}"
                )
        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(
                hidden_states,
                skip_allreduce=(
                    self.use_ep_shared_allreduce or self.use_unified_tp_allreduce
                ),
            )
            if self.use_unified_tp_allreduce:
                # Both paths are still TP-partial.  The shared-expert gate is
                # rank-consistent because hidden_states are replicated across
                # TP ranks, so it is safe to apply it before the single
                # all-reduce.
                experts_output = self._merge_shared_expert_output(
                    hidden_states, experts_output, shared_expert_output
                )
                experts_output = all_reduce(experts_output, group=Group.TP)
            elif self.use_ep_shared_allreduce:
                # EP mode: routed expert output is already complete
                # (EP combine via all_to_all / all_gather aggregated across ranks).
                # Only the shared expert output is TP-partial and needs all_reduce.
                shared_expert_output = self._gate_shared_expert_output(
                    hidden_states, shared_expert_output
                )
                shared_expert_output = all_reduce(shared_expert_output, group=Group.TP)
                experts_output = experts_output + shared_expert_output
            else:
                # Fallback path: each path is already complete independently.
                # This includes ffn_tp_size == 1 and routers that retain their
                # own finalize reduction, so only local merging remains.
                experts_output = self._merge_shared_expert_output(
                    hidden_states, experts_output, shared_expert_output
                )

        return experts_output


class DecodeLayerOutput:
    def __init__(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        self.hidden_states = hidden_states
        self.residual = residual


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get quant_config from model_config
        quant_config = config.quant_config
        if config.attn_config.use_mla:
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                global_weights=global_weights,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = CausalAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                layer_idx,
            )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )

        # 使用 RMSResNorm 来 fuse residual add 和 layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> DecodeLayerOutput:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return DecodeLayerOutput(hidden_states, residual)


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        self._mtp_target_hidden_states: Optional[torch.Tensor] = None
        self._capture_eagle3 = (
            getattr(model_config, "hc_mult", 1) == 3
            or os.environ.get("SP_TYPE", "").strip().lower() == "eagle3"
        )
        self._eagle3_capture_logged = False

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
        residual = torch.zeros_like(hidden_states)
        eagle3_hidden = []
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            output = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            if self._capture_eagle3 and i in (1, 46, 90):
                eagle3_hidden.append((hidden_states + residual).contiguous())

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(eagle3_hidden) == 3:
            self._mtp_target_hidden_states = torch.cat(eagle3_hidden, dim=-1)
        else:
            self._mtp_target_hidden_states = None
        if self._capture_eagle3 and not self._eagle3_capture_logged:
            logging.info(
                "EAGLE3 target hidden capture: hc_mult=%s, layers=%s, shape=%s",
                getattr(self.config, "hc_mult", None),
                len(eagle3_hidden),
                tuple(self._mtp_target_hidden_states.shape)
                if self._mtp_target_hidden_states is not None
                else None,
            )
            self._eagle3_capture_logged = True

        return PyModelOutputs(hidden_states)

    def get_mtp_target_hidden_states(self, num_tokens: int = -1) -> Optional[torch.Tensor]:
        hidden_states = self._mtp_target_hidden_states
        if hidden_states is None:
            return None
        if num_tokens < 0:
            return hidden_states
        return hidden_states.narrow(0, 0, min(num_tokens, hidden_states.size(0)))


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
