"""Static per-tensor FP8 (W8A8) loader for ModelOpt mixed-precision checkpoints.

Serves the FP8 group of ``modelopt_mixed`` checkpoints (e.g. nvidia
Qwen3.5-397B-A17B-NVFP4-V2), where attention / linear-attention /
shared-expert modules are quantized per-tensor static:
``.weight`` (fp8 e4m3, HF [N, K]) + ``.weight_scale`` (scalar fp32) +
``.input_scale`` (scalar fp32). The routed experts are NVFP4 and are handled
by MixedFp4Weight; ``ignore`` modules stay BF16 (plain load).

Fused multi-piece kernels (qkv, qkvz, w13) carry per-piece weight scales
which differ in the checkpoint; `_postprocess` dequant-requants each piece
to a shared max scale so a single scale serves the merged GEMM (same trick
as Fp8PerTensorCompressedWeight's qkv path). Fused input scales are equal in
modelopt checkpoints; the first piece's value is used (max for w13).

Delivered kernels follow the CUDA static-FP8 family convention ([N, K]),
consumed by CudaFp8PerTensorLinear via weight.T -> column-major mat2.
"""

import math
import re as _re
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import ModelOptMixedFp4Config, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight
from rtp_llm.model_loader.linear_attn_weight import (
    W8A8Fp8PerBlockLinearAttnAtomicWeight,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.w8a8_weight import W8A8Fp8AtomicWeight
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    get_tensor_from_scalar,
    identity,
    merge_te_qkv,
    pad_w13,
    sp_head_gemm_a8,
    sp_id,
)

W_SUFFIX = ".weight"
QS_SUFFIX = ".weight_scale"
ACT_S_SUFFIX = ".input_scale"


def _cat_scales(ts: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-piece scalar scales into a flat [P] vector."""
    return torch.cat([t.reshape(1) for t in ts], dim=0)


def _max_scales(ts: List[torch.Tensor]) -> torch.Tensor:
    """Max of per-piece scalar scales, as [1]."""
    return torch.stack([t.reshape(1) for t in ts]).max()


def _is_excluded(src_weight_info, exclude_modules: set) -> bool:
    """True if any ckpt tensor of this weight matches a quant exclusion glob.

    Mirrors MixedFp4Weight._is_excluded_from_quant: ckpt names are layer
    templates with ``{i}``/``{expert_id}`` placeholders, so sample a few
    concrete indices before glob-matching.
    """
    if not exclude_modules:
        return False
    import fnmatch

    for w in getattr(src_weight_info, "weights", []):
        template = w.name
        candidates = (
            [template]
            if "{" not in template
            else [
                _re.sub(r"\{[^}]+\}", str(idx), template) for idx in (0, 1)
            ]
        )
        for cand in candidates:
            if any(fnmatch.fnmatch(cand, pat) for pat in exclude_modules):
                return True
    return False


def _is_mixed_fp4_module(src_weight_info, quant_config) -> bool:
    """True if any ckpt tensor of this weight belongs to the NVFP4 group
    (``quant_config.fp4_targets``) and must be served by MixedFp4Weight."""
    targets = getattr(quant_config, "fp4_targets", None) or set()
    if not targets:
        return False
    for w in getattr(src_weight_info, "weights", []):
        template = w.name
        candidates = (
            [template]
            if "{" not in template
            else [_re.sub(r"\{[^}]+\}", str(idx), template) for idx in (0, 1)]
        )
        for cand in candidates:
            if any(cand == t or cand.startswith(t + ".") for t in targets):
                return True
    return False


def rescale_pieces_to_max(
    kernel: torch.Tensor, scales: torch.Tensor, boundaries: List[int]
) -> torch.Tensor:
    """Dequant-requant fp8 kernel rows to a shared max scale.

    kernel: [N, K] fp8; scales: [P] per-piece scalar scales; boundaries:
    row count of each piece along dim 0 (sum == N).
    """
    scales = scales.reshape(-1)
    s_max = scales.max()
    if bool(torch.all(scales == s_max)):
        return kernel
    out_rows = []
    start = 0
    for s, n in zip(scales.tolist(), boundaries):
        piece = kernel[start : start + n]
        out_rows.append((piece.to(torch.float32) * (s / s_max)).to(kernel.dtype))
        start += n
    assert start == kernel.shape[0], (
        f"piece boundaries {boundaries} sum {start} != kernel rows {kernel.shape[0]}"
    )
    return torch.cat(out_rows, dim=0).contiguous()


def _split_gate_rows(t, tp: int, tp_rank: int, **kwargs: Any):
    """Uniform q-head row split of the fused output gate [N_gate, K].

    Gate rows correspond 1:1 with q heads, so rank r owns the same
    contiguous head range as the q projection (sp_neg1-style ranges).
    """
    n_per = t.shape[0] // tp
    return t[tp_rank * n_per : (tp_rank + 1) * n_per].contiguous()


class MixedModelOptFp8AtomicWeight(W8A8Fp8AtomicWeight):
    """Per-tensor FP8 atomic weight with gate / input-scale split entries.

    Reuses the proven W8A8Fp8 per-tensor TP strategy (qkv head split, ffn
    inter-dim split, scales replicate) and adds the Qwen3.5 gate and the
    FP4-style input-scale keys (all replicate).
    """

    gpt_style_tp_strategy = dict(W8A8Fp8AtomicWeight.gpt_style_tp_strategy)
    gpt_style_tp_strategy.update(
        {
            W.attn_gate_w: _split_gate_rows,
            W.attn_gate_s: sp_id,
            W.attn_qkv_i_s: sp_id,
            W.attn_o_i_s: sp_id,
            W.ffn_w1_i_s: sp_id,
            W.ffn_w3_i_s: sp_id,
            W.ffn_w13_i_s: sp_id,
            W.ffn_w2_i_s: sp_id,
        }
    )


class _ReplicateScalarWeight(AtomicWeight):
    """Scalar (or per-piece scalar) scale that must NOT be TP-split.

    The parent _postprocess rescales fused pieces to a single max scalar
    anyway, so every rank derives the same scale; splitting a scalar via the
    gpt_style strategy dict would KeyError (no entry for scale keys).
    """

    def _get_split_func(self):
        return lambda t, **kwargs: t


class MixedModelOptFp8Weight(CompositeWeight, QuantWeight):
    """Static per-tensor FP8 weight for ModelOpt mixed-precision checkpoints."""

    # module name -> (kernel key, scale key, optional input-scale key)
    weight_map = {
        W.attn_qkv_w: (W.attn_qkv_w, W.attn_qkv_s, W.attn_qkv_i_s),
        W.attn_gate_w: (W.attn_gate_w, W.attn_gate_s, None),
        W.attn_o_w: (W.attn_o_w, W.attn_o_s, W.attn_o_i_s),
        W.ffn_w13: (W.ffn_w13, W.ffn_s13, W.ffn_w13_i_s),
        W.ffn_w1: (W.ffn_w1, W.ffn_s1, W.ffn_w1_i_s),
        W.ffn_w3: (W.ffn_w3, W.ffn_s3, W.ffn_w3_i_s),
        W.ffn_w2: (W.ffn_w2, W.ffn_s2, W.ffn_w2_i_s),
        W.linear_attn_qkvz_w: (W.linear_attn_qkvz_w, W.linear_attn_qkvz_s, None),
        W.linear_attn_out_w: (W.linear_attn_out_w, W.linear_attn_out_s, None),
    }

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, ModelOptMixedFp4Config
        ):
            return False
        if src_weight_info.name not in cls.weight_map:
            return False
        # Routed experts are NVFP4 -> MixedFp4Weight handles them.
        if src_weight_info.name in (W.moe_w1, W.moe_w2):
            return False
        if _is_mixed_fp4_module(src_weight_info, quant_config):
            return False
        # `ignore` modules stay BF16 (plain load, no quant wrapper).
        if _is_excluded(src_weight_info, quant_config.exclude_modules):
            return False
        return True

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        kernel: WeightModule = None
        scale: WeightModule = None
        input_scale: WeightModule = None
        # piece layout metadata for the _postprocess rescale
        self._piece_kind: Optional[str] = None
        self._la_config = None
        self._w13_dims = None

        name = src_weight_info.name
        if name == W.attn_qkv_w:
            kernel, scale, input_scale = self._get_qkv(src_weight_info)
            self._piece_kind = "qkv"
        elif name == W.attn_gate_w:
            kernel, scale = self._get_gate(src_weight_info)
        elif name == W.attn_o_w:
            kernel, scale, input_scale = self._get_single(
                src_weight_info, W.attn_o_w, W.attn_o_s, W.attn_o_i_s
            )
        elif name in (W.ffn_w1, W.ffn_w3, W.ffn_w2):
            kernel, scale, input_scale = self._get_single(
                src_weight_info, name, self.weight_map[name][1], self.weight_map[name][2]
            )
        elif name == W.ffn_w13:
            kernel, scale, input_scale = self._get_w13(src_weight_info)
            self._piece_kind = "w13"
        elif name == W.linear_attn_qkvz_w:
            kernel, scale = self._get_qkvz(src_weight_info)
            self._piece_kind = "qkvz"
            self._la_config = src_weight_info.config
        elif name == W.linear_attn_out_w:
            kernel, scale = self._get_linear_attn_out(src_weight_info)
        else:
            raise ValueError(f"Unsupported weight name {name}")

        sub_weights = {kernel.name: kernel}
        if scale is not None:
            sub_weights[scale.name] = scale
        if input_scale is not None:
            sub_weights[input_scale.name] = input_scale
        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = sub_weights[kernel.name]
        self.scale = sub_weights.get(scale.name) if scale is not None else None
        self.input_scale = (
            sub_weights.get(input_scale.name) if input_scale is not None else None
        )

    # ---- sub-weight construction helpers -------------------------------

    def _base_names(self, src_weight_info) -> List[str]:
        return [w.name[: -len(W_SUFFIX)] for w in src_weight_info.weights]

    def _get_qkv(self, src_weight_info: AttnAtomicWeight):
        weights = src_weight_info.weights
        assert len(weights) == 3, f"qkv expects 3 pieces, got {len(weights)}"
        base = self._base_names(src_weight_info)
        kernel = MixedModelOptFp8AtomicWeight(
            W.attn_qkv_w,
            # preserve merge_funs (q uses split_q_gate part=0 in Qwen3.5)
            [CkptWeightInfo(w.name, w.merge_fun) for w in weights],
            merge_te_qkv,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = MixedModelOptFp8AtomicWeight(
            W.attn_qkv_s,
            [CkptWeightInfo(n + QS_SUFFIX, identity) for n in base],
            _cat_scales,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = MixedModelOptFp8AtomicWeight(
            W.attn_qkv_i_s,
            [CkptWeightInfo(base[0] + ACT_S_SUFFIX, identity)],
            get_tensor_from_scalar,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return kernel, scale, input_scale

    def _get_gate(self, src_weight_info: AttnAtomicWeight):
        weights = src_weight_info.weights
        assert len(weights) == 1
        base = weights[0].name[: -len(W_SUFFIX)]
        # gate = split_q_gate(part=1) rows of q_proj (merge_fun preserved)
        kernel = MixedModelOptFp8AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo(weights[0].name, weights[0].merge_fun)],
            identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        # q and gate share q_proj's scalar weight scale (no row split).
        scale = MixedModelOptFp8AtomicWeight(
            W.attn_gate_s,
            [CkptWeightInfo(base + QS_SUFFIX, identity)],
            get_tensor_from_scalar,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return kernel, scale

    def _get_single(
        self,
        src_weight_info,
        kernel_key: str,
        scale_key: str,
        input_scale_key: Optional[str],
    ):
        weights = src_weight_info.weights
        assert len(weights) == 1
        base = weights[0].name[: -len(W_SUFFIX)]
        kernel = MixedModelOptFp8AtomicWeight(
            kernel_key,
            [CkptWeightInfo(weights[0].name, weights[0].merge_fun)],
            identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = MixedModelOptFp8AtomicWeight(
            scale_key,
            [CkptWeightInfo(base + QS_SUFFIX, identity)],
            get_tensor_from_scalar,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = None
        if input_scale_key is not None:
            input_scale = MixedModelOptFp8AtomicWeight(
                input_scale_key,
                [CkptWeightInfo(base + ACT_S_SUFFIX, identity)],
                get_tensor_from_scalar,
                data_type=torch.float32,
                config=src_weight_info.config,
            )
        return kernel, scale, input_scale

    def _get_w13(self, src_weight_info: FfnAtomicWeight):
        weights = src_weight_info.weights
        assert len(weights) == 2, f"w13 expects w1+w3 pieces, got {len(weights)}"
        base = self._base_names(src_weight_info)
        align = getattr(src_weight_info.config, "align_size", 0) or 0
        inter = getattr(src_weight_info.config, "inter_size", 0)
        self._w13_dims = (inter, inter, align)

        def merge_w13(ts: List[torch.Tensor]) -> torch.Tensor:
            return pad_w13(ts, align_size=align, dim=0)

        kernel = MixedModelOptFp8AtomicWeight(
            W.ffn_w13,
            [CkptWeightInfo(w.name, w.merge_fun) for w in weights],
            merge_w13,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = MixedModelOptFp8AtomicWeight(
            W.ffn_s13,
            [CkptWeightInfo(n + QS_SUFFIX, identity) for n in base],
            _cat_scales,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        # w1/w3 input scales are equal in modelopt checkpoints; take max.
        input_scale = MixedModelOptFp8AtomicWeight(
            W.ffn_w13_i_s,
            [CkptWeightInfo(n + ACT_S_SUFFIX, identity) for n in base],
            _max_scales,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return kernel, scale, input_scale

    def _get_qkvz(self, src_weight_info):
        weights = src_weight_info.weights
        assert len(weights) == 2, f"qkvz expects qkv+z pieces, got {len(weights)}"

        def merge_qkv_z(ts: List[torch.Tensor]) -> torch.Tensor:
            return torch.cat([ts[0], ts[1]], dim=0)

        base = self._base_names(src_weight_info)
        # Reuse the per-block linear-attn class: proven qkvz head splits.
        kernel = W8A8Fp8PerBlockLinearAttnAtomicWeight(
            W.linear_attn_qkvz_w,
            [CkptWeightInfo(w.name, w.merge_fun) for w in weights],
            merge_qkv_z,
            src_weight_info.config,
            torch.float8_e4m3fn,
        )
        # Scalar per-piece scales replicate (no TP split; postprocess
        # rescales to a single max scalar).
        scale = _ReplicateScalarWeight(
            W.linear_attn_qkvz_s,
            [CkptWeightInfo(n + QS_SUFFIX, identity) for n in base],
            _cat_scales,
            data_type=torch.float32,
        )
        return kernel, scale

    def _get_linear_attn_out(self, src_weight_info):
        weights = src_weight_info.weights
        assert len(weights) == 1
        base = weights[0].name[: -len(W_SUFFIX)]
        kernel = W8A8Fp8PerBlockLinearAttnAtomicWeight(
            W.linear_attn_out_w,
            [CkptWeightInfo(weights[0].name, weights[0].merge_fun)],
            identity,
            src_weight_info.config,
            torch.float8_e4m3fn,
        )
        scale = _ReplicateScalarWeight(
            W.linear_attn_out_s,
            [CkptWeightInfo(base + QS_SUFFIX, identity)],
            get_tensor_from_scalar,
            data_type=torch.float32,
        )
        return kernel, scale

    # ---- postprocess: rescale fused pieces to a shared max scale --------

    def _piece_boundaries(self, load_config: LoadConfig, n_rows: int) -> List[int]:
        if self._piece_kind == "qkv":
            # Same head math as Fp8PerTensorCompressedWeight._postprocess.
            head_size = load_config.size_per_head
            _kv_tp = math.gcd(load_config.head_num_kv, load_config.tp_size)
            head_num_kv = load_config.head_num_kv // _kv_tp
            head_num_q = load_config.head_num // load_config.tp_size
            return [
                head_num_q * head_size,
                head_num_kv * head_size,
                head_num_kv * head_size,
            ]
        if self._piece_kind == "qkvz":
            cfg = self._la_config
            n_qkv = (
                2 * cfg.linear_key_head_dim * cfg.linear_num_key_heads
                + cfg.linear_value_head_dim * cfg.linear_num_value_heads
            )
            n_z = cfg.linear_value_head_dim * cfg.linear_num_value_heads
            total = n_qkv + n_z
            frac = n_rows / total
            first = round(n_qkv * frac)
            return [first, n_rows - first]
        if self._piece_kind == "w13":
            # w13 = [up; gate] halves after pad/merge + TP row-split; the
            # per-rank row count is authoritative (inter_size is not always
            # present on the weight config).
            first = n_rows // 2
            return [first, n_rows - first]
        return [n_rows]

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel = processed_res.get(self.kernel.name)
        if kernel is not None and self.scale is not None:
            scale = processed_res[self.scale.name]
            if kernel.dim() == 2 and scale.numel() > 1:
                boundaries = self._piece_boundaries(
                    load_config, kernel.shape[0]
                )
                assert sum(boundaries) == kernel.shape[0], (
                    f"{self.kernel.name}: piece boundaries {boundaries} "
                    f"sum {sum(boundaries)} != kernel rows {kernel.shape[0]}"
                )
                processed_res[self.kernel.name] = rescale_pieces_to_max(
                    kernel, scale, boundaries
                )
                processed_res[self.scale.name] = (
                    scale.reshape(-1).max().reshape(1, 1)
                )
            else:
                processed_res[self.scale.name] = scale.reshape(1, 1)
        if self.input_scale is not None:
            iscale = processed_res.get(self.input_scale.name)
            if iscale is not None:
                processed_res[self.input_scale.name] = iscale.reshape(1, 1)
        return processed_res
