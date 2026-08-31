"""Minimal Qwen3 EAGLE3 draft model."""

from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import CausalAttention, DenseMLP, Embedding, RMSNorm
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3MoeEagle3Layer(nn.Module):
    """One EAGLE3 decoder layer with a pre-norm residual."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: dict[str, torch.Tensor],
        quant_config: Optional[object],
        hw_kernel_config: Optional[HWKernelConfig],
    ):
        super().__init__()
        self.self_attn = CausalAttention(
            config.getAttentionConfigs(parallelism_config.get_attn_tp_size()),
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            0,
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = hidden_states + residual
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3MoeEagle3Model(GptModelBase):
    """EAGLE3 encoder plus one draft decoder layer."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        moe_config: MoeConfig,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        layer_weights = weights.weights[0]
        self.input_norm = RMSNorm(
            layer_weights[W.eagle3_input_norm_gamma], eps=config.layernorm_eps
        )
        self.hidden_norm = RMSNorm(
            layer_weights[W.eagle3_fc_norm_gamma], eps=config.layernorm_eps
        )
        from rtp_llm.models_py.modules.factory import LinearFactory

        self.fc = LinearFactory.create_linear_from_weights(
            layer_weights, W.eagle3_fc_proj, quant_config=None, hw_kernel_config=None
        )
        self.layer = Qwen3MoeEagle3Layer(
            config,
            parallelism_config,
            layer_weights,
            quant_config=None,
            hw_kernel_config=py_hw_kernel_config,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_embeds = self.embed_tokens(inputs.input_ids)
        target_hidden = inputs.input_hiddens
        if target_hidden is None or target_hidden.numel() == 0:
            projected_hidden = input_embeds
        elif target_hidden.size(-1) == input_embeds.size(-1):
            # The first proposal consumes the target's three-layer EAGLE3
            # feature. Later proposal steps feed back the draft's own hidden.
            projected_hidden = target_hidden
        else:
            projected_hidden = self.fc(target_hidden)

        hidden_states = torch.cat(
            [self.input_norm(input_embeds), self.hidden_norm(projected_hidden)], dim=-1
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, 0)
        hidden_states = self.layer(
            hidden_states,
            projected_hidden,
            layer_fmha_impl,
            self.kv_cache.get_layer_cache(0) if self.kv_cache else None,
        )
        return PyModelOutputs(hidden_states)


__all__ = ["Qwen3MoeEagle3Model"]
