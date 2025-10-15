from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.mla import DeepSeekV2Attention
from rtp_llm.models_py.modules.ep.layers import FusedMoE
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

class DeepSeekV2MoeLayer(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.gate = LinearFactory.create_linear_from_weights(
                weights, W.moe_gate, None, None, config
            )
        self.fused_moe: FusedMoE = FusedMoE(config, weights, layer_id=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for FusedMoE implementation."""

        router_logits = self.gate(hidden_states)

        return self.fused_moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

class DeepSeekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.self_attn = DeepSeekV2Attention(config, weights, layer_idx)

        if len(config.moe_layer_index) > 0 and layer_idx < config.moe_layer_index[0]:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = DeepSeekV2MoeLayer(config, weights)
        self.add_shared_expert = config.moe_style == 2

        if self.add_shared_expert:
            self.shared_mlp = FusedSiluActDenseMLP(config, weights)

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_dense_layer:
            hidden_states = self.shared_mlp(hidden_states)
        else:
            experts_output = self.moe_mlp(hidden_states)
            if self.add_shared_expert:
                shared_mlp_output = self.shared_mlp(hidden_states)
                hidden_states = experts_output + shared_mlp_output
            else:
                hidden_states = experts_output
        hidden_states = residual + hidden_states

        return hidden_states


class DeepSeekV2Model(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.layer_num = config.layer_num
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))

        self.layers = nn.ModuleList(
            [
                DeepSeekV2DecoderLayer(
                    config,
                    weights.weights[idx],
                    idx,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        attention_inputs: PyAttentionInputs = inputs.attention_inputs

        fmha_impl = self.get_mla_impl(attention_inputs)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)

        return PyModelOutputs(hidden_states)


__all__ = [
    "DeepSeekV2Model",
]
