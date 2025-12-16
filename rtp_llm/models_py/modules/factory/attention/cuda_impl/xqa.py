import logging
from typing import Any

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    PyAttentionInputs,
    XQAAttnOp,
)


class XQAImpl(FMHADecodeImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            XQAAttnOp(config.gpt_init_params),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.XQA

    def support_cuda_graph(self) -> bool:
        return True

    def prepare_replay(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        new_offset = new_fmha_params.kv_cache_offset
        old_offset = self.fmha_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)

        assert self.rope_kvcache_impl is not None
        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)
