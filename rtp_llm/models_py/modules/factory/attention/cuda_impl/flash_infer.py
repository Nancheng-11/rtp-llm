import logging
from typing import Any

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FlashInferDecodeOp,
    FlashInferPrefillOp,
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
    PyAttentionInputs,
)


class FlashInferPrefillImpl(FMHAPrefillImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferPrefillOp(config.gpt_init_params),
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )
        self.support_ = self.support_ and (config.use_mla == False)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True


class FlashInferDecodeImpl(FMHADecodeImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferDecodeOp(config.gpt_init_params),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )
        self.seq_size_per_block = config.seq_size_per_block
        self.support_ = self.support_ and (config.use_mla == False)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True

    def prepare_replay(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        batch_size = attn_inputs.input_lengths.size(0)
        self.fmha_params.fill_params(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            batch_size,
            self.seq_size_per_block,
        )

        assert self.rope_kvcache_impl is not None
        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)
