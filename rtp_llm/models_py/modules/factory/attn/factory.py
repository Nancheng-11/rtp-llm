from typing import Callable, Dict, Type

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.attention import CausalAttention, MlaAttention


class AttentionFactory:
    """Factory class for creating attention modules based on key_str."""

    # Attention type registry - maps key_str to attention class and creation function
    ATTENTION_REGISTRY: Dict[str, Dict[str, any]] = {
        "causal": {
            "class": CausalAttention,
            "create_func": lambda config, weights, layer_idx: CausalAttention(
                config, weights
            ),
        },
        "mla": {
            "class": MlaAttention,
            "create_func": lambda config, weights, layer_idx: MlaAttention(
                config, weights, layer_idx
            ),
        },
    }

    @classmethod
    def create_attention(
        cls,
        key_str: str,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = 0,
    ) -> nn.Module:
        """
        Create attention module based on key_str.

        Args:
            key_str: String key identifying the attention type
            config: Model configuration
            weights: Model weights
            layer_idx: Layer index (used by some attention types)

        Returns:
        Attention module instance

        Raises:
            ValueError: If key_str is not supported
        """
        if key_str not in cls.ATTENTION_REGISTRY:
            available_types = list(cls.ATTENTION_REGISTRY.keys())
            raise ValueError(
                f"Unsupported attention type '{key_str}'. Available types: {available_types}"
            )

        attention_info = cls.ATTENTION_REGISTRY[key_str]
        attention_class = attention_info["class"]

        if attention_class is None:
            raise ImportError(
                f"Attention class for '{key_str}' is not available. Please check imports."
            )

        create_func = attention_info["create_func"]
        return create_func(config, weights, layer_idx)

    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported attention types."""
        return list(cls.ATTENTION_REGISTRY.keys())

    @classmethod
    def register_attention_type(
        cls, key_str: str, attention_class: Type[nn.Module], create_func: Callable
    ):
        """
        Register a new attention type.

        Args:
            key_str: String key for the attention type
            attention_class: Attention module class
            create_func: Function to create the attention instance
        """
        cls.ATTENTION_REGISTRY[key_str] = {
            "class": attention_class,
            "create_func": create_func,
        }


class FMHAImplFactory:
    """Factory class for creating FMHA implementations based on attention_type."""

    # FMHA implementation registry - maps attention_type to impl method
    FMHA_IMPL_REGISTRY: Dict[str, str] = {
        "causal": "get_fmha_impl",
        "mla": "get_mla_impl",
    }

    @classmethod
    def get_fmha_impl_method(cls, attention_type: str) -> str:
        """
        Get the appropriate FMHA implementation method based on attention_type.

        Args:
            attention_type: String identifying the attention type

        Returns:
            Method name to call for getting FMHA implementation

        Raises:
            ValueError: If attention_type is not supported
        """
        if attention_type not in cls.FMHA_IMPL_REGISTRY:
            available_types = list(cls.FMHA_IMPL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported attention type '{attention_type}'. Available types: {available_types}"
            )

        return cls.FMHA_IMPL_REGISTRY[attention_type]

    @classmethod
    def register_fmha_impl(cls, attention_type: str, impl_method: str):
        """
        Register a new FMHA implementation method for an attention type.

        Args:
            attention_type: String key for the attention type
            impl_method: Method name to call for getting FMHA implementation
        """
        cls.FMHA_IMPL_REGISTRY[attention_type] = impl_method

    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported attention types."""
        return list(cls.FMHA_IMPL_REGISTRY.keys())


__all__ = ["AttentionFactory", "FMHAImplFactory"]
