"""Factory for creating LLM providers."""

from typing import Optional

from src.core.config import VoiceLoopConfig
from src.core.protocols import LLMProvider


def create_llm_provider(config: VoiceLoopConfig) -> Optional[LLMProvider]:
    """
    Create an LLM provider based on configuration.

    Args:
        config: Voice loop configuration

    Returns:
        Configured LLM provider instance, or None if no LLM configured

    Raises:
        ValueError: If provider type is not supported
    """
    if config.llm.provider is None:
        return None

    if config.llm.provider == "openai":
        from .openai_provider import OpenAILLM

        return OpenAILLM(config.llm)

    elif config.llm.provider == "anthropic":
        from .anthropic_provider import AnthropicLLM

        return AnthropicLLM(config.llm)

    elif config.llm.provider == "local":
        from .local_provider import LocalLLM

        return LocalLLM(config.llm)

    else:
        raise ValueError(f"Unknown LLM provider: {config.llm.provider}")
