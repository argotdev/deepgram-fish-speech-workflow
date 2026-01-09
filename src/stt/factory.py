"""Factory for creating STT providers."""

from typing import TYPE_CHECKING

from src.core.config import VoiceLoopConfig
from src.core.protocols import STTProvider

if TYPE_CHECKING:
    pass


def create_stt_provider(config: VoiceLoopConfig) -> STTProvider:
    """
    Create an STT provider based on configuration.

    Args:
        config: Voice loop configuration

    Returns:
        Configured STT provider instance

    Raises:
        ValueError: If provider type is not supported
    """
    if config.stt_provider == "deepgram":
        from .deepgram_provider import DeepgramSTT

        return DeepgramSTT(config.deepgram)

    elif config.stt_provider == "whisper":
        from .whisper_local import LocalWhisperSTT

        return LocalWhisperSTT(config.whisper)

    else:
        raise ValueError(f"Unknown STT provider: {config.stt_provider}")
