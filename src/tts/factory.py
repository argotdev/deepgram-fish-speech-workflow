"""Factory for creating TTS providers."""

from src.core.config import VoiceLoopConfig
from src.core.protocols import TTSProvider


def create_tts_provider(config: VoiceLoopConfig) -> TTSProvider:
    """
    Create a TTS provider based on configuration.

    Args:
        config: Voice loop configuration

    Returns:
        Configured TTS provider instance

    Raises:
        ValueError: If provider type is not supported
    """
    if config.tts_provider == "fish_speech":
        from .fish_speech import FishSpeechTTS

        return FishSpeechTTS(config.fish_speech)

    elif config.tts_provider == "fish_audio_api":
        from .fish_audio_api import FishAudioAPI

        return FishAudioAPI(config.fish_speech)

    elif config.tts_provider == "pyttsx3":
        from .pyttsx3_provider import Pyttsx3TTS

        return Pyttsx3TTS()

    elif config.tts_provider == "edge_tts":
        from .edge_tts_provider import EdgeTTS

        return EdgeTTS()

    else:
        raise ValueError(f"Unknown TTS provider: {config.tts_provider}")
