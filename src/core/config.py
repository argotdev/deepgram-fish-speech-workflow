"""Configuration management using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file at module import time so all configs can access env vars
load_dotenv()


class DeepgramConfig(BaseSettings):
    """Deepgram STT configuration."""

    model_config = SettingsConfigDict(env_prefix="DEEPGRAM_")

    api_key: Optional[str] = None
    model: str = "nova-3"  # Nova-3: 54% lower WER than competitors
    language: str = "en-US"
    smart_format: bool = True
    interim_results: bool = True
    utterance_end_ms: int = 1000
    vad_events: bool = True
    diarize: bool = False


class WhisperConfig(BaseSettings):
    """Local Whisper STT configuration."""

    model_config = SettingsConfigDict(env_prefix="WHISPER_")

    model_size: str = "base"  # tiny, base, small, medium, large-v3
    device: str = "auto"  # auto, cpu, cuda
    compute_type: str = "auto"  # auto, float16, int8


class FishSpeechConfig(BaseSettings):
    """Fish Speech TTS configuration (local or cloud API)."""

    model_config = SettingsConfigDict(env_prefix="FISH_SPEECH_")

    # Cloud API settings (fish-audio-sdk, Python 3.9+)
    api_key: Optional[str] = None  # Fish Audio API key
    api_base_url: str = "https://api.fish.audio"
    reference_id: Optional[str] = None  # Voice reference ID from Fish Audio

    # Local model settings (requires Python 3.12+)
    model_path: Optional[Path] = None  # Path to Fish Speech model
    voice_id: str = "default"
    default_emotion: Optional[str] = None
    default_speed: float = 1.0
    device: str = "auto"  # auto, cpu, cuda


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: Optional[Literal["openai", "anthropic", "local"]] = None
    model: Optional[str] = None  # e.g., "gpt-4o", "claude-3-opus", "llama3"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For local LLM servers
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."


class AudioConfig(BaseSettings):
    """Audio I/O configuration."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_")

    input_sample_rate: int = 16000  # Standard for STT
    output_sample_rate: int = 44100  # Standard for playback
    input_channels: int = 1  # Mono
    output_channels: int = 1  # Mono
    chunk_duration_ms: int = 32  # Audio chunk size in ms (512 samples @ 16kHz for Deepgram)
    input_device: Optional[int] = None  # None = default device
    output_device: Optional[int] = None  # None = default device


class VADConfig(BaseSettings):
    """Voice Activity Detection configuration."""

    model_config = SettingsConfigDict(env_prefix="VAD_")

    enabled: bool = True
    threshold: float = 0.5  # Speech probability threshold
    min_speech_duration_ms: int = 250  # Minimum speech segment
    min_silence_duration_ms: int = 500  # Silence to end utterance
    padding_ms: int = 100  # Padding around speech segments


class VoiceLoopConfig(BaseSettings):
    """Main voice loop configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VOICE_LOOP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mode selection
    mode: Literal["online", "offline", "hybrid"] = "hybrid"

    # Provider selection
    stt_provider: Literal["deepgram", "whisper"] = "deepgram"
    tts_provider: Literal["fish_speech", "fish_audio_api", "pyttsx3", "edge_tts"] = "fish_audio_api"

    # Sub-configurations
    deepgram: DeepgramConfig = Field(default_factory=DeepgramConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    fish_speech: FishSpeechConfig = Field(default_factory=FishSpeechConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)

    # Pipeline settings
    streaming_mode: bool = True  # Enable streaming TTS for lower latency
    interruption_enabled: bool = True  # Allow user to interrupt TTS
    max_audio_buffer_seconds: float = 30.0  # Max audio buffer size
    echo_mode: bool = False  # Echo user speech without LLM (for testing)

    @model_validator(mode="after")
    def validate_config(self) -> "VoiceLoopConfig":
        """Validate configuration consistency."""
        # In online mode, Deepgram requires API key
        if self.mode == "online" and self.stt_provider == "deepgram":
            if not self.deepgram.api_key:
                raise ValueError("Deepgram API key required for online mode")

        # In offline mode, force whisper STT
        if self.mode == "offline" and self.stt_provider == "deepgram":
            self.stt_provider = "whisper"

        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "VoiceLoopConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def is_online(self) -> bool:
        """Check if we should use online services."""
        return self.mode in ("online", "hybrid")

    def is_offline(self) -> bool:
        """Check if we should use offline/local services."""
        return self.mode in ("offline", "hybrid")
