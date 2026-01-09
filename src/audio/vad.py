"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from src.core.config import VADConfig
from src.core.types import AudioChunk

logger = structlog.get_logger()

# Silero VAD expects 16kHz audio
SILERO_SAMPLE_RATE = 16000


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD.

    Silero VAD is a lightweight, accurate VAD model that runs efficiently
    on CPU. It's used to detect speech segments in audio streams.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize Silero VAD.

        Args:
            config: VAD configuration, uses defaults if not provided
        """
        self.config = config or VADConfig()
        self._model = None
        self._utils = None

    def _ensure_model_loaded(self):
        """Lazy load the Silero VAD model."""
        if self._model is None:
            try:
                import torch

                # Load Silero VAD from torch hub
                self._model, self._utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=True,  # Use ONNX for better CPU performance
                )
                logger.info("Silero VAD model loaded")
            except Exception as e:
                logger.error("Failed to load Silero VAD", error=str(e))
                raise

    def is_speech(self, audio: np.ndarray, sample_rate: int = SILERO_SAMPLE_RATE) -> bool:
        """
        Check if an audio segment contains speech.

        Args:
            audio: Audio samples (float32, mono)
            sample_rate: Sample rate of the audio

        Returns:
            True if speech is detected, False otherwise
        """
        self._ensure_model_loaded()

        import torch

        # Resample if needed
        if sample_rate != SILERO_SAMPLE_RATE:
            ratio = SILERO_SAMPLE_RATE / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)

        # Get speech probability
        speech_prob = self._model(audio_tensor, SILERO_SAMPLE_RATE).item()

        return speech_prob > self.config.threshold

    def get_speech_probability(
        self, audio: np.ndarray, sample_rate: int = SILERO_SAMPLE_RATE
    ) -> float:
        """
        Get the probability that an audio segment contains speech.

        Args:
            audio: Audio samples (float32, mono)
            sample_rate: Sample rate of the audio

        Returns:
            Speech probability (0.0 to 1.0)
        """
        self._ensure_model_loaded()

        import torch

        # Resample if needed
        if sample_rate != SILERO_SAMPLE_RATE:
            ratio = SILERO_SAMPLE_RATE / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        audio_tensor = torch.from_numpy(audio)
        return self._model(audio_tensor, SILERO_SAMPLE_RATE).item()

    async def filter_speech(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        """
        Filter an audio stream to only yield chunks containing speech.

        Implements buffering to handle minimum speech/silence durations.

        Args:
            audio_stream: Async iterator of AudioChunk objects

        Yields:
            AudioChunk objects that contain speech
        """
        speech_buffer: list[AudioChunk] = []
        silence_counter_ms = 0
        speech_counter_ms = 0
        in_speech = False

        async for chunk in audio_stream:
            chunk_duration_ms = int(chunk.duration_seconds * 1000)
            is_speech = self.is_speech(chunk.data, chunk.sample_rate)

            if is_speech:
                speech_counter_ms += chunk_duration_ms
                silence_counter_ms = 0
                speech_buffer.append(chunk)

                # Start speech segment if enough speech accumulated
                if not in_speech and speech_counter_ms >= self.config.min_speech_duration_ms:
                    in_speech = True
                    # Yield buffered speech
                    for buffered_chunk in speech_buffer:
                        yield buffered_chunk
                    speech_buffer = []

                elif in_speech:
                    yield chunk

            else:
                silence_counter_ms += chunk_duration_ms
                speech_counter_ms = 0

                if in_speech:
                    # Add padding during silence
                    if silence_counter_ms <= self.config.padding_ms:
                        yield chunk
                    # End speech segment if enough silence
                    elif silence_counter_ms >= self.config.min_silence_duration_ms:
                        in_speech = False
                        speech_buffer = []
                else:
                    # Not in speech, clear buffer if it gets too long
                    if len(speech_buffer) > 50:  # ~5 seconds at 100ms chunks
                        speech_buffer = speech_buffer[-10:]

    def reset(self):
        """Reset the VAD state (for starting a new utterance)."""
        if self._model is not None:
            self._model.reset_states()


class SimpleVAD:
    """
    Simple energy-based VAD as fallback when Silero is not available.

    Uses RMS energy threshold to detect speech. Less accurate than Silero
    but has no dependencies.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        # Energy threshold - can be tuned
        self.energy_threshold = 0.01

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio contains speech based on energy."""
        rms = np.sqrt(np.mean(audio**2))
        return rms > self.energy_threshold

    def get_speech_probability(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Get pseudo-probability based on energy level."""
        rms = np.sqrt(np.mean(audio**2))
        # Normalize to 0-1 range (assuming max RMS of ~0.5 for speech)
        return min(1.0, rms / 0.1)

    async def filter_speech(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Filter stream to only speech chunks."""
        async for chunk in audio_stream:
            if self.is_speech(chunk.data, chunk.sample_rate):
                yield chunk

    def reset(self):
        """No state to reset."""
        pass


def create_vad(config: Optional[VADConfig] = None, use_silero: bool = True):
    """
    Create a VAD instance.

    Args:
        config: VAD configuration
        use_silero: Whether to use Silero VAD (falls back to SimpleVAD if unavailable)

    Returns:
        VAD instance (SileroVAD or SimpleVAD)
    """
    if use_silero:
        try:
            vad = SileroVAD(config)
            vad._ensure_model_loaded()
            return vad
        except Exception as e:
            logger.warning("Silero VAD unavailable, using SimpleVAD", error=str(e))

    return SimpleVAD(config)
