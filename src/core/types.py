"""Core data types for the voice loop pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class WordInfo:
    """Word-level timing information from STT."""

    word: str
    start_time: float  # seconds from utterance start
    end_time: float
    confidence: float = 1.0


@dataclass
class AudioChunk:
    """A chunk of audio data flowing through the pipeline."""

    data: NDArray[np.float32]  # Audio samples (float32, mono, -1.0 to 1.0)
    sample_rate: int  # Usually 16000 for STT, 44100 for TTS output
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def duration_seconds(self) -> float:
        """Duration of this audio chunk in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Number of audio samples in this chunk."""
        return len(self.data)

    def resample(self, target_rate: int) -> "AudioChunk":
        """Resample audio to a different sample rate."""
        if self.sample_rate == target_rate:
            return self

        # Simple linear interpolation resampling
        ratio = target_rate / self.sample_rate
        new_length = int(len(self.data) * ratio)
        indices = np.linspace(0, len(self.data) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(self.data)), self.data)

        return AudioChunk(
            data=resampled.astype(np.float32),
            sample_rate=target_rate,
            timestamp=self.timestamp,
        )

    def to_int16(self) -> NDArray[np.int16]:
        """Convert float32 audio to int16 for playback/saving."""
        return (self.data * 32767).astype(np.int16)

    @classmethod
    def from_int16(
        cls, data: NDArray[np.int16], sample_rate: int, timestamp: Optional[datetime] = None
    ) -> "AudioChunk":
        """Create AudioChunk from int16 audio data."""
        float_data = data.astype(np.float32) / 32767.0
        return cls(
            data=float_data,
            sample_rate=sample_rate,
            timestamp=timestamp or datetime.now(),
        )


@dataclass
class TranscriptResult:
    """Result from speech-to-text transcription."""

    text: str
    is_final: bool  # True for final transcript, False for interim
    confidence: float = 1.0
    speaker_id: Optional[str] = None  # For diarization
    words: list[WordInfo] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the utterance based on word timings."""
        if not self.words:
            return None
        return self.words[-1].end_time - self.words[0].start_time


@dataclass
class TTSRequest:
    """Request for text-to-speech synthesis."""

    text: str
    voice_id: str = "default"
    emotion: Optional[str] = None  # Fish Speech emotion control (e.g., "happy", "sad", "neutral")
    speed: float = 1.0  # Speech rate multiplier
    pitch: float = 1.0  # Pitch multiplier (if supported)


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""

    audio_data: NDArray[np.float32]  # Audio samples (float32, mono)
    sample_rate: int
    request: TTSRequest  # Original request for reference

    @property
    def duration_seconds(self) -> float:
        """Duration of the synthesized audio in seconds."""
        return len(self.audio_data) / self.sample_rate

    def to_chunk(self) -> AudioChunk:
        """Convert to AudioChunk for playback."""
        return AudioChunk(
            data=self.audio_data,
            sample_rate=self.sample_rate,
        )


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
