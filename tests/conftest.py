"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import AudioChunk, TranscriptResult, TTSRequest


@pytest.fixture
def sample_audio_chunk() -> AudioChunk:
    """Create a sample audio chunk for testing."""
    # 1 second of 440Hz sine wave at 16kHz
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)

    return AudioChunk(data=data, sample_rate=sample_rate)


@pytest.fixture
def sample_transcript() -> TranscriptResult:
    """Create a sample transcript for testing."""
    return TranscriptResult(
        text="Hello, this is a test.",
        is_final=True,
        confidence=0.95,
    )


@pytest.fixture
def sample_tts_request() -> TTSRequest:
    """Create a sample TTS request for testing."""
    return TTSRequest(
        text="Hello, world!",
        voice_id="default",
        emotion="neutral",
        speed=1.0,
    )


def generate_test_audio(text: str, duration: float = 2.0, sample_rate: int = 16000) -> AudioChunk:
    """
    Generate test audio data.

    In a real implementation, this could use a TTS to generate actual speech.
    For testing, we generate a simple tone.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Generate a simple tone
    data = 0.3 * np.sin(2 * np.pi * 440 * t)

    return AudioChunk(data=data, sample_rate=sample_rate)
