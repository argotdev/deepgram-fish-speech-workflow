"""pyttsx3 offline TTS provider - uses system speech engines."""

from __future__ import annotations

import asyncio
import io
import tempfile
import wave
from collections.abc import AsyncIterator
from typing import Optional

import numpy as np
import structlog

from src.core.protocols import TTSProvider
from src.core.types import AudioChunk, TTSRequest, TTSResult

logger = structlog.get_logger()


class Pyttsx3TTS(TTSProvider):
    """
    Offline TTS using pyttsx3 (system TTS engines).

    Uses:
    - macOS: NSSpeechSynthesizer
    - Windows: SAPI5
    - Linux: espeak

    Pros: Truly offline, no API keys, works on Python 3.9
    Cons: Lower quality than neural TTS
    """

    OUTPUT_SAMPLE_RATE = 22050  # Default for most system TTS

    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize pyttsx3 TTS.

        Args:
            rate: Speech rate in words per minute (default 150)
            volume: Volume from 0.0 to 1.0
        """
        self._engine = None
        self._rate = rate
        self._volume = volume
        self._lock = asyncio.Lock()

    def _ensure_engine(self):
        """Lazy initialize the pyttsx3 engine."""
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)

            # Get available voices
            voices = self._engine.getProperty("voices")
            if voices:
                logger.info(
                    "pyttsx3 initialized",
                    voices_available=len(voices),
                    default_voice=voices[0].name if voices else "unknown",
                )

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.OUTPUT_SAMPLE_RATE

    @property
    def supported_emotions(self) -> list[str]:
        """pyttsx3 doesn't support emotions."""
        return []

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            request: TTSRequest with text and parameters

        Returns:
            TTSResult with synthesized audio
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, self._synthesize_impl, request)

        return TTSResult(
            audio_data=audio,
            sample_rate=self.OUTPUT_SAMPLE_RATE,
            request=request,
        )

    def _synthesize_impl(self, request: TTSRequest) -> np.ndarray:
        """Synchronous synthesis implementation."""
        self._ensure_engine()

        text = request.text.strip()
        if not text:
            return np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.1), dtype=np.float32)

        # Adjust rate based on speed parameter
        base_rate = self._rate
        adjusted_rate = int(base_rate * request.speed)
        self._engine.setProperty("rate", adjusted_rate)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            self._engine.save_to_file(text, temp_path)
            self._engine.runAndWait()

            # Load the audio file
            with wave.open(temp_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()

                audio_bytes = wf.readframes(n_frames)

            # Convert to float32
            if sample_width == 2:
                audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
                audio = audio_int.astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio_int = np.frombuffer(audio_bytes, dtype=np.uint8)
                audio = (audio_int.astype(np.float32) - 128) / 128.0
            else:
                audio_int = np.frombuffer(audio_bytes, dtype=np.int32)
                audio = audio_int.astype(np.float32) / 2147483648.0

            # Convert to mono if stereo
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            logger.debug(
                "pyttsx3 synthesis complete",
                text_length=len(text),
                audio_duration=len(audio) / framerate,
            )

            return audio

        except Exception as e:
            logger.error("pyttsx3 synthesis failed", error=str(e))
            # Return silence on error
            return np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.5), dtype=np.float32)

        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        """
        Stream synthesis by splitting into sentences.

        Args:
            request: TTSRequest with text

        Yields:
            AudioChunk objects
        """
        import re

        sentences = re.split(r"[.!?]+\s*", request.text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_request = TTSRequest(
                text=sentence + ".",
                voice_id=request.voice_id,
                emotion=request.emotion,
                speed=request.speed,
                pitch=request.pitch,
            )

            result = await self.synthesize(sentence_request)

            yield AudioChunk(
                data=result.audio_data,
                sample_rate=result.sample_rate,
            )

    async def close(self) -> None:
        """Clean up resources."""
        if self._engine:
            self._engine.stop()
            self._engine = None
        logger.debug("pyttsx3 TTS closed")
