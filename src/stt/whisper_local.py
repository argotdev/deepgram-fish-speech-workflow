"""Local Whisper STT using faster-whisper (CTranslate2)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Optional

import numpy as np
import structlog

from src.core.config import WhisperConfig
from src.core.protocols import STTProvider
from src.core.types import AudioChunk, TranscriptResult, WordInfo

logger = structlog.get_logger()


class LocalWhisperSTT(STTProvider):
    """
    Offline STT using faster-whisper (CTranslate2 optimized).

    Features:
    - Fully offline operation
    - Multiple model sizes (tiny to large-v3)
    - Word-level timestamps
    - Automatic language detection
    - GPU acceleration when available
    """

    def __init__(self, config: Optional[WhisperConfig] = None):
        """
        Initialize local Whisper STT.

        Args:
            config: Whisper configuration, uses defaults if not provided
        """
        self.config = config or WhisperConfig()
        self._model = None
        self._sample_rate = 16000  # Whisper requires 16kHz

    def _ensure_model_loaded(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            # Determine device and compute type
            device = self.config.device
            compute_type = self.config.compute_type

            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"

            logger.info(
                "Loading Whisper model",
                model_size=self.config.model_size,
                device=device,
                compute_type=compute_type,
            )

            self._model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=compute_type,
            )

            logger.info("Whisper model loaded")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptResult]:
        """
        Transcribe a stream of audio chunks.

        Since Whisper processes complete utterances, this buffers audio
        and transcribes when silence is detected or buffer is full.

        Args:
            audio_stream: Async iterator of AudioChunk objects

        Yields:
            TranscriptResult objects with transcribed text
        """
        self._ensure_model_loaded()

        # Buffer to accumulate audio
        audio_buffer: list[np.ndarray] = []
        buffer_duration = 0.0
        max_buffer_duration = 30.0  # Max seconds before forced transcription
        silence_threshold = 0.01  # RMS threshold for silence detection
        silence_duration = 0.0
        silence_trigger = 0.8  # Seconds of silence to trigger transcription

        async for chunk in audio_stream:
            # Resample if needed
            if chunk.sample_rate != self._sample_rate:
                resampled = chunk.resample(self._sample_rate)
                audio_data = resampled.data
            else:
                audio_data = chunk.data

            audio_buffer.append(audio_data)
            buffer_duration += len(audio_data) / self._sample_rate

            # Check for silence
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < silence_threshold:
                silence_duration += len(audio_data) / self._sample_rate
            else:
                silence_duration = 0.0

            # Transcribe if we have enough audio and silence, or buffer is full
            should_transcribe = (
                buffer_duration > 1.0 and silence_duration > silence_trigger
            ) or buffer_duration >= max_buffer_duration

            if should_transcribe and audio_buffer:
                # Combine buffer
                combined = np.concatenate(audio_buffer)

                # Transcribe
                result = await self._transcribe_audio(combined)

                if result.text.strip():
                    yield result

                # Reset buffer
                audio_buffer = []
                buffer_duration = 0.0
                silence_duration = 0.0

        # Transcribe any remaining audio
        if audio_buffer:
            combined = np.concatenate(audio_buffer)
            if len(combined) > self._sample_rate * 0.5:  # At least 0.5s
                result = await self._transcribe_audio(combined)
                if result.text.strip():
                    yield result

    async def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptResult:
        """
        Transcribe a single audio chunk.

        Args:
            chunk: AudioChunk containing the audio to transcribe

        Returns:
            TranscriptResult with the transcribed text
        """
        self._ensure_model_loaded()

        # Resample if needed
        if chunk.sample_rate != self._sample_rate:
            resampled = chunk.resample(self._sample_rate)
            audio_data = resampled.data
        else:
            audio_data = chunk.data

        return await self._transcribe_audio(audio_data)

    async def _transcribe_audio(self, audio: np.ndarray) -> TranscriptResult:
        """
        Internal method to transcribe audio array.

        Args:
            audio: Audio samples (float32, mono, 16kHz)

        Returns:
            TranscriptResult with transcribed text
        """
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def do_transcribe():
            segments, info = self._model.transcribe(
                audio,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
            )
            return list(segments), info

        segments, info = await loop.run_in_executor(None, do_transcribe)

        # Combine segments
        text_parts = []
        all_words = []

        for segment in segments:
            text_parts.append(segment.text)

            if segment.words:
                for word in segment.words:
                    all_words.append(
                        WordInfo(
                            word=word.word,
                            start_time=word.start,
                            end_time=word.end,
                            confidence=word.probability,
                        )
                    )

        text = " ".join(text_parts).strip()

        # Calculate average confidence
        avg_confidence = 1.0
        if all_words:
            avg_confidence = sum(w.confidence for w in all_words) / len(all_words)

        logger.debug(
            "Whisper transcription complete",
            text_length=len(text),
            duration=len(audio) / self._sample_rate,
            language=info.language,
            confidence=avg_confidence,
        )

        return TranscriptResult(
            text=text,
            is_final=True,
            confidence=avg_confidence,
            words=all_words,
            timestamp=datetime.now(),
        )

    async def close(self) -> None:
        """Clean up resources."""
        self._model = None
        logger.debug("Whisper STT closed")
