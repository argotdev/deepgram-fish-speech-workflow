"""Deepgram streaming STT provider (SDK 2.x)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Optional

import numpy as np
import structlog

from src.core.config import DeepgramConfig
from src.core.protocols import STTProvider
from src.core.types import AudioChunk, TranscriptResult, WordInfo

logger = structlog.get_logger()


class DeepgramSTT(STTProvider):
    """
    Deepgram Nova-2 streaming STT with 95%+ accuracy.

    Uses Deepgram SDK 2.x for Python 3.9 compatibility.

    Features:
    - Real-time streaming transcription via WebSocket
    - Interim and final results
    - Speaker diarization (100+ languages)
    - Smart formatting (punctuation, numerals)
    """

    def __init__(self, config: Optional[DeepgramConfig] = None):
        """
        Initialize Deepgram STT provider.

        Args:
            config: Deepgram configuration, uses defaults if not provided

        Raises:
            ValueError: If API key is not provided
        """
        self.config = config or DeepgramConfig()

        if not self.config.api_key:
            raise ValueError(
                "Deepgram API key required. Set DEEPGRAM_API_KEY environment variable."
            )

        self._client = None

    def _ensure_client(self):
        """Lazy initialize the Deepgram client."""
        if self._client is None:
            from deepgram import Deepgram

            self._client = Deepgram(self.config.api_key)
            logger.info("Deepgram client initialized (SDK 2.x)")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptResult]:
        """
        Transcribe a stream of audio chunks using Deepgram WebSocket API.

        Yields TranscriptResult objects as transcription progresses.
        Both interim (partial) and final results are yielded.

        Args:
            audio_stream: Async iterator of AudioChunk objects

        Yields:
            TranscriptResult objects with transcribed text
        """
        self._ensure_client()

        # Configure live transcription options
        options = {
            "model": self.config.model,
            "language": self.config.language,
            "smart_format": self.config.smart_format,
            "punctuate": True,  # Required for proper sentence detection
            "interim_results": self.config.interim_results,
            "utterance_end_ms": str(self.config.utterance_end_ms),
            "endpointing": str(self.config.utterance_end_ms),  # Backup endpoint detection
            "vad_events": self.config.vad_events,
            "diarize": self.config.diarize,
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
        }

        # Queue to pass results from callback to async iterator
        result_queue: asyncio.Queue[Optional[TranscriptResult]] = asyncio.Queue()

        # Track the latest interim transcript for utterance_end handling
        latest_interim: dict = {"text": "", "words": [], "confidence": 1.0, "speaker_id": None}

        async def on_message(result, **kwargs):
            """Handle transcription results."""
            try:
                # Handle dict or object responses
                if isinstance(result, dict):
                    channel = result.get("channel")
                else:
                    channel = getattr(result, "channel", None)

                if channel:
                    if isinstance(channel, dict):
                        alternatives = channel.get("alternatives", [])
                    else:
                        alternatives = getattr(channel, "alternatives", [])

                    if alternatives:
                        alt = alternatives[0]
                        if isinstance(alt, dict):
                            transcript_text = alt.get("transcript", "")
                        else:
                            transcript_text = getattr(alt, "transcript", "")

                        if transcript_text.strip():
                            # Extract word-level timing if available
                            words = []
                            alt_words = alt.get("words") if isinstance(alt, dict) else getattr(alt, "words", None)
                            if alt_words:
                                for word in alt_words:
                                    if isinstance(word, dict):
                                        words.append(
                                            WordInfo(
                                                word=word.get("word", ""),
                                                start_time=word.get("start", 0),
                                                end_time=word.get("end", 0),
                                                confidence=word.get("confidence", 1.0),
                                            )
                                        )
                                    else:
                                        words.append(
                                            WordInfo(
                                                word=getattr(word, "word", ""),
                                                start_time=getattr(word, "start", 0),
                                                end_time=getattr(word, "end", 0),
                                                confidence=getattr(word, "confidence", 1.0),
                                            )
                                        )

                            # Determine if this is a final result
                            if isinstance(result, dict):
                                is_final = result.get("is_final", False)
                                speech_final = result.get("speech_final", False)
                            else:
                                is_final = getattr(result, "is_final", False)
                                speech_final = getattr(result, "speech_final", False)

                            # Get speaker ID if diarization is enabled
                            speaker_id = None
                            alt_speaker = alt.get("speaker") if isinstance(alt, dict) else getattr(alt, "speaker", None)
                            if alt_speaker is not None:
                                speaker_id = str(alt_speaker)

                            # Store interim results for potential utterance_end finalization
                            if not is_final:
                                latest_interim["text"] = transcript_text
                                latest_interim["words"] = words
                                latest_interim["confidence"] = alt.get("confidence", 1.0) if isinstance(alt, dict) else getattr(alt, "confidence", 1.0)
                                latest_interim["speaker_id"] = speaker_id

                            # Treat speech_final OR is_final as final
                            actual_final = is_final or speech_final

                            transcript = TranscriptResult(
                                text=transcript_text,
                                is_final=actual_final,
                                confidence=alt.get("confidence", 1.0) if isinstance(alt, dict) else getattr(alt, "confidence", 1.0),
                                speaker_id=speaker_id,
                                words=words,
                                timestamp=datetime.now(),
                            )

                            await result_queue.put(transcript)

                            # Clear interim if we got a final
                            if actual_final:
                                latest_interim["text"] = ""

            except Exception as e:
                logger.error("Error processing Deepgram result", error=str(e))

        async def on_utterance_end(data=None, **kwargs):
            """Handle utterance end events - finalize pending transcripts."""
            try:
                # If we have accumulated interim text, emit it as final
                if latest_interim["text"].strip():
                    logger.debug("Utterance end detected, finalizing transcript", text=latest_interim["text"][:50])
                    transcript = TranscriptResult(
                        text=latest_interim["text"],
                        is_final=True,
                        confidence=latest_interim["confidence"],
                        speaker_id=latest_interim["speaker_id"],
                        words=latest_interim["words"],
                        timestamp=datetime.now(),
                    )
                    await result_queue.put(transcript)
                    latest_interim["text"] = ""
            except Exception as e:
                logger.error("Error handling utterance end", error=str(e))

        async def on_error(error, **kwargs):
            """Handle WebSocket errors."""
            logger.error("Deepgram WebSocket error", error=str(error))

        async def on_close(close=None, **kwargs):
            """Handle WebSocket close."""
            logger.debug("Deepgram WebSocket closed")
            await result_queue.put(None)

        logger.info(
            "Starting Deepgram streaming transcription",
            model=self.config.model,
            language=self.config.language,
            diarize=self.config.diarize,
        )

        try:
            # Create live transcription connection
            socket = await self._client.transcription.live(options)

            # Register event handlers
            socket.registerHandler(socket.event.TRANSCRIPT_RECEIVED, on_message)
            socket.registerHandler(socket.event.ERROR, on_error)
            socket.registerHandler(socket.event.CLOSE, on_close)

            # Register utterance end handler if available (SDK 2.x may use different names)
            try:
                if hasattr(socket.event, "UTTERANCE_END"):
                    socket.registerHandler(socket.event.UTTERANCE_END, on_utterance_end)
                elif hasattr(socket.event, "UtteranceEnd"):
                    socket.registerHandler(socket.event.UtteranceEnd, on_utterance_end)
            except Exception as e:
                logger.debug("Could not register utterance_end handler", error=str(e))

            # Task to send audio to Deepgram
            async def send_audio():
                try:
                    async for chunk in audio_stream:
                        # Convert float32 to int16 bytes
                        audio_int16 = (chunk.data * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()

                        # Send to Deepgram (sync method in SDK 2.x)
                        socket.send(audio_bytes)

                except Exception as e:
                    logger.error("Error sending audio to Deepgram", error=str(e))
                finally:
                    # Signal end of audio
                    await socket.finish()

            # Start sending audio in background
            send_task = asyncio.create_task(send_audio())

            # Track time since last transcript for timeout-based finalization
            last_transcript_time = datetime.now()
            silence_finalize_ms = self.config.utterance_end_ms + 500  # Extra buffer

            # Yield results as they come in
            while True:
                try:
                    result = await asyncio.wait_for(result_queue.get(), timeout=0.5)
                    if result is None:
                        break
                    last_transcript_time = datetime.now()
                    yield result
                except asyncio.TimeoutError:
                    # Check if we should finalize based on silence timeout
                    silence_duration = (datetime.now() - last_transcript_time).total_seconds() * 1000
                    if latest_interim["text"].strip() and silence_duration > silence_finalize_ms:
                        logger.debug("Timeout-based finalization", silence_ms=silence_duration)
                        transcript = TranscriptResult(
                            text=latest_interim["text"],
                            is_final=True,
                            confidence=latest_interim["confidence"],
                            speaker_id=latest_interim["speaker_id"],
                            words=latest_interim["words"],
                            timestamp=datetime.now(),
                        )
                        latest_interim["text"] = ""
                        last_transcript_time = datetime.now()
                        yield transcript

                    if send_task.done():
                        # Check for any remaining results
                        try:
                            while True:
                                result = result_queue.get_nowait()
                                if result is None:
                                    break
                                yield result
                        except asyncio.QueueEmpty:
                            # Final check for pending interim transcript
                            if latest_interim["text"].strip():
                                yield TranscriptResult(
                                    text=latest_interim["text"],
                                    is_final=True,
                                    confidence=latest_interim["confidence"],
                                    speaker_id=latest_interim["speaker_id"],
                                    words=latest_interim["words"],
                                    timestamp=datetime.now(),
                                )
                            break

            # Ensure send task is complete
            await send_task

        except Exception as e:
            logger.error("Deepgram streaming error", error=str(e))
            raise

    async def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptResult:
        """
        Transcribe a single audio chunk using Deepgram REST API.

        Args:
            chunk: AudioChunk containing the audio to transcribe

        Returns:
            TranscriptResult with the transcribed text
        """
        self._ensure_client()

        # Convert to int16 bytes
        audio_int16 = (chunk.data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        options = {
            "model": self.config.model,
            "language": self.config.language,
            "smart_format": self.config.smart_format,
            "diarize": self.config.diarize,
        }

        logger.debug("Transcribing audio chunk", duration=chunk.duration_seconds)

        # Transcribe
        source = {"buffer": audio_bytes, "mimetype": "audio/raw"}
        response = await self._client.transcription.prerecorded(source, options)

        # Extract result
        channel = response["results"]["channels"][0]
        alt = channel["alternatives"][0]

        words = []
        if alt.get("words"):
            for word in alt["words"]:
                words.append(
                    WordInfo(
                        word=word.get("word", ""),
                        start_time=word.get("start", 0),
                        end_time=word.get("end", 0),
                        confidence=word.get("confidence", 1.0),
                    )
                )

        return TranscriptResult(
            text=alt.get("transcript", ""),
            is_final=True,
            confidence=alt.get("confidence", 1.0),
            words=words,
            timestamp=datetime.now(),
        )

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None
        logger.debug("Deepgram STT closed")
