"""Fish Audio API provider - Cloud TTS using Fish Speech V1.5 (Direct HTTP)."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from typing import Optional

import httpx
import numpy as np
import structlog

from src.core.config import FishSpeechConfig
from src.core.protocols import TTSProvider
from src.core.types import AudioChunk, TTSRequest, TTSResult

logger = structlog.get_logger()


class FishAudioAPI(TTSProvider):
    """
    Fish Audio cloud TTS - Fish Speech V1.5 via direct HTTP API.

    Features:
    - High quality neural TTS (Fish Speech V1.5)
    - Voice cloning with reference audio
    - Emotion and style control
    - Works with Python 3.9+

    API key from: https://fish.audio
    """

    API_BASE = "https://api.fish.audio"
    OUTPUT_SAMPLE_RATE = 44100

    # Emotion tags supported by Fish Speech
    EMOTIONS = [
        "neutral", "happy", "sad", "angry", "fearful",
        "surprised", "disgusted", "calm", "serious", "excited"
    ]

    def __init__(self, config: Optional[FishSpeechConfig] = None):
        """
        Initialize Fish Audio API provider.

        Args:
            config: Fish Speech configuration with API key
        """
        self.config = config or FishSpeechConfig()
        self._client: Optional[httpx.AsyncClient] = None

        if not self.config.api_key:
            raise ValueError(
                "Fish Audio API key required. Set FISH_SPEECH_API_KEY environment variable "
                "or get one from https://fish.audio"
            )

    async def _ensure_client(self):
        """Lazy initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                },
                timeout=60.0,
            )
            logger.info("Fish Audio API client initialized")

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.OUTPUT_SAMPLE_RATE

    @property
    def supported_emotions(self) -> list[str]:
        """Supported emotion tags."""
        return self.EMOTIONS

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """
        Synthesize speech using Fish Audio API.

        Args:
            request: TTSRequest with text and parameters

        Returns:
            TTSResult with synthesized audio
        """
        await self._ensure_client()

        text = request.text.strip()
        if not text:
            return TTSResult(
                audio_data=np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.1), dtype=np.float32),
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                request=request,
            )

        # Add emotion tag if specified
        if request.emotion and request.emotion in self.EMOTIONS:
            text = f"[{request.emotion}]{text}"

        # Build request payload
        reference_id = self.config.reference_id
        if request.voice_id and request.voice_id != "default":
            reference_id = request.voice_id

        payload = {
            "text": text,
            "format": "mp3",
            "mp3_bitrate": 128,
        }

        if reference_id:
            payload["reference_id"] = reference_id

        try:
            response = await self._client.post("/v1/tts", json=payload)
            response.raise_for_status()

            audio_bytes = response.content
            audio = self._decode_audio(audio_bytes)

            # Apply speed adjustment if needed
            if request.speed != 1.0:
                audio = self._adjust_speed(audio, request.speed)

            logger.debug(
                "Fish Audio synthesis complete",
                text_length=len(text),
                audio_duration=len(audio) / self.OUTPUT_SAMPLE_RATE,
            )

            return TTSResult(
                audio_data=audio,
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                request=request,
            )

        except httpx.HTTPStatusError as e:
            logger.error("Fish Audio API HTTP error", status=e.response.status_code, detail=e.response.text)
            return TTSResult(
                audio_data=np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.5), dtype=np.float32),
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                request=request,
            )
        except Exception as e:
            logger.error("Fish Audio API error", error=str(e))
            return TTSResult(
                audio_data=np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.5), dtype=np.float32),
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                request=request,
            )

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode MP3 audio bytes to numpy array."""
        try:
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            audio_segment = audio_segment.set_frame_rate(self.OUTPUT_SAMPLE_RATE)
            audio_segment = audio_segment.set_channels(1)

            samples = np.array(audio_segment.get_array_of_samples())
            return samples.astype(np.float32) / 32768.0

        except Exception as e:
            logger.error("Audio decode error", error=str(e))
            return np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.5), dtype=np.float32)

    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio playback speed via resampling."""
        if speed == 1.0:
            return audio

        new_length = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        """
        Stream synthesis - yields full audio as single chunk.

        Fish Audio API doesn't support true streaming, so we synthesize
        the complete audio and yield it as one chunk for smooth playback.

        Args:
            request: TTSRequest with text

        Yields:
            Single AudioChunk with complete audio
        """
        result = await self.synthesize(request)

        if len(result.audio_data) == 0:
            return

        # Yield full audio as single chunk for smooth playback
        yield AudioChunk(
            data=result.audio_data,
            sample_rate=self.OUTPUT_SAMPLE_RATE,
        )

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.debug("Fish Audio API closed")
