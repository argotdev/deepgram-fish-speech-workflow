"""Edge TTS provider - Microsoft's free neural TTS."""

from __future__ import annotations

import asyncio
import io
import tempfile
from collections.abc import AsyncIterator
from typing import Optional

import numpy as np
import structlog

from src.core.protocols import TTSProvider
from src.core.types import AudioChunk, TTSRequest, TTSResult

logger = structlog.get_logger()


# Voice mapping for different styles/emotions
EDGE_VOICES = {
    "default": "en-US-JennyNeural",
    "male": "en-US-GuyNeural",
    "female": "en-US-JennyNeural",
    "aria": "en-US-AriaNeural",
    "davis": "en-US-DavisNeural",
    "jane": "en-US-JaneNeural",
    "jason": "en-US-JasonNeural",
    "sara": "en-US-SaraNeural",
    "tony": "en-US-TonyNeural",
    "nancy": "en-US-NancyNeural",
}

# Emotion/style mapping (for voices that support it)
EDGE_STYLES = {
    "neutral": "general",
    "happy": "cheerful",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "surprise": "excited",
    "friendly": "friendly",
    "chat": "chat",
    "customerservice": "customerservice",
    "newscast": "newscast",
}


class EdgeTTS(TTSProvider):
    """
    Microsoft Edge TTS - high quality neural voices.

    Features:
    - High quality neural TTS
    - Multiple voices and styles
    - Free (no API key required)
    - Requires internet connection

    Install: pip install edge-tts
    """

    OUTPUT_SAMPLE_RATE = 24000  # Edge TTS outputs 24kHz

    def __init__(self, voice: str = "default", rate: str = "+0%", volume: str = "+0%"):
        """
        Initialize Edge TTS.

        Args:
            voice: Voice name or alias (default, male, female, etc.)
            rate: Speech rate adjustment (e.g., "+10%", "-20%")
            volume: Volume adjustment (e.g., "+10%", "-20%")
        """
        self._voice = EDGE_VOICES.get(voice, voice)
        self._rate = rate
        self._volume = volume
        self._style = None

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.OUTPUT_SAMPLE_RATE

    @property
    def supported_emotions(self) -> list[str]:
        """Supported emotion styles."""
        return list(EDGE_STYLES.keys())

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """
        Synthesize speech using Edge TTS.

        Args:
            request: TTSRequest with text and parameters

        Returns:
            TTSResult with synthesized audio
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Install with: pip install edge-tts"
            )

        text = request.text.strip()
        if not text:
            return TTSResult(
                audio_data=np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.1), dtype=np.float32),
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                request=request,
            )

        # Determine voice
        voice = EDGE_VOICES.get(request.voice_id, self._voice)

        # Calculate rate adjustment from speed
        speed_percent = int((request.speed - 1.0) * 100)
        rate = f"{speed_percent:+d}%" if speed_percent != 0 else "+0%"

        # Build SSML if emotion/style is specified
        if request.emotion and request.emotion in EDGE_STYLES:
            style = EDGE_STYLES[request.emotion]
            # SSML with express-as for style
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
                   xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
                <voice name="{voice}">
                    <mstts:express-as style="{style}">
                        {text}
                    </mstts:express-as>
                </voice>
            </speak>
            """
            communicate = edge_tts.Communicate(ssml, voice=voice, rate=rate)
        else:
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)

        # Collect audio data
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            logger.warning("Edge TTS returned no audio")
            return TTSResult(
                audio_data=np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.5), dtype=np.float32),
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                request=request,
            )

        # Combine chunks and convert to numpy
        audio_bytes = b"".join(audio_chunks)
        audio = self._decode_mp3(audio_bytes)

        logger.debug(
            "Edge TTS synthesis complete",
            text_length=len(text),
            audio_duration=len(audio) / self.OUTPUT_SAMPLE_RATE,
            voice=voice,
        )

        return TTSResult(
            audio_data=audio,
            sample_rate=self.OUTPUT_SAMPLE_RATE,
            request=request,
        )

    def _decode_mp3(self, mp3_bytes: bytes) -> np.ndarray:
        """
        Decode MP3 bytes to numpy array.

        Args:
            mp3_bytes: MP3 audio data

        Returns:
            Audio samples as float32 numpy array
        """
        try:
            # Try pydub first (most reliable)
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            audio_segment = audio_segment.set_frame_rate(self.OUTPUT_SAMPLE_RATE)
            audio_segment = audio_segment.set_channels(1)

            samples = np.array(audio_segment.get_array_of_samples())
            audio = samples.astype(np.float32) / 32768.0

            return audio

        except ImportError:
            pass

        try:
            # Fallback to soundfile
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(mp3_bytes))
            if sr != self.OUTPUT_SAMPLE_RATE:
                # Simple resampling
                import scipy.signal
                audio = scipy.signal.resample(
                    audio, int(len(audio) * self.OUTPUT_SAMPLE_RATE / sr)
                )

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            return audio.astype(np.float32)

        except ImportError:
            pass

        # Last resort: save to temp file and read with wave
        logger.warning("Using temp file for MP3 decoding (install pydub for better performance)")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3_bytes)
            temp_path = f.name

        try:
            # Try ffmpeg conversion
            import subprocess
            import wave

            wav_path = temp_path.replace(".mp3", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_path, "-ar", str(self.OUTPUT_SAMPLE_RATE), wav_path],
                capture_output=True,
                check=True,
            )

            with wave.open(wav_path, "rb") as wf:
                audio_bytes = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            import os
            os.unlink(wav_path)
            return audio

        except Exception as e:
            logger.error("MP3 decoding failed", error=str(e))
            return np.zeros(int(self.OUTPUT_SAMPLE_RATE * 0.5), dtype=np.float32)

        finally:
            import os
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        """
        Stream synthesis using Edge TTS streaming.

        Args:
            request: TTSRequest with text

        Yields:
            AudioChunk objects
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Install with: pip install edge-tts"
            )

        text = request.text.strip()
        if not text:
            return

        voice = EDGE_VOICES.get(request.voice_id, self._voice)
        speed_percent = int((request.speed - 1.0) * 100)
        rate = f"{speed_percent:+d}%" if speed_percent != 0 else "+0%"

        communicate = edge_tts.Communicate(text, voice=voice, rate=rate)

        # Accumulate audio and yield in chunks
        buffer = b""
        chunk_size = 4800  # 0.2 seconds at 24kHz

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer += chunk["data"]

                # Yield when we have enough data
                while len(buffer) >= chunk_size * 2:  # *2 for MP3 overhead estimate
                    # Decode accumulated buffer
                    audio = self._decode_mp3(buffer)
                    buffer = b""

                    yield AudioChunk(
                        data=audio,
                        sample_rate=self.OUTPUT_SAMPLE_RATE,
                    )

        # Yield remaining
        if buffer:
            audio = self._decode_mp3(buffer)
            yield AudioChunk(
                data=audio,
                sample_rate=self.OUTPUT_SAMPLE_RATE,
            )

    async def close(self) -> None:
        """Clean up resources."""
        logger.debug("Edge TTS closed")


async def list_voices() -> list[dict]:
    """
    List all available Edge TTS voices.

    Returns:
        List of voice dictionaries with name, gender, locale info
    """
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "edge-tts not installed. Install with: pip install edge-tts"
        )

    voices = await edge_tts.list_voices()
    return [
        {
            "name": v["Name"],
            "short_name": v["ShortName"],
            "gender": v["Gender"],
            "locale": v["Locale"],
        }
        for v in voices
    ]
