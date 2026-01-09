"""Protocol definitions for provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .types import AudioChunk, Message, TranscriptResult, TTSRequest, TTSResult


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers."""

    @abstractmethod
    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptResult]:
        """
        Transcribe a stream of audio chunks.

        Yields TranscriptResult objects as transcription progresses.
        Results may be interim (is_final=False) or final (is_final=True).

        Args:
            audio_stream: Async iterator of AudioChunk objects

        Yields:
            TranscriptResult objects with transcribed text
        """
        ...

    @abstractmethod
    async def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptResult:
        """
        Transcribe a single audio chunk.

        For providers that don't support streaming, this processes
        a complete audio segment at once.

        Args:
            chunk: AudioChunk containing the audio to transcribe

        Returns:
            TranscriptResult with the transcribed text
        """
        ...

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass


class TTSProvider(ABC):
    """Abstract base class for Text-to-Speech providers."""

    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            request: TTSRequest with text and synthesis parameters

        Returns:
            TTSResult with synthesized audio
        """
        ...

    async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        """
        Synthesize speech from text in streaming mode.

        Yields audio chunks as they are generated for lower latency.
        Default implementation falls back to non-streaming synthesis.

        Args:
            request: TTSRequest with text and synthesis parameters

        Yields:
            AudioChunk objects as synthesis progresses
        """
        result = await self.synthesize(request)
        yield result.to_chunk()

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate of this TTS provider."""
        ...

    @property
    def supported_emotions(self) -> list[str]:
        """List of supported emotion tags. Override to specify available emotions."""
        return []

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """
        Generate a response to the given prompt.

        Args:
            prompt: User's input text
            context: Optional conversation history

        Returns:
            Generated response text
        """
        ...

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncIterator[str]:
        """
        Generate a response in streaming mode.

        Yields tokens/chunks as they are generated.
        Default implementation falls back to non-streaming generation.

        Args:
            prompt: User's input text
            context: Optional conversation history

        Yields:
            Response text chunks as they are generated
        """
        response = await self.generate(prompt, context)
        yield response

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
