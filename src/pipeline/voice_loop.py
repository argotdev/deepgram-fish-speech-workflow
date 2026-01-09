"""Voice loop pipeline orchestrator: STT -> LLM -> TTS."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Optional

import structlog

from src.audio.input import MicrophoneInput
from src.audio.output import SpeakerOutput
from src.audio.vad import create_vad
from src.core.config import VoiceLoopConfig
from src.core.protocols import LLMProvider, STTProvider, TTSProvider
from src.core.types import AudioChunk, Message, TranscriptResult, TTSRequest
from src.llm.factory import create_llm_provider
from src.stt.factory import create_stt_provider
from src.tts.factory import create_tts_provider

logger = structlog.get_logger()


class VoiceLoop:
    """
    Main voice loop orchestrator: STT -> LLM -> TTS.

    Manages the complete voice interaction pipeline:
    1. Captures audio from microphone
    2. Filters through VAD (voice activity detection)
    3. Transcribes speech to text via STT provider
    4. Generates responses via LLM (optional)
    5. Synthesizes speech via TTS provider
    6. Plays audio through speaker
    7. Handles interruptions (user speaking during TTS)
    """

    def __init__(
        self,
        config: Optional[VoiceLoopConfig] = None,
        stt: Optional[STTProvider] = None,
        tts: Optional[TTSProvider] = None,
        llm: Optional[LLMProvider] = None,
    ):
        """
        Initialize the voice loop.

        Args:
            config: Voice loop configuration
            stt: Custom STT provider (created from config if not provided)
            tts: Custom TTS provider (created from config if not provided)
            llm: Custom LLM provider (created from config if not provided)
        """
        self.config = config or VoiceLoopConfig()

        # Create providers
        self.stt = stt or create_stt_provider(self.config)
        self.tts = tts or create_tts_provider(self.config)
        self.llm = llm or create_llm_provider(self.config)

        # Audio I/O
        self.mic = MicrophoneInput(self.config.audio)
        self.speaker = SpeakerOutput(self.config.audio)

        # VAD
        self.vad = create_vad(self.config.vad) if self.config.vad.enabled else None

        # State
        self._running = False
        self._speaking = False
        self._conversation_context: list[Message] = []

        # Callbacks
        self._on_transcript: Optional[Callable[[TranscriptResult], Awaitable[None]]] = None
        self._on_response: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_speech_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_speech_end: Optional[Callable[[], Awaitable[None]]] = None

    @property
    def is_running(self) -> bool:
        """Check if the voice loop is currently running."""
        return self._running

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is currently playing."""
        return self._speaking

    def set_callbacks(
        self,
        on_transcript: Optional[Callable[[TranscriptResult], Awaitable[None]]] = None,
        on_response: Optional[Callable[[str], Awaitable[None]]] = None,
        on_speech_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_speech_end: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """
        Set event callbacks.

        Args:
            on_transcript: Called when transcript is received
            on_response: Called when LLM response is generated
            on_speech_start: Called when user starts speaking
            on_speech_end: Called when user stops speaking
        """
        self._on_transcript = on_transcript
        self._on_response = on_response
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end

    async def run(self):
        """
        Run the voice loop until stopped.

        Continuously listens for speech, transcribes, generates responses,
        and speaks them back.
        """
        self._running = True
        logger.info(
            "Starting voice loop",
            mode=self.config.mode,
            stt_provider=self.config.stt_provider,
            llm_provider=self.config.llm.provider,
            echo_mode=self.config.echo_mode,
        )

        try:
            # Get audio stream
            audio_stream = self.mic.stream()

            # Filter through VAD if enabled
            if self.vad:
                audio_stream = self.vad.filter_speech(audio_stream)

            # Transcribe stream
            async for transcript in self.stt.transcribe_stream(audio_stream):
                if not self._running:
                    break

                # Handle transcript
                await self._handle_transcript(transcript)

        except Exception as e:
            logger.error("Voice loop error", error=str(e))
            raise
        finally:
            self._running = False
            logger.info("Voice loop stopped")

    async def run_once(self) -> Optional[tuple[str, str]]:
        """
        Run a single voice interaction (one user turn + one assistant turn).

        Returns:
            Tuple of (user_text, assistant_response) or None if no speech detected
        """
        self._running = True
        user_text = None
        response_text = None

        try:
            # Collect audio until we get a final transcript
            audio_stream = self.mic.stream()

            if self.vad:
                audio_stream = self.vad.filter_speech(audio_stream)

            async for transcript in self.stt.transcribe_stream(audio_stream):
                if transcript.is_final and transcript.text.strip():
                    user_text = transcript.text

                    if self._on_transcript:
                        await self._on_transcript(transcript)

                    # Generate and speak response
                    response_text = await self._generate_response(user_text)

                    if self._on_response:
                        await self._on_response(response_text)

                    await self._speak_response(response_text)

                    # Stop after one complete interaction
                    self.mic.stop()
                    break

        except Exception as e:
            logger.error("Voice interaction error", error=str(e))
            raise
        finally:
            self._running = False

        if user_text and response_text:
            return (user_text, response_text)
        return None

    async def _handle_transcript(self, transcript: TranscriptResult):
        """
        Handle a transcript result.

        Args:
            transcript: Transcript from STT
        """
        # Call transcript callback
        if self._on_transcript:
            await self._on_transcript(transcript)

        # Only process final transcripts
        if not transcript.is_final:
            return

        text = transcript.text.strip()
        if not text:
            return

        logger.info("Transcript received", text=text, speaker=transcript.speaker_id)

        # Check for interruption
        if self._speaking and self.config.interruption_enabled:
            logger.debug("Interruption detected, stopping TTS")
            self.speaker.stop()
            self._speaking = False

        # Generate response
        response = await self._generate_response(text)

        # Call response callback
        if self._on_response:
            await self._on_response(response)

        # Speak response
        await self._speak_response(response)

    async def _generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.

        Uses LLM if configured, otherwise echoes input.

        Args:
            user_input: User's transcribed speech

        Returns:
            Response text
        """
        # Add to conversation context
        self._conversation_context.append(
            Message(role="user", content=user_input, timestamp=datetime.now())
        )

        # Echo mode for testing
        if self.config.echo_mode or self.llm is None:
            response = f"You said: {user_input}"
        else:
            # Generate LLM response
            try:
                response = await self.llm.generate(user_input, self._conversation_context)
            except Exception as e:
                logger.error("LLM generation failed", error=str(e))
                response = "I'm sorry, I couldn't generate a response."

        # Add response to context
        self._conversation_context.append(
            Message(role="assistant", content=response, timestamp=datetime.now())
        )

        # Limit context size
        max_context = 20
        if len(self._conversation_context) > max_context:
            self._conversation_context = self._conversation_context[-max_context:]

        logger.info("Response generated", response_length=len(response))

        return response

    async def _speak_response(self, text: str):
        """
        Speak a response using TTS.

        Mutes microphone during playback to prevent echo/feedback loop.

        Args:
            text: Text to speak
        """
        if not text.strip():
            return

        self._speaking = True

        # Mute microphone to prevent picking up TTS output
        self.mic.mute()

        try:
            # Use generic TTS settings - providers ignore unsupported options
            request = TTSRequest(
                text=text,
                voice_id="default",
                emotion=None,
                speed=1.0,
            )

            if self.config.streaming_mode:
                # Streaming TTS for lower latency
                audio_stream = self.tts.synthesize_stream(request)
                await self.speaker.play_stream(audio_stream)
            else:
                # Non-streaming TTS
                result = await self.tts.synthesize(request)
                await self.speaker.play(result)

        except Exception as e:
            logger.error("TTS playback failed", error=str(e))
        finally:
            self._speaking = False
            # Unmute microphone after TTS playback completes
            self.mic.unmute()

    def stop(self):
        """Stop the voice loop."""
        self._running = False
        self.mic.stop()
        self.speaker.stop()

    def clear_context(self):
        """Clear conversation context."""
        self._conversation_context.clear()
        logger.debug("Conversation context cleared")

    async def close(self):
        """Clean up all resources."""
        self.stop()

        if self.stt:
            await self.stt.close()
        if self.tts:
            await self.tts.close()
        if self.llm:
            await self.llm.close()

        logger.debug("Voice loop resources closed")
