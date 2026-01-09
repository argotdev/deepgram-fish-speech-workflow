#!/usr/bin/env python3
"""
Real-time Coaching Assistant

Continuous listening with context-aware feedback using:
- Deepgram Nova-3 for accurate speech recognition
- LLM (OpenAI/Anthropic) for intelligent coaching feedback
- Fish Speech V1.5 with emotion tags for natural, encouraging responses

Coaching Modes:
- Public Speaking: Pitch practice, presentation feedback, pacing
- Fitness: Workout counting, form reminders, motivation
- Language Learning: Pronunciation, grammar, vocabulary expansion
- Interview Prep: Answer structure, confidence building

Usage:
    # Public speaking coach (default)
    python examples/coaching_assistant.py

    # Fitness coach
    python examples/coaching_assistant.py --mode fitness

    # Language learning (specify target language)
    python examples/coaching_assistant.py --mode language --language spanish

    # Interview prep
    python examples/coaching_assistant.py --mode interview
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.core.config import VoiceLoopConfig, LLMConfig
from src.core.types import TranscriptResult, Message, TTSRequest
from src.audio.input import MicrophoneInput
from src.audio.output import SpeakerOutput
from src.audio.vad import create_vad
from src.stt.factory import create_stt_provider
from src.tts.factory import create_tts_provider
from src.llm.factory import create_llm_provider

console = Console()


# Coaching system prompts for different modes
COACHING_PROMPTS = {
    "speaking": """You are an expert public speaking and presentation coach. Your role is to help users practice and improve their presentations, pitches, and speeches.

Guidelines:
- Listen to their practice runs and provide constructive feedback
- Focus on: clarity, pacing, filler words (um, uh, like), structure, and engagement
- Be encouraging but honest - point out areas for improvement
- Give specific, actionable suggestions
- Track their progress across attempts and acknowledge improvements
- Keep feedback concise (2-3 sentences) unless they ask for detailed analysis

Emotion tags you can use: [happy], [excited], [calm], [serious], [friendly]
Use these naturally to make your coaching feel supportive.

Start by asking what they'd like to practice today.""",

    "fitness": """You are an energetic fitness coach helping users through their workouts. You provide motivation, count reps, remind about form, and keep energy high.

Guidelines:
- Be high-energy and motivating
- Count exercises when users tell you what they're doing
- Remind about proper form and breathing
- Provide encouragement, especially when they're struggling
- Suggest modifications if they mention difficulty
- Keep responses SHORT during active exercise (1-2 sentences)
- Celebrate completions and personal bests

Emotion tags you can use: [excited], [happy], [serious], [calm]
Use [excited] for motivation, [serious] for form corrections.

Start by asking what workout they're doing today.""",

    "language": """You are a patient and encouraging language learning coach. You help users practice their target language through conversation.

Guidelines:
- Gently correct pronunciation and grammar mistakes
- Provide the correct form and explain briefly why
- Expand their vocabulary by suggesting alternative words
- Encourage them to speak more, even with mistakes
- Mix encouragement with corrections (sandwich method)
- Occasionally ask them to repeat difficult words/phrases
- Keep explanations simple and brief

Emotion tags you can use: [friendly], [happy], [calm], [excited]
Use [friendly] for corrections, [excited] for celebrations.

The user is learning: {language}
Start by greeting them in their target language and asking what they'd like to practice.""",

    "interview": """You are a professional interview coach helping users prepare for job interviews. You conduct mock interviews and provide feedback.

Guidelines:
- Ask common interview questions appropriate to their field
- Evaluate their answers for: structure (STAR method), relevance, confidence
- Point out filler words, vague answers, or missed opportunities
- Suggest stronger ways to phrase their experiences
- Help them quantify achievements and tell compelling stories
- Be professional but supportive
- After each answer, give brief feedback then move to the next question

Emotion tags you can use: [serious], [friendly], [calm], [happy]
Use [serious] for professional questions, [friendly] for feedback.

Start by asking what role/industry they're interviewing for.""",
}


class CoachingAssistant:
    """Real-time coaching assistant with context-aware feedback."""

    def __init__(
        self,
        mode: str = "speaking",
        language: str = "Spanish",
        llm_provider: str = "openai",
    ):
        self.mode = mode
        self.language = language

        # Build configuration
        self.config = VoiceLoopConfig(
            mode="hybrid",
            stt_provider="deepgram",
            tts_provider="fish_audio_api",
            echo_mode=False,  # We want LLM responses, not echo
        )

        # Configure LLM
        self.config.llm.provider = llm_provider

        # Get coaching prompt
        system_prompt = COACHING_PROMPTS.get(mode, COACHING_PROMPTS["speaking"])
        if mode == "language":
            system_prompt = system_prompt.format(language=language)
        self.config.llm.system_prompt = system_prompt

        # Create providers
        self.stt = create_stt_provider(self.config)
        self.tts = create_tts_provider(self.config)
        self.llm = create_llm_provider(self.config)

        # Audio I/O
        self.mic = MicrophoneInput(self.config.audio)
        self.speaker = SpeakerOutput(self.config.audio)

        # VAD
        self.vad = create_vad(self.config.vad)

        # Conversation state
        self.conversation: list[Message] = []
        self.running = False
        self.speaking = False

        # Feedback timing - don't interrupt every sentence
        self.last_feedback_time: Optional[datetime] = None
        self.accumulated_text: list[str] = []
        self.silence_threshold = 2.0  # Seconds of silence before giving feedback

    async def run(self):
        """Run the coaching session."""
        self.running = True

        console.print(Panel(
            f"[bold]Coaching Mode: {self.mode.title()}[/bold]\n\n"
            "Speak naturally - I'll listen and provide feedback.\n"
            "Press Ctrl+C to end the session.",
            title="Real-time Coaching Assistant",
        ))

        # Start with a greeting from the coach (before starting STT)
        await self._coach_speak(await self._get_initial_greeting())

        console.print("[dim]Listening... (speak now)[/dim]\n")

        try:
            await self._listen_loop()
        except KeyboardInterrupt:
            console.print("\n[yellow]Ending coaching session...[/yellow]")
        finally:
            self.running = False
            await self._cleanup()

            # Session summary
            console.print(Panel(
                "[bold]Session Complete![/bold]\n\n"
                f"Total exchanges: {len(self.conversation) // 2}\n"
                "Great practice session! Keep it up!",
                title="Session Summary",
            ))

    async def _listen_loop(self):
        """Main listening loop with auto-reconnection."""
        while self.running:
            try:
                # Get fresh audio stream with VAD filtering
                audio_stream = self.mic.stream()
                if self.vad:
                    audio_stream = self.vad.filter_speech(audio_stream)

                # Process transcriptions
                async for transcript in self.stt.transcribe_stream(audio_stream):
                    if not self.running:
                        break

                    # Show interim results
                    if not transcript.is_final:
                        console.print(f"[dim]... {transcript.text}[/dim]", end="\r")
                        continue

                    # Handle final transcript
                    text = transcript.text.strip()
                    if not text:
                        continue

                    console.print(f"[green]You:[/green] {text}")
                    self.accumulated_text.append(text)

                    # Determine when to respond based on coaching mode
                    if self.mode == "interview":
                        # Interview mode: respond after each answer (back-and-forth)
                        await self._provide_feedback()
                    elif self.mode == "fitness":
                        # Fitness mode: respond after each command/update
                        await self._provide_feedback()
                    else:
                        # Speaking/language mode: wait for natural pause or trigger phrases
                        should_respond = (
                            text.endswith("?") or
                            any(phrase in text.lower() for phrase in [
                                "what do you think",
                                "how was that",
                                "any feedback",
                                "ready",
                                "let me try",
                                "done",
                                "finished",
                            ])
                        )
                        if should_respond or len(self.accumulated_text) >= 3:
                            await self._provide_feedback()

            except Exception as e:
                if "1011" in str(e) or "timeout" in str(e).lower():
                    # Deepgram timeout - reconnect
                    console.print("[dim]Reconnecting...[/dim]")
                    await asyncio.sleep(0.5)
                    # Recreate STT provider for fresh connection
                    await self.stt.close()
                    self.stt = create_stt_provider(self.config)
                    continue
                else:
                    raise

    async def _get_initial_greeting(self) -> str:
        """Get initial greeting from the coach."""
        # Use LLM to generate contextual greeting
        response = await self.llm.generate(
            "Start the coaching session with a brief, friendly greeting.",
            self.conversation,
        )
        return response

    async def _provide_feedback(self):
        """Provide coaching feedback on accumulated speech."""
        if not self.accumulated_text:
            return

        # Combine accumulated text
        full_text = " ".join(self.accumulated_text)
        self.accumulated_text.clear()

        # Add to conversation
        self.conversation.append(Message(
            role="user",
            content=full_text,
            timestamp=datetime.now(),
        ))

        # Generate coaching feedback
        feedback = await self.llm.generate(full_text, self.conversation)

        # Add coach response to conversation
        self.conversation.append(Message(
            role="assistant",
            content=feedback,
            timestamp=datetime.now(),
        ))

        # Display and speak feedback
        console.print(f"[blue]Coach:[/blue] {feedback}")
        await self._coach_speak(feedback)

        self.last_feedback_time = datetime.now()

    async def _coach_speak(self, text: str):
        """Speak coach response with appropriate emotion."""
        if not text.strip():
            return

        self.speaking = True
        self.mic.mute()  # Prevent echo

        try:
            # Extract emotion tag if present, otherwise use friendly default
            emotion = None
            if text.startswith("[") and "]" in text:
                end_bracket = text.index("]")
                emotion = text[1:end_bracket]
                text = text[end_bracket + 1:].strip()

            request = TTSRequest(
                text=text,
                voice_id="default",
                emotion=emotion,
                speed=1.0,
            )

            # Use streaming for lower latency
            audio_stream = self.tts.synthesize_stream(request)
            await self.speaker.play_stream(audio_stream)

        except Exception as e:
            console.print(f"[red]TTS error: {e}[/red]")
        finally:
            self.speaking = False
            self.mic.unmute()

    async def _cleanup(self):
        """Clean up resources."""
        self.mic.stop()
        await self.stt.close()
        await self.tts.close()
        if self.llm:
            await self.llm.close()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Coaching Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/coaching_assistant.py                    # Public speaking coach
  python examples/coaching_assistant.py --mode fitness     # Fitness coach
  python examples/coaching_assistant.py --mode language    # Language coach (Spanish)
  python examples/coaching_assistant.py --mode interview   # Interview prep
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["speaking", "fitness", "language", "interview"],
        default="speaking",
        help="Coaching mode (default: speaking)",
    )
    parser.add_argument(
        "--language",
        default="Spanish",
        help="Target language for language learning mode (default: Spanish)",
    )
    parser.add_argument(
        "--llm",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider for coaching (default: openai)",
    )

    args = parser.parse_args()

    # Show mode info
    mode_descriptions = {
        "speaking": "Public Speaking Coach - Practice presentations and pitches",
        "fitness": "Fitness Coach - Workout motivation and counting",
        "language": f"Language Coach - Practice {args.language} conversation",
        "interview": "Interview Coach - Mock interview preparation",
    }

    console.print(f"\n[bold]{mode_descriptions[args.mode]}[/bold]")
    console.print(f"[dim]Using {args.llm.title()} for coaching intelligence[/dim]\n")

    # Run coach
    coach = CoachingAssistant(
        mode=args.mode,
        language=args.language,
        llm_provider=args.llm,
    )

    asyncio.run(coach.run())


if __name__ == "__main__":
    main()
