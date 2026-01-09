#!/usr/bin/env python3
"""
Conversational AI Demo - STT -> LLM -> TTS

This example demonstrates full conversational AI:
1. Listen for speech via microphone
2. Transcribe using Deepgram or Whisper
3. Generate response using LLM (OpenAI, Anthropic, or local)
4. Speak response via Fish Speech TTS

Usage:
    # With OpenAI
    DEEPGRAM_API_KEY=your_key OPENAI_API_KEY=your_key python examples/conversational_ai.py

    # With Anthropic
    DEEPGRAM_API_KEY=your_key ANTHROPIC_API_KEY=your_key python examples/conversational_ai.py --llm anthropic

    # Fully offline (Whisper + Ollama)
    python examples/conversational_ai.py --offline --llm local
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.core.config import VoiceLoopConfig
from src.core.types import TranscriptResult
from src.pipeline.voice_loop import VoiceLoop

console = Console()


async def on_transcript(transcript: TranscriptResult):
    """Callback for transcript events."""
    if transcript.is_final:
        console.print(f"\n[bold green]You:[/bold green] {transcript.text}")
    else:
        console.print(f"[dim]Listening: {transcript.text}[/dim]", end="\r")


async def on_response(response: str):
    """Callback for response events."""
    console.print(f"\n[bold blue]Assistant:[/bold blue] {response}")


async def run_conversation(config: VoiceLoopConfig):
    """Run conversational AI loop."""
    provider_name = config.llm.provider or "echo"

    console.print(
        Panel(
            f"[bold]Conversational AI Started[/bold]\n\n"
            f"STT: {config.stt_provider}\n"
            f"LLM: {provider_name} ({config.llm.model or 'default'})\n"
            f"TTS: Fish Speech\n\n"
            "Speak naturally. The assistant will respond.\n"
            "Press Ctrl+C to stop.",
            title="Conversational AI Demo",
        )
    )

    loop = VoiceLoop(config)
    loop.set_callbacks(
        on_transcript=on_transcript,
        on_response=on_response,
    )

    try:
        await loop.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
    finally:
        await loop.close()
        console.print("[green]Conversation ended.[/green]")


def main():
    parser = argparse.ArgumentParser(description="Conversational AI Demo")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline mode (local Whisper + local LLM)",
    )
    parser.add_argument(
        "--llm",
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        help="Specific model to use (e.g., gpt-4o, claude-3-opus, llama3)",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful voice assistant. Keep responses concise and conversational, under 2-3 sentences when possible.",
        help="System prompt for the LLM",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size for offline mode",
    )

    args = parser.parse_args()

    # Build configuration
    config = VoiceLoopConfig(
        mode="offline" if args.offline else "hybrid",
        stt_provider="whisper" if args.offline else "deepgram",
        echo_mode=False,
    )

    # Configure LLM
    config.llm.provider = args.llm
    config.llm.system_prompt = args.system_prompt

    if args.model:
        config.llm.model = args.model
    else:
        # Set default models per provider
        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "local": "llama3",
        }
        config.llm.model = default_models.get(args.llm, "gpt-4o")

    # Update Whisper model if using offline mode
    if args.offline:
        config.whisper.model_size = args.whisper_model

    # Validate configuration
    if config.llm.provider == "openai":
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)
    elif config.llm.provider == "anthropic":
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set[/red]")
            sys.exit(1)

    if not args.offline:
        import os

        if not os.environ.get("DEEPGRAM_API_KEY"):
            console.print("[red]Error: DEEPGRAM_API_KEY environment variable not set[/red]")
            console.print("[yellow]Tip: Use --offline for fully offline operation[/yellow]")
            sys.exit(1)

    asyncio.run(run_conversation(config))


if __name__ == "__main__":
    main()
