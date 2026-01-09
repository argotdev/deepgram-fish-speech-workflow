#!/usr/bin/env python3
"""
Basic Voice Loop Demo - STT -> Echo -> TTS

This example demonstrates the basic voice loop functionality:
1. Listen for speech via microphone
2. Transcribe using Deepgram (online) or Whisper (offline)
3. Echo back what was said via Fish Speech TTS

Usage:
    # Online mode with Deepgram (requires API key)
    DEEPGRAM_API_KEY=your_key python examples/basic_loop.py

    # Offline mode with local Whisper
    python examples/basic_loop.py --offline

    # Single interaction mode
    python examples/basic_loop.py --once
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from src.core.config import VoiceLoopConfig
from src.core.types import TranscriptResult
from src.pipeline.voice_loop import VoiceLoop

console = Console()


async def on_transcript(transcript: TranscriptResult):
    """Callback for transcript events."""
    if transcript.is_final:
        console.print(f"[green]You:[/green] {transcript.text}")
    else:
        # Show interim results in dim
        console.print(f"[dim]... {transcript.text}[/dim]", end="\r")


async def on_response(response: str):
    """Callback for response events."""
    console.print(f"[blue]Assistant:[/blue] {response}")


async def run_continuous(config: VoiceLoopConfig):
    """Run continuous voice loop."""
    console.print(
        Panel(
            "[bold]Voice Loop Started[/bold]\n\n"
            "Speak into your microphone. The assistant will echo back what you say.\n"
            "Press Ctrl+C to stop.",
            title="Basic Voice Loop Demo",
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


async def run_once(config: VoiceLoopConfig):
    """Run a single voice interaction."""
    console.print(
        Panel(
            "[bold]Single Interaction Mode[/bold]\n\n"
            "Speak into your microphone. The assistant will echo back what you say.",
            title="Basic Voice Loop Demo",
        )
    )

    loop = VoiceLoop(config)
    loop.set_callbacks(
        on_transcript=on_transcript,
        on_response=on_response,
    )

    try:
        result = await loop.run_once()
        if result:
            user_text, response = result
            console.print(f"\n[bold]Interaction complete![/bold]")
        else:
            console.print("[yellow]No speech detected.[/yellow]")
    finally:
        await loop.close()


def main():
    parser = argparse.ArgumentParser(description="Basic Voice Loop Demo")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline mode (local Whisper instead of Deepgram)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single interaction instead of continuous loop",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size for offline mode",
    )
    parser.add_argument(
        "--tts",
        default="fish_audio_api",
        choices=["fish_audio_api", "edge_tts", "pyttsx3", "fish_speech"],
        help="TTS provider (fish_audio_api=Fish Speech V1.5 API, edge_tts=MS neural, pyttsx3=offline system)",
    )

    args = parser.parse_args()

    # Build configuration
    config = VoiceLoopConfig(
        mode="offline" if args.offline else "hybrid",
        stt_provider="whisper" if args.offline else "deepgram",
        tts_provider=args.tts,
        echo_mode=True,  # Echo mode for this demo
    )

    # Update VAD setting
    config.vad.enabled = not args.no_vad

    # Update Whisper model if using offline mode
    if args.offline:
        config.whisper.model_size = args.whisper_model

    # Show configuration
    console.print(f"[dim]Mode: {config.mode}[/dim]")
    console.print(f"[dim]STT: {config.stt_provider}[/dim]")
    console.print(f"[dim]TTS: {config.tts_provider}[/dim]")
    console.print(f"[dim]VAD: {'enabled' if config.vad.enabled else 'disabled'}[/dim]")
    console.print()

    # Run appropriate mode
    if args.once:
        asyncio.run(run_once(config))
    else:
        asyncio.run(run_continuous(config))


if __name__ == "__main__":
    main()
