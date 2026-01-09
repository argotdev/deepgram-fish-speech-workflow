#!/usr/bin/env python3
"""
Audio I/O Test - Basic loopback and device listing.

This example verifies that audio input/output works correctly on your system.
Run this before testing the full voice loop to ensure audio devices are configured.

Usage:
    python examples/audio_test.py --list-devices
    python examples/audio_test.py --loopback
    python examples/audio_test.py --vad-test
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table

from src.audio.input import MicrophoneInput
from src.audio.output import SpeakerOutput
from src.audio.vad import SimpleVAD, create_vad
from src.core.config import AudioConfig, VADConfig

console = Console()


def list_devices():
    """List all available audio devices."""
    console.print("\n[bold]Audio Input Devices:[/bold]")
    input_table = Table(show_header=True)
    input_table.add_column("ID", style="cyan")
    input_table.add_column("Name")
    input_table.add_column("Channels")
    input_table.add_column("Sample Rate")

    for device in MicrophoneInput.list_devices():
        input_table.add_row(
            str(device["id"]),
            device["name"],
            str(device["channels"]),
            str(int(device["sample_rate"])),
        )
    console.print(input_table)

    default_input = MicrophoneInput.get_default_device()
    if default_input:
        console.print(f"[green]Default input: {default_input['name']}[/green]\n")

    console.print("\n[bold]Audio Output Devices:[/bold]")
    output_table = Table(show_header=True)
    output_table.add_column("ID", style="cyan")
    output_table.add_column("Name")
    output_table.add_column("Channels")
    output_table.add_column("Sample Rate")

    for device in SpeakerOutput.list_devices():
        output_table.add_row(
            str(device["id"]),
            device["name"],
            str(device["channels"]),
            str(int(device["sample_rate"])),
        )
    console.print(output_table)

    default_output = SpeakerOutput.get_default_device()
    if default_output:
        console.print(f"[green]Default output: {default_output['name']}[/green]\n")


async def loopback_test(duration: float = 10.0):
    """
    Test audio loopback - record from mic and play back.

    Args:
        duration: Duration in seconds to record
    """
    console.print(f"\n[bold]Audio Loopback Test[/bold]")
    console.print(f"Recording for {duration} seconds, then playing back...")
    console.print("Speak into your microphone.\n")

    config = AudioConfig(input_sample_rate=16000, output_sample_rate=44100)
    mic = MicrophoneInput(config)
    speaker = SpeakerOutput(config)

    recorded_chunks: list[np.ndarray] = []
    total_samples = 0
    target_samples = int(config.input_sample_rate * duration)

    console.print("[yellow]Recording...[/yellow]")

    async for chunk in mic.stream():
        recorded_chunks.append(chunk.data)
        total_samples += len(chunk.data)

        # Show progress
        progress = min(100, int(total_samples / target_samples * 100))
        console.print(f"\r  Progress: {progress}%", end="")

        if total_samples >= target_samples:
            mic.stop()
            break

    console.print("\n[green]Recording complete![/green]")

    # Combine chunks
    if recorded_chunks:
        audio_data = np.concatenate(recorded_chunks)
        duration_actual = len(audio_data) / config.input_sample_rate
        console.print(f"Recorded {duration_actual:.2f} seconds of audio")

        # Play back
        console.print("\n[yellow]Playing back...[/yellow]")
        from src.core.types import AudioChunk

        playback_chunk = AudioChunk(
            data=audio_data.astype(np.float32),
            sample_rate=config.input_sample_rate,
        )
        await speaker.play(playback_chunk)
        console.print("[green]Playback complete![/green]")
    else:
        console.print("[red]No audio recorded![/red]")


async def vad_test(duration: float = 15.0):
    """
    Test Voice Activity Detection.

    Shows real-time speech detection status.

    Args:
        duration: Duration in seconds to test
    """
    console.print(f"\n[bold]VAD Test[/bold]")
    console.print(f"Testing for {duration} seconds...")
    console.print("Speak to see speech detection.\n")

    config = AudioConfig(input_sample_rate=16000)
    vad_config = VADConfig(threshold=0.5)

    mic = MicrophoneInput(config)

    # Try Silero VAD, fall back to simple
    try:
        vad = create_vad(vad_config, use_silero=True)
        console.print("[green]Using Silero VAD[/green]\n")
    except Exception:
        vad = SimpleVAD(vad_config)
        console.print("[yellow]Using Simple VAD (Silero unavailable)[/yellow]\n")

    total_samples = 0
    target_samples = int(config.input_sample_rate * duration)
    speech_segments = 0

    async for chunk in mic.stream():
        total_samples += len(chunk.data)

        # Check for speech
        prob = vad.get_speech_probability(chunk.data, chunk.sample_rate)
        is_speech = prob > vad_config.threshold

        if is_speech:
            speech_segments += 1
            bar = "=" * int(prob * 20)
            console.print(f"\r[green]SPEECH[/green] [{bar:<20}] {prob:.2f}", end="")
        else:
            bar = "-" * int(prob * 20)
            console.print(f"\r[dim]silence[/dim] [{bar:<20}] {prob:.2f}", end="")

        if total_samples >= target_samples:
            mic.stop()
            break

    console.print(f"\n\n[bold]Test complete![/bold]")
    console.print(f"Detected {speech_segments} speech chunks")


async def tone_test():
    """Generate and play a test tone to verify output."""
    console.print("\n[bold]Tone Test[/bold]")
    console.print("Playing a 440Hz test tone for 2 seconds...\n")

    config = AudioConfig(output_sample_rate=44100)
    speaker = SpeakerOutput(config)

    # Generate 440Hz sine wave
    duration = 2.0
    t = np.linspace(0, duration, int(config.output_sample_rate * duration), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)

    from src.core.types import AudioChunk

    chunk = AudioChunk(data=tone, sample_rate=config.output_sample_rate)
    await speaker.play(chunk)

    console.print("[green]Tone test complete![/green]")


def main():
    parser = argparse.ArgumentParser(description="Audio I/O Test")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--loopback", action="store_true", help="Run loopback test")
    parser.add_argument("--vad-test", action="store_true", help="Run VAD test")
    parser.add_argument("--tone", action="store_true", help="Play test tone")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
    elif args.loopback:
        asyncio.run(loopback_test(args.duration))
    elif args.vad_test:
        asyncio.run(vad_test(args.duration))
    elif args.tone:
        asyncio.run(tone_test())
    else:
        # Default: list devices and play tone
        list_devices()
        asyncio.run(tone_test())


if __name__ == "__main__":
    main()
