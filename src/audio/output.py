"""Speaker output handling using sounddevice."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Optional

import numpy as np
import sounddevice as sd
import structlog

from src.core.config import AudioConfig
from src.core.types import AudioChunk, TTSResult

logger = structlog.get_logger()


class SpeakerOutput:
    """Cross-platform audio playback using sounddevice."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize speaker output.

        Args:
            config: Audio configuration, uses defaults if not provided
        """
        self.config = config or AudioConfig()
        self._playing = False
        self._current_stream: Optional[sd.OutputStream] = None

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.config.output_sample_rate

    async def play(self, audio: TTSResult | AudioChunk) -> None:
        """
        Play audio through the speaker.

        Blocks until playback is complete or stop() is called.

        Args:
            audio: TTSResult or AudioChunk to play
        """
        if isinstance(audio, TTSResult):
            data = audio.audio_data
            sample_rate = audio.sample_rate
        else:
            data = audio.data
            sample_rate = audio.sample_rate

        # Resample if needed
        if sample_rate != self.sample_rate:
            ratio = self.sample_rate / sample_rate
            new_length = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, new_length)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
            sample_rate = self.sample_rate

        self._playing = True
        logger.debug(
            "Playing audio",
            duration=len(data) / sample_rate,
            sample_rate=sample_rate,
        )

        try:
            # Use asyncio-friendly playback
            loop = asyncio.get_event_loop()
            finished = asyncio.Event()

            def callback(outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
                nonlocal data
                if status:
                    logger.warning("Audio output status", status=str(status))

                if not self._playing:
                    outdata.fill(0)
                    raise sd.CallbackStop()

                chunk_size = min(frames, len(data))
                outdata[:chunk_size, 0] = data[:chunk_size]
                outdata[chunk_size:] = 0
                data = data[chunk_size:]

                if len(data) == 0:
                    raise sd.CallbackStop()

            def finished_callback():
                loop.call_soon_threadsafe(finished.set)

            self._current_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=self.config.output_channels,
                dtype=np.float32,
                device=self.config.output_device,
                callback=callback,
                finished_callback=finished_callback,
            )

            self._current_stream.start()
            await finished.wait()

        except sd.CallbackStop:
            pass
        finally:
            if self._current_stream:
                self._current_stream.close()
                self._current_stream = None
            self._playing = False

    async def play_stream(self, audio_stream: AsyncIterator[AudioChunk]) -> None:
        """
        Play a stream of audio chunks.

        Provides lower latency by starting playback before all audio is received.

        Args:
            audio_stream: Async iterator of AudioChunk objects
        """
        self._playing = True
        stream_active = True
        buffer: list[np.ndarray] = []
        buffer_lock = asyncio.Lock()

        async def fill_buffer():
            """Coroutine to fill the buffer from the stream."""
            nonlocal stream_active
            try:
                async for chunk in audio_stream:
                    if not stream_active:
                        break
                    async with buffer_lock:
                        # Resample if needed
                        if chunk.sample_rate != self.sample_rate:
                            resampled = chunk.resample(self.sample_rate)
                            buffer.append(resampled.data)
                        else:
                            buffer.append(chunk.data)
            finally:
                stream_active = False

        # Start filling buffer in background
        fill_task = asyncio.create_task(fill_buffer())

        # Wait for initial buffer
        await asyncio.sleep(0.1)

        try:
            while stream_active or buffer:
                async with buffer_lock:
                    if buffer:
                        data = buffer.pop(0)
                    else:
                        data = None

                if data is not None:
                    chunk = AudioChunk(data=data, sample_rate=self.sample_rate)
                    await self._play_chunk(chunk)
                else:
                    if not stream_active:
                        break
                    await asyncio.sleep(0.05)

        finally:
            self._playing = False
            stream_active = False
            fill_task.cancel()
            try:
                await fill_task
            except asyncio.CancelledError:
                pass

    async def _play_chunk(self, chunk: AudioChunk) -> None:
        """
        Play a single audio chunk without modifying _playing state.

        Used internally by play_stream to avoid state conflicts.
        """
        data = chunk.data
        sample_rate = chunk.sample_rate

        # Resample if needed
        if sample_rate != self.sample_rate:
            ratio = self.sample_rate / sample_rate
            new_length = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, new_length)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
            sample_rate = self.sample_rate

        loop = asyncio.get_event_loop()
        finished = asyncio.Event()

        def callback(outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
            nonlocal data
            if status:
                logger.warning("Audio output status", status=str(status))

            chunk_size = min(frames, len(data))
            outdata[:chunk_size, 0] = data[:chunk_size]
            outdata[chunk_size:] = 0
            data = data[chunk_size:]

            if len(data) == 0:
                raise sd.CallbackStop()

        def finished_callback():
            loop.call_soon_threadsafe(finished.set)

        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=self.config.output_channels,
                dtype=np.float32,
                device=self.config.output_device,
                callback=callback,
                finished_callback=finished_callback,
            )
            stream.start()
            await finished.wait()
            stream.close()
        except sd.CallbackStop:
            pass

    def stop(self) -> None:
        """Stop current playback immediately."""
        self._playing = False
        if self._current_stream:
            try:
                self._current_stream.abort()
            except Exception:
                pass

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio output devices."""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_output_channels"] > 0:
                devices.append(
                    {
                        "id": i,
                        "name": device["name"],
                        "channels": device["max_output_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )
        return devices

    @staticmethod
    def get_default_device() -> Optional[dict]:
        """Get the default output device info."""
        try:
            device_id = sd.default.device[1]
            if device_id is not None:
                device = sd.query_devices(device_id)
                return {
                    "id": device_id,
                    "name": device["name"],
                    "channels": device["max_output_channels"],
                    "sample_rate": device["default_samplerate"],
                }
        except Exception:
            pass
        return None
