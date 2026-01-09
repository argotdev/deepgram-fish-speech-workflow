"""Microphone input handling using sounddevice."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd
import structlog

from src.core.config import AudioConfig
from src.core.types import AudioChunk

logger = structlog.get_logger()


class MicrophoneInput:
    """Cross-platform microphone input using sounddevice."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize microphone input.

        Args:
            config: Audio configuration, uses defaults if not provided
        """
        self.config = config or AudioConfig()
        self._running = False
        self._muted = False  # Mute during TTS playback to prevent echo
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue[np.ndarray]] = None

    @property
    def sample_rate(self) -> int:
        """Input sample rate."""
        return self.config.input_sample_rate

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.sample_rate * self.config.chunk_duration_ms / 1000)

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """
        Stream audio chunks from the microphone.

        Yields AudioChunk objects as audio is captured.
        Continues until stop() is called.

        Yields:
            AudioChunk objects containing captured audio
        """
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._running = True

        def callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
            """Sounddevice callback - runs in separate thread."""
            if status:
                logger.warning("Audio input status", status=str(status))

            if self._running and self._loop and self._queue and not self._muted:
                # Copy data and send to async queue (skip if muted)
                data = indata.copy().flatten().astype(np.float32)
                self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

        logger.info(
            "Starting microphone input",
            sample_rate=self.sample_rate,
            chunk_samples=self.chunk_samples,
            device=self.config.input_device,
        )

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.config.input_channels,
                dtype=np.float32,
                blocksize=self.chunk_samples,
                device=self.config.input_device,
                callback=callback,
            ):
                while self._running:
                    try:
                        # Wait for audio data with timeout to allow checking _running
                        data = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                        yield AudioChunk(
                            data=data,
                            sample_rate=self.sample_rate,
                            timestamp=datetime.now(),
                        )
                    except asyncio.TimeoutError:
                        continue

        except Exception as e:
            logger.error("Microphone input error", error=str(e))
            raise
        finally:
            self._running = False
            logger.info("Microphone input stopped")

    def stop(self):
        """Stop the microphone input stream."""
        self._running = False

    def mute(self):
        """Mute the microphone (stop sending audio to queue)."""
        self._muted = True
        logger.debug("Microphone muted")

    def unmute(self):
        """Unmute the microphone (resume sending audio to queue)."""
        self._muted = False
        logger.debug("Microphone unmuted")

    @property
    def is_muted(self) -> bool:
        """Check if microphone is muted."""
        return self._muted

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    {
                        "id": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )
        return devices

    @staticmethod
    def get_default_device() -> Optional[dict]:
        """Get the default input device info."""
        try:
            device_id = sd.default.device[0]
            if device_id is not None:
                device = sd.query_devices(device_id)
                return {
                    "id": device_id,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"],
                }
        except Exception:
            pass
        return None
