"""Local LLM provider (Ollama/llama.cpp compatible)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Optional

import structlog

from src.core.config import LLMConfig
from src.core.protocols import LLMProvider
from src.core.types import Message

logger = structlog.get_logger()


class LocalLLM(LLMProvider):
    """
    Local LLM provider for Ollama or llama.cpp server.

    Connects to a local LLM server via OpenAI-compatible API.
    Default base URL is http://localhost:11434/v1 for Ollama.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None

    def _ensure_client(self):
        """Lazy initialize the client."""
        if self._client is None:
            import httpx

            base_url = self.config.base_url or "http://localhost:11434/v1"

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=60.0,
            )
            logger.info(
                "Local LLM client initialized",
                base_url=base_url,
                model=self.config.model,
            )

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using local LLM."""
        self._ensure_client()

        messages = [{"role": "system", "content": self.config.system_prompt}]

        if context:
            for msg in context:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.config.model or "llama3",
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        self._ensure_client()

        messages = [{"role": "system", "content": self.config.system_prompt}]

        if context:
            for msg in context:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        async with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": self.config.model or "llama3",
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    import json

                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
