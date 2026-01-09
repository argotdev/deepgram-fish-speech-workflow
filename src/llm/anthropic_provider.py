"""Anthropic LLM provider."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Optional

import structlog

from src.core.config import LLMConfig
from src.core.protocols import LLMProvider
from src.core.types import Message

logger = structlog.get_logger()


class AnthropicLLM(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None

    def _ensure_client(self):
        """Lazy initialize the Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required")

            self._client = AsyncAnthropic(api_key=api_key)
            logger.info("Anthropic client initialized", model=self.config.model)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Anthropic."""
        self._ensure_client()

        messages = []

        if context:
            for msg in context:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        response = await self._client.messages.create(
            model=self.config.model or "claude-3-5-sonnet-20241022",
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=messages,
        )

        return response.content[0].text

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        self._ensure_client()

        messages = []

        if context:
            for msg in context:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        async with self._client.messages.stream(
            model=self.config.model or "claude-3-5-sonnet-20241022",
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None
