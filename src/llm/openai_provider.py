"""OpenAI LLM provider."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Optional

import structlog

from src.core.config import LLMConfig
from src.core.protocols import LLMProvider
from src.core.types import Message

logger = structlog.get_logger()


class OpenAILLM(LLMProvider):
    """OpenAI/OpenAI-compatible LLM provider."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None

    def _ensure_client(self):
        """Lazy initialize the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
            )
            logger.info("OpenAI client initialized", model=self.config.model)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using OpenAI."""
        self._ensure_client()

        messages = [{"role": "system", "content": self.config.system_prompt}]

        if context:
            for msg in context:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.config.model or "gpt-4o",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.choices[0].message.content

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

        stream = await self._client.chat.completions.create(
            model=self.config.model or "gpt-4o",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None
