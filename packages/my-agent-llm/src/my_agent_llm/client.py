"""LLM facade for the OpenAI-compatible API used by the coding agent."""

from collections.abc import AsyncIterator, Iterator

from .config import Config
from .models import Message, Response, StreamChunk
from .providers import OpenAIProvider


class LLM:
    """Small model boundary that keeps SDK details out of the Agent core."""

    def __init__(self, *, config: Config | None = None, **kwargs):
        if config is None:
            config = Config(**kwargs)
        if not config.api_key:
            raise ValueError("No API key configured")
        self._provider = OpenAIProvider(config)
        self.config = config

    @property
    def model(self) -> str:
        if not self.config.model:
            raise ValueError("No model specified. Pass model=... or set Config.model.")
        return self.config.model

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Response:
        kwargs.setdefault(
            "temperature",
            temperature if temperature is not None else self.config.temperature,
        )
        if max_tokens is not None:
            kwargs.setdefault("max_tokens", max_tokens)
        elif self.config.max_tokens is not None:
            kwargs.setdefault("max_tokens", self.config.max_tokens)
        return self._provider.chat(
            messages, model=model or self.model, tools=tools, **kwargs
        )

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        return self._provider.stream(
            messages, model=model or self.model, tools=tools, **kwargs
        )

    async def achat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> Response:
        return await self._provider.achat(
            messages, model=model or self.model, tools=tools, **kwargs
        )

    async def achat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in self._provider.achat_stream(
            messages, model=model or self.model, tools=tools, **kwargs
        ):
            yield chunk
