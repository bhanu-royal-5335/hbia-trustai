"""
Layer 2: LLM Client
Multi-provider LLM client with fallback support (OpenAI + Anthropic).
"""
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dataclasses import dataclass
from app.core.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()


@dataclass
class GenerationResult:
    content: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_trace: Optional[str] = None


class LLMClient:
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def generate(
        self,
        messages: list,
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream: bool = False,
    ) -> GenerationResult | AsyncGenerator:
        model = model or settings.default_model
        provider = settings.default_llm_provider

        try:
            if provider == "openai" and self.openai:
                return await self._openai_generate(messages, model, temperature, max_tokens, stream)
            elif provider == "anthropic" and self.anthropic:
                return await self._anthropic_generate(messages, model, temperature, max_tokens)
            else:
                # Fallback
                if self.openai:
                    return await self._openai_generate(messages, model, temperature, max_tokens, stream)
                raise RuntimeError("No LLM provider configured")
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e), model=model)
            # Try fallback provider
            if provider == "openai" and self.anthropic:
                logger.info("falling_back_to_anthropic")
                return await self._anthropic_generate(
                    messages, "claude-3-5-sonnet-20241022", temperature, max_tokens
                )
            raise

    async def _openai_generate(
        self, messages, model, temperature, max_tokens, stream=False
    ) -> GenerationResult | AsyncGenerator:
        if stream:
            return self._openai_stream(messages, model, temperature, max_tokens)

        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return GenerationResult(
            content=response.choices[0].message.content,
            model_used=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    async def _openai_stream(self, messages, model, temperature, max_tokens) -> AsyncGenerator:
        stream = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _anthropic_generate(self, messages, model, temperature, max_tokens) -> GenerationResult:
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]

        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_msg,
            messages=user_msgs,
            temperature=temperature,
        )
        return GenerationResult(
            content=response.content[0].text,
            model_used=model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )


llm_client = LLMClient()
