from openai import AsyncOpenAI
from .base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider with async support"""

    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config['api_key'],
            timeout=self.timeout,
        )

    @staticmethod
    def _uses_max_completion_tokens(model_id, reasoning_effort):
        """Reasoning / GPT-5 family models require max_completion_tokens."""
        model = (model_id or "").lower()
        if reasoning_effort is not None:
            return True
        return any(
            model.startswith(prefix)
            for prefix in ("gpt-5", "o1", "o3", "o4")
        )

    @staticmethod
    def _extract_output(message):
        """Normalize OpenAI message into reasoning / generation fields."""
        raw_content = message.content or ""
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
        )
        return {
            "reasoning": reasoning,
            "generation": raw_content,
            "raw_generation": raw_content,
            "finish_reason": None,
        }

    async def generate(
        self,
        model_id,
        prompt,
        params,
        system_prompt=None,
        reasoning_effort=None,
        thinking_budget=0,
    ):
        """Generate completion using OpenAI asynchronously"""

        async def _generate():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": model_id,
                "messages": messages,
                "temperature": params.get("temperature", 0),
                "top_p": params.get("top_p", 1),
            }
            max_tokens = params.get("max_tokens", 512)
            if self._uses_max_completion_tokens(model_id, reasoning_effort):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort

            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            output = self._extract_output(choice.message)
            output["finish_reason"] = getattr(choice, "finish_reason", None)
            return output

        return await self._retry_with_backoff(_generate)
