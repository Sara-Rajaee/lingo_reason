from anthropic import AsyncAnthropic
from .base_provider import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider with async support."""

    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config["api_key"],
            timeout=self.timeout,
        )

    @staticmethod
    def _is_adaptive_only(model_id):
        """Fable / Mythos only support adaptive thinking (no temp/top_p)."""
        model = (model_id or "").lower()
        return "fable" in model or "mythos" in model

    @staticmethod
    def _extract_output(response):
        text_parts = []
        thinking_parts = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "redacted_thinking":
                continue
            elif btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
        generation = "".join(text_parts)
        reasoning = "".join(thinking_parts).strip() or None
        return {
            "reasoning": reasoning,
            "generation": generation,
            "raw_generation": generation,
            "finish_reason": getattr(response, "stop_reason", None),
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
        """Generate completion using Anthropic Messages API asynchronously."""

        async def _generate():
            kwargs = {
                "model": model_id,
                "max_tokens": params.get("max_tokens", 8192),
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            if self._is_adaptive_only(model_id):
                # Claude Fable 5 / Mythos 5: adaptive thinking always on;
                # temperature / top_p / top_k are rejected.
                effort = reasoning_effort or "high"
                kwargs["output_config"] = {"effort": effort}
                if params.get("reasoning", True):
                    # Request summarized thinking so we can store a reasoning trace.
                    kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            else:
                kwargs["temperature"] = params.get("temperature", 0)
                if "top_p" in params:
                    kwargs["top_p"] = params.get("top_p")
                if params.get("reasoning", True):
                    budget = thinking_budget or max(1024, int(params.get("max_tokens", 8192) // 2))
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }

            response = await self.client.messages.create(**kwargs)
            return self._extract_output(response)

        return await self._retry_with_backoff(_generate)
