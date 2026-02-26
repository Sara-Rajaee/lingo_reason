import re
from litellm import acompletion
from .base_provider import BaseProvider

# Matches <think>...</think> blocks (DeepSeek-R1 style, also used as fallback)
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class GptOssProvider(BaseProvider):
    """GPT-OSS provider via liteLLM proxy (vLLM-hosted models).

    Routes requests through a local OpenAI-compatible API server (e.g. vLLM).
    Model names are prefixed with ``hosted_vllm/`` for liteLLM routing.
    Reasoning traces are extracted via the ``reasoning_content`` field
    (OpenAI reasoning API) or ``<think>`` tags as fallback.

    Provider config (config/providers.yaml):
      - ``api_base``: proxy URL, default ``http://localhost:19743/v1/``
      - ``api_key``: optional, defaults to ``"no-key"`` (local proxy usually needs no auth)
    """

    def __init__(self, config):
        super().__init__(config)
        self.api_base = config.get("api_base", "http://localhost:19743/v1/")
        self.api_key = config.get("api_key", "no-key")

    @staticmethod
    def _extract_reasoning(message):
        """Extract reasoning trace and final answer from a model response message.

        Tries multiple strategies in order:
          1. ``reasoning_content`` field (OpenAI reasoning API)
          2. ``<think>...</think>`` tags in content (DeepSeek-R1 style)
          3. Treat entire content as the answer (no reasoning detected)
        """
        raw_content = message.content or ""

        # Strategy 1: reasoning_content field (OpenAI reasoning API)
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            return {
                "reasoning": reasoning_content,
                "generation": raw_content,
                "raw_generation": raw_content,
            }

        # Strategy 2: <think>...</think> tags
        think_match = THINK_TAG_RE.search(raw_content)
        if think_match:
            reasoning = think_match.group(1).strip()
            generation = THINK_TAG_RE.sub("", raw_content).strip()
            return {
                "reasoning": reasoning,
                "generation": generation,
                "raw_generation": raw_content,
            }

        # Strategy 3: no reasoning detected
        return {
            "reasoning": None,
            "generation": raw_content,
            "raw_generation": raw_content,
        }

    async def generate(self, model_id, prompt, params, reasoning_effort=None, thinking_budget=None):
        """Generate completion via liteLLM proxy (async)."""

        async def _generate():
            kwargs = dict(
                model=f"hosted_vllm/{model_id}",
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=params.get("temperature", 0),
                max_tokens=params.get("max_tokens", 4096),
                top_p=params.get("top_p", 1),
                num_retries=self.max_retries,
            )

            response = await acompletion(**kwargs)
            message = response.choices[0].message
            return self._extract_reasoning(message)

        return await self._retry_with_backoff(_generate)
