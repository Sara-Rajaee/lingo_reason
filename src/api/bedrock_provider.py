from litellm import acompletion
from .base_provider import BaseProvider


class BedrockProvider(BaseProvider):
    """Amazon Bedrock provider via liteLLM.

    Uses AWS credentials from the environment (or optional provider config):
      - ``AWS_ACCESS_KEY_ID`` / ``aws_access_key_id``
      - ``AWS_SECRET_ACCESS_KEY`` / ``aws_secret_access_key``
      - ``AWS_REGION`` / ``aws_region_name``

    Model IDs in ``models.yaml`` should be Bedrock inference-profile IDs, e.g.
    ``global.anthropic.claude-fable-5``. They are routed as
    ``bedrock/converse/<model_id>``.
    """

    def __init__(self, config):
        super().__init__(config)
        self.aws_region_name = config.get("aws_region_name") or config.get("region")
        self.aws_access_key_id = config.get("aws_access_key_id") or None
        self.aws_secret_access_key = config.get("aws_secret_access_key") or None
        # Empty env substitutions become ""; treat as unset.
        if not self.aws_access_key_id:
            self.aws_access_key_id = None
        if not self.aws_secret_access_key:
            self.aws_secret_access_key = None

    @staticmethod
    def _litellm_model(model_id):
        if model_id.startswith("bedrock/"):
            return model_id
        return f"bedrock/converse/{model_id}"

    @staticmethod
    def _is_fable_family(model_id):
        model = (model_id or "").lower()
        return "fable" in model or "mythos" in model

    @staticmethod
    def _is_claude_model(model_id):
        model = (model_id or "").lower()
        return "anthropic.claude" in model or "claude-" in model

    @staticmethod
    def _supports_sampling_params(model_id):
        """Recent Claude models on Bedrock reject temperature / top_p."""
        if BedrockProvider._is_fable_family(model_id):
            return False
        if BedrockProvider._is_claude_model(model_id):
            # Opus 4.8 / Sonnet 5+ reject sampling params via Converse.
            return False
        return True

    @staticmethod
    def _extract_output(message, finish_reason=None):
        raw_content = message.content or ""
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
        )
        thinking_blocks = getattr(message, "thinking_blocks", None) or []
        if not reasoning and thinking_blocks:
            parts = []
            for block in thinking_blocks:
                if isinstance(block, dict):
                    parts.append(block.get("thinking") or "")
                else:
                    parts.append(getattr(block, "thinking", "") or "")
            reasoning = "".join(parts).strip() or None
        return {
            "reasoning": reasoning,
            "generation": raw_content,
            "raw_generation": raw_content,
            "finish_reason": finish_reason,
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
        """Generate completion via liteLLM Bedrock Converse."""

        async def _generate():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": self._litellm_model(model_id),
                "messages": messages,
                "max_tokens": params.get("max_tokens", 8192),
                "timeout": self.timeout,
                # Prefer our BaseProvider retry/backoff (handles 429s better).
                "num_retries": 0,
                # Task defaults often include top_p/temperature that Claude rejects.
                "drop_params": True,
            }
            if self.aws_region_name:
                kwargs["aws_region_name"] = self.aws_region_name
            if self.aws_access_key_id:
                kwargs["aws_access_key_id"] = self.aws_access_key_id
            if self.aws_secret_access_key:
                kwargs["aws_secret_access_key"] = self.aws_secret_access_key

            if self._supports_sampling_params(model_id):
                kwargs["temperature"] = params.get("temperature", 0)
                if "top_p" in params:
                    kwargs["top_p"] = params.get("top_p")

            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            elif params.get("reasoning", False):
                kwargs["reasoning_effort"] = "high"

            response = await acompletion(**kwargs)
            choice = response.choices[0]
            return self._extract_output(
                choice.message,
                finish_reason=getattr(choice, "finish_reason", None),
            )

        return await self._retry_with_backoff(_generate)
