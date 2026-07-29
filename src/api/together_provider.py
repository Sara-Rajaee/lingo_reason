from together import AsyncTogether
from .base_provider import BaseProvider
import re


# Closed <think>...</think> blocks (DeepSeek-R1 / Qwen style).
_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.DOTALL)
# Unclosed leading <think> (streamed reasoning already separated, or truncated close).
_THINK_OPEN_ONLY_RE = re.compile(r"^<think>\s*", re.IGNORECASE)
# Close-only traces (opening tag was in the chat template).
_THINK_CLOSE_ONLY_RE = re.compile(r"^(.*?)</think>\s*(.*)$", re.DOTALL)


def strip_think_tags(text):
    """Remove <think> wrappers from model text; return (reasoning_or_None, generation)."""
    if not text:
        return None, ""
    s = str(text)

    match = _THINK_BLOCK_RE.search(s)
    if match:
        reasoning = match.group(1).strip() or None
        generation = _THINK_BLOCK_RE.sub("", s).strip()
        return reasoning, generation

    if "</think>" in s:
        close_match = _THINK_CLOSE_ONLY_RE.match(s)
        if close_match:
            reasoning = close_match.group(1).strip() or None
            generation = close_match.group(2).strip()
            return reasoning, generation

    if _THINK_OPEN_ONLY_RE.match(s):
        # Unclosed <think>: treat everything after the tag as the visible answer.
        generation = _THINK_OPEN_ONLY_RE.sub("", s, count=1).strip()
        return None, generation

    return None, s.strip()


class TogetherAIProvider(BaseProvider):
    """TogetherAI API provider with async support"""
    
    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncTogether(api_key=config['api_key'], timeout=self.timeout)
    
    def parse_reasoning(self, text, model_id):
        """
        Parse reasoning from TogetherAI model output.
        
        DeepSeek R1 / Qwen use <think>...</think> tags (sometimes unclosed).
        """
        return strip_think_tags(text)

    def _parse_output(self, model_id, raw_output, reasoning=None):
        """Split reasoning vs final answer for known Together model families."""
        model_id_lower = model_id.lower()
        if "deepseek-r1" in model_id_lower or "qwen" in model_id_lower:
            tag_reasoning, generation = strip_think_tags(raw_output)
            # Prefer channel reasoning when present; always strip tags from generation.
            return (reasoning or tag_reasoning), generation
        if "deepseek-v3" in model_id_lower or "deepseek-v4" in model_id_lower:
            # Still strip accidental think wrappers from content.
            tag_reasoning, generation = strip_think_tags(raw_output)
            return (reasoning or tag_reasoning), generation
        if reasoning:
            _, generation = strip_think_tags(raw_output)
            return reasoning, generation
        return strip_think_tags(raw_output)

    async def _consume_stream(self, stream):
        """Accumulate content and reasoning deltas from a chat completion stream."""
        content_parts = []
        reasoning_parts = []
        async for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            reasoning_delta = getattr(delta, "reasoning", None)
            content_delta = getattr(delta, "content", None)
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            if content_delta:
                content_parts.append(content_delta)
        return "".join(content_parts), ("".join(reasoning_parts) or None)

    async def generate(self, model_id, prompt, params, system_prompt=None, reasoning_effort=None, thinking_budget=None):
        """Generate completion using TogetherAI asynchronously"""
        
        async def _generate():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            use_stream = bool(params.get('stream', False))
            kwargs = dict(
                model=model_id,
                messages=messages,
                reasoning={"enabled": params.get('reasoning', True)},
                temperature=params.get('temperature', 0),
                max_tokens=params.get('max_tokens', 512),
                top_p=params.get('top_p', 1),
                stream=use_stream,
            )
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort

            response = await self.client.chat.completions.create(**kwargs)

            if use_stream:
                raw_output, streamed_reasoning = await self._consume_stream(response)
                reasoning, generation = self._parse_output(
                    model_id, raw_output, reasoning=streamed_reasoning
                )
            else:
                raw_output = response.choices[0].message.content
                message_reasoning = getattr(response.choices[0].message, "reasoning", None)
                reasoning, generation = self._parse_output(
                    model_id, raw_output, reasoning=message_reasoning
                )

            return {
                'reasoning': reasoning,
                'generation': generation,
                'raw_generation': raw_output
            }
        
        return await self._retry_with_backoff(_generate)
