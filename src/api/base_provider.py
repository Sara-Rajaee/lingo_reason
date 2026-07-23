from abc import ABC, abstractmethod
import asyncio
import re

class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, config):
        self.config = config
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
    
    @abstractmethod
    async def generate(self, model_id, prompt, params, system_prompt=None, reasoning_effort=None, thinking_budget=0):
        """
        Generate completion from the model asynchronously
        
        Args:
            model_id: Model identifier
            prompt: Input prompt
            params: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text string
        """
        pass

    @staticmethod
    def _is_rate_limit_error(exc):
        """Detect provider rate-limit / 429 responses."""
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status == 429:
            return True
        text = str(exc).lower()
        return any(
            needle in text
            for needle in (
                "rate limit",
                "too many requests",
                "429",
                "ratelimit",
            )
        )

    @staticmethod
    def _rate_limit_wait_seconds(exc, attempt):
        """Pick a wait time for rate limits; prefer Reset header, else ~60s * 2^attempt."""
        # Common SDK shapes: response.headers, headers, body with Retry-After
        headers = (
            getattr(exc, "headers", None)
            or getattr(getattr(exc, "response", None), "headers", None)
            or {}
        )
        reset = None
        if hasattr(headers, "get"):
            reset = (
                headers.get("x-ratelimit-reset")
                or headers.get("X-RateLimit-Reset")
                or headers.get("retry-after")
                or headers.get("Retry-After")
            )
        if reset is not None:
            try:
                return max(float(reset), 1.0)
            except (TypeError, ValueError):
                pass

        match = re.search(r"retry(?:ing)?(?: starting)?(?: from)?(?: in)? ~?(\d+)\s*s", str(exc), re.I)
        if match:
            return float(match.group(1)) * (2 ** attempt)

        # Together recommends retry starting from ~60s
        return 60.0 * (2 ** attempt)
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Helper method for async retry logic with exponential backoff.

        Each attempt is bounded by ``self.timeout`` via ``asyncio.wait_for``.
        Provider clients should also set HTTP-level timeouts so connections
        are cancelled promptly; this is a safety net for hung calls.

        Rate-limit errors (429) use a much longer backoff (~60s+) as providers
        like Together recommend, instead of the short network-error backoff.
        """
        for attempt in range(self.max_retries):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.timeout
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                if self._is_rate_limit_error(e):
                    wait_time = self._rate_limit_wait_seconds(e, attempt)
                else:
                    wait_time = 2 ** attempt * 0.1
                print(
                    f"Retry {attempt + 1}/{self.max_retries} after {wait_time:.1f}s "
                    f"due to: {type(e).__name__}: {e}"
                )
                await asyncio.sleep(wait_time)