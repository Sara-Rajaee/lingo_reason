from cohere import AsyncClientV2
from .base_provider import BaseProvider

class CohereAPIProvider(BaseProvider):
    """Cohere models with Cohere API"""

    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncClientV2(api_key=config['api_key'])

    async def generate(self, model_id, prompt, params, reasoning_effort=None, thinking_budget=0):
        """Generate completion using Cohere asynchronously"""
        
        async def _generate():
            response = await self.client.chat(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get('temperature', 0),
                max_tokens=params.get('max_tokens', 12024),
                p=params.get('top_p', 1),
                thinking={
                    "type": "enabled" if params.get('reasoning', True) else "disabled",
                    "token_budget": thinking_budget}
            )
            output = {"raw_generation": None, "generation": None, "reasoning": None}
            for content in response.message.content:
                if content.type == "thinking":
                    output["reasoning"] = content.thinking
                if content.type == "text":
                    output["generation"] = content.text
            return output
        
        return await self._retry_with_backoff(_generate)