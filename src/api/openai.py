from openai import AsyncOpenAI
from .base import BaseProvider

class OpenAIProvider(BaseProvider):
    """OpenAI API provider with async support"""
    
    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config['api_key'])
    
    async def generate(self, model_id, prompt, params):
        """Generate completion using OpenAI asynchronously"""
        
        async def _generate():
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get('temperature', 0.7),
                max_tokens=params.get('max_tokens', 512),
                top_p=params.get('top_p', 0.9),
            )
            return response.choices[0].message.content
        
        return await self._retry_with_backoff(_generate)