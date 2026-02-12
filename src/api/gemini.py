from google import genai
from google.genai import types
from .base import BaseProvider
import asyncio
import os

class GeminiProvider(BaseProvider):
    """Google Gemini API provider with new google-genai package"""
    
    def __init__(self, config):
        super().__init__(config)
        self.client = genai.Client(api_key=config['api_key'])
    
    async def generate(self, model_id, prompt, params):
        """Generate completion using Gemini asynchronously"""
        
        async def _generate():
            # New API uses async natively
            loop = asyncio.get_event_loop()
            
            def _sync_generate():
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=params.get('temperature', 0.7),
                        max_output_tokens=params.get('max_tokens', 512),
                        top_p=params.get('top_p', 0.9),
                    )
                )
                return response.text
            
            return await loop.run_in_executor(None, _sync_generate)
        
        return await self._retry_with_backoff(_generate)