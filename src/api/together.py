from together import AsyncTogether
from .base import BaseProvider
import re 

class TogetherAIProvider(BaseProvider):
    """TogetherAI API provider with async support"""
    
    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncTogether(api_key=config['api_key'])
    
    def parse_reasoning(self, text, model_id):
        """
        Parse reasoning from TogetherAI model output
        
        DeepSeek R1 uses <think>...</think> tags
        """
        # DeepSeek R1 reasoning pattern
        
        think_pattern = r'<think>\n(.*?)\n</think>\n'
        match = re.search(think_pattern, text, re.DOTALL)
        
        if match:
            reasoning = match.group(1).strip()
            generation = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
            return reasoning, generation
        

        
        # No reasoning found
        return None, text.strip()

    async def generate(self, model_id, prompt, params, reasoning_effort):
        """Generate completion using TogetherAI asynchronously"""
        
        async def _generate():
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                    ],
                reasoning={"enabled": params.get('reasoning', True)},
                temperature=params.get('temperature', 0),
                max_tokens=params.get('max_tokens', 512),
                top_p=params.get('top_p', 1),
            )
            raw_output = response.choices[0].message.content
            
            # Parse reasoning from output
            if "deepseek-r1" in model_id.lower():
                reasoning, generation = self.parse_reasoning(raw_output, model_id)
            elif "deepseek-v3" in model_id.lower():
                reasoning  = response.choices[0].message.reasoning
                generation = response.choices[0].message.content
            return {
                'reasoning': reasoning,
                'generation': generation,
                'raw_generation': raw_output
            }
        
        return await self._retry_with_backoff(_generate)
        