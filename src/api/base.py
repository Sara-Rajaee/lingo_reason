from abc import ABC, abstractmethod
import asyncio

class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, config):
        self.config = config
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
    
    @abstractmethod
    async def generate(self, model_id, prompt, params):
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
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Helper method for async retry logic with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = 2 ** attempt * 0.1
                print(f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s due to: {e}")
                await asyncio.sleep(wait_time)