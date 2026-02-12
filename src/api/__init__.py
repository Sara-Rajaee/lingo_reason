from .together import TogetherAIProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

class ProviderFactory:
    _providers = {
        'together_ai': TogetherAIProvider,
        'gemini': GeminiProvider,
        'openai': OpenAIProvider,
    }
    
    @staticmethod
    def get_provider(provider_name, config):
        provider_class = ProviderFactory._providers.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider_class(config)