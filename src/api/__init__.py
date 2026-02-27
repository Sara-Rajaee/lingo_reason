from .together_provider import TogetherAIProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .cohere_provider import CohereAPIProvider
from .gpt_oss_provider import GptOssProvider

class ProviderFactory:
    _providers = {
        'together_ai': TogetherAIProvider,
        'gemini': GeminiProvider,
        'openai': OpenAIProvider,
        'cohere': CohereAPIProvider,
        'gpt_oss': GptOssProvider,
    }

    @staticmethod
    def get_provider(provider_name, config):
        provider_class = ProviderFactory._providers.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider_class(config)
