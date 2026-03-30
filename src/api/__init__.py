class ProviderFactory:
    _provider_paths = {
        'together_ai': ('.together_provider', 'TogetherAIProvider'),
        'gemini': ('.gemini_provider', 'GeminiProvider'),
        'openai': ('.openai_provider', 'OpenAIProvider'),
        'cohere': ('.cohere_provider', 'CohereAPIProvider'),
        'gpt_oss': ('.gpt_oss_provider', 'GptOssProvider'),
    }

    @staticmethod
    def get_provider(provider_name, config):
        entry = ProviderFactory._provider_paths.get(provider_name)
        if not entry:
            raise ValueError(f"Unknown provider: {provider_name}")
        module_path, class_name = entry
        from importlib import import_module
        module = import_module(module_path, package='src.api')
        provider_class = getattr(module, class_name)
        return provider_class(config)
