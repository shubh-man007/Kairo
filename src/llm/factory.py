from typing import Optional
from src.llm.base import BaseLLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.anthropic_client import AnthropicClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClientFactory:
    _providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }

    @classmethod
    def create(
        cls,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ) -> BaseLLMClient:
        
        provider = provider.lower()
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported: {list(cls._providers.keys())}"
            )

        client_class = cls._providers[provider]
        # logger.info(f"Creating {provider} client with model: {model}")

        if provider == "openai":
            return client_class(model=model, temperature=temperature, api_key=api_key)
        elif provider == "anthropic":
            return client_class(model=model, temperature=temperature, api_key=api_key)
        else:
            raise ValueError(f"Provider {provider} not implemented")

    @classmethod
    def register_provider(cls, name: str, client_class: type[BaseLLMClient]) -> None:
        if not issubclass(client_class, BaseLLMClient):
            raise ValueError(
                f"Client class must inherit from BaseLLMClient, got {client_class}"
            )
        cls._providers[name.lower()] = client_class
        logger.info(f"Registered new provider: {name}")


def create_llm_client(
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> BaseLLMClient:
    
    return LLMClientFactory.create(
        provider=provider, model=model, temperature=temperature, api_key=api_key
    )

