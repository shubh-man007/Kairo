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

    @staticmethod
    def _infer_provider_from_model(model: str) -> Optional[str]:
        """
        Best‑effort detection of the provider from a model name.

        This is intentionally conservative – it only returns a provider when
        the mapping is very clear (e.g. 'claude-*' → Anthropic,
        'gpt-*' / 'gpt-4o*' → OpenAI). Otherwise it returns None and lets
        the explicit provider win.
        """
        if not model:
            return None

        m = model.lower()

        # Anthropic models are all "claude-*" today
        if "claude" in m:
            return "anthropic"

        # Common OpenAI chat / reasoning / embedding models
        if m.startswith("gpt-") or "gpt-" in m:
            return "openai"
        if m.startswith("o") and ("mini" in m or "preview" in m or "latest" in m):
            # e.g. "o3-mini", "o1-preview" – avoid being too clever otherwise
            return "openai"
        if "text-embedding" in m or "embedding" in m:
            return "openai"

        return None

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

        # Try to detect obvious provider/model mismatches early, before we ever
        # hit a provider API with an incompatible model name.
        if model:
            inferred = cls._infer_provider_from_model(model)
            if inferred is not None and inferred != provider:
                raise ValueError(
                    f"Model '{model}' appears to be a '{inferred}' model, "
                    f"but provider '{provider}' was requested. "
                    "Please align the model and provider in your settings "
                    "(e.g. use an OpenAI model for provider='openai' and "
                    "a Claude model for provider='anthropic')."
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

