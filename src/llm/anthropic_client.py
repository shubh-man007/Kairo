from typing import Optional, List
from anthropic import Anthropic
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):

        from src.config import get_settings

        settings = get_settings()
        api_key = api_key or settings.anthropic_api_key
        model = model or settings.anthropic_model

        if not api_key:
            raise ValueError("Anthropic API key is required")

        super().__init__(model=model, temperature=temperature, api_key=api_key)
        self.client = Anthropic(api_key=api_key)

        # Proactively verify that the model exists and is accessible for this key.
        try:
            models = self.client.models.list()
            available_ids = {m.id for m in models.data}
            if model not in available_ids:
                raise ValueError(
                    f"Anthropic model '{model}' is not available or not accessible "
                    "for the provided API key."
                )
        except Exception as e:
            logger.error(
                f"Anthropic model validation failed for '{model}'. "
                f"Make sure the model name is correct and your API key has access. "
                f"Original error: {e}"
            )
            # Re-raise to fail fast before any evaluation logic runs.
            raise

        logger.info(f"Initialized Anthropic client with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 4096,
                temperature=temperature if temperature is not None else self.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)
        return results

