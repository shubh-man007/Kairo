from typing import Optional, List
from openai import OpenAI
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        from src.config import get_settings

        settings = get_settings()
        api_key = api_key or settings.openai_api_key
        model = model or settings.openai_model

        if not api_key:
            raise ValueError("OpenAI API key is required")

        super().__init__(model=model, temperature=temperature, api_key=api_key)
        self.client = OpenAI(api_key=api_key)
        # logger.info(f"Initialized OpenAI client with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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

