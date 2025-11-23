from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseLLMClient(ABC):
    def __init__(self, model: str, temperature: float = 0.0, api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "provider": self.__class__.__name__,
        }

