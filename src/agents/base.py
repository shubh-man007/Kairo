from abc import ABC, abstractmethod
from typing import Optional
from src.models import Persona, Question, AgentResponse


class BaseAgent(ABC):
    @abstractmethod
    def respond(
        self,
        persona: Persona,
        question: Question,
    ) -> AgentResponse:
    
        pass

