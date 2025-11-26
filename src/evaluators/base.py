from abc import ABC, abstractmethod
from typing import Optional
from src.models import Persona, Question, AgentResponse
from src.rubrics.base import BaseRubric



# Evaluate an agent response and return a score (1-5).

# Args:
#     persona: Persona being evaluated
#     question: Question that was asked
#     response: Agent's response
#     rubric: Rubric to use for evaluation
#     score_examples: Optional examples for each score level

# Returns:
#     Score from 1.0 to 5.0

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        persona: Persona,
        question: Question,
        response: AgentResponse,
        rubric: BaseRubric,
        score_examples: Optional[dict[int, str]] = None,
    ) -> float:

        pass

    def get_display_name(self) -> str:
        """Name used in logs; override when an evaluator wraps an LLM client."""
        return self.__class__.__name__

    def get_last_evaluation_details(self):
        """Optional per-evaluator metadata for the most recent evaluation."""
        return None

