from typing import List, Optional
from src.evaluators.base import BaseEvaluator
from src.models import Persona, Question, AgentResponse
from src.rubrics.base import BaseRubric
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Ensemble evaluator that combines scores from multiple evaluators.
# Implements the ensembling mechanism from PersonaGym - averages
# scores from 2+ different evaluators to reduce bias.


class EnsembleEvaluator(BaseEvaluator):
    def __init__(self, evaluators: List[BaseEvaluator]):
        if len(evaluators) < 1:
            raise ValueError("At least one evaluator is required")
        self.evaluators = evaluators
        self._last_details = []

    def evaluate(
        self,
        persona: Persona,
        question: Question,
        response: AgentResponse,
        rubric: BaseRubric,
        score_examples: Optional[dict[int, str]] = None,
    ) -> float:

        scores = []
        details = []
        for i, evaluator in enumerate(self.evaluators, 1):
            evaluator_name = evaluator.get_display_name()
            try:
                score = evaluator.evaluate(
                    persona=persona,
                    question=question,
                    response=response,
                    rubric=rubric,
                    score_examples=score_examples,
                )
                scores.append(score)
                details.append({"name": evaluator_name, "score": score})
            except Exception as e:
                logger.warning(f"Evaluator {i} failed: {e}, skipping")
                details.append({"name": evaluator_name, "error": str(e)})

        self._last_details = details
        if not scores:
            logger.error("All evaluators failed, returning default score")
            self._last_details.append(
                {"name": "EnsembleEvaluator", "score": 3.0, "note": "default fallback"}
            )
            return 3.0  # Default middle score

        avg_score = sum(scores) / len(scores)
        return avg_score

    def get_individual_scores(
        self,
        persona: Persona,
        question: Question,
        response: AgentResponse,
        rubric: BaseRubric,
        score_examples: Optional[dict[int, str]] = None,
    ) -> List[float]:
    
        if not self._last_details:
            return []

        scores = []
        for detail in self._last_details:
            scores.append(detail.get("score"))

        return scores

    def get_last_evaluation_details(self):
        return [detail.copy() for detail in self._last_details]

