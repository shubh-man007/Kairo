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
        logger.info(f"Initialized ensemble with {len(evaluators)} evaluators")

    def evaluate(
        self,
        persona: Persona,
        question: Question,
        response: AgentResponse,
        rubric: BaseRubric,
        score_examples: Optional[dict[int, str]] = None,
    ) -> float:

        logger.info(
            f"Ensemble evaluation for persona={persona.name}, "
            f"task={rubric.task_name}, using {len(self.evaluators)} evaluators"
        )

        scores = []
        for i, evaluator in enumerate(self.evaluators, 1):
            try:
                score = evaluator.evaluate(
                    persona=persona,
                    question=question,
                    response=response,
                    rubric=rubric,
                    score_examples=score_examples,
                )
                scores.append(score)
                logger.info(f"Evaluator {i} score: {score:.2f}")
            except Exception as e:
                logger.warning(f"Evaluator {i} failed: {e}, skipping")

        if not scores:
            logger.error("All evaluators failed, returning default score")
            return 3.0  # Default middle score

        avg_score = sum(scores) / len(scores)
        logger.info(f"Ensemble average score: {avg_score:.2f} (from {len(scores)} evaluators)")

        return avg_score

    def get_individual_scores(
        self,
        persona: Persona,
        question: Question,
        response: AgentResponse,
        rubric: BaseRubric,
        score_examples: Optional[dict[int, str]] = None,
    ) -> List[float]:
    
        scores = []
        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(
                    persona=persona,
                    question=question,
                    response=response,
                    rubric=rubric,
                    score_examples=score_examples,
                )
                scores.append(score)
            except Exception as e:
                logger.warning(f"Evaluator failed: {e}")
                scores.append(None)  # Use None to indicate failure

        return scores

