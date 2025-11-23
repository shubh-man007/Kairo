import re
from typing import Optional
from src.evaluators.base import BaseEvaluator
from src.models import Persona, Question, AgentResponse
from src.rubrics.base import BaseRubric
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger
from src.config import get_settings, SCORE_MIN, SCORE_MAX

logger = get_logger(__name__)


class LLMEvaluator(BaseEvaluator):
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.settings = get_settings()

    def evaluate(
        self,
        persona: Persona,
        question: Question,
        response: AgentResponse,
        rubric: BaseRubric,
        score_examples: Optional[dict[int, str]] = None,
    ) -> float:

        logger.info(
            f"Evaluating response for persona={persona.name}, "
            f"task={rubric.task_name}"
        )

        prompt = rubric.format_evaluation_prompt(
            persona=persona,
            question=question,
            response=response,
            score_examples=score_examples,
        )

        try:
            evaluation_text = self.llm_client.generate(
                prompt=prompt,
                system_prompt=None,  # Prompt already contains instructions
                temperature=self.settings.evaluator_temperature,  
            )

            score = self._extract_score(evaluation_text)

            logger.info(f"Extracted score: {score}")
            return score

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return (SCORE_MIN + SCORE_MAX) / 2.0

    def _extract_score(self, evaluation_text: str) -> float:
        patterns = [
            r"final score is\s*([\d.]+)",
            r"score is\s*([\d.]+)",
            r"score:\s*([\d.]+)",
            r"rating:\s*([\d.]+)",
            r"([\d.]+)\s*out of\s*5",
            r"score\s*=\s*([\d.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    score = max(SCORE_MIN, min(SCORE_MAX, score))
                    return score
                except ValueError:
                    continue

        numbers = re.findall(r"\b([1-5](?:\.[0-9]+)?)\b", evaluation_text)
        if numbers:
            try:
                score = float(numbers[-1])  # Take last number found
                score = max(SCORE_MIN, min(SCORE_MAX, score))
                return score
            except ValueError:
                pass

        logger.warning("Could not extract score from evaluation text, using default")
        return (SCORE_MIN + SCORE_MAX) / 2.0

