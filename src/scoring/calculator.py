from typing import List, Dict
from src.models import EvaluationResult, TaskScore, PersonaScore
from src.config.constants import TASK_NAMES, SCORE_MIN, SCORE_MAX
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PersonaScoreCalculator:
    @staticmethod
    def calculate_persona_score(
        evaluation_results: List[EvaluationResult],
        persona_name: str,
    ) -> PersonaScore:
        
        if not evaluation_results:
            raise ValueError("No evaluation results provided")

        task_scores: Dict[str, List[float]] = {task: [] for task in TASK_NAMES}
        total_questions = len(evaluation_results)
        total_evaluations = 0

        for result in evaluation_results:
            for task_score in result.task_scores:
                task_type = task_score.task_type
                if task_type in task_scores:
                    task_scores[task_type].append(task_score.score)
                    total_evaluations += 1

        task_averages: Dict[str, float] = {}
        for task_type, scores in task_scores.items():
            if scores:
                task_averages[task_type] = sum(scores) / len(scores)
            else:
                task_averages[task_type] = (SCORE_MIN + SCORE_MAX) / 2.0

        if task_averages:
            overall_score = sum(task_averages.values()) / len(task_averages)
        else:
            overall_score = (SCORE_MIN + SCORE_MAX) / 2.0

        logger.info(
            f"Calculated PersonaScore for {persona_name}: "
            f"{overall_score:.2f} (from {total_evaluations} evaluations)"
        )

        return PersonaScore(
            persona_name=persona_name,
            overall_score=overall_score,
            task_averages=task_averages,
            total_questions=total_questions,
            total_evaluations=total_evaluations,
        )

    @staticmethod
    def calculate_task_average(
        evaluation_results: List[EvaluationResult],
        task_type: str,
    ) -> float:
        
        scores = []
        for result in evaluation_results:
            for task_score in result.task_scores:
                if task_score.task_type == task_type:
                    scores.append(task_score.score)

        if not scores:
            return (SCORE_MIN + SCORE_MAX) / 2.0

        return sum(scores) / len(scores)
