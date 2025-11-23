from typing import List, Dict
from src.models import TaskScore, EvaluationResult
from src.config.constants import SCORE_MIN, SCORE_MAX
from src.utils.logging import get_logger

logger = get_logger(__name__)


def aggregate_task_scores(
    task_scores: List[TaskScore],
) -> Dict[str, float]:
    
    task_totals: Dict[str, List[float]] = {}
    task_counts: Dict[str, int] = {}

    for task_score in task_scores:
        task_type = task_score.task_type
        if task_type not in task_totals:
            task_totals[task_type] = []
            task_counts[task_type] = 0

        task_totals[task_type].append(task_score.score)
        task_counts[task_type] += 1

    task_averages: Dict[str, float] = {}
    for task_type, scores in task_totals.items():
        if scores:
            task_averages[task_type] = sum(scores) / len(scores)
        else:
            task_averages[task_type] = (SCORE_MIN + SCORE_MAX) / 2.0

    return task_averages


def calculate_overall_average(scores: List[float]) -> float:
    if not scores:
        return (SCORE_MIN + SCORE_MAX) / 2.0

    return sum(scores) / len(scores)


def get_score_statistics(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {
            "mean": (SCORE_MIN + SCORE_MAX) / 2.0,
            "min": SCORE_MIN,
            "max": SCORE_MAX,
            "std_dev": 0.0,
        }

    mean = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std_dev = variance ** 0.5

    return {
        "mean": mean,
        "min": min_score,
        "max": max_score,
        "std_dev": std_dev,
    }

