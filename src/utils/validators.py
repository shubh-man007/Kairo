from typing import Any
from src.config.constants import SCORE_MIN, SCORE_MAX, TASK_NAMES


def validate_score(score: float) -> bool:
    return SCORE_MIN <= score <= SCORE_MAX


def validate_task_type(task_type: str) -> bool:
    return task_type in TASK_NAMES


def validate_not_empty(value: Any, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

