"""Linguistic Habits rubric."""

from src.rubrics.base import BaseRubric
from src.config.constants import TASK_LINGUISTIC_HABITS


class LinguisticHabitsRubric(BaseRubric):
    """Rubric for evaluating Linguistic Habits task."""

    def __init__(self):
        """Initialize Linguistic Habits rubric."""
        super().__init__(task_name=TASK_LINGUISTIC_HABITS)

    def get_task_description(self) -> str:
        """Get task description."""
        return (
            "The evaluation task of 'linguistic habit' assesses the persona's adherence "
            "to its characteristically unique syntax, tone, and lingo, ensuring that these "
            "elements are consistently utilized throughout the persona's dialogue. This includes "
            "avoiding generic language patterns (such as 'As a [persona]') and integrating "
            "specific idiomatic expressions, colloquialisms, or jargon that define the persona's "
            "distinctive verbal identity. The aim is to evaluate how effectively the persona "
            "maintains its linguistic uniqueness in various contexts."
        )

    def get_question_requirements(self) -> str:
        """Get question requirements."""
        return (
            "For questions to effectively evaluate a persona's response in terms of "
            '"Linguistic Habits," you must frame the question in a context that naturally '
            "prompts the persona to use its distinctive linguistic habits. This can test the "
            "persona's ability to maintain its unique language style in situations that are "
            "typical or challenging for its character. Also, design questions that allow for "
            "different aspects of the persona's linguistic habits to be showcased. This helps "
            "in evaluating the consistency and versatility of the persona in using its unique "
            "linguistic style across various scenarios."
        )

