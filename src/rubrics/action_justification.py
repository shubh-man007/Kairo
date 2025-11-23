"""Action Justification rubric."""

from src.rubrics.base import BaseRubric
from src.config.constants import TASK_ACTION_JUSTIFICATION


class ActionJustificationRubric(BaseRubric):
    """Rubric for evaluating Action Justification task."""

    def __init__(self):
        """Initialize Action Justification rubric."""
        super().__init__(task_name=TASK_ACTION_JUSTIFICATION)

    def get_task_description(self) -> str:
        """Get task description."""
        return (
            "Evaluate the persona's response to determine how effectively and convincingly "
            "it justifies a given action based on its described attributes and situation. "
            "The response should reflect the persona's internal reasoning and motivations "
            "for the action, aligning with its established characteristics and context."
        )

    def get_question_requirements(self) -> str:
        """Get question requirements."""
        return (
            "When crafting questions to evaluate a persona's ability to justify actions, "
            "ensure each question includes a detailed description of a specific situation along "
            "with a corresponding action the persona took. This action must be one that is "
            "expected to be taken by the persona in the given situation. The purpose of the "
            "question is to probe the persona to justify actions that it takes in given situations. "
            "DO NOT help the persona by telling it what attribute to align its response to. "
            "Simply ask the persona to justify the action in the given situation. The questions "
            "should be designed to challenge the persona to articulate a clear and reasonable "
            "justification for its actions, reflecting its defined attributes and the circumstances "
            "presented. This format will allow the evaluator to assess the depth of the persona's "
            "understanding of its own motivations and the consistency of its actions with its "
            "character traits and the context provided."
        )

