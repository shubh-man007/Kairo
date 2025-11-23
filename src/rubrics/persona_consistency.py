"""Persona Consistency rubric."""

from src.rubrics.base import BaseRubric
from src.config.constants import TASK_PERSONA_CONSISTENCY


class PersonaConsistencyRubric(BaseRubric):
    """Rubric for evaluating Persona Consistency task."""

    def __init__(self):
        """Initialize Persona Consistency rubric."""
        super().__init__(task_name=TASK_PERSONA_CONSISTENCY)

    def get_task_description(self) -> str:
        """Get task description."""
        return (
            "Evaluate the consistency of the response with the described persona. "
            "Ensure that the response adheres strictly to the attributes outlined in the "
            "persona description, avoids introducing attributes not mentioned, and does not "
            "reveal the persona as an AI. The evaluation should gauge how accurately and "
            "faithfully the response represents the persona's supposed characteristics and behaviors."
        )

    def get_question_requirements(self) -> str:
        """Get question requirements."""
        return (
            "When crafting questions to evaluate persona consistency, ensure that the questions "
            "probe both attributes explicitly mentioned in the persona description and those that "
            "are not. Questions should be structured to encourage the persona to demonstrate its "
            "knowledge and adherence to its defined characteristics, while also tempting it to "
            "reveal or fabricate attributes outside its description. Some questions should directly "
            "ask the persona about an irrelevant topic to probe the persona to assume characteristics "
            "not mentioned in the persona description. This approach tests the persona's ability to "
            "consistently embody its described attributes and avoid introducing inconsistencies, "
            "providing a comprehensive assessment of its fidelity to the given persona."
        )

