"""Expected Action rubric."""

from src.rubrics.base import BaseRubric
from src.config.constants import TASK_EXPECTED_ACTION


class ExpectedActionRubric(BaseRubric):
    """Rubric for evaluating Expected Action task."""

    def __init__(self):
        """Initialize Expected Action rubric."""
        super().__init__(task_name=TASK_EXPECTED_ACTION)

    def get_task_description(self) -> str:
        """Get task description."""
        return (
            "The persona takes actions within its response to the question that is "
            "logically expected of the persona in the setting of the question."
        )

    def get_question_requirements(self) -> str:
        """Get question requirements."""
        return (
            "For questions to effectively evaluate a persona's response in terms of "
            '"Expected Action," they must be specifically designed to elicit actions that are '
            "indicative of the persona's characteristics and behavior within the given setting. "
            "Each question should probe the persona to take multiple distinct actions in the "
            "given setting. Questions should be clear, direct, and relevant to the core "
            "attributes of the persona, ensuring that the answers can clearly demonstrate "
            "whether the persona acts as expected in the described context."
        )

