"""Base rubric class for evaluation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
from src.models import Persona, Question, AgentResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseRubric(ABC):
    """
    Abstract base class for evaluation rubrics.

    Each rubric defines scoring criteria and provides templates
    for evaluation prompts.
    """

    def __init__(self, task_name: str, template_path: Optional[Path] = None):
        """
        Initialize the rubric.

        Args:
            task_name: Name of the evaluation task
            template_path: Optional path to rubric template file
        """
        self.task_name = task_name
        self.template_path = template_path or self._get_default_template_path()
        self.template = self._load_template()

    def _get_default_template_path(self) -> Path:
        """Get the default template path for this rubric."""
        rubric_dir = Path(__file__).parent.parent.parent / "data" / "rubrics" / "examples"
        # Map task names to file names
        task_to_file = {
            "expected_action": "Expected Action.txt",
            "linguistic_habits": "Linguistic Habits.txt",
            "persona_consistency": "Persona Consistency.txt",
            "toxicity_control": "Toxicity.txt",
            "action_justification": "Action Justification.txt",
        }
        # Try direct lookup first
        task_key = self.task_name.lower()
        filename = task_to_file.get(task_key)
        
        # If not found, try with underscores replaced
        if not filename:
            task_key_normalized = task_key.replace("_", " ")
            # Try to match by converting to title case
            for key, value in task_to_file.items():
                if key.replace("_", " ").title() == task_key_normalized.title():
                    filename = value
                    break
        
        if not filename:
            raise ValueError(f"No template found for task: {self.task_name}")
        return rubric_dir / filename

    def _load_template(self) -> str:
        """Load the rubric template from file."""
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Template file not found: {self.template_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            raise

    def get_guidelines(self) -> Dict[int, str]:
        """
        Get scoring guidelines for each score level.

        Returns:
            Dictionary mapping score (1-5) to guideline description
        """
        # Extract guidelines from template
        guidelines = {}
        lines = self.template.split("\n")
        current_score = None
        current_text = []

        for line in lines:
            if line.strip().startswith("Score = "):
                if current_score is not None:
                    guidelines[current_score] = " ".join(current_text).strip()
                # Extract score number
                score_str = line.split("=")[1].split(":")[0].strip()
                try:
                    current_score = int(score_str)
                    current_text = []
                    # Get text after colon
                    if ":" in line:
                        text_after_colon = line.split(":", 1)[1].strip()
                        if text_after_colon:
                            current_text.append(text_after_colon)
                except ValueError:
                    continue
            elif current_score is not None and line.strip():
                current_text.append(line.strip())

        # Add last score
        if current_score is not None:
            guidelines[current_score] = " ".join(current_text).strip()

        return guidelines

    def format_evaluation_prompt(
        self,
        persona: Persona,
        question: Question,
        response: AgentResponse,
        score_examples: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        Format the evaluation prompt with persona, question, response, and examples.

        Args:
            persona: Persona being evaluated
            question: Question that was asked
            response: Agent's response
            score_examples: Optional dictionary mapping score (1-5) to example response

        Returns:
            Formatted evaluation prompt
        """
        # Format persona description
        persona_desc = f"{persona.name}: {persona.description}"
        if persona.attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in persona.attributes.items()])
            persona_desc += f" ({attrs})"

        # Format score examples
        score_example_text = ""
        if score_examples:
            score_example_lines = []
            for score in sorted(score_examples.keys()):
                score_example_lines.append(f"Score = {score}:")
                score_example_lines.append(score_examples[score])
                score_example_lines.append("")
            score_example_text = "\n".join(score_example_lines)

        # Replace placeholders in template
        prompt = self.template.replace("{persona}", persona_desc)
        prompt = prompt.replace("{question}", question.text)
        prompt = prompt.replace("{response}", response.text)
        prompt = prompt.replace("{score_example}", score_example_text)

        return prompt

    @abstractmethod
    def get_task_description(self) -> str:
        """
        Get the task description for this rubric.

        Returns:
            Task description text
        """
        pass

    def get_question_requirements(self) -> str:
        """
        Get question requirements for this task.

        Returns:
            Question requirements text
        """
        # Default implementation - can be overridden
        return ""

