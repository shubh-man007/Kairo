"""Example generator for rubric scoring examples."""

from typing import Dict
from src.models import Persona, Question
from src.rubrics.base import BaseRubric
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExampleGenerator:
    """
    Generates custom scoring examples for each (persona, question) pair.

    This implements the Ξ_t component from PersonaGym - generating
    persona-specific examples at each score level (1-5).
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize the example generator.

        Args:
            llm_client: LLM client for generating examples
        """
        self.llm_client = llm_client

    def generate_examples(
        self,
        rubric: BaseRubric,
        persona: Persona,
        question: Question,
    ) -> Dict[int, str]:
        """
        Generate scoring examples for each score level (1-5).

        Args:
            rubric: The rubric to generate examples for
            persona: Persona description
            question: Question being evaluated

        Returns:
            Dictionary mapping score (1-5) to example response text
        """
        logger.info(
            f"Generating examples for persona={persona.name}, "
            f"task={rubric.task_name}, question={question.text}"
        )

        # Format persona description
        persona_desc = f"{persona.name}: {persona.description}"
        if persona.attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in persona.attributes.items()])
            persona_desc += f" ({attrs})"

        # Get scoring guidelines
        guidelines = rubric.get_guidelines()

        # Build prompt for example generation
        prompt = self._build_example_generation_prompt(
            rubric=rubric,
            persona_desc=persona_desc,
            question=question.text,
            guidelines=guidelines,
        )

        # Generate examples using LLM
        system_prompt = (
            "You are an expert at creating realistic example responses for persona evaluation. "
            "Generate example responses that demonstrate different quality levels for the given persona and question."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.9,  # Creative generation
            )
            examples = self._parse_examples(response)
            logger.info(f"Generated {len(examples)} examples")
            return examples
        except Exception as e:
            logger.error(f"Error generating examples: {e}")
            # Return empty examples on error
            return {i: "" for i in range(1, 6)}

    def _build_example_generation_prompt(
        self,
        rubric: BaseRubric,
        persona_desc: str,
        question: str,
        guidelines: Dict[int, str],
    ) -> str:
        """Build the prompt for generating examples."""
        prompt = f"""Generate 5 example responses for a persona evaluation task.

Task: {rubric.task_name}
Task Description: {rubric.get_task_description()}

Persona Description:
{persona_desc}

Question:
{question}

For each score level (1-5), generate a realistic example response that demonstrates that score level according to the following criteria:

"""
        for score in range(1, 6):
            prompt += f"Score {score}: {guidelines.get(score, 'N/A')}\n\n"

        prompt += """Generate one example response for each score level (1-5). Format your response as:

Score 1:
[Example response text here]

Score 2:
[Example response text here]

Score 3:
[Example response text here]

Score 4:
[Example response text here]

Score 5:
[Example response text here]

Make sure each example is realistic and clearly demonstrates the quality level indicated by its score."""

        return prompt

    def _parse_examples(self, response: str) -> Dict[int, str]:
        """
        Parse examples from LLM response.

        Args:
            response: LLM response text

        Returns:
            Dictionary mapping score (1-5) to example text
        """
        examples = {}
        lines = response.split("\n")
        current_score = None
        current_text = []

        for line in lines:
            line_lower = line.lower().strip()
            # Check if line indicates a score
            if "score" in line_lower and any(str(i) in line_lower for i in range(1, 6)):
                # Save previous score if exists
                if current_score is not None:
                    examples[current_score] = "\n".join(current_text).strip()
                # Extract new score
                for i in range(1, 6):
                    if str(i) in line_lower:
                        current_score = i
                        current_text = []
                        break
            elif current_score is not None:
                # Add line to current example
                if line.strip():
                    current_text.append(line.strip())

        # Save last score
        if current_score is not None:
            examples[current_score] = "\n".join(current_text).strip()

        # Ensure we have all 5 scores
        for i in range(1, 6):
            if i not in examples:
                examples[i] = f"Example response for score {i}"

        return examples

