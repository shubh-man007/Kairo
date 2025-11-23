from typing import List
from src.models import Persona, Environment, Question
from src.rubrics.base import BaseRubric
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger
from src.config.constants import DEFAULT_QUESTIONS_PER_TASK

logger = get_logger(__name__)


class QuestionGenerator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        num_questions_per_task: int = DEFAULT_QUESTIONS_PER_TASK,
    ):
        self.llm_client = llm_client
        self.num_questions_per_task = num_questions_per_task

    def generate_questions(
        self,
        persona: Persona,
        environment: Environment,
        rubric: BaseRubric,
    ) -> List[Question]:
        
        logger.info(
            f"Generating questions for persona={persona.name}, "
            f"environment={environment.name}, task={rubric.task_name}"
        )

        persona_desc = f"{persona.name}: {persona.description}"
        if persona.attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in persona.attributes.items()])
            persona_desc += f" ({attrs})"

=        prompt = self._build_generation_prompt(
            persona_desc=persona_desc,
            environment=environment,
            rubric=rubric,
        )

        system_prompt = (
            "You are an expert at creating evaluation questions for persona agents. "
            "Generate questions that effectively probe the persona's behavior and characteristics."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.9,  
            )

            # Parse questions from response
            questions = self._parse_questions(
                response=response,
                persona=persona,
                environment=environment,
                rubric=rubric,
            )

            logger.info(f"Generated {len(questions)} questions")
            return questions

        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            # Return a default question on error
            return [
                Question(
                    text=f"How would {persona.name} behave in {environment.name}?",
                    task_type=rubric.task_name,
                    environment_name=environment.name,
                    persona_name=persona.name,
                )
            ]

    def _build_generation_prompt(
        self,
        persona_desc: str,
        environment: Environment,
        rubric: BaseRubric,
    ) -> str:
        """Build the prompt for question generation."""
        prompt = f"""Generate {self.num_questions_per_task} evaluation questions for the following persona and environment.

Persona:
{persona_desc}

Environment:
{environment.name}: {environment.description}

Task: {rubric.task_name}
Task Description: {rubric.get_task_description()}

Question Requirements:
{rubric.get_question_requirements()}

Generate {self.num_questions_per_task} questions that:
1. Are specific to this persona and environment
2. Effectively probe the task being evaluated
3. Are clear and direct
4. Test different aspects of the persona's behavior

Format your response as a numbered list, one question per line."""

        return prompt

    def _parse_questions(
        self,
        response: str,
        persona: Persona,
        environment: Environment,
        rubric: BaseRubric,
    ) -> List[Question]:
        
        questions = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line = line.lstrip("- ").lstrip("• ").lstrip("* ")
            for i in range(1, 20):
                line = line.lstrip(f"{i}. ").lstrip(f"{i}) ")

            line = line.strip()
            if not line or len(line) < 10: 
                continue

            if "?" in line or line[0].isupper():
                try:
                    question = Question(
                        text=line,
                        task_type=rubric.task_name,
                        environment_name=environment.name,
                        persona_name=persona.name,
                        quality_criteria=rubric.get_question_requirements(),
                    )
                    questions.append(question)
                except Exception as e:
                    logger.warning(f"Error creating question from line '{line}': {e}")
                    continue

            if len(questions) >= self.num_questions_per_task:
                break

        if not questions:
            questions.append(
                Question(
                    text=f"How would {persona.name} behave in {environment.name}?",
                    task_type=rubric.task_name,
                    environment_name=environment.name,
                    persona_name=persona.name,
                )
            )

        return questions[: self.num_questions_per_task]

