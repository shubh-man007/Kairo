from typing import List, Optional
from src.models import (
    Persona,
    Environment,
    Question,
    AgentResponse,
    EvaluationResult,
    TaskScore,
)
from src.rubrics import (
    BaseRubric,
    ExpectedActionRubric,
    LinguisticHabitsRubric,
    PersonaConsistencyRubric,
    ToxicityControlRubric,
    ActionJustificationRubric,
    ExampleGenerator,
)
from src.pipeline import EnvironmentSelector, QuestionGenerator, AgentGenerator
from src.agents.base import BaseAgent
from src.evaluators import BaseEvaluator
from src.scoring import PersonaScoreCalculator
from src.models import PersonaScore
from src.llm.base import BaseLLMClient
from src.config.constants import TASK_NAMES
from src.utils.logging import get_logger

logger = get_logger(__name__)



# Orchestrates the complete evaluation pipeline.

# Coordinates all components to perform end-to-end evaluation:
# 1. Environment selection
# 2. Question generation
# 3. Agent response generation
# 4. Example generation
# 5. Evaluation
# 6. Score calculation


class EvaluationOrchestrator:
    def __init__(
        self,
        generator_client: BaseLLMClient,
        agent_client: BaseLLMClient,
        evaluator: BaseEvaluator,
        environment_pool: List[Environment],
        example_generator_client: Optional[BaseLLMClient] = None,
        agent: Optional[BaseAgent] = None,
    ):
        
        self.generator_client = generator_client
        self.agent_client = agent_client
        self.evaluator = evaluator
        self.environment_pool = environment_pool
        self.example_generator_client = example_generator_client or generator_client
        self.agent = agent

        self.environment_selector = EnvironmentSelector(
            llm_client=generator_client,
            environment_pool=environment_pool,
        )
        self.question_generator = QuestionGenerator(llm_client=generator_client)
        
        if agent:
            self.agent_generator = None
            logger.info("Using provided BaseAgent instance")
        else:
            self.agent_generator = AgentGenerator(llm_client=agent_client)
            logger.info("Using AgentGenerator with LLM client")
        
        self.example_generator = ExampleGenerator(
            llm_client=self.example_generator_client
        )

        self.rubrics = {
            TASK_NAMES[0]: ExpectedActionRubric(),
            TASK_NAMES[1]: LinguisticHabitsRubric(),
            TASK_NAMES[2]: PersonaConsistencyRubric(),
            TASK_NAMES[3]: ToxicityControlRubric(),
            TASK_NAMES[4]: ActionJustificationRubric(),
        }

        logger.info("Initialized EvaluationOrchestrator")

    def evaluate_persona(
        self,
        persona: Persona,
        num_environments: Optional[int] = None,
        num_questions_per_task: Optional[int] = None,
    ) -> List[EvaluationResult]:
        
        logger.info(f"Starting evaluation for persona: {persona.name}")

        if num_environments:
            self.environment_selector.num_environments = num_environments

        selected_environments = self.environment_selector.select_environments(persona)
        logger.info(f"Selected {len(selected_environments)} environments")

        all_results = []

        for environment in selected_environments:
            logger.info(f"Processing environment: {environment.name}")

            for task_name in TASK_NAMES:
                rubric = self.rubrics[task_name]
                logger.info(f"Processing task: {task_name}")

                if num_questions_per_task:
                    self.question_generator.num_questions_per_task = (
                        num_questions_per_task
                    )

                questions = self.question_generator.generate_questions(
                    persona=persona,
                    environment=environment,
                    rubric=rubric,
                )

                for question in questions:
                    if self.agent:
                        response = self.agent.respond(
                            persona=persona, question=question
                        )
                    else:
                        response = self.agent_generator.generate_response(
                            persona=persona, question=question
                        )

                    examples = self.example_generator.generate_examples(
                        rubric=rubric, persona=persona, question=question
                    )

                    task_scores = []
                    score = self.evaluator.evaluate(
                        persona=persona,
                        question=question,
                        response=response,
                        rubric=rubric,
                        score_examples=examples,
                    )

                    individual_scores = None
                    if hasattr(self.evaluator, "get_individual_scores"):
                        individual_scores = self.evaluator.get_individual_scores(
                            persona=persona,
                            question=question,
                            response=response,
                            rubric=rubric,
                            score_examples=examples,
                        )

                    task_score = TaskScore(
                        task_type=task_name,
                        score=score,
                        evaluator_scores=individual_scores,
                    )
                    task_scores.append(task_score)

                    result = EvaluationResult(
                        question_text=question.text,
                        response_text=response.text,
                        persona_name=persona.name,
                        environment_name=environment.name,
                        task_scores=task_scores,
                    )
                    all_results.append(result)

        logger.info(
            f"Completed evaluation: {len(all_results)} results for {persona.name}"
        )
        return all_results

    def evaluate_persona_with_score(
        self,
        persona: Persona,
        num_environments: Optional[int] = None,
        num_questions_per_task: Optional[int] = None,
    ) -> tuple[List[EvaluationResult], PersonaScore]:
        
        results = self.evaluate_persona(
            persona=persona,
            num_environments=num_environments,
            num_questions_per_task=num_questions_per_task,
        )

        persona_score = PersonaScoreCalculator.calculate_persona_score(
            evaluation_results=results,
            persona_name=persona.name,
        )

        return results, persona_score

