import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Persona, Environment, Question
from src.agents import WeatherAgent
from src.rubrics import ExpectedActionRubric, ExampleGenerator
from src.pipeline import EnvironmentSelector, QuestionGenerator, AgentGenerator
from src.evaluators import LLMEvaluator, EnsembleEvaluator
from src.scoring import PersonaScoreCalculator
from src.llm import create_llm_client
from src.config import get_settings
from src.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def load_persona(file_path: Path) -> Persona:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Persona(**data)


def load_environments(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Environment(**env) for env in data]


def test_phase1():
    try:
        from src.config import get_settings, TASK_NAMES, SCORE_RANGE
        from src.models import Persona, Environment, Question, AgentResponse
        
        settings = get_settings()
        print(f"[OK] Configuration loaded")
        print(f"  - OpenAI model: {settings.openai_model}")
        print(f"  - Generator temperature: {settings.generator_temperature}")
        
        print(f"[OK] Constants loaded")
        print(f"  - Tasks: {len(TASK_NAMES)}")
        print(f"  - Score range: {SCORE_RANGE}")
        
        print(f"[OK] Data models working")
        return True
    except Exception as e:
        print(f"[ERROR] Phase 1 test failed: {e}")
        return False


def test_phase2(persona, environments):
    try:
        settings = get_settings()
        generator_client = create_llm_client(
            provider="openai",
            model=settings.generator_model,
            temperature=0.9,
        )

        print("\n[INFO] Testing rubrics...")
        rubric = ExpectedActionRubric()
        guidelines = rubric.get_guidelines()
        assert len(guidelines) == 5, "Should have 5 score levels"
        print(f"[OK] Rubric loaded: {rubric.task_name}")

        print("\n[INFO] Testing environment selector...")
        selector = EnvironmentSelector(
            llm_client=generator_client,
            environment_pool=environments,
            num_environments=2,
        )
        selected = selector.select_environments(persona)
        assert len(selected) > 0, "Should select at least one environment"
        print(f"[OK] Selected {len(selected)} environments")

        print("\n[INFO] Testing question generator...")
        qgen = QuestionGenerator(llm_client=generator_client, num_questions_per_task=1)
        questions = qgen.generate_questions(
            persona=persona,
            environment=selected[0],
            rubric=rubric,
        )
        assert len(questions) > 0, "Should generate at least one question"
        print(f"[OK] Generated {len(questions)} questions")

        print("\n[INFO] Testing example generator...")
        example_gen = ExampleGenerator(llm_client=generator_client)
        examples = example_gen.generate_examples(
            rubric=rubric,
            persona=persona,
            question=questions[0],
        )
        assert len(examples) == 5, "Should generate 5 examples"
        print(f"[OK] Generated {len(examples)} examples")

        return True, questions[0]

    except Exception as e:
        print(f"[ERROR] Phase 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_phase3(persona, question):
    try:
        settings = get_settings()
        agent_client = create_llm_client(
            provider="openai",
            model=settings.generator_model,
            temperature=settings.generator_temperature,
        )

        print("\n[INFO] Testing agent generator...")
        agent_gen = AgentGenerator(llm_client=agent_client)
        response = agent_gen.generate_response(persona=persona, question=question)
        assert response.text is not None, "Should generate response"
        print(f"[OK] Generated response ({len(response.text)} chars)")

        print("\n[INFO] Testing evaluator...")
        evaluator1 = LLMEvaluator(
            llm_client=create_llm_client(
                provider="openai",
                model=settings.evaluator_model_1,
                temperature=0.0,
            )
        )
        rubric = ExpectedActionRubric()
        score = evaluator1.evaluate(
            persona=persona,
            question=question,
            response=response,
            rubric=rubric,
        )
        assert 1.0 <= score <= 5.0, "Score should be in valid range"
        print(f"[OK] Evaluated response: {score:.2f}/5.0")

        print("\n[INFO] Testing ensemble evaluator...")
        evaluator2 = LLMEvaluator(
            llm_client=create_llm_client(
                provider="openai",
                model=settings.evaluator_model_1,
                temperature=0.0,
            )
        )
        ensemble = EnsembleEvaluator(evaluators=[evaluator1, evaluator2])
        ensemble_score = ensemble.evaluate(
            persona=persona,
            question=question,
            response=response,
            rubric=rubric,
        )
        assert 1.0 <= ensemble_score <= 5.0, "Score should be in valid range"
        print(f"[OK] Ensemble score: {ensemble_score:.2f}/5.0")

        return True

    except Exception as e:
        print(f"[ERROR] Phase 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent(persona, question):
    try:
        settings = get_settings()
        agent_client = create_llm_client(
            provider="openai",
            model=settings.generator_model,
            temperature=settings.generator_temperature,
        )

        print("\n[INFO] Creating weather agent...")
        weather_agent = WeatherAgent(llm_client=agent_client)
        print("[OK] Weather agent created")

        print("\n[INFO] Testing location extraction...")
        test_question = Question(
            text="What's the weather like in New York?",
            task_type="expected_action",
            environment_name="test",
            persona_name=persona.name,
        )
        location = weather_agent._extract_location(test_question.text)
        print(f"[OK] Extracted location: {location}")

        print("\n[INFO] Testing location formatting...")
        formatted = weather_agent._format_location("San Francisco")
        assert formatted == "San+Francisco", "Should replace spaces with +"
        print(f"[OK] Formatted location: {formatted}")

        print("\n[INFO] Testing weather agent response...")
        response = weather_agent.respond(persona=persona, question=question)
        assert response.text is not None, "Should generate response"
        print(f"[OK] Generated response ({len(response.text)} chars)")
        print(f"  Location used: {response.metadata.get('location_used')}")
        print(f"  Weather data fetched: {response.metadata.get('weather_data_fetched')}")

        return True

    except Exception as e:
        print(f"[ERROR] Weather agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline(persona, environments):
    try:
        from src.pipeline import EvaluationOrchestrator

        settings = get_settings()
        generator_client = create_llm_client(
            provider="openai",
            model=settings.generator_model,
            temperature=0.9,
        )
        agent_client = create_llm_client(
            provider="openai",
            model=settings.generator_model,
            temperature=settings.generator_temperature,
        )

        weather_agent = WeatherAgent(llm_client=agent_client)

        evaluator1 = LLMEvaluator(
            llm_client=create_llm_client(
                provider="openai",
                model=settings.evaluator_model_1,
                temperature=0.0,
            )
        )
        evaluator2 = LLMEvaluator(
            llm_client=create_llm_client(
                provider="openai",
                model=settings.evaluator_model_1,
                temperature=0.0,
            )
        )
        ensemble = EnsembleEvaluator(evaluators=[evaluator1, evaluator2])

        print("\n[INFO] Creating orchestrator with weather agent...")
        orchestrator = EvaluationOrchestrator(
            generator_client=generator_client,
            agent_client=agent_client,
            evaluator=ensemble,
            environment_pool=environments,
            agent=weather_agent,
        )
        print("[OK] Orchestrator created")

        print("\n[INFO] Running mini evaluation...")
        print("  This will test the full pipeline with reduced parameters")
        results, persona_score = orchestrator.evaluate_persona_with_score(
            persona=persona,
            num_environments=1,
            num_questions_per_task=1,
        )

        print(f"\n[OK] Evaluation complete!")
        print(f"  Total questions: {persona_score.total_questions}")
        print(f"  Total evaluations: {persona_score.total_evaluations}")
        print(f"  Overall PersonaScore: {persona_score.overall_score:.2f}/5.0")

        return True

    except Exception as e:
        print(f"[ERROR] Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    data_dir = Path(__file__).parent.parent / "data"
    persona_path = data_dir / "personas" / "weather_persona.json"
    environments_path = data_dir / "environments" / "weather_environments.json"

    try:
        persona = load_persona(persona_path)
        environments = load_environments(environments_path)
        print(f"\n[OK] Loaded persona: {persona.name}")
        print(f"[OK] Loaded {len(environments)} environments")
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        return 1

    results = {}

    results["phase1"] = test_phase1()

    phase2_result, question = test_phase2(persona, environments)
    results["phase2"] = phase2_result

    if not question:
        print("\n[ERROR] Cannot continue without question from Phase 2")
        return 1

    results["phase3"] = test_phase3(persona, question)

    results["weather_agent"] = test_agent(persona, question)

    results["full_pipeline"] = test_pipeline(persona, environments)

    for test_name, passed in results.items():
        status = "[SUCCESS]" if passed else "[FAILED]"
        print(f"{status} {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[SUCCESS] Done")
        return 0
    else:
        print("\n[ERROR] Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

