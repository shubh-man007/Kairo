import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Persona
from src.pipeline import EvaluationOrchestrator
from src.agents import WeatherAgent
from src.evaluators import LLMEvaluator, EnsembleEvaluator
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
    from src.models import Environment
    return [Environment(**env) for env in data]


def main():
    print(">> Kairo Eval")
    data_dir = Path(__file__).parent.parent / "data"
    persona_path = data_dir / "personas" / "weather_persona.json"
    environments_path = data_dir / "environments" / "weather_environments.json"

    try:
        persona = load_persona(persona_path)
        print(f"\n[OK] Loaded persona: {persona.name}")
    except Exception as e:
        print(f"\n[ERROR] Error loading persona: {e}")
        return 1

    try:
        environments = load_environments(environments_path)
        print(f"[OK] Loaded {len(environments)} environments")
    except Exception as e:
        print(f"\n[ERROR] Error loading environments: {e}")
        return 1

    settings = get_settings()
    
    print("\n[INFO] Initializing LLM clients...")
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

    print("[INFO] Creating weather agent...")
    weather_agent = WeatherAgent(llm_client=agent_client)

    print("[INFO] Creating evaluators...")
    evaluator1 = LLMEvaluator(
        llm_client=create_llm_client(
            provider="openai",
            model=settings.evaluator_model_1,
            temperature=0.0,
        )
    )

    evaluator2 = LLMEvaluator(
        llm_client=create_llm_client(
            # Use Anthropic (Claude) for the second evaluator by default.
            provider="anthropic",
            model=settings.evaluator_model_2,
            temperature=0.0,
        )
    )
    
    ensemble_evaluator = EnsembleEvaluator(evaluators=[evaluator1, evaluator2])

    print("[INFO] Creating evaluation orchestrator...")
    orchestrator = EvaluationOrchestrator(
        generator_client=generator_client,
        agent_client=agent_client,
        evaluator=ensemble_evaluator,
        environment_pool=environments,
        agent=weather_agent,  
    )

    
    print(">> Starting Evaluation")
    print("\n[INFO] Running evaluation with:")
    print(f"  - Persona: {persona.name}")
    print(f"  - Environments: 1 (reduced for testing)")
    print(f"  - Questions per task: 1 (reduced for testing)")
    print(f"  - Total tasks: 5")

    try:
        results, persona_score = orchestrator.evaluate_persona_with_score(
            persona=persona,
            num_environments=1,  
            num_questions_per_task=1,  
        )

        print(">> Evaluation Results")
        print(f"\n[SUCCESS] Completed evaluation!")
        print(f"  Total questions evaluated: {persona_score.total_questions}")
        print(f"  Total evaluations: {persona_score.total_evaluations}")
        print(f"\n  Overall PersonaScore: {persona_score.overall_score:.2f}/5.0")
        
        print(f"\n  Task Averages:")
        for task, avg_score in persona_score.task_averages.items():
            print(f"    - {task}: {avg_score:.2f}/5.0")

        print(f"\n[INFO] Sample results:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n  Result {i}:")
            print(f"    Question: {result.question_text}")
            print(f"    Response: {result.response_text}")
            if result.task_scores:
                print(f"    Score: {result.task_scores[0].score:.2f}/5.0")

        print("[SUCCESS] Evaluation complete!")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

