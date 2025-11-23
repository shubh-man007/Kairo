import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    evaluator_model_1: str = os.getenv("EVALUATOR_MODEL_1", "gpt-4o-mini")
    evaluator_model_2: str = os.getenv("EVALUATOR_MODEL_2", "claude-3-5-sonnet-20241022")
    generator_model: str = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
    generator_temperature: float = float(os.getenv("GENERATOR_TEMPERATURE", "0.9"))
    evaluator_temperature: float = float(os.getenv("EVALUATOR_TEMPERATURE", "0.0"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE", "logs/evaluation.log")

    def validate(self) -> None:
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError(
                "At least one API key (OpenAI or Anthropic) must be provided"
            )

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate()
    return _settings

