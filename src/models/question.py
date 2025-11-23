from typing import Optional
from pydantic import BaseModel, Field
from src.config.constants import TASK_NAMES


class Question(BaseModel):
    text: str = Field(..., description="The question text")
    task_type: str = Field(..., description="Type of evaluation task")
    environment_name: str = Field(..., description="Name of the environment")
    persona_name: str = Field(..., description="Name of the persona")
    quality_criteria: Optional[str] = Field(
        default=None, description="Quality criteria for this question"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.task_type not in TASK_NAMES:
            raise ValueError(
                f"task_type must be one of {TASK_NAMES}, got {self.task_type}"
            )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What would you do if you needed to provide a weather forecast for a major storm?",
                "task_type": "expected_action",
                "environment_name": "weather_forecast",
                "persona_name": "weather_forecaster",
                "quality_criteria": "Tests whether agent takes logically appropriate actions",
            }
        }

