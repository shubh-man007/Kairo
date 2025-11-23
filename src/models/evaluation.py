from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from src.config.constants import SCORE_MIN, SCORE_MAX


class TaskScore(BaseModel):
    task_type: str = Field(..., description="Type of evaluation task")
    score: float = Field(..., ge=SCORE_MIN, le=SCORE_MAX, description="Score (1-5)")
    evaluator_scores: Optional[List[float]] = Field(
        default=None, description="Individual scores from each evaluator"
    )
    evaluator_models: Optional[List[str]] = Field(
        default=None, description="Models used for evaluation"
    )
    justification: Optional[str] = Field(
        default=None, description="Justification for the score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_type": "expected_action",
                "score": 4.5,
                "evaluator_scores": [4.0, 5.0],
                "evaluator_models": ["gpt-4o-mini", "claude-3-5-sonnet"],
                "justification": "Agent took appropriate actions for a weather forecaster",
            }
        }


class EvaluationResult(BaseModel):
    question_id: Optional[str] = Field(
        default=None, description="Identifier for the question"
    )
    question_text: str = Field(..., description="The question text")
    response_text: str = Field(..., description="The agent's response")
    persona_name: str = Field(..., description="Name of the persona")
    environment_name: str = Field(..., description="Name of the environment")
    task_scores: List[TaskScore] = Field(
        ..., description="Scores for each evaluation task"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the evaluation was performed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional evaluation metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question_id": "q_001",
                "question_text": "What would you do if...",
                "response_text": "I would analyze...",
                "persona_name": "weather_forecaster",
                "environment_name": "weather_forecast",
                "task_scores": [
                    {
                        "task_type": "expected_action",
                        "score": 4.5,
                        "evaluator_scores": [4.0, 5.0],
                    }
                ],
                "timestamp": "2024-01-15T10:30:00",
            }
        }


class PersonaScore(BaseModel):
    persona_name: str = Field(..., description="Name of the persona")
    overall_score: float = Field(
        ..., ge=SCORE_MIN, le=SCORE_MAX, description="Overall PersonaScore (1-5)"
    )
    task_averages: Dict[str, float] = Field(
        ..., description="Average score for each task type"
    )
    total_questions: int = Field(..., description="Total number of questions evaluated")
    total_evaluations: int = Field(
        ..., description="Total number of evaluations performed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the score was calculated"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "persona_name": "weather_forecaster",
                "overall_score": 4.2,
                "task_averages": {
                    "expected_action": 4.5,
                    "linguistic_habits": 4.0,
                    "persona_consistency": 4.3,
                    "toxicity_control": 5.0,
                    "action_justification": 3.5,
                },
                "total_questions": 50,
                "total_evaluations": 250,
                "timestamp": "2024-01-15T10:30:00",
            }
        }

