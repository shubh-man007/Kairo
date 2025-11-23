from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    text: str = Field(..., description="The agent's response text")
    question_id: Optional[str] = Field(
        default=None, description="Identifier for the question"
    )
    persona_name: str = Field(..., description="Name of the persona used")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata (model used, tokens, etc.)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the response was generated"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I would analyze the storm's trajectory using meteorological data...",
                "question_id": "q_001",
                "persona_name": "weather_forecaster",
                "metadata": {"model": "gpt-4o-mini", "temperature": 0.9},
                "timestamp": "2024-01-15T10:30:00",
            }
        }

