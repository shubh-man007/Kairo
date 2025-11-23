from typing import Optional
from pydantic import BaseModel, Field


class Environment(BaseModel):
    name: str = Field(..., description="Name of the environment")
    description: str = Field(..., description="Description of the environment")
    domain: Optional[str] = Field(
        default=None, description="Domain category (e.g., 'weather', 'travel')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "weather_forecast",
                "description": "Environment for weather forecasting scenarios",
                "domain": "weather",
            }
        }

