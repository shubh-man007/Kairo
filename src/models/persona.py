from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Persona(BaseModel):
    name: str = Field(..., description="Name or identifier of the persona")
    description: str = Field(..., description="Detailed description of the persona")
    attributes: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional persona attributes (age, location, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "weather_forecaster",
                "description": "A professional weather forecaster with expertise in meteorology",
                "attributes": {
                    "age": 35,
                    "location": "New York",
                    "expertise": ["weather prediction", "climate analysis"],
                },
            }
        }

