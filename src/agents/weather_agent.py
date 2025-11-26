import os
import requests
from typing import Optional
from dotenv import load_dotenv
from src.agents.base import BaseAgent
from src.models import Persona, Question, AgentResponse
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger
from src.config import get_settings

load_dotenv()
logger = get_logger(__name__)

# Weather agent that uses AccuWeather API to fetch weather data and generates persona-conditioned responses.
class WeatherAgent(BaseAgent):
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.settings = get_settings()
        self.api_key = os.getenv("ACCUWEATHER_API_KEY", "")
        
        if not self.api_key:
            logger.warning("ACCUWEATHER_API_KEY not found in environment variables")

        self.base_url = "https://dataservice.accuweather.com"
        self.location_url = f"{self.base_url}/locations/v1/cities/search"
        self.weather_url = f"{self.base_url}/forecasts/v1/daily/5day"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def respond(
        self,
        persona: Persona,
        question: Question,
    ) -> AgentResponse:
        logger.info(
            f"WeatherAgent responding to question: {question.text}"
        )

        location = self._extract_location(question.text)
        
        weather_data = None
        if location and self.api_key:
            try:
                weather_data = self._fetch_weather_data(location)
            except Exception as e:
                logger.warning(f"Failed to fetch weather data: {e}")

        system_prompt = self._build_system_prompt(persona, weather_data)

        try:
            response_text = self.llm_client.generate(
                prompt=question.text,
                system_prompt=system_prompt,
                temperature=self.settings.generator_temperature,
            )

            response = AgentResponse(
                text=response_text,
                persona_name=persona.name,
                metadata={
                    "model": self.llm_client.model,
                    "temperature": self.settings.generator_temperature,
                    "task_type": question.task_type,
                    "environment": question.environment_name,
                    "location_used": location,
                    "weather_data_fetched": weather_data is not None,
                },
            )

            logger.info(f"Generated response ({len(response_text)} chars)")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return AgentResponse(
                text=f"Error generating response: {str(e)}",
                persona_name=persona.name,
                metadata={"error": str(e)},
            )

    def _extract_location(self, question_text: str) -> Optional[str]:
        question_lower = question_text.lower()
        common_cities = [
            "new york", "london", "paris", "tokyo", "sydney",
            "mumbai", "delhi", "bangalore", "ahmedabad", "pune",
            "chicago", "los angeles", "san francisco", "boston",
        ]
        
        for city in common_cities:
            if city in question_lower:
                return city.title()
        
        return None
    
    def _format_location(self, location: str) -> str:
        return location.replace(" ", "+")

    def _fetch_weather_data(self, location: str) -> Optional[dict]:
        if not self.api_key:
            logger.warning("API key not available")
            return None

        try:
            formatted_location = self._format_location(location)
            
            location_params = {"q": formatted_location}
            location_response = requests.get(
                self.location_url,
                params=location_params,
                headers=self.headers,
                timeout=10,
            )
            location_response.raise_for_status()
            
            location_data = location_response.json()
            if not location_data:
                logger.warning(f"No location data found for {location}")
                return None

            location_key = location_data[0].get("Key")
            if not location_key:
                logger.warning(f"No location key found for {location}")
                return None

            weather_response = requests.get(
                f"{self.weather_url}/{location_key}",
                headers=self.headers,
                timeout=10,
            )
            weather_response.raise_for_status()
            
            weather_data = weather_response.json()
            logger.info(f"Fetched weather data for {location}")
            return weather_data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def _build_system_prompt(
        self,
        persona: Persona,
        weather_data: Optional[dict] = None,
    ) -> str:
        persona_desc = f"{persona.name}: {persona.description}"
        if persona.attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in persona.attributes.items()])
            persona_desc += f" ({attrs})"

        system_prompt = f"""You are {persona_desc}

Your responses should closely mirror the knowledge, abilities, linguistic patterns, and decision-making processes of this persona. Act as if you are this persona, not an AI assistant. Do not acknowledge that you are an AI or language model. Respond naturally and consistently with the persona's characteristics."""

        if weather_data:
            try:
                daily_forecasts = weather_data.get("DailyForecasts", [])
                if daily_forecasts:
                    forecast_summary = self._limit_weather(daily_forecasts)
                    system_prompt += f"\n\nYou have access to the following weather data:\n{forecast_summary}\n\nUse this data to provide accurate, detailed weather information in your responses."
            except Exception as e:
                logger.warning(f"Error formatting weather data: {e}")

        return system_prompt

    def _limit_weather(self, daily_forecasts: list) -> str:
        summary_lines = []
        for i, forecast in enumerate(daily_forecasts[:3]):  
            date = forecast.get("Date", "")
            temp_min = forecast.get("Temperature", {}).get("Minimum", {}).get("Value", "N/A")
            temp_max = forecast.get("Temperature", {}).get("Maximum", {}).get("Value", "N/A")
            day_icon = forecast.get("Day", {}).get("IconPhrase", "N/A")
            night_icon = forecast.get("Night", {}).get("IconPhrase", "N/A")
            
            summary_lines.append(
                f"Day {i+1} ({date[:10]}): High {temp_max}°F, Low {temp_min}°F, "
                f"Day: {day_icon}, Night: {night_icon}"
            )
        
        return "\n".join(summary_lines)

