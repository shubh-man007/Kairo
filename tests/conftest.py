import pytest
from unittest.mock import Mock, MagicMock
from src.models import Persona, Environment, Question, AgentResponse
from src.config.constants import TASK_EXPECTED_ACTION


@pytest.fixture
def sample_persona():
    return Persona(
        name="weather_forecaster",
        description="A professional weather forecaster with expertise in meteorology",
        attributes={
            "age": 35,
            "location": "New York",
            "expertise": ["weather prediction", "climate analysis"],
        },
    )


@pytest.fixture
def sample_environment():
    return Environment(
        name="weather_forecast",
        description="Environment for weather forecasting scenarios",
        domain="weather",
    )


@pytest.fixture
def sample_question(sample_persona, sample_environment):
    return Question(
        text="What would you do if you needed to provide a weather forecast for a major storm?",
        task_type=TASK_EXPECTED_ACTION,
        environment_name=sample_environment.name,
        persona_name=sample_persona.name,
        quality_criteria="Tests whether agent takes logically appropriate actions",
    )


@pytest.fixture
def sample_response(sample_question):
    return AgentResponse(
        text="I would analyze the storm's trajectory using meteorological data and provide a detailed forecast.",
        question_id="q_001",
        persona_name=sample_question.persona_name,
        metadata={"model": "gpt-4o-mini", "temperature": 0.9},
    )


@pytest.fixture
def mock_openai_client():
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock OpenAI response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Mock Anthropic response"
    mock_client.messages.create.return_value = mock_response
    return mock_client

