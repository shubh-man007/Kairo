"""Unit tests for pipeline components."""

import pytest
from unittest.mock import Mock, patch
from src.pipeline import (
    EnvironmentSelector,
    QuestionGenerator,
    AgentGenerator,
)
from src.models import Persona, Environment, Question
from src.rubrics import ExpectedActionRubric
from src.config.constants import TASK_EXPECTED_ACTION


class TestEnvironmentSelector:
    """Tests for EnvironmentSelector."""

    @pytest.fixture
    def sample_environments(self):
        """Sample environment pool."""
        return [
            Environment(name="env1", description="Environment 1", domain="test"),
            Environment(name="env2", description="Environment 2", domain="test"),
            Environment(name="env3", description="Environment 3", domain="test"),
            Environment(name="weather_forecast", description="Weather forecasting", domain="weather"),
        ]

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = Mock()
        client.generate.return_value = "env1\nenv2\nweather_forecast"
        return client

    def test_environment_selector_initialization(self, mock_llm_client, sample_environments):
        """Test environment selector initialization."""
        selector = EnvironmentSelector(
            llm_client=mock_llm_client,
            environment_pool=sample_environments,
        )
        assert selector.llm_client == mock_llm_client
        assert len(selector.environment_pool) == 4

    def test_select_environments(self, mock_llm_client, sample_environments, sample_persona):
        """Test selecting environments."""
        selector = EnvironmentSelector(
            llm_client=mock_llm_client,
            environment_pool=sample_environments,
            num_environments=3,
        )

        selected = selector.select_environments(persona=sample_persona)

        assert len(selected) <= 3
        assert all(isinstance(env, Environment) for env in selected)
        mock_llm_client.generate.assert_called_once()

    def test_select_environments_error_handling(self, sample_environments, sample_persona):
        """Test error handling in environment selection."""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("API error")
        selector = EnvironmentSelector(
            llm_client=mock_client,
            environment_pool=sample_environments,
            num_environments=2,
        )

        selected = selector.select_environments(persona=sample_persona)

        # Should fallback to first N environments
        assert len(selected) == 2
        assert selected == sample_environments[:2]

    def test_parse_selected_environments(self, mock_llm_client, sample_environments):
        """Test parsing selected environments."""
        selector = EnvironmentSelector(
            llm_client=mock_llm_client,
            environment_pool=sample_environments,
        )

        response = "env1\nenv2\nweather_forecast"
        selected = selector._parse_selected_environments(response)

        assert len(selected) > 0
        assert all(env.name in ["env1", "env2", "weather_forecast"] for env in selected)


class TestQuestionGenerator:
    """Tests for QuestionGenerator."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = Mock()
        client.generate.return_value = """1. What would you do in this situation?
2. How would you handle this scenario?
3. What actions would you take?"""
        return client

    def test_question_generator_initialization(self, mock_llm_client):
        """Test question generator initialization."""
        generator = QuestionGenerator(llm_client=mock_llm_client)
        assert generator.llm_client == mock_llm_client
        assert generator.num_questions_per_task == 10

    def test_generate_questions(
        self, mock_llm_client, sample_persona, sample_environment
    ):
        """Test generating questions."""
        generator = QuestionGenerator(llm_client=mock_llm_client, num_questions_per_task=3)
        rubric = ExpectedActionRubric()

        questions = generator.generate_questions(
            persona=sample_persona,
            environment=sample_environment,
            rubric=rubric,
        )

        assert len(questions) > 0
        assert all(isinstance(q, Question) for q in questions)
        assert all(q.task_type == TASK_EXPECTED_ACTION for q in questions)
        assert all(q.persona_name == sample_persona.name for q in questions)
        assert all(q.environment_name == sample_environment.name for q in questions)

    def test_generate_questions_error_handling(
        self, sample_persona, sample_environment
    ):
        """Test error handling in question generation."""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("API error")
        generator = QuestionGenerator(llm_client=mock_client)
        rubric = ExpectedActionRubric()

        questions = generator.generate_questions(
            persona=sample_persona,
            environment=sample_environment,
            rubric=rubric,
        )

        # Should return at least a default question
        assert len(questions) > 0

    def test_parse_questions(self, mock_llm_client, sample_persona, sample_environment):
        """Test parsing questions from response."""
        generator = QuestionGenerator(llm_client=mock_llm_client)
        rubric = ExpectedActionRubric()

        response = """1. What would you do?
2. How would you handle this?
3. What actions would you take?"""

        questions = generator._parse_questions(
            response=response,
            persona=sample_persona,
            environment=sample_environment,
            rubric=rubric,
        )

        assert len(questions) > 0
        assert all("?" in q.text or q.text[0].isupper() for q in questions)


class TestAgentGenerator:
    """Tests for AgentGenerator."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = Mock()
        client.model = "gpt-4o-mini"
        client.generate.return_value = "This is a test response from the agent."
        return client

    def test_agent_generator_initialization(self, mock_llm_client):
        """Test agent generator initialization."""
        generator = AgentGenerator(llm_client=mock_llm_client)
        assert generator.llm_client == mock_llm_client

    def test_generate_response(self, mock_llm_client, sample_persona, sample_question):
        """Test generating agent response."""
        generator = AgentGenerator(llm_client=mock_llm_client)

        response = generator.generate_response(
            persona=sample_persona,
            question=sample_question,
        )

        from src.models import AgentResponse
        assert isinstance(response, AgentResponse)
        assert response.text is not None
        assert response.persona_name == sample_persona.name
        assert response.metadata is not None
        mock_llm_client.generate.assert_called_once()

    def test_generate_response_error_handling(self, sample_persona, sample_question):
        """Test error handling in response generation."""
        mock_client = Mock()
        mock_client.model = "gpt-4o-mini"
        mock_client.generate.side_effect = Exception("API error")
        generator = AgentGenerator(llm_client=mock_client)

        response = generator.generate_response(
            persona=sample_persona,
            question=sample_question,
        )

        assert response.text is not None
        assert "Error" in response.text or "error" in response.text.lower()

    def test_build_system_prompt(self, mock_llm_client, sample_persona):
        """Test building system prompt."""
        generator = AgentGenerator(llm_client=mock_llm_client)
        prompt = generator._build_system_prompt(persona=sample_persona)

        assert sample_persona.name in prompt
        assert sample_persona.description in prompt
        assert "AI" in prompt or "assistant" in prompt.lower()

