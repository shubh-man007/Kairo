"""Unit tests for data models."""

import pytest
from datetime import datetime
from src.models import (
    Persona,
    Environment,
    Question,
    AgentResponse,
    EvaluationResult,
    TaskScore,
    PersonaScore,
)
from src.config.constants import (
    TASK_EXPECTED_ACTION,
    TASK_LINGUISTIC_HABITS,
    SCORE_MIN,
    SCORE_MAX,
)


class TestPersona:
    """Tests for Persona model."""

    def test_persona_creation(self):
        """Test creating a persona."""
        persona = Persona(
            name="test_persona",
            description="A test persona",
            attributes={"age": 30},
        )
        assert persona.name == "test_persona"
        assert persona.description == "A test persona"
        assert persona.attributes == {"age": 30}

    def test_persona_without_attributes(self):
        """Test creating a persona without attributes."""
        persona = Persona(name="test", description="Test")
        assert persona.attributes is None


class TestEnvironment:
    """Tests for Environment model."""

    def test_environment_creation(self):
        """Test creating an environment."""
        env = Environment(
            name="test_env",
            description="Test environment",
            domain="test",
        )
        assert env.name == "test_env"
        assert env.description == "Test environment"
        assert env.domain == "test"

    def test_environment_without_domain(self):
        """Test creating an environment without domain."""
        env = Environment(name="test", description="Test")
        assert env.domain is None


class TestQuestion:
    """Tests for Question model."""

    def test_question_creation(self, sample_question):
        """Test creating a question."""
        assert sample_question.text is not None
        assert sample_question.task_type == TASK_EXPECTED_ACTION
        assert sample_question.environment_name == "weather_forecast"
        assert sample_question.persona_name == "weather_forecaster"

    def test_question_invalid_task_type(self):
        """Test that invalid task type raises error."""
        with pytest.raises(ValueError):
            Question(
                text="Test question",
                task_type="invalid_task",
                environment_name="test",
                persona_name="test",
            )


class TestAgentResponse:
    """Tests for AgentResponse model."""

    def test_response_creation(self, sample_response):
        """Test creating an agent response."""
        assert sample_response.text is not None
        assert sample_response.persona_name == "weather_forecaster"
        assert isinstance(sample_response.timestamp, datetime)

    def test_response_without_metadata(self):
        """Test creating a response without metadata."""
        response = AgentResponse(
            text="Test response",
            persona_name="test",
        )
        assert response.metadata is None


class TestTaskScore:
    """Tests for TaskScore model."""

    def test_task_score_creation(self):
        """Test creating a task score."""
        score = TaskScore(
            task_type=TASK_EXPECTED_ACTION,
            score=4.5,
            evaluator_scores=[4.0, 5.0],
            evaluator_models=["model1", "model2"],
        )
        assert score.task_type == TASK_EXPECTED_ACTION
        assert score.score == 4.5
        assert score.evaluator_scores == [4.0, 5.0]

    def test_task_score_boundaries(self):
        """Test that scores are within valid range."""
        # Valid scores
        TaskScore(task_type=TASK_EXPECTED_ACTION, score=SCORE_MIN)
        TaskScore(task_type=TASK_EXPECTED_ACTION, score=SCORE_MAX)
        TaskScore(task_type=TASK_EXPECTED_ACTION, score=3.0)

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            TaskScore(task_type=TASK_EXPECTED_ACTION, score=SCORE_MIN - 1)
        with pytest.raises(Exception):
            TaskScore(task_type=TASK_EXPECTED_ACTION, score=SCORE_MAX + 1)


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_evaluation_result_creation(self, sample_question, sample_response):
        """Test creating an evaluation result."""
        task_score = TaskScore(
            task_type=TASK_EXPECTED_ACTION,
            score=4.5,
        )
        result = EvaluationResult(
            question_text=sample_question.text,
            response_text=sample_response.text,
            persona_name=sample_question.persona_name,
            environment_name=sample_question.environment_name,
            task_scores=[task_score],
        )
        assert len(result.task_scores) == 1
        assert result.task_scores[0].score == 4.5


class TestPersonaScore:
    """Tests for PersonaScore model."""

    def test_persona_score_creation(self):
        """Test creating a persona score."""
        score = PersonaScore(
            persona_name="test_persona",
            overall_score=4.2,
            task_averages={
                TASK_EXPECTED_ACTION: 4.5,
                TASK_LINGUISTIC_HABITS: 4.0,
            },
            total_questions=50,
            total_evaluations=250,
        )
        assert score.overall_score == 4.2
        assert score.total_questions == 50
        assert len(score.task_averages) == 2

