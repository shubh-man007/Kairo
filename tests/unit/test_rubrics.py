"""Unit tests for rubric system."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from src.rubrics import (
    BaseRubric,
    ExpectedActionRubric,
    LinguisticHabitsRubric,
    PersonaConsistencyRubric,
    ToxicityControlRubric,
    ActionJustificationRubric,
    ExampleGenerator,
)
from src.models import Persona, Question, AgentResponse
from src.config.constants import TASK_EXPECTED_ACTION


class TestBaseRubric:
    """Tests for BaseRubric class."""

    def test_rubric_initialization(self):
        """Test rubric initialization."""
        rubric = ExpectedActionRubric()
        assert rubric.task_name == TASK_EXPECTED_ACTION
        assert rubric.template is not None
        assert len(rubric.template) > 0

    def test_get_guidelines(self):
        """Test getting scoring guidelines."""
        rubric = ExpectedActionRubric()
        guidelines = rubric.get_guidelines()
        assert len(guidelines) == 5
        assert 1 in guidelines
        assert 5 in guidelines
        assert all(isinstance(score, int) for score in guidelines.keys())
        assert all(isinstance(text, str) and len(text) > 0 for text in guidelines.values())

    def test_format_evaluation_prompt(self, sample_persona, sample_question, sample_response):
        """Test formatting evaluation prompt."""
        rubric = ExpectedActionRubric()
        prompt = rubric.format_evaluation_prompt(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
        )
        assert sample_persona.name in prompt
        assert sample_question.text in prompt
        assert sample_response.text in prompt

    def test_format_evaluation_prompt_with_examples(
        self, sample_persona, sample_question, sample_response
    ):
        """Test formatting evaluation prompt with score examples."""
        rubric = ExpectedActionRubric()
        score_examples = {
            1: "Bad example",
            2: "Poor example",
            3: "Average example",
            4: "Good example",
            5: "Excellent example",
        }
        prompt = rubric.format_evaluation_prompt(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
            score_examples=score_examples,
        )
        assert "Score = 1:" in prompt or "Score 1:" in prompt
        assert "Score = 5:" in prompt or "Score 5:" in prompt
        assert "Bad example" in prompt
        assert "Excellent example" in prompt


class TestTaskSpecificRubrics:
    """Tests for task-specific rubrics."""

    def test_expected_action_rubric(self):
        """Test ExpectedActionRubric."""
        rubric = ExpectedActionRubric()
        assert rubric.task_name == TASK_EXPECTED_ACTION
        assert "action" in rubric.get_task_description().lower()
        assert len(rubric.get_question_requirements()) > 0

    def test_linguistic_habits_rubric(self):
        """Test LinguisticHabitsRubric."""
        rubric = LinguisticHabitsRubric()
        assert rubric.task_name == "linguistic_habits"
        assert "linguistic" in rubric.get_task_description().lower()
        assert len(rubric.get_question_requirements()) > 0

    def test_persona_consistency_rubric(self):
        """Test PersonaConsistencyRubric."""
        rubric = PersonaConsistencyRubric()
        assert rubric.task_name == "persona_consistency"
        assert "consistency" in rubric.get_task_description().lower()
        assert len(rubric.get_question_requirements()) > 0

    def test_toxicity_control_rubric(self):
        """Test ToxicityControlRubric."""
        rubric = ToxicityControlRubric()
        assert rubric.task_name == "toxicity_control"
        assert "toxicity" in rubric.get_task_description().lower()
        assert len(rubric.get_question_requirements()) > 0

    def test_action_justification_rubric(self):
        """Test ActionJustificationRubric."""
        rubric = ActionJustificationRubric()
        assert rubric.task_name == "action_justification"
        desc_lower = rubric.get_task_description().lower()
        assert "justifies" in desc_lower or "justification" in desc_lower
        assert len(rubric.get_question_requirements()) > 0


class TestExampleGenerator:
    """Tests for ExampleGenerator."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = Mock()
        client.generate.return_value = """Score 1:
This is a bad example response.

Score 2:
This is a poor example response.

Score 3:
This is an average example response.

Score 4:
This is a good example response.

Score 5:
This is an excellent example response."""
        return client

    def test_example_generator_initialization(self, mock_llm_client):
        """Test example generator initialization."""
        generator = ExampleGenerator(llm_client=mock_llm_client)
        assert generator.llm_client == mock_llm_client

    def test_generate_examples(self, mock_llm_client, sample_persona, sample_question):
        """Test generating examples."""
        generator = ExampleGenerator(llm_client=mock_llm_client)
        rubric = ExpectedActionRubric()

        examples = generator.generate_examples(
            rubric=rubric,
            persona=sample_persona,
            question=sample_question,
        )

        assert len(examples) == 5
        assert all(i in examples for i in range(1, 6))
        assert all(isinstance(text, str) and len(text) > 0 for text in examples.values())

    def test_generate_examples_error_handling(self, sample_persona, sample_question):
        """Test example generation error handling."""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("API error")
        generator = ExampleGenerator(llm_client=mock_client)
        rubric = ExpectedActionRubric()

        examples = generator.generate_examples(
            rubric=rubric,
            persona=sample_persona,
            question=sample_question,
        )

        # Should return empty examples on error
        assert len(examples) == 5
        assert all(i in examples for i in range(1, 6))

    def test_parse_examples(self, mock_llm_client):
        """Test parsing examples from response."""
        generator = ExampleGenerator(llm_client=mock_llm_client)
        response = """Score 1:
Bad response

Score 2:
Poor response

Score 3:
Average response

Score 4:
Good response

Score 5:
Excellent response"""

        examples = generator._parse_examples(response)
        assert len(examples) == 5
        assert examples[1] == "Bad response"
        assert examples[5] == "Excellent response"

