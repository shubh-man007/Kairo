"""Unit tests for scoring system."""

import pytest
from datetime import datetime
from src.scoring import (
    PersonaScoreCalculator,
    aggregate_task_scores,
    calculate_overall_average,
    get_score_statistics,
)
from src.models import EvaluationResult, TaskScore
from src.config.constants import TASK_EXPECTED_ACTION, TASK_LINGUISTIC_HABITS


class TestPersonaScoreCalculator:
    """Tests for PersonaScoreCalculator."""

    @pytest.fixture
    def sample_evaluation_results(self):
        """Sample evaluation results."""
        results = []
        for i in range(3):
            result = EvaluationResult(
                question_text=f"Question {i}",
                response_text=f"Response {i}",
                persona_name="test_persona",
                environment_name="test_env",
                task_scores=[
                    TaskScore(task_type=TASK_EXPECTED_ACTION, score=4.0 + i * 0.5),
                    TaskScore(task_type=TASK_LINGUISTIC_HABITS, score=3.5 + i * 0.3),
                ],
            )
            results.append(result)
        return results

    def test_calculate_persona_score(self, sample_evaluation_results):
        """Test calculating PersonaScore."""
        score = PersonaScoreCalculator.calculate_persona_score(
            evaluation_results=sample_evaluation_results,
            persona_name="test_persona",
        )

        assert score.persona_name == "test_persona"
        assert score.total_questions == 3
        assert score.total_evaluations == 6  # 3 questions * 2 tasks
        assert score.overall_score > 0
        assert TASK_EXPECTED_ACTION in score.task_averages
        assert TASK_LINGUISTIC_HABITS in score.task_averages

    def test_calculate_persona_score_empty(self):
        """Test calculating PersonaScore with empty results."""
        with pytest.raises(ValueError, match="No evaluation results"):
            PersonaScoreCalculator.calculate_persona_score(
                evaluation_results=[],
                persona_name="test",
            )

    def test_calculate_task_average(self, sample_evaluation_results):
        """Test calculating task average."""
        avg = PersonaScoreCalculator.calculate_task_average(
            evaluation_results=sample_evaluation_results,
            task_type=TASK_EXPECTED_ACTION,
        )
        assert avg > 0
        assert 1.0 <= avg <= 5.0

    def test_calculate_task_average_no_scores(self):
        """Test calculating task average with no scores."""
        results = [
            EvaluationResult(
                question_text="Q1",
                response_text="R1",
                persona_name="test",
                environment_name="test",
                task_scores=[],
            )
        ]
        avg = PersonaScoreCalculator.calculate_task_average(
            evaluation_results=results,
            task_type=TASK_EXPECTED_ACTION,
        )
        # Should return default middle score
        assert avg == 3.0


class TestAggregator:
    """Tests for aggregation utilities."""

    def test_aggregate_task_scores(self):
        """Test aggregating task scores."""
        task_scores = [
            TaskScore(task_type=TASK_EXPECTED_ACTION, score=4.0),
            TaskScore(task_type=TASK_EXPECTED_ACTION, score=5.0),
            TaskScore(task_type=TASK_LINGUISTIC_HABITS, score=3.0),
            TaskScore(task_type=TASK_LINGUISTIC_HABITS, score=4.0),
        ]

        averages = aggregate_task_scores(task_scores)

        assert TASK_EXPECTED_ACTION in averages
        assert TASK_LINGUISTIC_HABITS in averages
        assert averages[TASK_EXPECTED_ACTION] == 4.5
        assert averages[TASK_LINGUISTIC_HABITS] == 3.5

    def test_aggregate_task_scores_empty(self):
        """Test aggregating empty task scores."""
        averages = aggregate_task_scores([])
        assert averages == {}

    def test_calculate_overall_average(self):
        """Test calculating overall average."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        avg = calculate_overall_average(scores)
        assert avg == 3.0

    def test_calculate_overall_average_empty(self):
        """Test calculating average with empty list."""
        avg = calculate_overall_average([])
        assert avg == 3.0  # Default middle score

    def test_get_score_statistics(self):
        """Test getting score statistics."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = get_score_statistics(scores)

        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["std_dev"] > 0

    def test_get_score_statistics_empty(self):
        """Test getting statistics with empty list."""
        stats = get_score_statistics([])
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["std_dev"] == 0.0

    def test_get_score_statistics_single_value(self):
        """Test getting statistics with single value."""
        stats = get_score_statistics([4.5])
        assert stats["mean"] == 4.5
        assert stats["min"] == 4.5
        assert stats["max"] == 4.5
        assert stats["std_dev"] == 0.0

