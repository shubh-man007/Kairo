import pytest
from unittest.mock import Mock, patch
from src.evaluators import LLMEvaluator, EnsembleEvaluator, BaseEvaluator
from src.models import Persona, Question, AgentResponse
from src.rubrics import ExpectedActionRubric
from src.config.constants import TASK_EXPECTED_ACTION, SCORE_MIN, SCORE_MAX


class TestLLMEvaluator:
    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        client.model = "gpt-4o-mini"
        client.generate.return_value = "The response is good. Therefore, the final score is 4."
        return client

    @pytest.fixture
    def evaluator(self, mock_llm_client):
        return LLMEvaluator(llm_client=mock_llm_client)

    def test_evaluator_initialization(self, evaluator, mock_llm_client):
        assert evaluator.llm_client == mock_llm_client

    def test_evaluate(self, evaluator, sample_persona, sample_question, sample_response):
        rubric = ExpectedActionRubric()
        score = evaluator.evaluate(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
            rubric=rubric,
        )
        assert SCORE_MIN <= score <= SCORE_MAX
        evaluator.llm_client.generate.assert_called_once()

    def test_extract_score_final_score_pattern(self, evaluator):
        text = "The response is excellent. Therefore, the final score is 5."
        score = evaluator._extract_score(text)
        assert score == 5.0

    def test_extract_score_colon_pattern(self, evaluator):
        text = "After careful consideration, score: 3.5"
        score = evaluator._extract_score(text)
        assert score == 3.5

    def test_extract_score_number_only(self, evaluator):
        """Test score extraction with just a number."""
        text = "The response shows moderate quality. Rating: 2.5 out of 5"
        score = evaluator._extract_score(text)
        assert score == 2.5

    def test_extract_score_no_match(self, evaluator):
        """Test score extraction when no pattern matches."""
        text = "This is some text without a score."
        score = evaluator._extract_score(text)
        # Should return default middle score
        assert score == (SCORE_MIN + SCORE_MAX) / 2.0

    def test_extract_score_clamping(self, evaluator):
        """Test that scores are clamped to valid range."""
        text = "The final score is 10"  # Out of range
        score = evaluator._extract_score(text)
        assert score == SCORE_MAX

        text = "The final score is 0"  # Out of range
        score = evaluator._extract_score(text)
        assert score == SCORE_MIN

    def test_evaluate_error_handling(self, sample_persona, sample_question, sample_response):
        """Test error handling in evaluation."""
        mock_client = Mock()
        mock_client.model = "gpt-4o-mini"
        mock_client.generate.side_effect = Exception("API error")
        evaluator = LLMEvaluator(llm_client=mock_client)
        rubric = ExpectedActionRubric()

        score = evaluator.evaluate(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
            rubric=rubric,
        )
        # Should return middle score on error
        assert score == (SCORE_MIN + SCORE_MAX) / 2.0


class TestEnsembleEvaluator:
    """Tests for EnsembleEvaluator."""

    @pytest.fixture
    def mock_evaluator1(self):
        """Mock evaluator 1."""
        evaluator = Mock(spec=BaseEvaluator)
        evaluator.evaluate.return_value = 4.0
        return evaluator

    @pytest.fixture
    def mock_evaluator2(self):
        """Mock evaluator 2."""
        evaluator = Mock(spec=BaseEvaluator)
        evaluator.evaluate.return_value = 5.0
        return evaluator

    @pytest.fixture
    def ensemble(self, mock_evaluator1, mock_evaluator2):
        """Create ensemble evaluator."""
        return EnsembleEvaluator(evaluators=[mock_evaluator1, mock_evaluator2])

    def test_ensemble_initialization(self, ensemble, mock_evaluator1, mock_evaluator2):
        """Test ensemble initialization."""
        assert len(ensemble.evaluators) == 2
        assert ensemble.evaluators[0] == mock_evaluator1
        assert ensemble.evaluators[1] == mock_evaluator2

    def test_ensemble_empty_evaluators(self):
        """Test that empty evaluators list raises error."""
        with pytest.raises(ValueError, match="At least one evaluator"):
            EnsembleEvaluator(evaluators=[])

    def test_ensemble_evaluate(self, ensemble, sample_persona, sample_question, sample_response):
        """Test ensemble evaluation."""
        rubric = ExpectedActionRubric()
        score = ensemble.evaluate(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
            rubric=rubric,
        )
        # Should be average of 4.0 and 5.0
        assert score == 4.5
        ensemble.evaluators[0].evaluate.assert_called_once()
        ensemble.evaluators[1].evaluate.assert_called_once()

    def test_ensemble_evaluate_single_evaluator(self, mock_evaluator1):
        """Test ensemble with single evaluator."""
        ensemble = EnsembleEvaluator(evaluators=[mock_evaluator1])
        rubric = ExpectedActionRubric()
        score = ensemble.evaluate(
            persona=Mock(),
            question=Mock(),
            response=Mock(),
            rubric=rubric,
        )
        assert score == 4.0

    def test_ensemble_evaluate_with_failure(self, mock_evaluator1, sample_persona, sample_question, sample_response):
        """Test ensemble when one evaluator fails."""
        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.side_effect = Exception("Error")
        ensemble = EnsembleEvaluator(evaluators=[mock_evaluator1, mock_evaluator2])
        rubric = ExpectedActionRubric()

        score = ensemble.evaluate(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
            rubric=rubric,
        )
        # Should use score from working evaluator
        assert score == 4.0

    def test_ensemble_evaluate_all_fail(self, sample_persona, sample_question, sample_response):
        """Test ensemble when all evaluators fail."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator1.evaluate.side_effect = Exception("Error 1")
        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.side_effect = Exception("Error 2")
        ensemble = EnsembleEvaluator(evaluators=[mock_evaluator1, mock_evaluator2])
        rubric = ExpectedActionRubric()

        score = ensemble.evaluate(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
            rubric=rubric,
        )
        # Should return default score
        assert score == 3.0

    def test_get_individual_scores(self, ensemble, sample_persona, sample_question, sample_response):
        """Test getting individual scores."""
        rubric = ExpectedActionRubric()
        scores = ensemble.get_individual_scores(
            persona=sample_persona,
            question=sample_question,
            response=sample_response,
            rubric=rubric,
        )
        assert len(scores) == 2
        assert scores[0] == 4.0
        assert scores[1] == 5.0

