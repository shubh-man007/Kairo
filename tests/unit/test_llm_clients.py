"""Unit tests for LLM clients."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm import OpenAIClient, AnthropicClient, create_llm_client
from src.llm.factory import LLMClientFactory


class TestOpenAIClient:
    """Tests for OpenAI client."""

    @patch("src.llm.openai_client.OpenAI")
    def test_openai_client_initialization(self, mock_openai_class):
        """Test OpenAI client initialization."""
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance

        client = OpenAIClient(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key="test-key",
        )
        assert client.model == "gpt-4o-mini"
        assert client.temperature == 0.0
        assert client.api_key == "test-key"

    @patch("src.llm.openai_client.OpenAI")
    def test_openai_generate(self, mock_openai_class):
        """Test OpenAI generate method."""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance

        client = OpenAIClient(api_key="test-key")
        response = client.generate("Test prompt")
        assert response == "Test response"

    @patch("src.llm.openai_client.OpenAI")
    def test_openai_generate_with_system_prompt(self, mock_openai_class):
        """Test OpenAI generate with system prompt."""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance

        client = OpenAIClient(api_key="test-key")
        response = client.generate("Test prompt", system_prompt="You are a helpful assistant")
        assert response == "Test response"
        # Verify system prompt was included
        call_args = mock_client_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"

    @patch("src.config.settings._settings", None)  # Clear cached settings
    @patch("src.config.get_settings")
    def test_openai_missing_api_key(self, mock_get_settings):
        """Test that missing API key raises error."""
        mock_settings = Mock()
        mock_settings.openai_api_key = ""  # Empty OpenAI key
        mock_settings.anthropic_api_key = "test-anthropic-key"  # Set Anthropic key so validate() passes
        mock_settings.openai_model = "gpt-4o-mini"
        mock_get_settings.return_value = mock_settings
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIClient()


class TestAnthropicClient:
    """Tests for Anthropic client."""

    @patch("src.llm.anthropic_client.Anthropic")
    def test_anthropic_client_initialization(self, mock_anthropic_class):
        """Test Anthropic client initialization."""
        mock_client_instance = Mock()
        mock_anthropic_class.return_value = mock_client_instance

        client = AnthropicClient(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            api_key="test-key",
        )
        assert client.model == "claude-sonnet-4-5-20250929"
        assert client.temperature == 0.0
        assert client.api_key == "test-key"

    @patch("src.llm.anthropic_client.Anthropic")
    def test_anthropic_generate(self, mock_anthropic_class):
        """Test Anthropic generate method."""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_client_instance.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client_instance

        client = AnthropicClient(api_key="test-key")
        response = client.generate("Test prompt")
        assert response == "Test response"

    @patch("src.config.settings._settings", None)  # Clear cached settings
    @patch("src.config.get_settings")
    def test_anthropic_missing_api_key(self, mock_get_settings):
        """Test that missing API key raises error."""
        mock_settings = Mock()
        mock_settings.anthropic_api_key = ""  # Empty Anthropic key
        mock_settings.openai_api_key = "test-openai-key"  # Set OpenAI key so validate() passes
        mock_settings.anthropic_model = "claude-sonnet-4-5-20250929"
        mock_get_settings.return_value = mock_settings
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            AnthropicClient()


class TestLLMClientFactory:
    """Tests for LLM client factory."""

    @patch("src.llm.openai_client.OpenAI")
    def test_create_openai_client(self, mock_openai_class):
        """Test creating OpenAI client via factory."""
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance

        client = LLMClientFactory.create(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            api_key="test-key",
        )
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4o-mini"

    @patch("src.llm.anthropic_client.Anthropic")
    def test_create_anthropic_client(self, mock_anthropic_class):
        """Test creating Anthropic client via factory."""
        mock_client_instance = Mock()
        mock_anthropic_class.return_value = mock_client_instance

        client = LLMClientFactory.create(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            api_key="test-key",
        )
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-4-5-20250929"

    def test_create_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClientFactory.create(provider="invalid_provider", api_key="test-key")

    @patch("src.llm.openai_client.OpenAI")
    def test_create_llm_client_convenience_function(self, mock_openai_class):
        """Test convenience function for creating clients."""
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance

        client = create_llm_client(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        assert isinstance(client, OpenAIClient)

