"""Tests for OpenAI embedding adapter."""

from unittest.mock import Mock, patch

import pytest

from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.exceptions import InvalidFieldError, MissingConfigError, SearchError


class TestOpenAIEmbeddingAdapter:
    """Tests for OpenAI embedding adapter."""

    def test_initialization(self):
        """Test adapter initialization with valid model."""
        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

        assert adapter.model_name == "text-embedding-3-small"
        assert adapter.embedding_dimension == 1536

    def test_initialization_invalid_model(self):
        """Test adapter initialization with unknown model."""
        with pytest.raises(InvalidFieldError, match="Unknown embedding dimension"):
            OpenAIEmbeddingAdapter(model_name="unknown-model")

    def test_supported_models(self):
        """Test that all supported models initialize correctly."""
        models = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        for model_name, expected_dim in models.items():
            adapter = OpenAIEmbeddingAdapter(model_name=model_name)
            assert adapter.embedding_dimension == expected_dim

    @patch("crossvector.embeddings.openai.OpenAI")
    @patch("crossvector.embeddings.openai.settings")
    def test_lazy_client_initialization(self, mock_settings, mock_openai_class):
        """Test that client is lazily initialized."""
        mock_settings.OPENAI_API_KEY = "test-key"
        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

        # Client should not be initialized yet
        assert adapter._client is None

        # Access client property
        mock_openai_class.return_value = Mock()
        client = adapter.client

        # Now it should be initialized
        assert client is not None
        mock_openai_class.assert_called_once()

    @patch("crossvector.embeddings.openai.settings")
    def test_client_initialization_no_api_key(self, mock_settings):
        """Test that client initialization fails without API key."""
        mock_settings.OPENAI_API_KEY = None

        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        with pytest.raises(MissingConfigError, match="API key not configured"):
            _ = adapter.client

    @patch("crossvector.embeddings.openai.OpenAI")
    @patch("crossvector.embeddings.openai.settings")
    def test_get_embeddings_success(self, mock_settings, mock_openai_class):
        """Test successful embedding generation."""
        mock_settings.OPENAI_API_KEY = "test-key"

        # Mock OpenAI response
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1, 0.2, 0.3]

        mock_response = Mock()
        mock_response.data = [mock_embedding_obj, mock_embedding_obj]

        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        texts = ["Hello world", "Test text"]

        embeddings = adapter.get_embeddings(texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch("crossvector.embeddings.openai.OpenAI")
    @patch("crossvector.embeddings.openai.settings")
    def test_get_embeddings_empty_list(self, mock_settings, mock_openai_class):
        """Test embedding generation with empty text list."""
        mock_settings.OPENAI_API_KEY = "test-key"

        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

        embeddings = adapter.get_embeddings([])

        assert embeddings == []

    @patch("crossvector.embeddings.openai.OpenAI")
    @patch("crossvector.embeddings.openai.settings")
    def test_get_embeddings_newline_replacement(self, mock_settings, mock_openai_class):
        """Test that newlines are replaced before sending to API."""
        mock_settings.OPENAI_API_KEY = "test-key"

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1, 0.2]

        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]

        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        texts = ["Hello\nworld\ntest"]

        adapter.get_embeddings(texts)

        # Check that newlines were replaced
        call_args = mock_client.embeddings.create.call_args
        assert "\n" not in call_args.kwargs["input"][0]
        assert call_args.kwargs["input"][0] == "Hello world test"

    @patch("crossvector.embeddings.openai.OpenAI")
    @patch("crossvector.embeddings.openai.settings")
    def test_get_embeddings_api_error(self, mock_settings, mock_openai_class):
        """Test handling of API errors."""
        mock_settings.OPENAI_API_KEY = "test-key"

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        with pytest.raises(SearchError, match="Embedding generation failed"):
            adapter.get_embeddings(["test"])


@pytest.mark.integration
class TestOpenAIEmbeddingIntegration:
    """Integration tests for OpenAI embedding adapter (requires API key)."""

    def test_real_embedding_generation(self, openai_api_key):
        """Test real embedding generation with OpenAI API."""
        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a programming language.",
        ]

        embeddings = adapter.get_embeddings(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536

        # Check that embeddings are normalized (approximately)
        magnitude = sum(x**2 for x in embeddings[0]) ** 0.5
        assert 0.95 < magnitude < 1.05  # Should be close to 1

    def test_real_embedding_similarity(self, openai_api_key):
        """Test that similar texts have similar embeddings."""
        adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

        texts = [
            "The cat sits on the mat.",
            "A feline rests on the rug.",  # Similar meaning
            "Python programming language.",  # Different meaning
        ]

        embeddings = adapter.get_embeddings(texts)

        # Compute cosine similarity
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x**2 for x in a) ** 0.5
            mag_b = sum(x**2 for x in b) ** 0.5
            return dot_product / (mag_a * mag_b)

        sim_01 = cosine_similarity(embeddings[0], embeddings[1])  # Similar texts
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])  # Different texts

        # Similar texts should have higher similarity
        assert sim_01 > sim_02
        assert sim_01 > 0.6  # Should be reasonably similar
