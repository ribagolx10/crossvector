"""Tests for Gemini embedding adapter."""

import os
from unittest.mock import MagicMock, patch

import pytest

from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.exceptions import InvalidFieldError


class TestGeminiEmbeddingAdapter:
    """Test suite for GeminiEmbeddingAdapter."""

    @pytest.fixture
    def mock_genai(self):
        """Mock google.genai module."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("llm_scraper.vectors.embeddings.gemini.GeminiEmbeddingAdapter.client") as mock_client_prop:
                mock_client = MagicMock()
                mock_client_prop.return_value = mock_client
                yield mock_client

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            adapter = GeminiEmbeddingAdapter()
            assert adapter.model_name == "models/gemini-embedding-001"
            assert adapter.embedding_dimension == 768
            assert adapter.task_type == "retrieval_document"

    def test_initialization_custom_model(self):
        """Test initialization with specific model."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            adapter = GeminiEmbeddingAdapter(model_name="text-embedding-004")
            assert adapter.model_name == "models/text-embedding-004"
            assert adapter.embedding_dimension == 768

    def test_dynamic_dimensionality_valid(self):
        """Test valid dynamic dimensionality for gemini-embedding-001."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            # Test 768
            adapter = GeminiEmbeddingAdapter(model_name="gemini-embedding-001", output_dimensionality=768)
            assert adapter.embedding_dimension == 768

            # Test 1536
            adapter = GeminiEmbeddingAdapter(model_name="gemini-embedding-001", output_dimensionality=1536)
            assert adapter.embedding_dimension == 1536

            # Test 3072
            adapter = GeminiEmbeddingAdapter(model_name="gemini-embedding-001", output_dimensionality=3072)
            assert adapter.embedding_dimension == 3072

    def test_dynamic_dimensionality_invalid(self):
        """Test invalid dynamic dimensionality raises error."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with pytest.raises(InvalidFieldError, match="Invalid output_dimensionality"):
                GeminiEmbeddingAdapter(
                    model_name="gemini-embedding-001",
                    output_dimensionality=1024,  # Invalid
                )

    def test_dynamic_dimensionality_ignored_for_other_models(self):
        """Test dynamic dimensionality is ignored for other models."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            adapter = GeminiEmbeddingAdapter(model_name="text-embedding-004", output_dimensionality=1536)
            # Should fallback to default 768
            assert adapter.embedding_dimension == 768

    def test_get_embeddings(self):
        """Test get_embeddings calls API correctly."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            # Mock the module import
            mock_types = MagicMock()
            with patch.dict(
                "sys.modules", {"google": MagicMock(), "google.genai": MagicMock(), "google.genai.types": mock_types}
            ):
                adapter = GeminiEmbeddingAdapter()

                # Create a fresh mock for the client
                mock_client = MagicMock()
                adapter._client = mock_client

                # Mock response
                mock_result = MagicMock()
                mock_embedding = MagicMock()
                mock_embedding.values = [0.1, 0.2, 0.3]
                mock_result.embeddings = [mock_embedding]

                # Setup client mock return value
                mock_client.models.embed_content.return_value = mock_result

                # Call method
                texts = ["hello world"]
                embeddings = adapter.get_embeddings(texts)

                # Verify
                assert len(embeddings) == 1
                assert embeddings[0] == [0.1, 0.2, 0.3]

                # Verify API call
                mock_client.models.embed_content.assert_called_once()
                call_args = mock_client.models.embed_content.call_args
                assert call_args.kwargs["model"] == "models/gemini-embedding-001"
                assert call_args.kwargs["contents"] == "hello world"

    def test_get_embeddings_with_dimensionality(self):
        """Test get_embeddings passes output_dimensionality to API."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            # Mock types
            mock_types = MagicMock()
            mock_genai_module = MagicMock()
            mock_genai_module.types = mock_types

            with patch.dict(
                "sys.modules",
                {"google": MagicMock(), "google.genai": mock_genai_module, "google.genai.types": mock_types},
            ):
                adapter = GeminiEmbeddingAdapter(output_dimensionality=1536)

                # Create a fresh mock for the client
                mock_client = MagicMock()
                adapter._client = mock_client

                # Mock response
                mock_result = MagicMock()
                mock_embedding = MagicMock()
                mock_embedding.values = [0.1] * 1536
                mock_result.embeddings = [mock_embedding]

                # Setup client mock return value
                mock_client.models.embed_content.return_value = mock_result

                # Call method
                adapter.get_embeddings(["test"])

                # Verify API call
                mock_client.models.embed_content.assert_called_once()

                # Check if config was created with correct params
                mock_types.EmbedContentConfig.assert_called_with(
                    task_type="retrieval_document", output_dimensionality=1536
                )
