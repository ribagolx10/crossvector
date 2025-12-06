"""Extended tests for utils and exceptions modules."""

from crossvector.exceptions import (
    CrossVectorError,
    InvalidFieldError,
    MissingConfigError,
    MissingFieldError,
)
from crossvector.utils import chunk_iter, extract_pk, generate_pk, load_custom_pk_factory


class TestChunkIter:
    """Tests for chunk_iter utility."""

    def test_chunk_iter_basic(self):
        """Test basic chunking."""
        seq = [1, 2, 3, 4, 5, 6, 7]
        chunks = list(chunk_iter(seq, 3))
        assert chunks == [[1, 2, 3], [4, 5, 6], [7]]

    def test_chunk_iter_exact_multiple(self):
        """Test chunking when length is multiple of size."""
        seq = [1, 2, 3, 4]
        chunks = list(chunk_iter(seq, 2))
        assert chunks == [[1, 2], [3, 4]]

    def test_chunk_iter_single_chunk(self):
        """Test when chunk size is larger than sequence."""
        seq = [1, 2, 3]
        chunks = list(chunk_iter(seq, 10))
        assert chunks == [[1, 2, 3]]

    def test_chunk_iter_size_one(self):
        """Test chunking with size 1."""
        seq = [1, 2, 3]
        chunks = list(chunk_iter(seq, 1))
        assert chunks == [[1], [2], [3]]

    def test_chunk_iter_zero_size(self):
        """Test chunking with zero size yields entire sequence."""
        seq = [1, 2, 3]
        chunks = list(chunk_iter(seq, 0))
        assert chunks == [[1, 2, 3]]

    def test_chunk_iter_negative_size(self):
        """Test chunking with negative size yields entire sequence."""
        seq = [1, 2, 3]
        chunks = list(chunk_iter(seq, -1))
        assert chunks == [[1, 2, 3]]

    def test_chunk_iter_empty_sequence(self):
        """Test chunking empty sequence."""
        chunks = list(chunk_iter([], 3))
        assert chunks == []

    def test_chunk_iter_strings(self):
        """Test chunking with strings."""
        seq = "abcdefg"
        chunks = list(chunk_iter(seq, 3))
        assert chunks == ["abc", "def", "g"]


class TestExtractPk:
    """Tests for extract_pk utility."""

    def test_extract_pk_from_object(self):
        """Test extracting pk from object with id attribute."""

        class MockDoc:
            id = "doc-123"

        pk = extract_pk(MockDoc())
        assert pk == "doc-123"

    def test_extract_pk_from_kwargs_id(self):
        """Test extracting from kwargs with id key."""
        pk = extract_pk(id="doc-123")
        assert pk == "doc-123"

    def test_extract_pk_from_kwargs_pk(self):
        """Test extracting from kwargs with pk key."""
        pk = extract_pk(pk="doc-123")
        assert pk == "doc-123"

    def test_extract_pk_from_kwargs_underscore_id(self):
        """Test extracting from kwargs with _id key."""
        pk = extract_pk(**{"_id": "doc-123"})
        assert pk == "doc-123"

    def test_extract_pk_kwargs_precedence(self):
        """Test that kwargs takes precedence over object."""

        class MockDoc:
            id = "doc-from-obj"

        pk = extract_pk(MockDoc(), id="doc-from-kwargs")
        assert pk == "doc-from-kwargs"

    def test_extract_pk_none_returns_none(self):
        """Test extracting pk when not available returns None."""
        pk = extract_pk()
        assert pk is None

    def test_extract_pk_object_without_id_returns_none(self):
        """Test extracting from object without id attribute."""

        class MockDoc:
            pass

        pk = extract_pk(MockDoc())
        assert pk is None

    def test_extract_pk_priority_underscore_id_over_id(self):
        """Test that _id has priority over id."""
        pk = extract_pk(**{"_id": "from-underscore", "id": "from-id"})
        assert pk == "from-underscore"

    def test_extract_pk_priority_underscore_id_over_pk(self):
        """Test that _id has priority over pk."""
        pk = extract_pk(**{"_id": "from-underscore", "pk": "from-pk"})
        assert pk == "from-underscore"


class TestGeneratePk:
    """Tests for generate_pk utility."""

    def test_generate_pk_uuid_default(self):
        """Test UUID generation (default mode)."""
        pk1 = generate_pk("text", [0.1, 0.2])
        pk2 = generate_pk("text", [0.1, 0.2])
        # UUIDs should be different each time
        assert pk1 != pk2
        assert len(pk1) == 32  # UUID hex is 32 chars

    def test_generate_pk_hash_text(self):
        """Test hash_text mode."""
        text = "test document"
        # Generate pk and verify it's a valid string
        pk = generate_pk(text, [0.1])
        assert isinstance(pk, str)
        assert len(pk) > 0

    def test_generate_pk_with_metadata(self):
        """Test that metadata parameter is accepted."""
        pk = generate_pk("text", [0.1], {"source": "api"})
        assert isinstance(pk, str)
        assert len(pk) > 0

    def test_generate_pk_none_text_vector(self):
        """Test pk generation with None text and vector."""
        pk = generate_pk(None, None)
        assert isinstance(pk, str)
        assert len(pk) > 0


class TestLoadCustomPkFactory:
    """Tests for load_custom_pk_factory."""

    def test_load_custom_pk_factory_none(self):
        """Test loading with None path."""
        fn = load_custom_pk_factory(None)
        assert fn is None

    def test_load_custom_pk_factory_empty_string(self):
        """Test loading with empty string."""
        fn = load_custom_pk_factory("")
        assert fn is None

    def test_load_custom_pk_factory_invalid_path(self):
        """Test loading with invalid path."""
        fn = load_custom_pk_factory("nonexistent.module.function")
        assert fn is None

    def test_load_custom_pk_factory_invalid_module(self):
        """Test loading with invalid module."""
        fn = load_custom_pk_factory("this.does.not.exist.function")
        assert fn is None


class TestExceptions:
    """Tests for exception classes."""

    def test_cross_vector_error_basic(self):
        """Test CrossVectorError creation."""
        error = CrossVectorError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_missing_config_error(self):
        """Test MissingConfigError with message and config_key."""
        error = MissingConfigError(
            "API key not configured",
            config_key="OPENAI_API_KEY",
        )
        assert "API key not configured" in str(error)
        assert error.details["config_key"] == "OPENAI_API_KEY"

    def test_missing_config_error_with_suggestion(self):
        """Test MissingConfigError with suggestion."""
        error = MissingConfigError(
            "Package not installed",
            config_key="google-genai",
            suggestion="pip install google-genai",
        )
        assert error.details["config_key"] == "google-genai"
        assert error.details["suggestion"] == "pip install google-genai"

    def test_missing_field_error(self):
        """Test MissingFieldError."""
        error = MissingFieldError(
            "Field is required",
            field="vector",
        )
        assert "Field is required" in str(error)
        assert error.details["field"] == "vector"

    def test_invalid_field_error(self):
        """Test InvalidFieldError with expected value."""
        error = InvalidFieldError(
            "Invalid dimension",
            field="dim",
            value=1000,
            expected=[768, 1536, 3072],
        )
        assert "Invalid dimension" in str(error)
        assert error.details["field"] == "dim"
        assert error.details["value"] == 1000
        assert error.details["expected"] == [768, 1536, 3072]

    def test_invalid_field_error_without_expected(self):
        """Test InvalidFieldError without expected value."""
        error = InvalidFieldError(
            "Invalid model",
            field="model_name",
            value="unknown",
        )
        assert error.details["field"] == "model_name"
        assert error.details["value"] == "unknown"

    def test_exception_inheritance_chain(self):
        """Test exception inheritance."""
        assert issubclass(CrossVectorError, Exception)
        assert issubclass(MissingConfigError, CrossVectorError)
        assert issubclass(MissingFieldError, CrossVectorError)
        assert issubclass(InvalidFieldError, CrossVectorError)

    def test_exception_message_format(self):
        """Test that exception messages are properly formatted."""
        error = MissingConfigError("Test message")
        # Should be able to convert to string
        assert isinstance(str(error), str)
        assert len(str(error)) > 0
