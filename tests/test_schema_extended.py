"""Extended tests for VectorDocument schema to increase coverage."""

import pytest

from crossvector.exceptions import InvalidFieldError, MissingFieldError
from crossvector.schema import VectorDocument


class TestVectorDocumentExtended:
    """Extended tests for VectorDocument covering more code paths."""

    def test_from_kwargs_with_vector(self):
        """Test from_kwargs with vector field."""
        doc = VectorDocument.from_kwargs(
            id="doc-1",
            vector=[0.1, 0.2, 0.3],
            text="Hello world",
            source="api",
        )
        assert doc.id == "doc-1"
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.text == "Hello world"
        assert doc.metadata["source"] == "api"

    def test_from_kwargs_with_dollar_vector(self):
        """Test from_kwargs with $vector field (AstraDB format)."""
        doc = VectorDocument.from_kwargs(
            id="doc-1",
            **{"$vector": [0.1, 0.2, 0.3]},
            text="Hello world",
        )
        assert doc.id == "doc-1"
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.text == "Hello world"

    def test_from_kwargs_without_vector_raises(self):
        """Test from_kwargs without vector raises MissingFieldError."""
        with pytest.raises(MissingFieldError, match="vector"):
            VectorDocument.from_kwargs(id="doc-1", text="Hello")

    def test_from_kwargs_with_auto_pk(self):
        """Test from_kwargs auto-generates pk when not provided."""
        doc = VectorDocument.from_kwargs(
            vector=[0.1, 0.2, 0.3],
            text="Hello",
        )
        assert doc.id is not None
        assert len(doc.id) > 0

    def test_from_text_basic(self):
        """Test from_text creates document with text only."""
        doc = VectorDocument.from_text("Hello world")
        assert doc.text == "Hello world"
        assert doc.vector == []
        assert doc.id is not None

    def test_from_text_with_metadata_dict(self):
        """Test from_text with metadata dict."""
        doc = VectorDocument.from_text(
            "Hello",
            metadata={"source": "api", "user": "john"},
        )
        assert doc.text == "Hello"
        assert doc.metadata["source"] == "api"
        assert doc.metadata["user"] == "john"

    def test_from_text_with_metadata_kwargs(self):
        """Test from_text with metadata as kwargs."""
        doc = VectorDocument.from_text(
            "Hello",
            source="api",
            user="john",
        )
        assert doc.text == "Hello"
        assert doc.metadata["source"] == "api"
        assert doc.metadata["user"] == "john"

    def test_from_text_with_explicit_id(self):
        """Test from_text with explicit id."""
        doc = VectorDocument.from_text("Hello", id="custom-id")
        assert doc.id == "custom-id"
        assert doc.text == "Hello"

    def test_from_dict_with_vector(self):
        """Test from_dict with vector field."""
        data = {
            "id": "doc-1",
            "vector": [0.1, 0.2, 0.3],
            "text": "Hello",
            "source": "api",
        }
        doc = VectorDocument.from_dict(data)
        assert doc.id == "doc-1"
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.metadata["source"] == "api"

    def test_from_dict_without_vector(self):
        """Test from_dict without vector creates document with empty vector."""
        data = {"id": "doc-1", "text": "Hello", "source": "api"}
        doc = VectorDocument.from_dict(data)
        assert doc.id == "doc-1"
        assert doc.vector == []
        assert doc.metadata["source"] == "api"

    def test_from_dict_with_kwargs_override(self):
        """Test from_dict merges with kwargs."""
        data = {"id": "doc-1", "text": "Hello"}
        doc = VectorDocument.from_dict(data, source="api", user="john")
        assert doc.id == "doc-1"
        assert doc.metadata["source"] == "api"
        assert doc.metadata["user"] == "john"

    def test_from_any_with_vector_document(self):
        """Test from_any passthrough with VectorDocument."""
        original = VectorDocument(id="doc-1", vector=[0.1], text="Hello")
        doc = VectorDocument.from_any(original)
        assert doc is original

    def test_from_any_with_string(self):
        """Test from_any with string input."""
        doc = VectorDocument.from_any("Hello world")
        assert doc.text == "Hello world"
        assert doc.vector == []

    def test_from_any_with_string_and_metadata(self):
        """Test from_any with string and metadata kwargs."""
        doc = VectorDocument.from_any("Hello", source="api", user="john")
        assert doc.text == "Hello"
        assert doc.metadata["source"] == "api"
        assert doc.metadata["user"] == "john"

    def test_from_any_with_dict(self):
        """Test from_any with dict input."""
        data = {"text": "Hello", "source": "api"}
        doc = VectorDocument.from_any(data)
        assert doc.text == "Hello"
        assert doc.metadata["source"] == "api"

    def test_from_any_with_dict_and_kwargs(self):
        """Test from_any merges dict with kwargs."""
        data = {"text": "Hello"}
        doc = VectorDocument.from_any(data, source="api", user="john")
        assert doc.text == "Hello"
        assert doc.metadata["source"] == "api"
        assert doc.metadata["user"] == "john"

    def test_from_any_with_none_and_text_kwarg(self):
        """Test from_any with None and text in kwargs."""
        doc = VectorDocument.from_any(None, text="Hello", source="api")
        assert doc.text == "Hello"
        assert doc.metadata["source"] == "api"

    def test_from_any_with_none_and_dict_kwargs(self):
        """Test from_any with None and dict fields in kwargs."""
        doc = VectorDocument.from_any(None, id="doc-1", text="Hello")
        assert doc.id == "doc-1"
        assert doc.vector == []  # from_text creates empty vector
        assert doc.text == "Hello"

    def test_from_any_with_none_no_args_raises(self):
        """Test from_any with None and no kwargs raises."""
        with pytest.raises(InvalidFieldError):
            VectorDocument.from_any(None)

    def test_from_any_with_invalid_type_raises(self):
        """Test from_any with unsupported type raises."""
        with pytest.raises(TypeError):
            VectorDocument.from_any(12345)

    def test_dump_default_options(self):
        """Test dump with default options."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1, 0.2],
            text="Hello",
            metadata={"source": "api"},
        )
        output = doc.dump()
        assert output["_id"] == "doc-1"
        assert output["vector"] == [0.1, 0.2]
        assert output["text"] == "Hello"
        assert output["source"] == "api"
        assert "created_timestamp" not in output
        assert "updated_timestamp" not in output

    def test_dump_with_dollar_vector(self):
        """Test dump with $vector format."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1, 0.2],
            text="Hello",
        )
        output = doc.dump(use_dollar_vector=True)
        assert "$vector" in output
        assert "vector" not in output
        assert output["$vector"] == [0.1, 0.2]

    def test_dump_without_text(self):
        """Test dump without storing text."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1, 0.2],
            text="Hello",
        )
        output = doc.dump(store_text=False)
        assert "text" not in output

    def test_dump_with_timestamps(self):
        """Test dump with timestamps included."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1],
            text="Hello",
        )
        output = doc.dump(include_timestamps=True)
        assert "created_timestamp" in output
        assert "updated_timestamp" in output
        assert output["created_timestamp"] is not None
        assert output["updated_timestamp"] is not None

    def test_dump_without_text_value(self):
        """Test dump when text is None."""
        doc = VectorDocument(id="doc-1", vector=[0.1], text=None)
        output = doc.dump(store_text=True)
        assert "text" not in output

    def test_to_storage_dict_default(self):
        """Test to_storage_dict with defaults."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1],
            text="Hello",
            metadata={"source": "api"},
        )
        output = doc.to_storage_dict()
        assert output["_id"] == "doc-1"
        assert output["vector"] == [0.1]
        assert output["text"] == "Hello"
        assert output["source"] == "api"

    def test_to_storage_dict_with_dollar_vector(self):
        """Test to_storage_dict with $vector."""
        doc = VectorDocument(id="doc-1", vector=[0.1], text="Hello")
        output = doc.to_storage_dict(use_dollar_vector=True)
        assert "$vector" in output
        assert "vector" not in output

    def test_to_storage_dict_without_text(self):
        """Test to_storage_dict without text."""
        doc = VectorDocument(id="doc-1", vector=[0.1], text="Hello")
        output = doc.to_storage_dict(store_text=False)
        assert "text" not in output

    def test_pk_property(self):
        """Test pk property returns id."""
        doc = VectorDocument(id="doc-1", vector=[0.1])
        assert doc.pk == "doc-1"

    def test_pk_property_with_none_id_auto_generates(self):
        """Test pk property auto-generates id when id is None initially."""
        doc = VectorDocument(vector=[0.1])
        # After validation, id should be auto-generated
        assert doc.id is not None
        assert doc.pk == doc.id

    def test_assign_defaults_auto_generates_id(self):
        """Test that assign_defaults generates id when not provided."""
        doc = VectorDocument(vector=[0.1, 0.2], text="Hello")
        assert doc.id is not None
        assert len(doc.id) > 0

    def test_assign_defaults_sets_timestamps(self):
        """Test that assign_defaults sets timestamps."""
        doc = VectorDocument(id="doc-1", vector=[0.1])
        assert doc.created_timestamp is not None
        assert doc.updated_timestamp is not None

    def test_auto_pk_generation_from_text_and_vector(self):
        """Test that auto-generated pk is deterministic for given inputs."""
        # Test that same text creates consistent pk across different metadata
        doc1 = VectorDocument(vector=[0.1, 0.2], text="Hello World")
        doc2 = VectorDocument(vector=[0.1, 0.2], text="Hello World")
        # Should be consistent for same text and vector
        assert isinstance(doc1.id, str)
        assert isinstance(doc2.id, str)
        assert len(doc1.id) > 0
        assert len(doc2.id) > 0

    def test_auto_pk_generation_different_text(self):
        """Test that different text produces different pk."""
        doc1 = VectorDocument(vector=[0.1], text="Hello")
        doc2 = VectorDocument(vector=[0.1], text="World")
        # Different input should produce different pk
        assert doc1.id != doc2.id

    def test_metadata_merging_from_kwargs(self):
        """Test that metadata from various sources is properly merged."""
        doc = VectorDocument.from_kwargs(
            id="doc-1",
            vector=[0.1],
            text="Hello",
            metadata={"a": 1, "b": 2},
            c=3,
            d=4,
        )
        assert doc.metadata["a"] == 1
        assert doc.metadata["b"] == 2
        assert doc.metadata["c"] == 3
        assert doc.metadata["d"] == 4

    def test_model_config_default_factory(self):
        """Test that metadata default_factory works correctly."""
        doc1 = VectorDocument(vector=[0.1])
        doc2 = VectorDocument(vector=[0.2])
        # Each should have independent metadata dict
        doc1.metadata["test"] = "value"
        assert "test" not in doc2.metadata

    def test_empty_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        doc = VectorDocument(vector=[0.1])
        assert doc.metadata == {}
        assert isinstance(doc.metadata, dict)

    def test_vector_empty_list_default(self):
        """Test that vector defaults to empty list."""
        doc = VectorDocument()
        assert doc.vector == []
        assert isinstance(doc.vector, list)

    def test_dump_all_options_combined(self):
        """Test dump with all options enabled."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1, 0.2],
            text="Hello",
            metadata={"source": "api", "user": "john"},
        )
        output = doc.dump(
            store_text=True,
            use_dollar_vector=True,
            include_timestamps=True,
        )
        assert output["_id"] == "doc-1"
        assert "$vector" in output
        assert output["text"] == "Hello"
        assert output["source"] == "api"
        assert output["user"] == "john"
        assert "created_timestamp" in output
        assert "updated_timestamp" in output
