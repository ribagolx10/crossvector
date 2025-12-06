"""Tests for VectorDocument schema."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from crossvector.exceptions import MissingFieldError
from crossvector.schema import VectorDocument


class TestVectorDocumentCreation:
    """Tests for VectorDocument creation and initialization."""

    def test_create_minimal(self):
        doc = VectorDocument(vector=[0.1, 0.2, 0.3])
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.id is not None  # Auto-generated
        assert doc.text is None
        assert doc.metadata == {}
        assert doc.created_timestamp is not None
        assert doc.updated_timestamp is not None

    def test_create_with_all_fields(self):
        doc = VectorDocument(
            id="test-123",
            vector=[0.1, 0.2],
            text="Test text",
            metadata={"key": "value"},
        )
        assert doc.id == "test-123"
        assert doc.vector == [0.1, 0.2]
        assert doc.text == "Test text"
        assert doc.metadata == {"key": "value"}

    def test_pk_property(self):
        doc = VectorDocument(id="test-pk", vector=[0.1])
        assert doc.pk == "test-pk"

    def test_pk_property_raises_when_none(self):
        doc = VectorDocument(vector=[0.1])
        doc.id = None
        with pytest.raises(MissingFieldError, match="Document ID not set"):
            _ = doc.pk

    def test_auto_generate_id(self):
        doc = VectorDocument(vector=[0.1, 0.2], text="test")
        assert doc.id is not None
        assert len(doc.id) > 0

    def test_timestamps_auto_set(self):
        before = datetime.now(timezone.utc).timestamp()
        doc = VectorDocument(vector=[0.1])
        after = datetime.now(timezone.utc).timestamp()
        assert before <= doc.created_timestamp <= after
        assert before <= doc.updated_timestamp <= after

    def test_timestamps_preserved_if_set(self):
        custom_ts = 1000.0
        doc = VectorDocument(
            vector=[0.1],
            created_timestamp=custom_ts,
        )
        assert doc.created_timestamp == custom_ts
        # updated_timestamp should still be set to current time
        assert doc.updated_timestamp > custom_ts


class TestFromKwargs:
    """Tests for VectorDocument.from_kwargs() class method."""

    def test_from_kwargs_basic(self):
        doc = VectorDocument.from_kwargs(
            id="test-1",
            vector=[0.1, 0.2],
            text="Hello",
            metadata={"source": "test"},
        )
        assert doc.id == "test-1"
        assert doc.vector == [0.1, 0.2]
        assert doc.text == "Hello"
        assert doc.metadata == {"source": "test"}

    def test_from_kwargs_with_underscore_id(self):
        doc = VectorDocument.from_kwargs(_id="test-2", vector=[0.1])
        assert doc.id == "test-2"

    def test_from_kwargs_with_pk(self):
        doc = VectorDocument.from_kwargs(pk="test-3", vector=[0.1])
        assert doc.id == "test-3"

    def test_from_kwargs_with_dollar_vector(self):
        doc = VectorDocument.from_kwargs(id="test", **{"$vector": [0.5, 0.6]})
        assert doc.vector == [0.5, 0.6]

    def test_from_kwargs_missing_vector(self):
        with pytest.raises(MissingFieldError, match="vector"):
            VectorDocument.from_kwargs(id="test", text="hello")

    def test_from_kwargs_extra_fields_to_metadata(self):
        doc = VectorDocument.from_kwargs(
            vector=[0.1],
            text="test",
            custom_field="value",
            another_field=123,
        )
        assert doc.metadata["custom_field"] == "value"
        assert doc.metadata["another_field"] == 123

    def test_from_kwargs_metadata_not_overwritten(self):
        doc = VectorDocument.from_kwargs(
            vector=[0.1],
            metadata={"existing": "value"},
            new_field="new_value",
        )
        assert doc.metadata["existing"] == "value"
        assert doc.metadata["new_field"] == "new_value"


class TestFromText:
    """Tests for VectorDocument.from_text() class method."""

    def test_from_text_basic(self):
        doc = VectorDocument.from_text("Hello world")
        assert doc.text == "Hello world"
        assert doc.vector == []
        assert doc.id is not None

    def test_from_text_with_id(self):
        doc = VectorDocument.from_text("Hello", id="custom-id")
        assert doc.id == "custom-id"
        assert doc.text == "Hello"

    def test_from_text_with_metadata_dict(self):
        doc = VectorDocument.from_text("Hello", metadata={"source": "api"})
        assert doc.metadata == {"source": "api"}

    def test_from_text_with_metadata_kwargs(self):
        doc = VectorDocument.from_text("Hello", source="api", user_id="123")
        assert doc.metadata["source"] == "api"
        assert doc.metadata["user_id"] == "123"

    def test_from_text_metadata_merge(self):
        doc = VectorDocument.from_text(
            "Hello",
            metadata={"existing": "value"},
            new_field="new",
        )
        assert doc.metadata["existing"] == "value"
        assert doc.metadata["new_field"] == "new"


class TestFromDict:
    """Tests for VectorDocument.from_dict() class method."""

    def test_from_dict_with_vector(self):
        data = {"id": "test", "vector": [0.1, 0.2], "text": "Hello"}
        doc = VectorDocument.from_dict(data)
        assert doc.id == "test"
        assert doc.vector == [0.1, 0.2]
        assert doc.text == "Hello"

    def test_from_dict_with_dollar_vector(self):
        data = {"id": "test", "$vector": [0.3, 0.4]}
        doc = VectorDocument.from_dict(data)
        assert doc.vector == [0.3, 0.4]

    def test_from_dict_kwargs_override(self):
        data = {"id": "test", "vector": [0.1], "source": "original"}
        doc = VectorDocument.from_dict(data, source="overridden")
        assert doc.metadata["source"] == "overridden"

    def test_from_dict_without_vector(self):
        data = {"text": "Hello", "source": "api"}
        doc = VectorDocument.from_dict(data)
        # Should create doc with empty vector
        assert doc.text == "Hello"
        assert doc.metadata["source"] == "api"


class TestToStorageDict:
    """Tests for to_storage_dict() method."""

    def test_to_storage_dict_basic(self):
        doc = VectorDocument(id="test", vector=[0.1, 0.2], text="Hello")
        result = doc.to_storage_dict()
        assert result["_id"] == "test"
        assert result["vector"] == [0.1, 0.2]
        assert result["text"] == "Hello"

    def test_to_storage_dict_with_dollar_vector(self):
        doc = VectorDocument(id="test", vector=[0.1])
        result = doc.to_storage_dict(use_dollar_vector=True)
        assert result["$vector"] == [0.1]
        assert "vector" not in result

    def test_to_storage_dict_without_text(self):
        doc = VectorDocument(id="test", vector=[0.1], text="Hello")
        result = doc.to_storage_dict(store_text=False)
        assert "text" not in result

    def test_to_storage_dict_with_metadata(self):
        doc = VectorDocument(
            id="test",
            vector=[0.1],
            metadata={"key": "value", "num": 123},
        )
        result = doc.to_storage_dict()
        assert result["key"] == "value"
        assert result["num"] == 123


class TestModelDump:
    """Tests for model_dump() method (Pydantic standard)."""

    def test_model_dump_basic(self):
        doc = VectorDocument(id="test", vector=[0.1], text="Hello")
        result = doc.model_dump()
        assert result["id"] == "test"
        assert result["vector"] == [0.1]
        assert result["text"] == "Hello"
        assert "created_timestamp" in result
        assert "updated_timestamp" in result

    def test_model_dump_exclude_none(self):
        doc = VectorDocument(id="test", vector=[0.1])  # text is None
        result = doc.model_dump(exclude_none=True)
        assert "text" not in result
        assert "id" in result

    def test_model_dump_include_fields(self):
        doc = VectorDocument(id="test", vector=[0.1], text="Hello")
        result = doc.model_dump(include={"id", "text"})
        assert "id" in result
        assert "text" in result
        assert "vector" not in result

    def test_model_dump_exclude_fields(self):
        doc = VectorDocument(id="test", vector=[0.1], text="Hello")
        result = doc.model_dump(exclude={"vector", "created_timestamp"})
        assert "id" in result
        assert "text" in result
        assert "vector" not in result
        assert "created_timestamp" not in result


class TestModelDumpJson:
    """Tests for model_dump_json() method."""

    def test_model_dump_json_basic(self):
        doc = VectorDocument(id="test", vector=[0.1, 0.2])
        json_str = doc.model_dump_json()
        assert isinstance(json_str, str)
        assert "test" in json_str
        assert "0.1" in json_str


class TestEquality:
    """Tests for document equality."""

    def test_equal_documents(self):
        doc1 = VectorDocument(id="test", vector=[0.1], text="Hello")
        doc2 = VectorDocument(id="test", vector=[0.1], text="Hello")
        # Pydantic BaseModel uses field comparison
        assert doc1.id == doc2.id
        assert doc1.vector == doc2.vector
        assert doc1.text == doc2.text

    def test_different_documents(self):
        doc1 = VectorDocument(id="test1", vector=[0.1])
        doc2 = VectorDocument(id="test2", vector=[0.1])
        assert doc1.id != doc2.id


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_vector(self):
        doc = VectorDocument(vector=[])
        assert doc.vector == []

    def test_empty_metadata(self):
        doc = VectorDocument(vector=[0.1], metadata={})
        assert doc.metadata == {}

    def test_large_vector(self):
        large_vector = [0.1] * 1536  # Common embedding size
        doc = VectorDocument(vector=large_vector)
        assert len(doc.vector) == 1536

    def test_nested_metadata(self):
        doc = VectorDocument(
            vector=[0.1],
            metadata={
                "user": {"id": "123", "name": "Test"},
                "tags": ["tag1", "tag2"],
            },
        )
        assert doc.metadata["user"]["id"] == "123"
        assert doc.metadata["tags"] == ["tag1", "tag2"]

    @patch("crossvector.schema.generate_pk")
    def test_custom_pk_generation(self, mock_generate):
        mock_generate.return_value = "custom-generated-id"
        doc = VectorDocument(vector=[0.1], text="test")
        assert doc.id == "custom-generated-id"
        mock_generate.assert_called_once()
