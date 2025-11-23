"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from crossvector.schema import (
    Document,
    SearchRequest,
    UpsertRequest,
    VectorRequest,
)


class TestDocumentSchema:
    """Tests for Document schema."""

    def test_valid_document(self):
        """Test creating a valid document."""
        doc = Document(id="doc1", text="Sample text", metadata={"key": "value"})

        assert doc.id == "doc1"
        assert doc.text == "Sample text"
        assert doc.metadata == {"key": "value"}

    def test_document_with_empty_metadata(self):
        """Test document with empty metadata."""
        doc = Document(id="doc1", text="Sample text")

        assert doc.metadata == {}

    def test_document_missing_required_fields(self):
        """Test document validation fails without required fields."""
        with pytest.raises(ValidationError):
            Document(id="doc1")  # Missing text

    def test_document_auto_id_generation(self):
        """Test that ID is automatically generated if missing."""
        doc = Document(text="Sample text")
        assert doc.id is not None
        assert len(doc.id) == 64  # SHA256 hash length

    def test_document_timestamps(self):
        """Test that timestamps are automatically generated."""
        doc = Document(text="Sample text")

        assert doc.created_timestamp is not None
        assert doc.updated_timestamp is not None

        # Both should be float (Unix timestamp)
        assert isinstance(doc.created_timestamp, float)
        assert isinstance(doc.updated_timestamp, float)

        # Should be reasonable values (after year 2020)
        assert doc.created_timestamp > 1577836800  # 2020-01-01
        assert doc.updated_timestamp > 1577836800

        # For a new document, created_timestamp and updated_timestamp should be the same
        assert doc.created_timestamp == doc.updated_timestamp

    def test_document_timestamp_update(self):
        """Test that updated_timestamp is refreshed when document is recreated."""
        import time

        doc1 = Document(id="test-id", text="Sample text")
        created_ts_1 = doc1.created_timestamp
        updated_ts_1 = doc1.updated_timestamp

        time.sleep(0.01)  # Small delay

        # Recreate with same ID but preserve created_timestamp
        doc2 = Document(id="test-id", text="Sample text", created_timestamp=created_ts_1)

        assert doc2.created_timestamp == created_ts_1  # Should preserve original created_timestamp
        assert doc2.updated_timestamp > updated_ts_1  # Should have new updated_timestamp (later)

    def test_document_reserved_fields_warning(self):
        """Test that using reserved fields in metadata triggers a warning."""
        import warnings

        # Should trigger warning for legacy fields
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            doc = Document(text="Sample text", metadata={"created_at": "custom_value", "updated_at": "another_value"})

            # Check that warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "reserved timestamp fields" in str(w[0].message)

            # Check that automatic timestamps are still set with _cv_ prefix
            assert doc.created_timestamp is not None
            assert doc.updated_timestamp is not None

    def test_document_user_timestamps_preserved(self):
        """Test that user's own created_at/updated_at in metadata are preserved."""
        doc = Document(
            text="Sample text",
            metadata={
                "created_at": "2024-01-15T10:00:00Z",  # User's article timestamp
                "updated_at": "2024-11-20T15:30:00Z",  # User's article timestamp
            },
        )

        # CrossVector timestamps should exist with _cv_ prefix
        assert doc.created_timestamp is not None
        assert doc.updated_timestamp is not None

        # User's timestamps should still be accessible (though warned about)
        # They will be in metadata but overridden when stored

    def test_document_metadata_types(self):
        """Test various metadata types."""
        doc = Document(
            id="doc1",
            text="Sample text",
            metadata={"string": "value", "number": 42, "float": 3.14, "bool": True, "nested": {"key": "value"}},
        )

        assert doc.metadata["string"] == "value"
        assert doc.metadata["number"] == 42
        assert doc.metadata["bool"] is True
        assert doc.metadata["nested"]["key"] == "value"


class TestUpsertRequest:
    """Tests for UpsertRequest schema."""

    def test_valid_upsert_request(self):
        """Test creating a valid upsert request."""
        docs = [
            Document(id="doc1", text="Text 1"),
            Document(id="doc2", text="Text 2"),
        ]
        request = UpsertRequest(documents=docs)

        assert len(request.documents) == 2
        assert request.documents[0].id == "doc1"

    def test_empty_upsert_request(self):
        """Test upsert request with empty documents list."""
        request = UpsertRequest(documents=[])

        assert request.documents == []

    def test_upsert_request_validation(self):
        """Test upsert request validation."""
        # Invalid document in list
        with pytest.raises(ValidationError):
            UpsertRequest(documents=[{"id": "doc1"}])  # Not a Document object


class TestSearchRequest:
    """Tests for SearchRequest schema."""

    def test_valid_search_request(self):
        """Test creating a valid search request."""
        request = SearchRequest(query="test query", limit=10)

        assert request.query == "test query"
        assert request.limit == 10

    def test_search_request_defaults(self):
        """Test search request with default values."""
        request = SearchRequest(query="test query")

        assert request.limit == 5  # Default
        assert request.fields == {"text", "metadata"}  # Default

    def test_search_request_custom_fields(self):
        """Test search request with custom fields."""
        request = SearchRequest(query="test query", fields={"text", "score", "custom_field"})

        assert "text" in request.fields
        assert "score" in request.fields
        assert "custom_field" in request.fields

    def test_search_request_empty_query(self):
        """Test search request with empty query."""
        request = SearchRequest(query="")

        assert request.query == ""


class TestVectorRequest:
    """Tests for VectorRequest schema."""

    def test_upsert_operation(self):
        """Test vector store request with upsert operation."""
        docs = [Document(id="doc1", text="Text 1")]
        upsert_params = UpsertRequest(documents=docs)

        request = VectorRequest(operation="upsert", params=upsert_params)

        assert request.operation == "upsert"
        assert isinstance(request.params, UpsertRequest)

    def test_search_operation(self):
        """Test vector store request with search operation."""
        search_params = SearchRequest(query="test")

        request = VectorRequest(operation="search", params=search_params)

        assert request.operation == "search"
        assert isinstance(request.params, SearchRequest)

    def test_invalid_operation(self):
        """Test vector store request with invalid operation."""
        with pytest.raises(ValidationError):
            VectorRequest(operation="invalid", params=SearchRequest(query="test"))


class TestSchemaIntegration:
    """Integration tests for schemas."""

    def test_full_workflow_schemas(self):
        """Test a complete workflow using all schemas."""
        # Create documents
        docs = [Document(id=f"doc_{i}", text=f"Sample text {i}", metadata={"index": i}) for i in range(5)]

        # Create upsert request
        upsert_req = UpsertRequest(documents=docs)
        assert len(upsert_req.documents) == 5

        # Create search request
        search_req = SearchRequest(query="sample", limit=3, fields={"text", "metadata", "score"})
        assert search_req.limit == 3

        # Create vector store requests
        upsert_vector_req = VectorRequest(operation="upsert", params=upsert_req)
        search_vector_req = VectorRequest(operation="search", params=search_req)

        assert upsert_vector_req.operation == "upsert"
        assert search_vector_req.operation == "search"

    def test_document_serialization(self):
        """Test document serialization to dict."""
        doc = Document(id="doc1", text="Sample text", metadata={"key": "value", "num": 42})

        doc_dict = doc.model_dump()

        assert doc_dict["id"] == "doc1"
        assert doc_dict["text"] == "Sample text"
        assert doc_dict["metadata"]["key"] == "value"
        assert doc_dict["metadata"]["num"] == 42

    def test_request_serialization(self):
        """Test request serialization."""
        request = SearchRequest(query="test", limit=10, fields={"text", "score"})

        req_dict = request.model_dump()

        assert req_dict["query"] == "test"
        assert req_dict["limit"] == 10
        assert "text" in req_dict["fields"]
        assert "score" in req_dict["fields"]
