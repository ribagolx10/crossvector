"""Tests for VectorEngine core functionality."""

from crossvector import (
    SearchRequest,
    UpsertRequest,
    VectorEngine,
)
from crossvector.abc import EmbeddingAdapter, VectorDBAdapter


class MockEmbeddingAdapter(EmbeddingAdapter):
    """Mock embedding adapter for testing."""

    def __init__(self, dimension=1536):
        super().__init__("mock-model")
        self._dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def get_embeddings(self, texts):
        # Return mock embeddings (simple normalized vectors)
        return [[0.5] * self._dimension for _ in texts]


class MockDBAdapter(VectorDBAdapter):
    """Mock database adapter for testing."""

    def __init__(self):
        self.documents = {}
        self.collection_initialized = False

    def initialize(
        self, collection_name: str, embedding_dimension: int, metric: str = "cosine", store_text: bool = True
    ):
        self.collection_initialized = True
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.store_text = store_text

    def get_collection(self, collection_name: str, embedding_dimension: int, metric: str = "cosine"):
        return f"mock_collection_{collection_name}"

    def upsert(self, documents):
        for doc in documents:
            # Simulate store_text behavior
            if not self.store_text:
                # If store_text is False, we shouldn't store the text content
                # But for the mock, we might want to verify it's NOT there.
                # Let's remove 'text' key if it exists, or set it to None
                if "text" in doc:
                    doc = doc.copy()
                    del doc["text"]
            self.documents[doc["_id"]] = doc

    def search(self, vector, limit: int, fields):
        # Return first N documents
        results = list(self.documents.values())[:limit]
        return results

    def get(self, id: str):
        return self.documents.get(id)

    def count(self) -> int:
        return len(self.documents)

    def delete_one(self, id: str) -> int:
        if id in self.documents:
            del self.documents[id]
            return 1
        return 0

    def delete_many(self, ids) -> int:
        count = 0
        for id in ids:
            if id in self.documents:
                del self.documents[id]
                count += 1
        return count


class TestVectorEngine:
    """Test suite for VectorEngine."""

    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        assert engine.embedding_adapter == embedding_adapter
        assert engine.db_adapter == db_adapter
        assert engine.collection_name == "test_collection"
        assert db_adapter.collection_initialized

    def test_upsert_documents(self, sample_documents):
        """Test upserting documents."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        # Upsert documents
        result = engine.upsert(UpsertRequest(documents=sample_documents[:3]))

        assert result["status"] == "success"
        assert result["count"] == 3
        assert db_adapter.count() == 3

    def test_upsert_empty_documents(self):
        """Test upserting empty document list."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        result = engine.upsert(UpsertRequest(documents=[]))

        assert result["status"] == "noop"
        assert result["count"] == 0

    def test_search_documents(self, sample_documents):
        """Test searching documents."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        # First upsert some documents
        engine.upsert(UpsertRequest(documents=sample_documents[:3]))

        # Then search
        results = engine.search(SearchRequest(query="test query", limit=2))

        assert len(results) <= 2

    def test_get_document(self, sample_documents):
        """Test retrieving a document by ID."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        # Upsert a document
        engine.upsert(UpsertRequest(documents=[sample_documents[0]]))

        # Get it back
        doc = engine.get(sample_documents[0].id)

        assert doc is not None
        assert doc["_id"] == sample_documents[0].id

    def test_count_documents(self, sample_documents):
        """Test counting documents."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        # Initially 0
        assert engine.count() == 0

        # Upsert 3 documents
        engine.upsert(UpsertRequest(documents=sample_documents[:3]))

        # Should be 3
        assert engine.count() == 3

    def test_delete_one_document(self, sample_documents):
        """Test deleting a single document."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        # Upsert documents
        engine.upsert(UpsertRequest(documents=sample_documents[:3]))
        assert engine.count() == 3

        # Delete one
        deleted_count = engine.delete_one(sample_documents[0].id)

        assert deleted_count == 1
        assert engine.count() == 2

    def test_delete_many_documents(self, sample_documents):
        """Test deleting multiple documents."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        # Upsert documents
        engine.upsert(UpsertRequest(documents=sample_documents[:5]))
        assert engine.count() == 5

        # Delete multiple
        ids_to_delete = [sample_documents[0].id, sample_documents[1].id]
        deleted_count = engine.delete_many(ids_to_delete)

        assert deleted_count == 2
        assert engine.count() == 3

    def test_delete_many_empty_list(self):
        """Test deleting with empty ID list."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        deleted_count = engine.delete_many([])

        assert deleted_count == 0

    def test_document_format(self, sample_documents):
        """Test that documents are formatted correctly for DB adapter."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        engine = VectorEngine(
            embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name="test_collection"
        )

        # Upsert a document
        doc = sample_documents[0]
        engine.upsert(UpsertRequest(documents=[doc]))

        # Check the stored document format
        stored_doc = db_adapter.documents[doc.id]

        assert "_id" in stored_doc
        assert "vector" in stored_doc
        assert "text" in stored_doc
        assert stored_doc["_id"] == doc.id
        assert stored_doc["text"] == doc.text
        assert len(stored_doc["vector"]) == embedding_adapter.embedding_dimension

    def test_upsert_without_store_text(self, sample_documents):
        """Test upserting documents with store_text=False."""
        embedding_adapter = MockEmbeddingAdapter()
        db_adapter = MockDBAdapter()

        # Initialize engine with store_text=False
        engine = VectorEngine(
            embedding_adapter=embedding_adapter,
            db_adapter=db_adapter,
            collection_name="test_collection",
            store_text=False,
        )

        # Upsert a document
        doc = sample_documents[0]
        engine.upsert(UpsertRequest(documents=[doc]))

        # Check the stored document format
        stored_doc = db_adapter.documents[doc.id]

        assert "_id" in stored_doc
        assert "vector" in stored_doc
        # Text should NOT be present
        assert "text" not in stored_doc
        assert stored_doc["_id"] == doc.id
        assert len(stored_doc["vector"]) == embedding_adapter.embedding_dimension

    def test_auto_generated_id(self):
        """Test that ID is automatically generated if not provided."""
        from crossvector.schema import Document

        text = "This is a test document without ID."
        doc = Document(text=text)

        assert doc.id is not None
        # Verify it's a valid SHA256 hash (64 hex chars)
        assert len(doc.id) == 64

        # Verify determinism
        doc2 = Document(text=text)
        assert doc.id == doc2.id
