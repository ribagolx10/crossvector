"""Extended tests for VectorEngine to increase coverage."""

from crossvector.engine import VectorEngine
from crossvector.schema import VectorDocument


class MockEmbedding:
    """Minimal mock embedding adapter."""

    def __init__(self, dim=1536):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def get_embeddings(self, texts):
        return [[0.1 * (i + 1) for _ in range(self._dim)] for i in range(len(texts))]


class MockDB:
    """Minimal mock database adapter."""

    def __init__(self):
        self.docs = {}
        self.collection_initialized = False
        self.store_text = True

    def initialize(self, collection_name, dim, metric="cosine", store_text=True, **kwargs):
        self.collection_initialized = True
        self.collection_name = collection_name
        self.dim = dim
        self.store_text = store_text

    def add_collection(self, collection_name, dim, metric="cosine"):
        pass

    def get_collection(self, collection_name):
        return {}

    def get_or_create_collection(self, collection_name, dim, metric="cosine"):
        return {}

    def bulk_create(self, docs, **kwargs):
        for doc in docs:
            self.docs[doc.pk] = doc
        return docs

    def count(self):
        return len(self.docs)

    def get(self, pk):
        return self.docs.get(pk)

    def search(self, vector, limit=10, **kwargs):
        return list(self.docs.values())[:limit]

    def delete(self, ids):
        if isinstance(ids, str):
            ids = [ids]
        deleted = 0
        for pk in ids:
            if pk in self.docs:
                del self.docs[pk]
                deleted += 1
        return deleted

    def create(self, doc):
        self.docs[doc.pk] = doc
        return doc

    def update(self, doc):
        if doc.pk not in self.docs:
            raise ValueError(f"Document {doc.pk} not found")
        self.docs[doc.pk] = doc
        return doc

    def drop_collection(self, collection_name):
        self.docs.clear()
        return True

    def clear_collection(self):
        count = len(self.docs)
        self.docs.clear()
        return count


class TestVectorEngineExtended:
    """Extended tests for VectorEngine error handling and edge cases."""

    def test_engine_with_custom_collection_name(self):
        """Test engine with custom collection name."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(
            embedding=emb,
            db=db,
            collection_name="custom_collection",
        )
        assert engine.collection_name == "custom_collection"
        assert db.collection_initialized

    def test_engine_with_store_text_false(self):
        """Test engine initialized with store_text=False."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(
            embedding=emb,
            db=db,
            store_text=False,
        )
        assert engine.store_text is False
        assert db.store_text is False

    def test_create_with_explicit_id(self):
        """Test creating document with explicit id."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        doc = engine.create({"id": "my-doc", "text": "Hello"})
        assert doc.pk == "my-doc"
        assert doc.text == "Hello"

    def test_create_from_text_string(self):
        """Test creating document from plain text string."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        doc = engine.create("Simple text")
        assert doc.text == "Simple text"
        assert doc.vector is not None
        assert len(doc.vector) == 1536

    def test_create_with_metadata(self):
        """Test creating document with metadata."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        doc = engine.create("Text", metadata={"source": "api", "user": "john"})
        assert doc.metadata["source"] == "api"
        assert doc.metadata["user"] == "john"

    def test_bulk_create_mixed_types(self):
        """Test bulk_create with mixed input types."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        inputs = [
            "Plain text document",
            {"id": "doc-2", "text": "From dict"},
            VectorDocument(id="doc-3", text="From object", vector=[]),
        ]

        docs = engine.bulk_create(inputs)
        assert len(docs) == 3
        assert docs[0].text == "Plain text document"
        assert docs[1].pk == "doc-2"
        assert docs[2].pk == "doc-3"

    def test_search_with_where_filter(self):
        """Test search with where filter."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        # Create documents with metadata
        engine.create("Doc1", metadata={"category": "news"})
        engine.create("Doc2", metadata={"category": "blog"})

        # Search with filter
        results = engine.search("query", where={"category": "news"})
        assert isinstance(results, list)

    def test_get_existing_document(self):
        """Test getting existing document."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        doc = engine.create("My document")
        retrieved = engine.get(doc.pk)
        assert retrieved.pk == doc.pk
        assert retrieved.text == "My document"

    def test_update_existing_document(self):
        """Test updating existing document."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        engine.create({"id": "doc-1", "text": "Original"})
        updated = engine.update({"id": "doc-1", "text": "Updated"})
        assert updated.pk == "doc-1"
        assert updated.text == "Updated"

    def test_delete_single_document(self):
        """Test deleting single document."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        doc = engine.create("Text")
        assert engine.count() == 1

        deleted = engine.delete(doc.pk)
        assert deleted == 1
        assert engine.count() == 0

    def test_delete_multiple_documents(self):
        """Test deleting multiple documents."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        docs = engine.bulk_create(["Text1", "Text2", "Text3"])
        pks = [d.pk for d in docs]

        deleted = engine.delete(pks[:2])
        assert deleted == 2
        assert engine.count() == 1

    def test_count_documents(self):
        """Test counting documents in collection."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        assert engine.count() == 0
        engine.bulk_create(["Text1", "Text2", "Text3"])
        assert engine.count() == 3

    def test_count_after_delete(self):
        """Test count after deleting documents."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        docs = engine.bulk_create(["Text1", "Text2"])
        assert engine.count() == 2

        engine.delete([docs[0].pk])
        assert engine.count() == 1

    def test_embedding_property(self):
        """Test accessing embedding property."""
        emb = MockEmbedding(dim=512)
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        assert engine.embedding == emb
        assert engine.embedding.dim == 512

    def test_db_property(self):
        """Test accessing db property."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        assert engine.db == db

    def test_create_with_text_only(self):
        """Test creating document with text only."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        # Document with text but no explicit id
        doc = engine.create({"text": "doc-text"})
        assert doc.pk is not None
        assert doc.text == "doc-text"
        assert doc.vector is not None  # Should have embedding

    def test_bulk_create_with_custom_metadata(self):
        """Test bulk_create preserves metadata properly."""
        emb = MockEmbedding()
        db = MockDB()
        engine = VectorEngine(embedding=emb, db=db)

        docs = engine.bulk_create(
            [
                {"text": "Doc1", "source": "api", "priority": 1},
                {"text": "Doc2", "source": "web", "priority": 2},
            ]
        )

        assert docs[0].metadata["source"] == "api"
        assert docs[0].metadata["priority"] == 1
        assert docs[1].metadata["source"] == "web"
        assert docs[1].metadata["priority"] == 2
