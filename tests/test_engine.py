"""Tests for VectorEngine core functionality."""

from typing import Any, Dict, List, Sequence, Set, Union

from crossvector import VectorEngine
from crossvector.abc import EmbeddingAdapter, VectorDBAdapter
from crossvector.exceptions import CrossVectorError
from crossvector.schema import VectorDocument


class MockEmbeddingAdapter(EmbeddingAdapter):
    """Mock embedding adapter for testing."""

    def __init__(self, dimension=1536):
        super().__init__("mock-model")
        self._dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def get_embeddings(self, texts):
        # Deterministic per-text embedding (value derived from text ord sums)
        vectors = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 100
            base = (seed / 100.0) or 0.01
            vectors.append([base] * self._dimension)
        return vectors


class MockDBAdapter(VectorDBAdapter):
    """Mock database adapter for testing."""

    use_dollar_vector = True

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

    def add_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        pass

    def get_collection(self, collection_name: str):
        return f"mock_collection_{collection_name}"

    def get_or_create_collection(self, collection_name: str, dimension: int, metric: str = "cosine"):
        return f"mock_collection_{collection_name}"

    def upsert(self, documents: List[VectorDocument], batch_size: int | None = None) -> List[VectorDocument]:
        result = []
        for doc in documents:
            doc_dict = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=True)
            self.documents[doc.pk] = doc_dict
            result.append(doc)
        return result

    def bulk_create(
        self,
        documents: List[VectorDocument],
        ignore_conflicts: bool = False,
        update_conflicts: bool = False,
    ) -> List[VectorDocument]:
        result = []
        for doc in documents:
            if doc.pk in self.documents:
                if ignore_conflicts:
                    continue
                elif update_conflicts:
                    doc_dict = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=True)
                    self.documents[doc.pk] = doc_dict
                    result.append(doc)
                else:
                    raise ValueError(f"Document with pk {doc.pk} already exists")
            else:
                doc_dict = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=True)
                self.documents[doc.pk] = doc_dict
                result.append(doc)
        return result

    def bulk_update(
        self,
        documents: List[VectorDocument],
        batch_size: int | None = None,
        ignore_conflicts: bool = False,
    ) -> List[VectorDocument]:
        result = []
        for doc in documents:
            if doc.pk not in self.documents:
                if not ignore_conflicts:
                    raise ValueError(f"Document with pk {doc.pk} not found")
            else:
                doc_dict = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=True)
                self.documents[doc.pk].update(doc_dict)
                result.append(doc)
        return result

    def search(
        self,
        vector: List[float],
        limit: int,
        offset: int = 0,
        where: Dict[str, Any] | None = None,
        fields: Set[str] | None = None,
    ) -> List[VectorDocument]:
        # Convert stored dicts back to VectorDocuments
        all_docs = []
        for pk, doc_dict in self.documents.items():
            vector = doc_dict.get("$vector") or doc_dict.get("vector") or []
            text = doc_dict.get("text")
            metadata = {k: v for k, v in doc_dict.items() if k not in ("_id", "$vector", "vector", "text")}
            all_docs.append(VectorDocument(id=pk, vector=vector, text=text, metadata=metadata))

        # Apply offset and limit
        return all_docs[offset : offset + limit]

    def get(self, pk: str) -> VectorDocument:
        doc_dict = self.documents.get(pk)
        if not doc_dict:
            raise ValueError(f"Document with pk {pk} not found")

        vector = doc_dict.get("$vector") or doc_dict.get("vector") or []
        text = doc_dict.get("text")
        metadata = {k: v for k, v in doc_dict.items() if k not in ("_id", "$vector", "vector", "text")}
        return VectorDocument(id=pk, vector=vector, text=text, metadata=metadata)

    def count(self) -> int:
        return len(self.documents)

    def delete(self, ids: Union[str, Sequence[str]]) -> int:
        id_list = [ids] if isinstance(ids, str) else list(ids or [])
        deleted = 0
        for _id in id_list:
            if _id in self.documents:
                del self.documents[_id]
                deleted += 1
        return deleted

    def create(self, document: VectorDocument) -> VectorDocument:
        if document.pk in self.documents:
            raise ValueError(f"Document with pk {document.pk} already exists")
        doc_dict = document.to_storage_dict(store_text=self.store_text, use_dollar_vector=True)
        self.documents[document.pk] = doc_dict
        return document

    def get_or_create(self, document: VectorDocument) -> tuple[VectorDocument, bool]:
        if document.pk in self.documents:
            return self.get(document.pk), False
        return self.create(document), True

    def update(self, document: VectorDocument) -> VectorDocument:
        if document.pk not in self.documents:
            raise CrossVectorError(f"Document with pk {document.pk} not found")
        doc_dict = document.to_storage_dict(store_text=self.store_text, use_dollar_vector=True)
        self.documents[document.pk].update(doc_dict)
        return document

    def update_or_create(self, document: VectorDocument) -> tuple[VectorDocument, bool]:
        if document.pk in self.documents:
            return self.update(document), False
        return self.create(document), True

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the collection (clear all documents)."""
        self.documents.clear()
        self.collection_initialized = False
        return True

    def clear_collection(self) -> int:
        """Clear all documents from the collection."""
        count = len(self.documents)
        self.documents.clear()
        return count


class TestVectorEngine:
    """Test suite for VectorEngine."""

    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        assert engine.embedding == embedding
        assert engine.db == db
        assert engine.collection_name == "test_collection"
        assert db.collection_initialized

    def test_bulk_create_from_texts(self, sample_documents):
        """Test creating documents from text strings."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # Create documents from plain strings
        texts = sample_documents["texts"][:3]
        result = engine.bulk_create(texts)

        assert len(result) == 3
        assert db.count() == 3
        assert all(isinstance(doc, VectorDocument) for doc in result)

    def test_bulk_create_empty_list(self):
        """Test creating from empty list."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        result = engine.bulk_create([])

        assert len(result) == 0
        assert db.count() == 0

    def test_search_documents(self, sample_documents):
        """Test searching documents."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # First create some documents
        docs = [{"id": sample_documents["pks"][i], "text": sample_documents["texts"][i]} for i in range(3)]
        engine.bulk_create(docs)

        # Then search
        results = engine.search("test query", limit=2)

        assert len(results) <= 2
        assert all(isinstance(doc, VectorDocument) for doc in results)

    def test_get_document(self, sample_documents):
        """Test retrieving a document by ID."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # Create a document
        engine.bulk_create([{"id": sample_documents["pks"][0], "text": sample_documents["texts"][0]}])

        # Get it back
        doc = engine.get(sample_documents["pks"][0])

        assert doc is not None
        assert isinstance(doc, VectorDocument)
        assert doc.pk == sample_documents["pks"][0]

    def test_count_documents(self, sample_documents):
        """Test counting documents."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # Initially 0
        assert engine.count() == 0

        # Create 3 documents
        docs = [{"id": sample_documents["pks"][i], "text": sample_documents["texts"][i]} for i in range(3)]
        engine.bulk_create(docs)

        # Should be 3
        assert engine.count() == 3

    def test_delete_one_document(self, sample_documents):
        """Test deleting a single document."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # Create documents
        docs = [{"id": sample_documents["pks"][i], "text": sample_documents["texts"][i]} for i in range(3)]
        engine.bulk_create(docs)
        assert engine.count() == 3

        # Delete single document
        deleted_count = engine.delete(sample_documents["pks"][0])

        assert deleted_count == 1
        assert engine.count() == 2

    def test_delete_many_documents(self, sample_documents):
        """Test deleting multiple documents."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # Create documents
        docs = [{"id": sample_documents["pks"][i], "text": sample_documents["texts"][i]} for i in range(5)]
        engine.bulk_create(docs)
        assert engine.count() == 5

        # Delete multiple
        ids_to_delete = [sample_documents["pks"][0], sample_documents["pks"][1]]
        deleted_count = engine.delete(ids_to_delete)

        assert deleted_count == 2
        assert engine.count() == 3

    def test_delete_empty_list(self):
        """Test deleting with empty ID list."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        deleted_count = engine.delete([])

        assert deleted_count == 0

    def test_document_format(self, sample_documents):
        """Test that documents are formatted correctly for DB adapter."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # Create a document
        engine.bulk_create([{"id": sample_documents["pks"][0], "text": sample_documents["texts"][0]}])

        # Check the stored document format
        stored_doc = db.documents[sample_documents["pks"][0]]

        assert "_id" in stored_doc
        assert "$vector" in stored_doc or "vector" in stored_doc
        assert stored_doc["_id"] == sample_documents["pks"][0]
        vector_key = "$vector" if "$vector" in stored_doc else "vector"
        assert len(stored_doc[vector_key]) == embedding.embedding_dimension

    def test_create_without_store_text(self, sample_documents):
        """Test creating documents with store_text=False."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        # Initialize engine with store_text=False
        engine = VectorEngine(
            embedding=embedding,
            db=db,
            collection_name="test_collection",
            store_text=False,
        )

        # Create a document
        engine.bulk_create([{"id": sample_documents["pks"][0], "text": sample_documents["texts"][0]}])

        # Check the stored document format
        stored_doc = db.documents[sample_documents["pks"][0]]

        assert "_id" in stored_doc
        vector_key = "$vector" if "$vector" in stored_doc else "vector"
        assert vector_key in stored_doc
        # Text should NOT be present
        assert "text" not in stored_doc
        assert stored_doc["_id"] == sample_documents["pks"][0]
        assert len(stored_doc[vector_key]) == embedding.embedding_dimension

    def test_auto_generated_pk(self):
        """Test that pk is automatically generated if not provided."""
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()

        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")

        # Create document without providing pk (just text string)
        text = "This is a test document without ID."
        docs = engine.bulk_create([text])

        assert len(docs) == 1
        assert docs[0].pk is not None
        # Verify pk was generated
        assert isinstance(docs[0].pk, str)
        assert len(docs[0].pk) > 0

    def test_create_single_document(self):
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()
        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")
        doc = engine.create("Simple text document")
        assert doc.pk in db.documents
        assert doc.vector and len(doc.vector) == embedding.embedding_dimension

    def test_update_document_regenerates_embedding(self):
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()
        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")
        original = engine.create({"id": "doc-upd", "text": "First version"})
        original_vector = original.vector.copy()
        updated = engine.update({"id": "doc-upd"}, text="Second version")
        assert updated.pk == original.pk
        assert updated.text == "Second version"
        assert updated.vector != original_vector  # different seed value expected

    def test_get_or_create_creates_new(self):
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()
        db.supports_vector_search = True
        db.supports_metadata_only = True
        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")
        doc, created = engine.get_or_create("Hello world", metadata={"lang": "en"})
        assert created is True
        assert doc.pk in db.documents

    def test_get_or_create_returns_existing_by_pk(self):
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()
        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")
        first = engine.create({"id": "pk-1", "text": "Alpha text"})
        second, created = engine.get_or_create({"id": "pk-1", "text": "Alpha text"})
        assert created is False
        assert second.pk == first.pk

    def test_update_or_create_updates_existing(self):
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()
        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")
        updated, created = engine.update_or_create(
            {"id": "doc-x"}, text="Version B", defaults={"metadata": {"tier": "gold"}}
        )
        assert created is False
        assert updated.text == "Version B"
        assert db.documents[updated.pk]["tier"] == "gold"

    def test_update_or_create_creates_new(self):
        embedding = MockEmbeddingAdapter()
        db = MockDBAdapter()
        engine = VectorEngine(embedding=embedding, db=db, collection_name="test_collection")
        doc, created = engine.update_or_create(
            {"id": "new-doc", "text": "Brand new"}, create_defaults={"metadata": {"owner": "tester"}}
        )
        assert created is True
        stored = db.documents[doc.pk]
        assert stored.get("owner") == "tester"
