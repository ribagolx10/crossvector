"""Integration tests for Milvus with Query DSL and VectorEngine.

Targets real Milvus. Configure using TEST_ env vars first,
with static default collection name.
"""

import pytest
from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.dbs.milvus import MilvusAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.exceptions import MissingConfigError
from crossvector.querydsl import Q

load_dotenv()


@pytest.fixture(scope="module")
def milvus_engine():
    """Create VectorEngine with Milvus adapter for testing."""
    try:
        embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        db = MilvusAdapter()
        engine = VectorEngine(
            db=db,
            embedding=embedding,
            collection_name="test_crossvector",
            store_text=True,
        )

        # Clean up before tests
        try:
            engine.drop_collection("test_crossvector")
        except Exception:
            pass

        # Reinitialize
        engine = VectorEngine(
            db=db,
            embedding=embedding,
            collection_name="test_crossvector",
            store_text=True,
        )

        yield engine

        # Cleanup after tests
        try:
            engine.drop_collection("test_crossvector")
        except Exception:
            pass

    except (MissingConfigError, Exception) as e:
        pytest.skip(f"Milvus not available: {e}")


@pytest.fixture(scope="module")
def sample_docs(milvus_engine):
    """Insert sample documents for testing."""
    docs = [
        {
            "id": "doc1",
            "text": "AI and machine learning basics",
            "metadata": {"category": "tech", "year": 2024, "score": 95},
        },
        {"id": "doc2", "text": "Python programming guide", "metadata": {"category": "tech", "year": 2023, "score": 88}},
        {
            "id": "doc3",
            "text": "Cooking recipes collection",
            "metadata": {"category": "food", "year": 2024, "score": 75},
        },
        {
            "id": "doc4",
            "text": "Travel destinations Europe",
            "metadata": {"category": "travel", "year": 2022, "score": 82},
        },
        {"id": "doc5", "text": "Database design patterns", "metadata": {"category": "tech", "year": 2024, "score": 91}},
    ]

    created = milvus_engine.bulk_create(docs)
    return created


class TestMilvus:
    """Milvus integration tests (search, filters, constraints)."""

    def test_eq_operator(self, milvus_engine, sample_docs):
        """Test $eq operator with Q object."""
        results = milvus_engine.search(query="technology", where=Q(category="tech"), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("category") == "tech" for doc in results)

    def test_ne_operator(self, milvus_engine, sample_docs):
        """Test $ne operator."""
        results = milvus_engine.search(query="programming", where=Q(category__ne="tech"), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") != "tech" for doc in results)

    def test_gt_operator(self, milvus_engine, sample_docs):
        """Test $gt operator for numeric comparison."""
        results = milvus_engine.search(query="latest", where=Q(year__gt=2023), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("year") > 2023 for doc in results)

    def test_gte_operator(self, milvus_engine, sample_docs):
        """Test $gte operator."""
        results = milvus_engine.search(query="excellent", where=Q(score__gte=90), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") >= 90 for doc in results)

    def test_lt_operator(self, milvus_engine, sample_docs):
        """Test $lt operator."""
        results = milvus_engine.search(query="older", where=Q(year__lt=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("year") < 2024 for doc in results)

    def test_lte_operator(self, milvus_engine, sample_docs):
        """Test $lte operator."""
        results = milvus_engine.search(query="moderate", where=Q(score__lte=85), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") <= 85 for doc in results)

    def test_in_operator(self, milvus_engine, sample_docs):
        """Test $in operator."""
        results = milvus_engine.search(query="content", where=Q(category__in=["tech", "food"]), limit=10)
        assert len(results) == 4
        assert all(doc.metadata.get("category") in ["tech", "food"] for doc in results)

    def test_nin_operator(self, milvus_engine, sample_docs):
        """Test $nin operator."""
        results = milvus_engine.search(query="exploration", where=Q(category__nin=["tech", "food"]), limit=10)
        assert len(results) == 1
        assert results[0].metadata.get("category") == "travel"

    def test_and_combination(self, milvus_engine, sample_docs):
        """Test combining filters with AND."""
        results = milvus_engine.search(query="technology", where=Q(category="tech") & Q(year__gte=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    def test_or_combination(self, milvus_engine, sample_docs):
        """Test combining filters with OR."""
        results = milvus_engine.search(query="content", where=Q(category="food") | Q(category="travel"), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") in ["food", "travel"] for doc in results)

    def test_complex_combination(self, milvus_engine, sample_docs):
        """Test complex AND/OR combinations."""
        results = milvus_engine.search(
            query="information", where=(Q(category="tech") & Q(year=2024)) | Q(category="travel"), limit=10
        )
        assert len(results) == 3  # 2 tech docs from 2024 + 1 travel doc

    def test_metadata_only_search_supported(self, milvus_engine, sample_docs):
        """Test that metadata-only search works (Milvus supports query without vector)."""
        # Milvus supports metadata-only search via query() method
        results = milvus_engine.search(where=Q(category="tech"), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("category") == "tech" for doc in results)

    def test_universal_dict_format(self, milvus_engine, sample_docs):
        """Test using universal dict format instead of Q objects."""
        where_dict = {"category": {"$eq": "tech"}, "year": {"$gte": 2024}}
        results = milvus_engine.search(query="technology", where=where_dict, limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    @pytest.mark.skip(reason="Nested metadata support needs investigation")
    def test_nested_metadata(self, milvus_engine):
        """Test nested metadata queries with JSON field access.

        Milvus stores metadata as JSON field, queries use metadata['key'] syntax.
        """
        # Create doc with nested metadata
        _ = milvus_engine.create(
            {"id": "nested1", "text": "Test nested metadata", "metadata": {"info": {"lang": "en", "tier": "gold"}}}
        )

        # Query nested field - Milvus flattens to metadata['info.lang']
        # Note: nested support depends on how Milvus stores JSON
        results = milvus_engine.search(query="test", where=Q(info__lang="en"), limit=10)
        assert len(results) >= 1
        found = any(d.id == "nested1" for d in results)
        assert found

        # Cleanup
        milvus_engine.delete("nested1")

    def test_range_query(self, milvus_engine, sample_docs):
        """Test range queries (between values)."""
        results = milvus_engine.search(query="content", where=Q(score__gte=80) & Q(score__lte=90), limit=10)
        assert len(results) == 2
        assert all(80 <= doc.metadata.get("score") <= 90 for doc in results)

    def test_metadata_search_capability(self, milvus_engine, sample_docs):
        """Test that Milvus supports metadata-only search."""
        # Adapter should support metadata-only search
        assert milvus_engine.supports_metadata_only

        # Direct adapter call without vector should work
        results = milvus_engine.db.search(vector=None, where=Q(category="tech"), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("category") == "tech" for doc in results)

    def test_boolean_expression_compilation(self, milvus_engine, sample_docs):
        """Test that Q objects compile to Milvus boolean expressions."""
        where_q = Q(category="tech") & Q(score__gte=90)
        compiled = milvus_engine.db.where_compiler.to_where(where_q)

        # Milvus uses boolean expressions with && and metadata['key'] syntax
        assert isinstance(compiled, str)
        assert "metadata['category']" in compiled or "category" in compiled
        assert "&&" in compiled or "and" in compiled.lower()

        # Verify it works end-to-end
        results = milvus_engine.search(query="technology", where=where_q, limit=10)
        assert len(results) == 2

    def test_crud_create_update_delete(self, milvus_engine):
        """Basic CRUD: create, update, get_or_create, update_or_create, delete."""
        # Create (Milvus requires vectors, done via engine.create)
        doc = milvus_engine.create(text="CRUD doc", metadata={"owner": "tester", "tier": "bronze"})
        assert doc.id

        # Update
        milvus_engine.update({"id": doc.id}, text="CRUD doc updated", metadata={"tier": "silver"})
        fetched = milvus_engine.get(doc.id)
        assert fetched.text == "CRUD doc updated"

        # get_or_create existing
        got, created = milvus_engine.get_or_create({"id": doc.id}, defaults={"text": "should not change"})
        assert not created and got.id == doc.id

        # update_or_create new
        uoc, created2 = milvus_engine.update_or_create(
            {"id": "crud-new-1"}, create_defaults={"text": "uoc created", "metadata": {"owner": "tester"}}
        )
        assert created2 and uoc.id == "crud-new-1"

        # Delete
        deleted = milvus_engine.delete(doc.id, uoc.id)
        assert deleted >= 0
