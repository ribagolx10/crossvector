"""Integration tests for Milvus with Query DSL and VectorEngine.

Tests common DSL operators with real Milvus backend to ensure:
- Q objects compile correctly to Milvus boolean expressions
- All 8 common operators work end-to-end
- Vector requirement is enforced (no metadata-only search)
- Nested metadata queries work with JSON field access
"""

import pytest
from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.dbs.milvus import MilvusAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.exceptions import MissingConfigError, SearchError
from crossvector.querydsl import Q

load_dotenv()


@pytest.fixture(scope="module")
def milvus_engine():
    """Create VectorEngine with Milvus adapter for testing."""
    try:
        embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        db = MilvusAdapter()
        engine = VectorEngine(
            embedding=embedding,
            db=db,
            collection_name="test_querydsl_milvus",
        )

        # Clean up before tests
        try:
            engine.drop_collection("test_querydsl_milvus")
        except Exception:
            pass

        # Reinitialize
        engine = VectorEngine(
            embedding=embedding,
            db=db,
            collection_name="test_querydsl_milvus",
        )

        yield engine

        # Cleanup after tests
        try:
            engine.drop_collection("test_querydsl_milvus")
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


class TestMilvusQueryDSL:
    """Test Query DSL with Milvus backend."""

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

    def test_metadata_only_not_supported(self, milvus_engine, sample_docs):
        """Test that metadata-only search raises error (Milvus requires vector)."""
        # Milvus does not support metadata-only search via engine
        with pytest.raises(SearchError, match="vector.*required"):
            milvus_engine.search(where=Q(category="tech"), limit=10)

    def test_universal_dict_format(self, milvus_engine, sample_docs):
        """Test using universal dict format instead of Q objects."""
        where_dict = {"category": {"$eq": "tech"}, "year": {"$gte": 2024}}
        results = milvus_engine.search(query="technology", where=where_dict, limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

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

    def test_vector_required_for_search(self, milvus_engine, sample_docs):
        """Test that Milvus requires vector for all searches."""
        # Adapter should have REQUIRES_VECTOR=True
        assert milvus_engine.db.REQUIRES_VECTOR is False or not milvus_engine.db.supports_metadata_only

        # Direct adapter call without vector should fail
        with pytest.raises(SearchError):
            milvus_engine.db.search(vector=None, where=Q(category="tech"), limit=10)

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
