"""Integration tests for ChromaDB with Query DSL and VectorEngine.

Tests common DSL operators with real ChromaDB backend to ensure:
- Q objects compile correctly to ChromaDB filter format
- All 8 common operators work end-to-end
- Flattened metadata behavior is handled correctly
- Metadata-only search is supported
"""

import pytest
from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.dbs.chroma import ChromaAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.exceptions import MissingConfigError
from crossvector.querydsl import Q

load_dotenv()


@pytest.fixture(scope="module")
def chroma_engine():
    """Create VectorEngine with ChromaDB adapter for testing."""
    try:
        embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        db = ChromaAdapter()
        engine = VectorEngine(
            embedding=embedding,
            db=db,
            collection_name="test_querydsl_chroma",
        )

        # Clean up before tests
        try:
            engine.drop_collection("test_querydsl_chroma")
        except Exception:
            pass

        # Reinitialize
        engine = VectorEngine(
            embedding=embedding,
            db=db,
            collection_name="test_querydsl_chroma",
        )

        yield engine

        # Cleanup after tests
        try:
            engine.drop_collection("test_querydsl_chroma")
        except Exception:
            pass

    except (MissingConfigError, Exception) as e:
        pytest.skip(f"ChromaDB not available: {e}")


@pytest.fixture(scope="module")
def sample_docs(chroma_engine):
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

    created = chroma_engine.bulk_create(docs)
    return created


class TestChromaQueryDSL:
    """Test Query DSL with ChromaDB backend."""

    def test_eq_operator(self, chroma_engine, sample_docs):
        """Test $eq operator with Q object."""
        results = chroma_engine.search(where=Q(category="tech"), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("category") == "tech" for doc in results)

    def test_ne_operator(self, chroma_engine, sample_docs):
        """Test $ne operator."""
        results = chroma_engine.search(where=Q(category__ne="tech"), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") != "tech" for doc in results)

    def test_gt_operator(self, chroma_engine, sample_docs):
        """Test $gt operator for numeric comparison."""
        results = chroma_engine.search(where=Q(year__gt=2023), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("year") > 2023 for doc in results)

    def test_gte_operator(self, chroma_engine, sample_docs):
        """Test $gte operator."""
        results = chroma_engine.search(where=Q(score__gte=90), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") >= 90 for doc in results)

    def test_lt_operator(self, chroma_engine, sample_docs):
        """Test $lt operator."""
        results = chroma_engine.search(where=Q(year__lt=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("year") < 2024 for doc in results)

    def test_lte_operator(self, chroma_engine, sample_docs):
        """Test $lte operator."""
        results = chroma_engine.search(where=Q(score__lte=85), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") <= 85 for doc in results)

    def test_in_operator(self, chroma_engine, sample_docs):
        """Test $in operator."""
        results = chroma_engine.search(where=Q(category__in=["tech", "food"]), limit=10)
        assert len(results) == 4
        assert all(doc.metadata.get("category") in ["tech", "food"] for doc in results)

    def test_nin_operator(self, chroma_engine, sample_docs):
        """Test $nin operator."""
        results = chroma_engine.search(where=Q(category__nin=["tech", "food"]), limit=10)
        assert len(results) == 1
        assert results[0].metadata.get("category") == "travel"

    def test_and_combination(self, chroma_engine, sample_docs):
        """Test combining filters with AND."""
        results = chroma_engine.search(where=Q(category="tech") & Q(year__gte=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    def test_or_combination(self, chroma_engine, sample_docs):
        """Test combining filters with OR."""
        results = chroma_engine.search(where=Q(category="food") | Q(category="travel"), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") in ["food", "travel"] for doc in results)

    def test_complex_combination(self, chroma_engine, sample_docs):
        """Test complex AND/OR combinations."""
        results = chroma_engine.search(where=(Q(category="tech") & Q(year=2024)) | Q(category="travel"), limit=10)
        assert len(results) == 3  # 2 tech docs from 2024 + 1 travel doc

    def test_metadata_only_search(self, chroma_engine, sample_docs):
        """Test metadata-only search (no vector, no text query)."""
        # ChromaDB supports metadata-only search via get()
        results = chroma_engine.db.search(vector=None, where=Q(category="tech") & Q(score__gte=90), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("score") >= 90 for doc in results)

    def test_universal_dict_format(self, chroma_engine, sample_docs):
        """Test using universal dict format instead of Q objects."""
        where_dict = {"category": {"$eq": "tech"}, "year": {"$gte": 2024}}
        results = chroma_engine.search(where=where_dict, limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    def test_flattened_nested_metadata(self, chroma_engine):
        """Test nested metadata with ChromaDB's flattened storage.

        ChromaDB flattens nested metadata, so info.lang becomes "info.lang" key.
        Query using dot notation in field name.
        """
        # Create doc with nested metadata
        _ = chroma_engine.create(
            {"id": "nested1", "text": "Test nested metadata", "metadata": {"info": {"lang": "en", "tier": "gold"}}}
        )

        # Query using dot notation (flattened key)
        results = chroma_engine.search(where=Q(info__lang="en"), limit=10)
        assert len(results) >= 1
        # ChromaDB stores as "info.lang" key, so check flattened structure
        found = any(d.id == "nested1" and d.metadata.get("info", {}).get("lang") == "en" for d in results)
        assert found

        # Cleanup
        chroma_engine.delete("nested1")

    def test_range_query(self, chroma_engine, sample_docs):
        """Test range queries (between values)."""
        results = chroma_engine.search(where=Q(score__gte=80) & Q(score__lte=90), limit=10)
        assert len(results) == 2
        assert all(80 <= doc.metadata.get("score") <= 90 for doc in results)

    def test_requires_and_wrapper(self, chroma_engine, sample_docs):
        """Test that ChromaDB compiler wraps multiple fields in $and."""
        # Multiple top-level fields should be wrapped in $and by compiler
        where_q = Q(category="tech", year=2024)
        compiled = chroma_engine.db.where_compiler.to_where(where_q)

        # ChromaDB requires $and wrapper for multiple fields
        assert "$and" in compiled or len(compiled) == 1

        # Verify it works end-to-end
        results = chroma_engine.search(where=where_q, limit=10)
        assert len(results) == 2
