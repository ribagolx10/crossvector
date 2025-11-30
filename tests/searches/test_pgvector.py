"""Integration tests for PgVector with Query DSL and VectorEngine.

Tests common DSL operators with real PgVector backend to ensure:
- Q objects compile correctly to PostgreSQL WHERE clauses
- All 8 common operators work end-to-end
- Nested JSONB metadata queries function properly
- Numeric casting works for comparisons
- Metadata-only search is supported
"""

import pytest
from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.dbs.pgvector import PgVectorAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.exceptions import MissingConfigError
from crossvector.querydsl import Q

load_dotenv()


@pytest.fixture(scope="module")
def pgvector_engine():
    """Create VectorEngine with PgVector adapter for testing."""
    try:
        embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        db = PgVectorAdapter()
        engine = VectorEngine(
            embedding=embedding,
            db=db,
            collection_name="test_querydsl_pgvector",
        )

        # Clean up before tests
        try:
            engine.drop_collection("test_querydsl_pgvector")
        except Exception:
            pass

        # Reinitialize
        engine = VectorEngine(
            embedding=embedding,
            db=db,
            collection_name="test_querydsl_pgvector",
        )

        yield engine

        # Cleanup after tests
        try:
            engine.drop_collection("test_querydsl_pgvector")
        except Exception:
            pass

    except (MissingConfigError, Exception) as e:
        pytest.skip(f"PgVector not available: {e}")


@pytest.fixture(scope="module")
def sample_docs(pgvector_engine):
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

    created = pgvector_engine.bulk_create(docs)
    return created


class TestPgVectorQueryDSL:
    """Test Query DSL with PgVector backend."""

    def test_eq_operator(self, pgvector_engine, sample_docs):
        """Test $eq operator with Q object."""
        results = pgvector_engine.search(where=Q(category="tech"), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("category") == "tech" for doc in results)

    def test_ne_operator(self, pgvector_engine, sample_docs):
        """Test $ne operator."""
        results = pgvector_engine.search(where=Q(category__ne="tech"), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") != "tech" for doc in results)

    def test_gt_operator(self, pgvector_engine, sample_docs):
        """Test $gt operator for numeric comparison with casting."""
        results = pgvector_engine.search(where=Q(year__gt=2023), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("year") > 2023 for doc in results)

    def test_gte_operator(self, pgvector_engine, sample_docs):
        """Test $gte operator with numeric casting."""
        results = pgvector_engine.search(where=Q(score__gte=90), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") >= 90 for doc in results)

    def test_lt_operator(self, pgvector_engine, sample_docs):
        """Test $lt operator."""
        results = pgvector_engine.search(where=Q(year__lt=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("year") < 2024 for doc in results)

    def test_lte_operator(self, pgvector_engine, sample_docs):
        """Test $lte operator."""
        results = pgvector_engine.search(where=Q(score__lte=85), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") <= 85 for doc in results)

    def test_in_operator(self, pgvector_engine, sample_docs):
        """Test $in operator."""
        results = pgvector_engine.search(where=Q(category__in=["tech", "food"]), limit=10)
        assert len(results) == 4
        assert all(doc.metadata.get("category") in ["tech", "food"] for doc in results)

    def test_nin_operator(self, pgvector_engine, sample_docs):
        """Test $nin operator."""
        results = pgvector_engine.search(where=Q(category__nin=["tech", "food"]), limit=10)
        assert len(results) == 1
        assert results[0].metadata.get("category") == "travel"

    def test_and_combination(self, pgvector_engine, sample_docs):
        """Test combining filters with AND."""
        results = pgvector_engine.search(where=Q(category="tech") & Q(year__gte=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    def test_or_combination(self, pgvector_engine, sample_docs):
        """Test combining filters with OR."""
        results = pgvector_engine.search(where=Q(category="food") | Q(category="travel"), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") in ["food", "travel"] for doc in results)

    def test_complex_combination(self, pgvector_engine, sample_docs):
        """Test complex AND/OR combinations."""
        results = pgvector_engine.search(where=(Q(category="tech") & Q(year=2024)) | Q(category="travel"), limit=10)
        assert len(results) == 3  # 2 tech docs from 2024 + 1 travel doc

    def test_metadata_only_search(self, pgvector_engine, sample_docs):
        """Test metadata-only search (no vector, no text query)."""
        # PgVector supports metadata-only search via JSONB filters
        results = pgvector_engine.db.search(vector=None, where=Q(category="tech") & Q(score__gte=90), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("score") >= 90 for doc in results)

    def test_universal_dict_format(self, pgvector_engine, sample_docs):
        """Test using universal dict format instead of Q objects."""
        where_dict = {"category": {"$eq": "tech"}, "year": {"$gte": 2024}}
        results = pgvector_engine.search(where=where_dict, limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    def test_nested_metadata_jsonb(self, pgvector_engine):
        """Test nested metadata queries with JSONB #>> operator.

        PgVector uses JSONB, supports nested path queries like metadata #>> '{info,lang}'.
        """
        # Create doc with nested metadata
        _ = pgvector_engine.create(
            {"id": "nested1", "text": "Test nested metadata", "metadata": {"info": {"lang": "en", "tier": "gold"}}}
        )

        # Query nested field using __ syntax (compiles to JSONB path)
        results = pgvector_engine.search(where=Q(info__lang="en"), limit=10)
        assert len(results) >= 1
        found = any(d.id == "nested1" and d.metadata.get("info", {}).get("lang") == "en" for d in results)
        assert found

        # Test deeper nesting
        _ = pgvector_engine.create(
            {"id": "nested2", "text": "Deep nested test", "metadata": {"data": {"user": {"name": "Alice"}}}}
        )

        results = pgvector_engine.search(where=Q(data__user__name="Alice"), limit=10)
        assert len(results) >= 1
        found = any(d.id == "nested2" for d in results)
        assert found

        # Cleanup
        pgvector_engine.delete(["nested1", "nested2"])

    def test_range_query(self, pgvector_engine, sample_docs):
        """Test range queries (between values) with numeric casting."""
        results = pgvector_engine.search(where=Q(score__gte=80) & Q(score__lte=90), limit=10)
        assert len(results) == 2
        assert all(80 <= doc.metadata.get("score") <= 90 for doc in results)

    def test_numeric_casting(self, pgvector_engine, sample_docs):
        """Test that numeric comparisons use ::numeric casting."""
        where_q = Q(score__gt=85) & Q(year__gte=2023)
        compiled = pgvector_engine.db.where_compiler.to_where(where_q)

        # Should contain ::numeric casting for numeric comparisons
        assert "::numeric" in compiled
        assert "metadata" in compiled  # Uses JSONB metadata field

        # Verify it works end-to-end
        results = pgvector_engine.search(where=where_q, limit=10)
        assert len(results) == 3

    def test_sql_where_clause_generation(self, pgvector_engine, sample_docs):
        """Test that Q objects compile to proper SQL WHERE clauses."""
        where_q = Q(category="tech") & Q(score__gte=90)
        compiled = pgvector_engine.db.where_compiler.to_where(where_q)

        # Should be a SQL WHERE clause string
        assert isinstance(compiled, str)
        assert "AND" in compiled
        # JSONB operators
        assert "->>" in compiled or "#>>" in compiled

        # Verify it works end-to-end
        results = pgvector_engine.search(where=where_q, limit=10)
        assert len(results) == 2

    def test_jsonb_path_operator(self, pgvector_engine):
        """Test JSONB path operator for nested fields."""
        _ = pgvector_engine.create(
            {"id": "jsonb1", "text": "JSONB path test", "metadata": {"config": {"enabled": True, "level": 5}}}
        )

        # Query with nested path - should use #>> operator
        where_q = Q(config__level__gte=3)
        compiled = pgvector_engine.db.where_compiler.to_where(where_q)

        # Should use JSONB path operator for nested access
        assert "#>>" in compiled

        results = pgvector_engine.search(where=where_q, limit=10)
        assert len(results) >= 1
        found = any(d.id == "jsonb1" for d in results)
        assert found

        # Cleanup
        pgvector_engine.delete("jsonb1")
