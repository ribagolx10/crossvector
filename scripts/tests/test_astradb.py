"""Integration tests for AstraDB with Query DSL and VectorEngine.

This suite targets real AstraDB. Configure using TEST_ env vars first,
with static defaults for collection naming.
"""

import pytest
from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.dbs.astradb import AstraDBAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.exceptions import MissingConfigError
from crossvector.querydsl import Q

load_dotenv()


@pytest.fixture(scope="module")
def astradb_engine():
    """Create VectorEngine with AstraDB adapter for testing."""
    try:
        embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        db = AstraDBAdapter()
        engine = VectorEngine(
            embedding=embedding,
            db=db,
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
            embedding=embedding,
            db=db,
            collection_name="test_crossvector",
            store_text=True,
        )

        yield engine

        # Cleanup after tests
        try:
            engine.drop_collection("test_crossvector")
        except Exception:
            pass

    except MissingConfigError as e:
        pytest.skip(f"AstraDB credentials not configured: {e}")


@pytest.fixture(scope="module")
def sample_docs(astradb_engine):
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

    created = astradb_engine.bulk_create(docs, ignore_conflicts=True, update_conflicts=True)
    return created


class TestAstraDB:
    """AstraDB integration tests (search and filters)."""

    def test_eq_operator(self, astradb_engine, sample_docs):
        """Test $eq operator with Q object."""
        results = astradb_engine.search(where=Q(category="tech"), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("category") == "tech" for doc in results)

    def test_ne_operator(self, astradb_engine, sample_docs):
        """Test $ne operator."""
        results = astradb_engine.search(where=Q(category__ne="tech"), limit=10)
        assert len(results) >= 2
        assert all(doc.metadata.get("category") != "tech" for doc in results)

    def test_gt_operator(self, astradb_engine, sample_docs):
        """Test $gt operator for numeric comparison."""
        results = astradb_engine.search(where=Q(year__gt=2023), limit=10)
        assert len(results) == 3
        assert all(doc.metadata.get("year") > 2023 for doc in results)

    def test_gte_operator(self, astradb_engine, sample_docs):
        """Test $gte operator."""
        results = astradb_engine.search(where=Q(score__gte=90), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") >= 90 for doc in results)

    def test_lt_operator(self, astradb_engine, sample_docs):
        """Test $lt operator."""
        results = astradb_engine.search(where=Q(year__lt=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("year") < 2024 for doc in results)

    def test_lte_operator(self, astradb_engine, sample_docs):
        """Test $lte operator."""
        results = astradb_engine.search(where=Q(score__lte=85), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("score") <= 85 for doc in results)

    def test_in_operator(self, astradb_engine, sample_docs):
        """Test $in operator."""
        results = astradb_engine.search(where=Q(category__in=["tech", "food"]), limit=10)
        assert len(results) == 4
        assert all(doc.metadata.get("category") in ["tech", "food"] for doc in results)

    def test_nin_operator(self, astradb_engine, sample_docs):
        """Test $nin operator."""
        results = astradb_engine.search(where=Q(category__nin=["tech", "food"]), limit=10)
        assert len(results) >= 1
        assert any(r.metadata.get("category") == "travel" for r in results)

    def test_and_combination(self, astradb_engine, sample_docs):
        """Test combining filters with AND."""
        results = astradb_engine.search(where=Q(category="tech") & Q(year__gte=2024), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    def test_or_combination(self, astradb_engine, sample_docs):
        """Test combining filters with OR."""
        results = astradb_engine.search(where=Q(category="food") | Q(category="travel"), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") in ["food", "travel"] for doc in results)

    def test_complex_combination(self, astradb_engine, sample_docs):
        """Test complex AND/OR combinations."""
        results = astradb_engine.search(where=(Q(category="tech") & Q(year=2024)) | Q(category="travel"), limit=10)
        assert len(results) == 3  # 2 tech docs from 2024 + 1 travel doc

    def test_metadata_only_search(self, astradb_engine, sample_docs):
        """Test metadata-only search (no vector, no text query)."""
        # AstraDB supports metadata-only search
        results = astradb_engine.db.search(vector=None, where=Q(category="tech") & Q(score__gte=90), limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("score") >= 90 for doc in results)

    def test_universal_dict_format(self, astradb_engine, sample_docs):
        """Test using universal dict format instead of Q objects."""
        where_dict = {"category": {"$eq": "tech"}, "year": {"$gte": 2024}}
        results = astradb_engine.search(where=where_dict, limit=10)
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "tech" and doc.metadata.get("year") >= 2024 for doc in results)

    def test_nested_metadata(self, astradb_engine):
        """Test nested metadata queries with dot notation."""
        # Create doc with nested metadata
        _ = astradb_engine.create(
            {"id": "nested1", "text": "Test nested metadata", "metadata": {"info": {"lang": "en", "tier": "gold"}}}
        )

        # Query nested field using __ syntax
        results = astradb_engine.search(where=Q(info__lang="en"), limit=10)
        assert len(results) >= 1
        found = any(d.id == "nested1" for d in results)
        assert found

        # Cleanup
        astradb_engine.delete("nested1")

    def test_range_query(self, astradb_engine, sample_docs):
        """Test range queries (between values)."""
        results = astradb_engine.search(where=Q(score__gte=80) & Q(score__lte=90), limit=10)
        assert len(results) == 2
        assert all(80 <= doc.metadata.get("score") <= 90 for doc in results)

    def test_crud_create_update_delete(self, astradb_engine):
        """Basic CRUD: create, update, get_or_create, update_or_create, delete."""
        # Create
        doc = astradb_engine.create(text="CRUD doc", metadata={"owner": "tester", "tier": "bronze"})
        assert doc.id

        # Update
        astradb_engine.update({"id": doc.id}, text="CRUD doc updated", metadata={"tier": "silver"})
        fetched = astradb_engine.get(doc.id)
        assert fetched.text == "CRUD doc updated"

        # get_or_create existing
        got, created = astradb_engine.get_or_create({"id": doc.id}, defaults={"text": "should not change"})
        assert not created and got.id == doc.id

        # update_or_create new
        uoc, created2 = astradb_engine.update_or_create(
            {"id": "crud-new-1"}, create_defaults={"text": "uoc created", "metadata": {"owner": "tester"}}
        )
        assert created2 and uoc.id == "crud-new-1"

        # Delete
        deleted = astradb_engine.delete([doc.id, uoc.id])
        assert deleted >= 0
