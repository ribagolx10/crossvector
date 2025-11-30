"""Pytest configuration and fixtures for vector store tests."""

import math
import os
from typing import Any, Dict, List, Optional, Set

import pytest
from dotenv import load_dotenv

from crossvector.engine import VectorEngine
from crossvector.querydsl.q import Q
from crossvector.schema import VectorDocument

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Vector databases enable semantic search.",
        "Python is a great programming language.",
        "Machine learning requires large amounts of data.",
    ]


@pytest.fixture(scope="session")
def sample_documents(sample_texts):
    """Sample document data for testing (texts, metadatas, pks)."""
    return {
        "texts": sample_texts,
        "metadatas": [{"index": i, "category": "test"} for i in range(len(sample_texts))],
        "pks": [f"doc_{i}" for i in range(len(sample_texts))],
    }


# In-memory mock adapter for Query DSL testing
class InMemoryAdapter:
    """Simple in-memory adapter to test Query DSL without external backends."""

    name = "inmemory"
    supports_metadata_only = True

    def __init__(self) -> None:
        self._docs: Dict[str, VectorDocument] = {}

    def create(self, docs: List[VectorDocument] | VectorDocument | Dict[str, Any]) -> List[VectorDocument]:
        normalized: List[VectorDocument] = []
        if isinstance(docs, list):
            normalized = docs
        elif isinstance(docs, VectorDocument):
            normalized = [docs]
        elif isinstance(docs, dict):
            normalized = [VectorDocument.from_kwargs(**docs)]
        else:
            raise TypeError("Unsupported document type for create")
        for d in normalized:
            self._docs[d.id] = d
        return normalized

    def delete(self, ids: List[str]) -> int:
        count = 0
        for _id in ids:
            if _id in self._docs:
                del self._docs[_id]
                count += 1
        return count

    def get(self, _id: str) -> Optional[VectorDocument]:
        return self._docs.get(_id)

    def count(self) -> int:
        return len(self._docs)

    def search(
        self,
        vector: Optional[List[float]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        where: Optional[Dict[str, Any]] = None,
        fields: Optional[Set[str]] = None,
    ) -> List[VectorDocument]:
        items = list(self._docs.values())

        def match(doc: VectorDocument) -> bool:
            meta = doc.metadata or {}

            def eval_condition(key: str, cond: Dict[str, Any]) -> bool:
                val = meta
                parts = key.split("__") if "__" in key else key.split(".")
                for part in parts:
                    if isinstance(val, dict):
                        val = val.get(part)
                    else:
                        val = None
                        break
                if "$eq" in cond:
                    return val == cond["$eq"]
                if "$ne" in cond:
                    return val != cond["$ne"]
                if "$gt" in cond:
                    return val is not None and val > cond["$gt"]
                if "$gte" in cond:
                    return val is not None and val >= cond["$gte"]
                if "$lt" in cond:
                    return val is not None and val < cond["$lt"]
                if "$lte" in cond:
                    return val is not None and val <= cond["$lte"]
                if "$in" in cond:
                    return val in cond["$in"]
                if "$nin" in cond:
                    return val not in cond["$nin"]
                return True

            def eval_where(w: Dict[str, Any]) -> bool:
                if "$and" in w:
                    return all(eval_where(x) for x in w["$and"])
                if "$or" in w:
                    return any(eval_where(x) for x in w["$or"])
                return all(eval_condition(k, (v if isinstance(v, dict) else {"$eq": v})) for k, v in w.items())

            return eval_where(where) if where else True

        if where:
            if isinstance(where, Q):
                where = where.to_dict()
            items = [d for d in items if match(d)]

        def cosine(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        if vector is not None:
            items.sort(key=lambda d: cosine(vector, d.vector or []), reverse=True)

        start = offset
        end = start + (limit if limit is not None else len(items))
        return items[start:end]


class FixedEmbedding:
    """Deterministic embedding for testing without external API calls."""

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            h = abs(hash(t))
            vec = [
                ((h >> 0) & 0xFF) / 255.0,
                ((h >> 8) & 0xFF) / 255.0,
                ((h >> 16) & 0xFF) / 255.0,
                ((h >> 24) & 0xFF) / 255.0,
            ]
            out.append(vec)
        return out


@pytest.fixture(scope="module")
def mock_engine():
    """Build VectorEngine with in-memory adapter and fixed embeddings."""
    adapter = InMemoryAdapter()
    embedding = FixedEmbedding()
    engine = VectorEngine(db=adapter, embedding=embedding)

    # Seed with test documents
    docs = [
        VectorDocument(
            id="doc1",
            text="AI in 2024",
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={"category": "tech", "year": 2024, "score": 91},
        ),
        VectorDocument(
            id="doc2",
            text="Cooking tips",
            vector=[0.0, 0.1, 0.0, 0.2],
            metadata={"category": "food", "year": 2023, "score": 85},
        ),
        VectorDocument(
            id="doc3",
            text="Travel guide",
            vector=[0.2, 0.0, 0.1, 0.0],
            metadata={"category": "travel", "year": 2022, "score": 78},
        ),
        VectorDocument(
            id="doc4",
            text="Tech gadgets",
            vector=[0.3, 0.2, 0.1, 0.0],
            metadata={"category": "tech", "year": 2024, "score": 88},
        ),
        VectorDocument(
            id="doc5",
            text="Healthy recipes",
            vector=[0.05, 0.05, 0.05, 0.05],
            metadata={"category": "food", "year": 2024, "score": 92},
        ),
    ]
    engine.db.create(docs)
    return engine


@pytest.fixture(scope="module")
def sample_docs(mock_engine):
    """Return seeded test documents from mock engine."""
    return [
        mock_engine.get("doc1"),
        mock_engine.get("doc2"),
        mock_engine.get("doc3"),
        mock_engine.get("doc4"),
        mock_engine.get("doc5"),
    ]


@pytest.fixture
def mock_embeddings(sample_texts):
    """Mock embeddings for testing without API calls."""
    import random

    dimension = 1536  # OpenAI text-embedding-3-small dimension

    def generate_mock_embedding():
        # Generate random normalized vector
        vec = [random.gauss(0, 1) for _ in range(dimension)]
        magnitude = sum(x**2 for x in vec) ** 0.5
        return [x / magnitude for x in vec]

    return [generate_mock_embedding() for _ in sample_texts]


@pytest.fixture
def openai_api_key():
    """OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture
def astradb_credentials():
    """AstraDB credentials from environment."""
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    if not token or not endpoint:
        pytest.skip("AstraDB credentials not set")
    return {"token": token, "endpoint": endpoint}


@pytest.fixture
def chromadb_cloud_credentials():
    """ChromaDB Cloud credentials from environment."""
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")
    if not api_key:
        pytest.skip("ChromaDB credentials not set")
    return {"api_key": api_key, "tenant": tenant, "database": database}


@pytest.fixture
def milvus_credentials():
    """Milvus credentials from environment."""
    endpoint = os.getenv("MILVUS_API_ENDPOINT")
    if not endpoint:
        pytest.skip("Milvus credentials not set")
    return {
        "endpoint": endpoint,
        "user": os.getenv("MILVUS_USER"),
        "password": os.getenv("MILVUS_PASSWORD"),
    }


@pytest.fixture
def pgvector_credentials():
    """PGVector credentials from environment."""
    host = os.getenv("PGVECTOR_HOST", "localhost")
    return {
        "host": host,
        "port": os.getenv("PGVECTOR_PORT", "5432"),
        "dbname": os.getenv("PGVECTOR_DBNAME", "postgres"),
        "user": os.getenv("PGVECTOR_USER", "postgres"),
        "password": os.getenv("PGVECTOR_PASSWORD", "postgres"),
    }


# Cleanup fixtures
@pytest.fixture
def cleanup_collection():
    """Fixture to clean up test collections after tests."""
    collections_to_cleanup = []

    def register_cleanup(adapter, collection_name):
        collections_to_cleanup.append((adapter, collection_name))

    yield register_cleanup

    # Cleanup after test
    for adapter, collection_name in collections_to_cleanup:
        try:
            if hasattr(adapter, "db"):
                # AstraDB
                if collection_name in adapter.db.list_collection_names():
                    adapter.db.drop_collection(collection_name)
            elif hasattr(adapter, "client"):
                # ChromaDB or Milvus
                if hasattr(adapter.client, "delete_collection"):
                    adapter.client.delete_collection(collection_name)
        except Exception as e:
            print(f"Cleanup warning: {e}")
