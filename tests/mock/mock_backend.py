from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set

from crossvector.engine import VectorEngine
from crossvector.querydsl.q import Q
from crossvector.schema import VectorDocument


class InMemoryAdapter:
    """Simple in-memory adapter to test Query DSL without external backends.

    - Stores documents in a list
    - Supports create, delete, search with Q/dict filters
    - Similarity: cosine similarity on `vector`
    - Supports metadata-only search (no vector provided)
    """

    name = "inmemory"
    supports_metadata_only = True

    def __init__(self) -> None:
        self._docs: Dict[str, VectorDocument] = {}

    # Minimal API used by VectorEngine
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

        # Filter by where
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
                # leaf conditions
                return all(eval_condition(k, (v if isinstance(v, dict) else {"$eq": v})) for k, v in w.items())

            return eval_where(where) if where else True

        if where:
            # Accept Q or dict
            if isinstance(where, Q):
                where = where.to_dict()
            items = [d for d in items if match(d)]

        # Similarity sort if vector provided
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

        # Offset/limit
        start = offset
        end = start + (limit if limit is not None else len(items))
        return items[start:end]


def build_mock_engine() -> VectorEngine:
    # Deterministic embedding adapter stub: no external calls
    class FixedEmbedding:
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

    adapter = InMemoryAdapter()
    embedding = FixedEmbedding()
    return VectorEngine(db=adapter, embedding=embedding)


def seed_mock_docs(engine: VectorEngine) -> List[VectorDocument]:
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
    return docs


# Pytest fixtures
try:
    import pytest
except Exception:
    pytest = None

if pytest:

    @pytest.fixture(scope="module")
    def mock_engine():
        eng = build_mock_engine()
        seed_mock_docs(eng)
        return eng

    @pytest.fixture(scope="module")
    def sample_docs(mock_engine):
        return [
            mock_engine.get("doc1"),
            mock_engine.get("doc2"),
            mock_engine.get("doc3"),
            mock_engine.get("doc4"),
            mock_engine.get("doc5"),
        ]
