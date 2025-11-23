# Database Adapters

## AstraDB

```python
from crossvector.dbs.astradb import AstraDBAdapter

adapter = AstraDBAdapter()
adapter.initialize(
    collection_name="my_collection",
    embedding_dimension=1536,
    metric="cosine"
)
```

## ChromaDB

```python
from crossvector.dbs.chroma import ChromaDBAdapter

# Local mode
adapter = ChromaDBAdapter()

# Cloud mode (auto-detected from env vars)
# CHROMA_API_KEY, CHROMA_CLOUD_TENANT, CHROMA_CLOUD_DATABASE
adapter = ChromaDBAdapter()

adapter.initialize(
    collection_name="my_collection",
    embedding_dimension=1536
)
```

## Milvus

```python
from crossvector.dbs.milvus import MilvusDBAdapter

adapter = MilvusDBAdapter()
adapter.initialize(
    collection_name="my_collection",
    embedding_dimension=1536,
    metric="cosine"
)
```

## PGVector

```python
from crossvector.dbs.pgvector import PGVectorAdapter

adapter = PGVectorAdapter()
adapter.initialize(
    table_name="my_vectors",
    embedding_dimension=1536,
    metric="cosine"
)
```

## Creating a Custom Database Adapter

```python
from crossvector.abc import VectorDBAdapter
from typing import Any, Dict, List, Set

class MyCustomDBAdapter(VectorDBAdapter):
    def initialize(self, collection_name: str, embedding_dimension: int, metric: str = "cosine"):
        # Your implementation
        pass

    def get_collection(self, collection_name: str, embedding_dimension: int, metric: str = "cosine"):
        # Your implementation
        pass

    def upsert(self, documents: List[Dict[str, Any]]):
        # Your implementation
        pass

    def search(self, vector: List[float], limit: int, fields: Set[str]) -> List[Dict[str, Any]]:
        # Your implementation
        pass

    def get(self, id: str) -> Dict[str, Any] | None:
        # Your implementation
        pass

    def count(self) -> int:
        # Your implementation
        pass

    def delete_one(self, id: str) -> int:
        # Your implementation
        pass

    def delete_many(self, ids: List[str]) -> int:
        # Your implementation
        pass
```
