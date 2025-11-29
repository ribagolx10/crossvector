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

See the main README or `crossvector.abc.VectorDBAdapter` for the complete interface.

```python
from crossvector.abc import VectorDBAdapter
from crossvector.schema import VectorDocument
from typing import Any, Dict, List, Set, Optional, Union, Sequence, Tuple

class MyCustomDBAdapter(VectorDBAdapter):
    """Custom vector database adapter implementation."""
    
    use_dollar_vector: bool = False  # Set to True if your DB uses '$vector'
    
    def initialize(
        self, 
        collection_name: str, 
        embedding_dimension: int, 
        metric: str = "cosine",
        **kwargs: Any
    ) -> None:
        """Initialize database and ensure collection is ready."""
        pass

    def search(
        self,
        vector: List[float],
        limit: int,
        offset: int = 0,
        where: Dict[str, Any] | None = None,
        fields: Set[str] | None = None,
    ) -> List[VectorDocument]:
        """Perform vector similarity search."""
        # Should return List[VectorDocument]
        pass

    def get(self, *args, **kwargs) -> VectorDocument:
        """Retrieve a single document by primary key."""
        # Should return VectorDocument instance
        pass

    def upsert(
        self, 
        documents: List[VectorDocument], 
        batch_size: int = None
    ) -> List[VectorDocument]:
        """Insert new documents or update existing ones."""
        # Should return List[VectorDocument]
        pass

    def delete(self, ids: Union[str, Sequence[str]]) -> int:
        """Delete document(s) by primary key."""
        # Should return count of deleted documents
        pass

    def count(self) -> int:
        """Count total documents in current collection."""
        pass

    # ... and more methods (see VectorDBAdapter ABC)
```
