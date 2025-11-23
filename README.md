# CrossVector

## Cross-platform Vector Database Engine

A flexible, production-ready vector database engine with pluggable adapters for
multiple vector databases (AstraDB, ChromaDB, Milvus, PGVector) and embedding
providers (OpenAI, Gemini, and more).

Simplify your vector search infrastructure with a single, unified API across all
major vector databases.

## Features

- **Pluggable Architecture**: Easy adapter pattern for both databases and embeddings
- **Multiple Vector Databases**: AstraDB, ChromaDB, Milvus, PGVector
- **Multiple Embedding Providers**: OpenAI (Gemini coming soon)
- **Install Only What You Need**: Optional dependencies per adapter
- **Type-Safe**: Full Pydantic validation
- **Consistent API**: Same interface across all adapters

## Supported Vector Databases

| Database | Status | Features |
| ---------- | -------- | ---------- |
| **AstraDB** | ✅ Production | Cloud-native Cassandra, lazy initialization |
| **ChromaDB** | ✅ Production | Cloud/HTTP/Local modes, auto-fallback |
| **Milvus** | ✅ Production | Auto-indexing, schema validation |
| **PGVector** | ✅ Production | PostgreSQL extension, JSONB metadata |

## Supported Embedding Providers

| Provider | Status | Models |
| ---------- | -------- | -------- |
| **OpenAI** | ✅ Production | text-embedding-3-small, 3-large, ada-002 |
| **Gemini** | ✅ Production | text-embedding-004, gemini-embedding-001 |

## Installation

### Minimal (core only)

```bash
pip install crossvector
```

### With specific adapters

```bash
# AstraDB + OpenAI
pip install crossvector[astradb,openai]

# ChromaDB + OpenAI
pip install crossvector[chromadb,openai]

# All databases + OpenAI
pip install crossvector[all-dbs,openai]

# Everything
pip install crossvector[all]
```

## Quick Start

```python
from crossvector import VectorEngine, Document, UpsertRequest, SearchRequest
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.astradb import AstraDBAdapter

# Initialize engine
engine = VectorEngine(
    embedding_adapter=OpenAIEmbeddingAdapter(model_name="text-embedding-3-small"),
    db_adapter=AstraDBAdapter(),
    collection_name="my_documents"
)

# Upsert documents
docs = [
    Document(id="doc1", text="The quick brown fox", metadata={"category": "animals"}),
    Document(id="doc2", text="Artificial intelligence", metadata={"category": "tech"}),
]
result = engine.upsert(UpsertRequest(documents=docs))
print(f"Inserted {result['count']} documents")

# Search
results = engine.search(SearchRequest(query="AI and ML", limit=5))
for doc in results:
    print(f"Score: {doc.get('$similarity', 'N/A')}, Text: {doc.get('text')}")

# Get document by ID
doc = engine.get("doc1")

# Count documents
count = engine.count()

# Delete documents
engine.delete_one("doc1")
engine.delete_many(["doc2", "doc3"])
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# OpenAI (for embeddings)
OPENAI_API_KEY=sk-...

# AstraDB
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_API_ENDPOINT=https://...
ASTRA_DB_COLLECTION_NAME=my_collection

# ChromaDB Cloud
CHROMA_API_KEY=...
CHROMA_CLOUD_TENANT=...
CHROMA_CLOUD_DATABASE=...

# Milvus
MILVUS_API_ENDPOINT=https://...
MILVUS_USER=...
MILVUS_PASSWORD=...

# PGVector
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DBNAME=vectordb
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=...

# Vector metric (cosine, dot_product, euclidean)
VECTOR_METRIC=cosine
```

## Database-Specific Examples

### AstraDB

```python
from crossvector.dbs.astradb import AstraDBAdapter

adapter = AstraDBAdapter()
adapter.initialize(
    collection_name="my_collection",
    embedding_dimension=1536,
    metric="cosine"
)
```

### ChromaDB

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

### Milvus

```python
from crossvector.dbs.milvus import MilvusDBAdapter

adapter = MilvusDBAdapter()
adapter.initialize(
    collection_name="my_collection",
    embedding_dimension=1536,
    metric="cosine"
)
```

### PGVector

```python
from crossvector.dbs.pgvector import PGVectorAdapter

adapter = PGVectorAdapter()
adapter.initialize(
    table_name="my_vectors",
    embedding_dimension=1536,
    metric="cosine"
)
```

## Custom Adapters

### Create Custom Database Adapter

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

### Create Custom Embedding Adapter

```python
from crossvector.abc import EmbeddingAdapter
from typing import List

class MyCustomEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Initialize your client

    @property
    def embedding_dimension(self) -> int:
        return 768  # Your model's dimension

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Your implementation
        pass
```

## Document Format

All adapters expect documents in this standard format:

```python
{
    "_id": "unique-doc-id",              # Document ID (string)
    "$vector": [0.1, 0.2, ...],         # Embedding vector (List[float])
    "text": "original text content",     # Original text
    "any_field": "value",                # Additional metadata fields
    "another_field": 123,
}
```

## Development

```bash
# Clone repository
git clone https://github.com/thewebscraping/crossvector.git
cd crossvector

# Install with dev dependencies
pip install -e ".[all,dev]"

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .

# Setup pre-commit hooks
pre-commit install
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific adapter tests
pytest tests/test_gemini_embeddings.py
pytest tests/test_openai_embeddings.py
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- [x] Gemini embedding adapter
- [ ] Qdrant adapter (not supported yet)
- [ ] Pinecone adapter (not supported yet)
- [ ] Weaviate adapter (not supported yet)
- [ ] Async support
- [ ] Batch operations optimization
- [ ] Advanced filtering
- [ ] Hybrid search (vector + keyword)
- [ ] Rerank support (planned)
- [ ] Additional embedding providers (e.g., Cohere, Mistral, Ollama)

## Support

For issues and questions:

- GitHub Issues: <https://github.com/thewebscraping/crossvector/issues>
- Email: <thetwofarm@gmail.com>
