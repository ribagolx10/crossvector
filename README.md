# CrossVector

[![Beta Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/thewebscraping/crossvector)
[![Not Production Ready](https://img.shields.io/badge/production-not%20ready-red)](https://github.com/thewebscraping/crossvector#%EF%B8%8F-beta-status---not-production-ready)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Cross-platform Vector Database Engine

A flexible vector database engine **currently in beta** with pluggable adapters for
multiple vector databases (AstraDB, ChromaDB, Milvus, PGVector) and embedding
providers (OpenAI, Gemini, and more).

Simplify your vector search infrastructure with a single, unified API across all
major vector databases.

## ⚠️ Beta Status - Not Production Ready

> **WARNING**: CrossVector is currently in **BETA** and under active development.
>
> **DO NOT USE IN PRODUCTION** until a stable 1.0 release.
>
> **Risks:**
>
> - API may change without notice
> - Database schema may evolve, requiring migrations
> - Features may be added, removed, or modified
> - Bugs and edge cases are still being discovered
> - Performance optimizations are ongoing
>
> **Use Cases:**
>
> - ✅ Experimentation and prototyping
> - ✅ Development and testing
> - ✅ Learning and exploration
> - ❌ Production applications
> - ❌ Mission-critical systems
> - ❌ Customer-facing services
>
> **Recommendations:**
>
> - Pin to specific version: `crossvector==0.x.x`
> - Monitor the [CHANGELOG](CHANGELOG.md) for breaking changes
> - Test thoroughly before upgrading
> - Join discussions in [GitHub Issues](https://github.com/thewebscraping/crossvector/issues)
> - Wait for 1.0 stable release for production use

## Features

- **Pluggable Architecture**: Easy adapter pattern for both databases and embeddings
- **Multiple Vector Databases**: AstraDB, ChromaDB, Milvus, PGVector
- **Multiple Embedding Providers**: OpenAI, Gemini
- **Smart Document Handling**: Auto-generated IDs (SHA256), optional text storage
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
    collection_name="my_documents",
    store_text=True  # Optional: Set to False to not store original text
)

# Upsert documents
docs = [
    Document(text="The quick brown fox", metadata={"category": "animals"}), # ID auto-generated
    Document(id="doc2", text="Artificial intelligence", metadata={"category": "tech"}),
]
result = engine.upsert(UpsertRequest(documents=docs))
print(f"Inserted {result['count']} documents")

# Search
results = engine.search(SearchRequest(query="AI and ML", limit=5))
for doc in results:
    print(f"Score: {doc.get('score', 'N/A')}, Text: {doc.get('text')}")

# Get document by ID
doc = engine.get("doc2")

# Count documents
count = engine.count()

# Delete documents
engine.delete_one("doc2")
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# OpenAI (for embeddings)
OPENAI_API_KEY=sk-...

# Gemini (for embeddings)
GOOGLE_API_KEY=...

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
# Store original text in database (true/false)
VECTOR_STORE_TEXT=true
```

## Database-Specific Examples

### AstraDB

```python
from crossvector.dbs.astradb import AstraDBAdapter

adapter = AstraDBAdapter()
adapter.initialize(
    collection_name="my_collection",
    embedding_dimension=1536,
    metric="cosine",
    store_text=True  # Optional: Set to False to save space
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
    embedding_dimension=1536,
    store_text=True  # Optional
)
```

### Milvus

```python
from crossvector.dbs.milvus import MilvusDBAdapter

adapter = MilvusDBAdapter()
adapter.initialize(
    collection_name="my_collection",
    embedding_dimension=1536,
    metric="cosine",
    store_text=True  # Optional
)
```

### PGVector

```python
from crossvector.dbs.pgvector import PGVectorAdapter

adapter = PGVectorAdapter()
adapter.initialize(
    table_name="my_vectors",
    embedding_dimension=1536,
    metric="cosine",
    store_text=True  # Optional
)
```

## Custom Adapters

### Create Custom Database Adapter

```python
from crossvector.abc import VectorDBAdapter
from typing import Any, Dict, List, Set

class MyCustomDBAdapter(VectorDBAdapter):
    def initialize(self, collection_name: str, embedding_dimension: int, metric: str = "cosine", store_text: bool = True):
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

## JSON Format Specification

CrossVector uses a standardized JSON format across all vector databases. Here's the complete specification:

### 1. User Level (Creating Documents)

When you create documents, use the `Document` class:

```python
from crossvector import Document

# Option 1: With explicit ID
doc = Document(
    id="my-custom-id",
    text="The content of my document",
    metadata={
        "category": "example",
        "source": "manual",
        "tags": ["important", "review"]
    }
)

# Option 2: Auto-generated ID (SHA256 hash of text)
doc = Document(
    text="Another document without ID",
    metadata={"category": "auto"}
)
# doc.id will be a 64-character SHA256 hash

# Timestamps are automatically generated
print(doc.created_timestamp)  # Unix timestamp: 1732349789.123456
print(doc.updated_timestamp)  # Unix timestamp: 1732349789.123456

# Convert to datetime if needed
from datetime import datetime, timezone
created_dt = datetime.fromtimestamp(doc.created_timestamp, tz=timezone.utc)
print(created_dt)  # 2024-11-23 11:16:29.123456+00:00

# You can safely use your own created_at/updated_at in metadata!
doc_with_article_timestamps = Document(
    text="My article content",
    metadata={
        "title": "My Article",
        "created_at": "2024-01-15T10:00:00Z",  # ✅ Your article's timestamp (ISO 8601)
        "updated_at": "2024-11-20T15:30:00Z",  # ✅ Your article's timestamp (ISO 8601)
        "author": "John Doe"
    }
)
# Both timestamps coexist:
# - doc.created_timestamp: CrossVector internal tracking (Unix timestamp float)
# - metadata["created_at"]: Your article's timestamp (any format you want)
```

**Auto-Generated Fields:**

- `id`: SHA256 hash of text if not provided
- `created_timestamp`: Unix timestamp (float) when document was created
- `updated_timestamp`: Unix timestamp (float), updated on every modification

**✅ Why Float/Unix Timestamp?**

- **Compact**: `1732349789.123456` vs `"2024-11-23T11:16:29.123456+00:00"`
- **Efficient**: Easy to compare and sort (`<`, `>`, `==`)
- **Universal**: Works across all programming languages
- **Smaller storage**: Numbers are more efficient than strings

**✅ No Conflicts:**
CrossVector uses `created_timestamp` and `updated_timestamp` (float), so you can freely use `created_at`, `updated_at`, or any other timestamp fields in your metadata with any format (ISO 8601, RFC 3339, custom, etc.).

### 2. Engine Level (Internal Format)

When `VectorEngine.upsert()` processes documents, it converts them to this standardized format before passing to database adapters:

```python
{
    "_id": "unique-doc-id",           # Document identifier (string)
    "vector": [0.1, 0.2, ...],        # Embedding vector (List[float])
    "text": "original text",           # Original text content (if store_text=True)
    # Metadata fields (flattened at root level)
    "category": "example",
    "source": "manual",
    "tags": ["important", "review"],
    "created_timestamp": 1732349789.123456,  # CrossVector timestamp (float)
    "updated_timestamp": 1732349789.123456,  # CrossVector timestamp (float)
    # User's own timestamps (if any) - any format is fine
    "created_at": "2024-01-15T10:00:00Z",  # Your article timestamp (ISO 8601)
    "updated_at": "2024-11-20T15:30:00Z",  # Your article timestamp (ISO 8601)
    "published_date": "2024-01-15"         # Or any other format
}
```

**Key Points:**

- Field `_id`: Document unique identifier
- Field `vector`: Embedding vector (replaces `$vector` in older versions)
- Field `text`: Stored separately from metadata
- Fields `created_timestamp` and `updated_timestamp`: Automatic CrossVector tracking (Unix timestamp float)
- User metadata (including user's own timestamps in any format) are preserved
- Metadata fields are stored at root level (not nested)

### 3. Storage Level (Database-Specific)

Each database adapter translates the engine format to its native storage format:

#### **PGVector**

```sql
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE,
    vector vector(1536),
    text TEXT,                    -- Separate column
    metadata JSONB                -- All metadata fields
);
```

Storage format:

```python
{
    "doc_id": "unique-doc-id",
    "vector": [0.1, 0.2, ...],
    "text": "original text",
    "metadata": {                  # Nested in JSONB
        "category": "example",
        "source": "manual",
        "tags": ["important"]
    }
}
```

#### **Milvus**

```python
schema = {
    "doc_id": VARCHAR(255),        # Primary key
    "vector": FLOAT_VECTOR(1536),
    "text": VARCHAR(65535),        # Separate field (if store_text=True)
    "metadata": JSON               # All metadata fields
}
```

Storage format:

```python
{
    "doc_id": "unique-doc-id",
    "vector": [0.1, 0.2, ...],
    "text": "original text",
    "metadata": {                  # Nested in JSON field
        "category": "example",
        "source": "manual"
    }
}
```

#### **ChromaDB**

ChromaDB uses separate arrays for each field:

```python
{
    "ids": ["unique-doc-id"],
    "embeddings": [[0.1, 0.2, ...]],
    "documents": ["original text"],      # Separate array (if store_text=True)
    "metadatas": [{                      # Flattened metadata (no nesting)
        "category": "example",
        "source": "manual",
        "tags.0": "important",           # Nested lists/dicts are flattened
        "tags.1": "review"
    }]
}
```

**Note**: ChromaDB doesn't support nested metadata, so we auto-flatten it.

#### **AstraDB**

AstraDB stores everything at the document root level:

```python
{
    "_id": "unique-doc-id",
    "$vector": [0.1, 0.2, ...],
    "text": "original text",          # At root level (if store_text=True)
    "category": "example",            # Metadata at root level
    "source": "manual",
    "tags": ["important", "review"]
}
```

### 4. Search Results Format

When you call `search()` or `get()`, results are returned in a unified format:

```python
# Search results
results = engine.search(SearchRequest(query="example", limit=5))

# Each result:
{
    "id": "unique-doc-id",           # Document ID
    "score": 0.92,                   # Similarity score (lower = more similar for some metrics)
    "text": "original text",         # If requested in fields
    "metadata": {                    # Original metadata structure
        "category": "example",
        "source": "manual",
        "tags": ["important"]
    }
}
```

### 5. Example: Complete Flow

```python
from crossvector import VectorEngine, Document, UpsertRequest, SearchRequest
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.pgvector import PGVectorAdapter

engine = VectorEngine(
    embedding_adapter=OpenAIEmbeddingAdapter(),
    db_adapter=PGVectorAdapter(),
    collection_name="docs",
    store_text=True
)

# 1. Create documents (User Level)
docs = [
    Document(
        text="Python is a programming language",
        metadata={"lang": "en", "category": "tech"}
    )
]

# 2. Upsert (Engine Level conversion happens automatically)
engine.upsert(UpsertRequest(documents=docs))

# 3. Search (Results in unified format)
results = engine.search(SearchRequest(
    query="programming languages",
    limit=5,
    fields={"text", "metadata"}  # Specify what to return
))

# 4. Use results
for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']}")
    print(f"Text: {result.get('text', 'N/A')}")
    print(f"Metadata: {result.get('metadata', {})}")
```

### Summary Table

| Level | Format | Key Fields | Notes |
|-------|--------|-----------|-------|
| **User** | `Document` object | `id`, `text`, `metadata` | Pydantic validation, auto-generated ID |
| **Engine** | Python dict | `_id`, `vector`, `text`, metadata fields | Standardized across all DBs |
| **PGVector** | SQL row | `doc_id`, `vector`, `text`, `metadata` (JSONB) | Text in separate column |
| **Milvus** | JSON document | `doc_id`, `vector`, `text`, `metadata` (JSON) | Text in VARCHAR field |
| **ChromaDB** | Arrays | `ids`, `embeddings`, `documents`, `metadatas` | Flattened metadata |
| **AstraDB** | JSON document | `_id`, `$vector`, `text`, metadata at root | Everything at root level |
| **Search Results** | Python dict | `id`, `score`, `text`, `metadata` | Unified format |

**Note**: The `text` field is optional and controlled by the `store_text` parameter. If `store_text=False`, the text will not be stored in any database.

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
