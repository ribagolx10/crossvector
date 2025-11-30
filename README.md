# CrossVector

[![Beta Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/thewebscraping/crossvector)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A unified Python library for vector database operations with pluggable backends and embedding providers.**

CrossVector provides a consistent, high-level API across multiple vector databases (AstraDB, ChromaDB, Milvus, PgVector) and embedding providers (OpenAI, Gemini), allowing you to switch between backends without rewriting your application code.

## ⚠️ Beta Status

> **WARNING**: CrossVector is currently in **BETA**. Do not use in production until version 1.0 release.
>
> - API may change without notice
> - Database schemas may evolve
> - Features are still being tested and refined
>
> **Recommended for:**
>
> - ✅ Prototyping and experimentation
> - ✅ Development and testing environments
> - ✅ Learning vector databases
>
> **Not recommended for:**
>
> - ❌ Production applications
> - ❌ Mission-critical systems

---

## Features

### 🔌 Pluggable Architecture

- **4 Vector Databases**: AstraDB, ChromaDB, Milvus, PgVector
- **2 Embedding Providers**: OpenAI, Gemini
- Switch backends without code changes

### 🎯 Unified API

- Consistent interface across all adapters
- Django-style `get`, `get_or_create`, `update_or_create` semantics
- Flexible document input formats: `str`, `dict`, or `VectorDocument`

### 🔍 Advanced Querying

- **Query DSL**: Type-safe filter composition with `Q` objects
- **Universal operators**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
- **Nested metadata**: Dot-notation paths for hierarchical data
- **Metadata-only search**: Query without vector similarity (where supported)

### 🚀 Performance Optimized

- Automatic batch embedding generation
- Bulk operations: `bulk_create`, `bulk_update`, `upsert`
- Configurable batch sizes and conflict resolution

### 🛡️ Type-Safe & Validated

- Full Pydantic validation
- Structured exceptions with detailed context
- Centralized logging with configurable levels

### ⚙️ Flexible Configuration

- Environment variable support via `.env`
- Multiple primary key strategies: UUID, hash-based, int64, custom
- Optional text storage to optimize space

---

## Installation

### Core Package (Minimal)

```bash
pip install crossvector
```

### With Specific Backends

```bash
# AstraDB + OpenAI
pip install crossvector[astradb,openai]

# ChromaDB + OpenAI
pip install crossvector[chromadb,openai]

# Milvus + Gemini
pip install crossvector[milvus,gemini]

# PgVector + OpenAI
pip install crossvector[pgvector,openai]
```

### All Backends and Providers

```bash
# Everything
pip install crossvector[all]

# All databases only
pip install crossvector[all-dbs,openai]

# All embeddings only
pip install crossvector[astradb,all-embeddings]
```

---

## Quick Start

### Basic Usage

```python
from crossvector import VectorEngine
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Initialize engine
engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(model_name="text-embedding-3-small"),
    db=PgVectorAdapter(),
    collection_name="my_documents",
    store_text=True
)

# Create documents (flexible input formats)
doc1 = engine.create(text="Python is a programming language")
doc2 = engine.create({"text": "Artificial intelligence", "metadata": {"category": "tech"}})
doc3 = engine.create(text="Machine learning basics", metadata={"level": "beginner"})

print(f"Created documents: {doc1.id}, {doc2.id}, {doc3.id}")

# Search by text (automatic embedding generation)
results = engine.search("programming languages", limit=5)
for doc in results:
    print(f"[{doc.metadata.get('score', 0):.3f}] {doc.text}")

# Search by vector (skip embedding step)
vector = engine.embedding.get_embeddings(["my query"])[0]
results = engine.search(vector, limit=3)

# Get document by ID
doc = engine.get(doc1.id)
print(f"Retrieved: {doc.text}")

# Count documents
total = engine.count()
print(f"Total documents: {total}")

# Delete documents
engine.delete(doc1.id)
engine.delete([doc2.id, doc3.id])  # Batch delete
```

### Flexible Input Formats

CrossVector accepts multiple document input formats for maximum convenience:

```python
# String input (text only)
doc1 = engine.create("Simple text document")

# Dict input with metadata
doc2 = engine.create({
    "text": "Document with metadata",
    "metadata": {"source": "api", "author": "user123"}
})

# Dict input with metadata as kwargs
doc3 = engine.create(
    text="Document with inline metadata",
    source="web",
    category="blog"
)

# VectorDocument instance
from crossvector import VectorDocument
doc4 = engine.create(
    VectorDocument(
        id="custom-id",
        text="Full control document",
        metadata={"priority": "high"}
    )
)

# Provide pre-computed vector (skip embedding)
doc5 = engine.create(
    text="Document with vector",
    vector=[0.1, 0.2, ...],  # 1536-dim for OpenAI
    metadata={"source": "external"}
)
```

### Django-Style Operations

```python
# Get or create pattern
doc, created = engine.get_or_create(
    text="My document",
    metadata={"topic": "AI"}
)
if created:
    print("Created new document")
else:
    print("Document already exists")

# Update or create pattern
doc, created = engine.update_or_create(
    {"id": "doc-123"},
    text="Updated content",
    defaults={"metadata": {"updated": True}}
)

# Get with metadata filters
doc = engine.get(source="api", status="active")  # Must return exactly one

# Bulk operations
docs = [
    {"text": "Doc 1", "metadata": {"idx": 1}},
    {"text": "Doc 2", "metadata": {"idx": 2}},
    {"text": "Doc 3", "metadata": {"idx": 3}},
]
created_docs = engine.bulk_create(docs, batch_size=100)

# Upsert (insert or update)
docs = engine.upsert([
    {"id": "doc-1", "text": "Updated doc 1"},
    {"id": "doc-2", "text": "New doc 2"},
])
```

---

## Advanced Querying

### Query DSL with Q Objects

CrossVector provides a powerful Query DSL for composing complex filters:

```python
from crossvector.querydsl.q import Q

# Simple equality
results = engine.search("AI", where=Q(category="tech"))

# Comparison operators
results = engine.search(
    "articles",
    where=Q(score__gte=0.8) & Q(views__lt=1000)
)

# Range queries
results = engine.search(
    "products",
    where=Q(price__gte=100) & Q(price__lte=500)
)

# IN / NOT IN
results = engine.search(
    "users",
    where=Q(role__in=["admin", "moderator"]) & Q(status__ne="banned")
)

# Boolean combinations
high_quality = Q(rating__gte=4.5) & Q(reviews__gte=10)
featured = Q(featured__eq=True)
results = engine.search("items", where=high_quality | featured)

# Negation
results = engine.search("posts", where=~Q(status="archived"))

# Nested metadata (dot notation)
results = engine.search(
    "documents",
    where=Q(info__lang__eq="en") & Q(info__tier__eq="gold")
)
```

### Universal Filter Format

You can also use dict-based filters with universal operators:

```python
# Equality and comparison
results = engine.search("query", where={
    "category": {"$eq": "tech"},
    "score": {"$gt": 0.8},
    "views": {"$lte": 1000}
})

# IN / NOT IN
results = engine.search("query", where={
    "status": {"$in": ["active", "pending"]},
    "priority": {"$nin": ["low"]}
})

# Nested paths
results = engine.search("query", where={
    "user.role": {"$eq": "admin"},
    "user.verified": {"$eq": True}
})

# Multiple conditions (implicit AND)
results = engine.search("query", where={
    "category": {"$eq": "blog"},
    "published": {"$eq": True},
    "views": {"$gte": 100}
})
```

### Metadata-Only Search

Search by metadata filters without vector similarity:

```python
# Find all documents with specific metadata
docs = engine.search(
    query=None,  # No vector search
    where={"status": {"$eq": "published"}},
    limit=50
)

# Complex metadata queries
docs = engine.search(
    query=None,
    where=Q(category="tech") & Q(featured=True) & Q(score__gte=0.9),
    limit=100
)
```

### Supported Operators

All backends support these universal operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal to | `{"age": {"$eq": 25}}` or `Q(age=25)` |
| `$ne` | Not equal to | `{"status": {"$ne": "inactive"}}` or `Q(status__ne="inactive")` |
| `$gt` | Greater than | `{"score": {"$gt": 0.8}}` or `Q(score__gt=0.8)` |
| `$gte` | Greater than or equal | `{"price": {"$gte": 100}}` or `Q(price__gte=100)` |
| `$lt` | Less than | `{"age": {"$lt": 18}}` or `Q(age__lt=18)` |
| `$lte` | Less than or equal | `{"priority": {"$lte": 5}}` or `Q(priority__lte=5)` |
| `$in` | In array | `{"role": {"$in": ["admin", "mod"]}}` or `Q(role__in=["admin"])` |
| `$nin` | Not in array | `{"status": {"$nin": ["banned"]}}` or `Q(status__nin=["banned"])` |

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Gemini
GOOGLE_API_KEY=AI...

# AstraDB
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_API_ENDPOINT=https://...
ASTRA_DB_COLLECTION_NAME=vectors

# ChromaDB (Cloud)
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...

# ChromaDB (Self-hosted)
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Milvus
MILVUS_API_ENDPOINT=https://...
MILVUS_API_KEY=...

# PgVector
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DBNAME=vector_db
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=postgres

# Vector settings
VECTOR_STORE_TEXT=true
VECTOR_METRIC=cosine
VECTOR_SEARCH_LIMIT=10
PRIMARY_KEY_MODE=uuid
LOG_LEVEL=INFO
```

### Primary Key Strategies

CrossVector supports multiple primary key generation strategies:

```python
from crossvector.settings import settings

# UUID (default) - random UUID
settings.PRIMARY_KEY_MODE = "uuid"

# Hash text - deterministic from text content
settings.PRIMARY_KEY_MODE = "hash_text"

# Hash vector - deterministic from vector values
settings.PRIMARY_KEY_MODE = "hash_vector"

# Sequential int64
settings.PRIMARY_KEY_MODE = "int64"

# Auto - hash text if available, else hash vector, else UUID
settings.PRIMARY_KEY_MODE = "auto"

# Custom factory function
settings.PRIMARY_KEY_FACTORY = "mymodule.generate_custom_id"
```

---

## Backend-Specific Features

### Backend Capabilities

Different backends have varying feature support:

| Feature | AstraDB | ChromaDB | Milvus | PgVector |
|---------|---------|----------|--------|----------|
| Vector Search | ✅ | ✅ | ✅ | ✅ |
| Metadata-Only Search | ✅ | ✅ | ❌ | ✅ |
| Nested Metadata | ✅ | ✅* | ❌ | ✅ |
| Numeric Comparisons | ✅ | ✅ | ✅ | ✅ |
| Text Storage | ✅ | ✅ | ✅ | ✅ |

*ChromaDB supports nested metadata via dot-notation when metadata is flattened.

### AstraDB

```python
from crossvector.dbs.astradb import AstraDBAdapter

db = AstraDBAdapter()
engine = VectorEngine(embedding=embedding, db=db)

# Features:
# - Serverless, auto-scaling
# - Native JSON metadata support
# - Nested field queries with dot notation
# - Metadata-only search
```

### ChromaDB

```python
from crossvector.dbs.chroma import ChromaAdapter

# Cloud mode
db = ChromaAdapter()  # Uses CHROMA_API_KEY from env

# Self-hosted mode
db = ChromaAdapter()  # Uses CHROMA_HOST/PORT from env

# Local persistence mode
db = ChromaAdapter()  # Uses CHROMA_PERSIST_DIR from env

engine = VectorEngine(embedding=embedding, db=db)

# Features:
# - Multiple deployment modes (cloud/HTTP/local)
# - Automatic client fallback
# - Flattened metadata with dot-notation support
```

### Milvus

```python
from crossvector.dbs.milvus import MilvusAdapter

db = MilvusAdapter()
engine = VectorEngine(embedding=embedding, db=db)

# Features:
# - High performance at scale
# - Automatic index creation
# - Boolean expression filters
# - Requires vector for all searches (no metadata-only)
```

### PgVector

```python
from crossvector.dbs.pgvector import PgVectorAdapter

db = PgVectorAdapter()
engine = VectorEngine(embedding=embedding, db=db)

# Features:
# - PostgreSQL extension
# - JSONB metadata storage
# - Nested field support with #>> operator
# - Automatic numeric type casting
# - Metadata-only search
# - Auto-creates database if missing
```

---

## Embedding Providers

### OpenAI

```python
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Default model (text-embedding-3-small, 1536 dims)
embedding = OpenAIEmbeddingAdapter()

# Larger model (text-embedding-3-large, 3072 dims)
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-large")

# Legacy model (text-embedding-ada-002, 1536 dims)
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-ada-002")
```

### Gemini

```python
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

# Default model (gemini-embedding-001)
embedding = GeminiEmbeddingAdapter()

# With custom dimensions (768, 1536, 3072)
embedding = GeminiEmbeddingAdapter(
    model_name="gemini-embedding-001",
    dim=1536
)

# With task type
embedding = GeminiEmbeddingAdapter(
    task_type="retrieval_document"  # or "retrieval_query", "semantic_similarity"
)
```

---

## Error Handling

CrossVector provides structured exceptions with detailed context:

```python
from crossvector.exceptions import (
    DoesNotExist,
    MultipleObjectsReturned,
    DocumentExistsError,
    MissingFieldError,
    InvalidFieldError,
    CollectionNotFoundError,
    MissingConfigError,
)

# Catch specific errors
try:
    doc = engine.get(id="nonexistent")
except DoesNotExist as e:
    print(f"Document not found: {e.details}")

# Multiple results when expecting one
try:
    doc = engine.get(status="active")  # Multiple matches
except MultipleObjectsReturned as e:
    print(f"Multiple documents matched: {e.details}")

# Missing configuration
try:
    db = PgVectorAdapter()
except MissingConfigError as e:
    print(f"Missing config: {e.details['config_key']}")
    print(f"Hint: {e.details['hint']}")

# Invalid field or operator
try:
    results = engine.search("query", where={"field": {"$regex": "pattern"}})
except InvalidFieldError as e:
    print(f"Unsupported operator: {e.message}")
```

---

## Logging

Configure logging via environment variable:

```bash
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

Or programmatically:

```python
from crossvector.settings import settings
settings.LOG_LEVEL = "DEBUG"

# Logs include:
# - Engine initialization
# - Embedding generation
# - Database operations
# - Query compilation
# - Error details
```

---

## Testing

Run tests with pytest:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_engine.py

# With coverage
pytest tests/ --cov=crossvector --cov-report=html

# Integration tests (requires real backends)
python scripts/backend.py --backend pgvector --embedding-provider openai
python scripts/backend.py --backend astradb --embedding-provider openai
python scripts/backend.py --backend milvus --embedding-provider openai
python scripts/backend.py --backend chroma --embedding-provider openai
```

---

## Examples

### Full CRUD Example

```python
from crossvector import VectorEngine
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.astradb import AstraDBAdapter
from crossvector.querydsl.q import Q

# Initialize
engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(),
    db=AstraDBAdapter(),
    collection_name="articles"
)

# Create
article1 = engine.create(
    text="Introduction to Python programming",
    metadata={"category": "tutorial", "level": "beginner", "views": 1500}
)

article2 = engine.create(
    text="Advanced machine learning techniques",
    metadata={"category": "tutorial", "level": "advanced", "views": 3200}
)

article3 = engine.create(
    text="Best practices for API design",
    metadata={"category": "guide", "level": "intermediate", "views": 2100}
)

# Search with filters
results = engine.search(
    "machine learning tutorials",
    where=Q(category="tutorial") & Q(level__in=["beginner", "intermediate"]),
    limit=5
)

# Update
article1.metadata["views"] = 2000
engine.update(article1)

# Batch update
updates = [
    {"id": article2.id, "metadata": {"featured": True}},
    {"id": article3.id, "metadata": {"featured": True}},
]
engine.bulk_update(updates)

# Get or create
doc, created = engine.get_or_create(
    text="Python best practices",
    metadata={"category": "guide", "level": "intermediate"}
)

# Delete
engine.delete(article1.id)
engine.delete([article2.id, article3.id])

# Count
total = engine.count()
print(f"Total articles: {total}")
```

### Switching Backends

```python
# Same code works across all backends - just swap the adapter

# PgVector
from crossvector.dbs.pgvector import PgVectorAdapter
engine = VectorEngine(embedding=embedding, db=PgVectorAdapter())

# ChromaDB
from crossvector.dbs.chroma import ChromaAdapter
engine = VectorEngine(embedding=embedding, db=ChromaAdapter())

# Milvus
from crossvector.dbs.milvus import MilvusAdapter
engine = VectorEngine(embedding=embedding, db=MilvusAdapter())

# AstraDB
from crossvector.dbs.astradb import AstraDBAdapter
engine = VectorEngine(embedding=embedding, db=AstraDBAdapter())

# All operations remain the same!
results = engine.search("query", limit=10)
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        VectorEngine                          │
│  (Unified API, automatic embedding, flexible input)         │
└───────────────────┬──────────────────┬──────────────────────┘
                    │                  │
        ┌───────────▼──────────┐  ┌───▼──────────────────┐
        │  EmbeddingAdapter    │  │   VectorDBAdapter    │
        │  (OpenAI, Gemini)    │  │  (Astra, Chroma...)  │
        └──────────────────────┘  └──────────┬───────────┘
                                              │
                                   ┌──────────▼──────────┐
                                   │  WhereCompiler      │
                                   │  (Query DSL → SQL)  │
                                   └─────────────────────┘
```

### Query Processing Flow

```
User Input (Q or dict)
    ↓
Normalize to Universal Dict Format
    ↓
Backend-Specific Compiler
    ↓
Native Filter (SQL, Milvus expr, Chroma dict)
    ↓
Database Query
    ↓
VectorDocument Results
```

---

## Roadmap

- [ ] **v1.0 Stable Release**
  - API freeze and backwards compatibility guarantee
  - Production-ready documentation
  - Performance benchmarks

- [ ] **Additional Backends**
  - Pinecone
  - Weaviate
  - Qdrant
  - MongoDB
  - Elasticsearch
  - OpenSearch

- [ ] **Enhanced Features**
  - Hybrid search (vector + keyword)
  - Reranking support (Cohere, Jina)
  - Async/await support
  - Streaming search results
  - Pagination helpers

- [ ] **Developer Experience**
  - CLI tool for management
  - Migration utilities
  - Schema validation and linting
  - Interactive query builder

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](docs/contributing.md) for detailed guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and migration guides.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/thewebscraping/crossvector/issues)
- **Documentation**: [GitHub Wiki](https://github.com/thewebscraping/crossvector/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/thewebscraping/crossvector/discussions)

---

## Acknowledgments

- Built with [Pydantic](https://docs.pydantic.dev/) for validation
- Inspired by Django ORM's elegant API design
- Thanks to all vector database and embedding providers for their excellent SDKs

---

**Made with ❤️ by the [Two Farm](https://www.linkedin.com/in/thetwofarm/)**
