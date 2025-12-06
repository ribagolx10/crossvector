# CrossVector

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/thewebscraping/crossvector)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-365%20passing-brightgreen)](https://github.com/thewebscraping/crossvector)

**A unified Python library for vector database operations with pluggable backends and embedding providers.**

CrossVector provides a consistent, high-level API across multiple vector databases (AstraDB, ChromaDB, Milvus, PgVector) and embedding providers (OpenAI, Gemini), allowing you to switch between backends without rewriting your application code.

## 🎯 Recommended Backends

Based on our comprehensive benchmarking, we recommend:

### **For Production:**

- **🥇 ChromaDB Cloud** - Best for cloud deployments
  - Hosted solution with excellent performance
  - Easy setup and management
  - Built-in scaling and backups
  - Good for: SaaS applications, MVPs, rapid prototyping

- **🥈 PgVector** - Best for self-hosted/on-premise
  - Excellent performance (6-10 docs/sec bulk insert)
  - Very fast metadata queries (<1ms)
  - PostgreSQL reliability and ecosystem
  - Good for: Enterprise, existing PostgreSQL infrastructure, cost-sensitive deployments

### **Also Supported:**

- **AstraDB** - DataStax managed Cassandra with vector support
- **Milvus** - Purpose-built vector database for large-scale deployments

See our [benchmarking guide](docs/benchmarking.md) for detailed performance comparisons.

---

## Features

### 🔌 Pluggable Architecture

- **4 Vector Databases**: AstraDB, ChromaDB, Milvus, PgVector
- **2 Embedding Providers**: OpenAI, Gemini
- Switch backends without code changes
- Lazy initialization pattern for optimal resource usage

### 🎯 Unified API

- Consistent interface across all adapters
- Django-style `get`, `get_or_create`, `update_or_create` semantics
- Flexible document input formats: `str`, `dict`, or `VectorDocument`
- Standardized error handling with contextual exceptions

### 🔍 Advanced Querying

- **Query DSL**: Type-safe filter composition with `Q` objects
- **Universal operators**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
- **Nested metadata**: Dot-notation paths for hierarchical data
- **Metadata-only search**: Query without vector similarity (where supported)

### 🚀 Performance Optimized

- Automatic batch embedding generation
- Bulk operations: `bulk_create`, `bulk_update`, `upsert`
- Configurable batch sizes and conflict resolution
- Lazy client initialization for faster startup

### 🛡️ Type-Safe & Validated

- Full Pydantic v2 validation
- Structured exceptions with detailed context
- Centralized logging with configurable levels
- Explicit configuration validation with helpful error messages

### ⚙️ Flexible Configuration

- Environment variable support via `.env`
- Multiple primary key strategies: UUID, hash-based, int64, custom
- Optional text storage to optimize space
- Strict config validation prevents silent failures

---

## Installation

### Core Package (Minimal)

```bash
pip install crossvector
```

### With Specific Backends

```bash
# Recommended: PgVector + Gemini (free tier)
pip install crossvector[pgvector,gemini]

# Alternative: ChromaDB + Gemini (cloud or local)
pip install crossvector[chromadb,gemini]

# With OpenAI (requires paid API key)
pip install crossvector[pgvector,openai]
pip install crossvector[chromadb,openai]

# Milvus + Gemini
pip install crossvector[milvus,gemini]

# AstraDB + OpenAI
pip install crossvector[astradb,openai]
```

### All Backends

```bash
# Install everything
pip install crossvector[all]

# All databases only
pip install crossvector[all-dbs,openai]

# All embeddings only
pip install crossvector[astradb,all-embeddings]
```

---

## Quick Start

> 💡 **Recommended**: Use `GeminiEmbeddingAdapter` for most use cases - free tier, faster search (1.5x), smaller vectors (768 vs 1536 dims). See [benchmarks](benchmark.md) for details.

### Basic Usage

```python
from crossvector import VectorEngine
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Initialize engine with Gemini (recommended: free tier, fast performance)
engine = VectorEngine(
    embedding=GeminiEmbeddingAdapter(),  # Free tier, 1536-dim vectors
    db=PgVectorAdapter(),
    collection_name="my_documents",
    store_text=True
)

# Alternative: OpenAI (requires paid API key, 1536-dim vectors)
# from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
# embedding = OpenAIEmbeddingAdapter()

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
engine.delete(doc2.id, doc3.id)  # Batch delete
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
# OpenAI Embedding Provider
OPENAI_API_KEY=sk-...

# Gemini Embedding Provider
GEMINI_API_KEY=AI...

# Optional: Override default embedding model (adapter-specific)
VECTOR_EMBEDDING_MODEL=text-embedding-3-small

# AstraDB Backend
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_API_ENDPOINT=https://...apps.astra.datastax.com

# ChromaDB Cloud Backend
CHROMA_API_KEY=ck-...
CHROMA_TENANT=...
CHROMA_DATABASE=Test

# ChromaDB Self-hosted (HTTP)
CHROMA_HOST=localhost
CHROMA_PORT=8000

# ChromaDB Local (Persistent)
CHROMA_PERSIST_DIR=./chroma_data

# Note: Cannot set both CHROMA_HOST and CHROMA_PERSIST_DIR
# Choose one based on deployment mode

# Milvus Backend
MILVUS_API_ENDPOINT=https://...
MILVUS_API_KEY=...

# PgVector Backend
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=postgres

# Vector Configuration (applies to all backends)
VECTOR_COLLECTION_NAME=vector_db
VECTOR_STORE_TEXT=false
VECTOR_METRIC=cosine
VECTOR_DIM=1536
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
| Metadata-Only Search | ✅ | ✅ | ✅ | ✅ |
| Nested Metadata | ✅ | ✅ | ✅ | ✅ |
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

# Cloud mode (requires CHROMA_API_KEY)
db = ChromaAdapter()

# Self-hosted HTTP mode (requires CHROMA_HOST, must not set CHROMA_PERSIST_DIR)
db = ChromaAdapter()

# Local persistence mode (requires CHROMA_PERSIST_DIR, must not set CHROMA_HOST)
db = ChromaAdapter()

engine = VectorEngine(embedding=embedding, db=db)

# Features:
# - Multiple deployment modes (cloud/HTTP/local)
# - Strict config validation (prevents conflicting settings)
# - Explicit import pattern for better code clarity
# - Flattened metadata with dot-notation support
# - Lazy client initialization

# Important: Cannot set both CHROMA_HOST and CHROMA_PERSIST_DIR
# Choose one deployment mode explicitly to avoid errors
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

> 💡 **Recommended**: Start with **Gemini** for free tier and faster performance. See [benchmark comparison](benchmark.md).

### Gemini (Recommended)

```python
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

# Default model (gemini-embedding-001, 1536 dims)
embedding = GeminiEmbeddingAdapter()

# Explicit model specification
embedding = GeminiEmbeddingAdapter(model_name="models/text-embedding-004", dim=768)
```

**Why Choose Gemini:**
- ✅ **Free tier**: 1,500 requests/min (vs OpenAI paid only)
- ✅ **Faster search**: 234ms avg (1.5x faster than OpenAI)
- ✅ **Efficient**: 768 dims = 50% less storage than OpenAI
- ✅ **Quality**: Comparable accuracy to OpenAI

**Configuration:**
```bash
GEMINI_API_KEY=AI...  # Get free key at https://makersuite.google.com/app/apikey
```

**Supported Models:**
- `gemini-embedding-001` (768 dims, **recommended**)
- `models/text-embedding-004` (768 dims)

### OpenAI (Alternative)

```python
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Default model (text-embedding-3-small, 1536 dims)
embedding = OpenAIEmbeddingAdapter()

# Explicit model specification
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-large")
```

**When to Use OpenAI:**
- ✅ Need 1536 or 3072 dimensions
- ✅ Already have OpenAI API budget
- ✅ Prefer OpenAI ecosystem integration

**Configuration:**
```bash
OPENAI_API_KEY=sk-...  # Paid API key from https://platform.openai.com
```

**Supported Models:**
- `text-embedding-3-small` (1536 dims, default)
- `text-embedding-3-large` (3072 dims)
- `text-embedding-ada-002` (1536 dims, legacy)

- `gemini-embedding-001` (1536 dims, default)
- `text-embedding-005` (768 dims)
- `text-embedding-004` (768 dims, legacy)

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
### Real Environment Tests (Opt-in)

Integration tests that exercise real backends live under `scripts/tests/` to avoid running in GitHub Actions by default.

- Location: `scripts/tests/`
- Run manually when services/credentials are available

Static defaults used in tests:
- AstraDB collection: `test_crossvector`
- Chroma collection: `test_crossvector`
- Milvus collection: `test_crossvector`
- PgVector table: `test_crossvector`

Run examples:
```zsh
pytest scripts/tests -q
pytest scripts/tests/test_pgvector.py -q
```

Environment setup examples:
```zsh
# OpenAI (embeddings)
export OPENAI_API_KEY=sk-...
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# AstraDB
export ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
export ASTRA_DB_API_ENDPOINT=https://...apps.astra.datastax.com

# Chroma (local/cloud)
export CHROMA_HOST=api.trychroma.com
export CHROMA_API_KEY=ck-...
export CHROMA_TENANT=...
export CHROMA_DATABASE=Test

# Milvus
export MILVUS_API_ENDPOINT=http://localhost:19530
export MILVUS_API_TOKEN=...

# PgVector
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=5432
export VECTOR_COLLECTION_NAME=vectordb
export PGVECTOR_USER=postgres
export PGVECTOR_PASSWORD=postgres
```

Run tests with pytest:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_engine.py

# With coverage
pytest tests/ --cov=crossvector --cov-report=html

# Integration tests with real backends (requires credentials)
pytest scripts/tests/test_pgvector.py -v
pytest scripts/tests/test_astradb.py -v
pytest scripts/tests/test_milvus.py -v
pytest scripts/tests/test_chroma.py -v
```

---

## Benchmarking

CrossVector includes a comprehensive benchmarking tool to compare performance across different database backends and embedding providers.

### Quick Start

```bash
# Quick test with 10 documents (recommended first run)
python scripts/benchmark.py --num-docs 10

# Full benchmark with 1000 documents
python scripts/benchmark.py

# Test specific backends and embeddings
python scripts/benchmark.py --backends pgvector milvus --embedding-providers openai

# Custom output file
python scripts/benchmark.py --output results/my_benchmark.md
```

### What Gets Benchmarked

The benchmark tool measures performance across 7 key operations:

1. **Bulk Create** - Batch insertion with automatic embedding generation
2. **Individual Create** - Single document creation performance
3. **Vector Search** - Semantic similarity search with embeddings
4. **Metadata-Only Search** - Filtering without vector similarity
5. **Query DSL Operators** - Testing all 10 operators (eq, ne, gt, gte, lt, lte, in, nin, and, or)
6. **Update Operations** - Document update performance
7. **Delete Operations** - Batch deletion throughput

### Supported Backends

- **PgVector** - PostgreSQL with vector extension
- **AstraDB** - DataStax Astra vector database
- **Milvus** - Open-source vector database
- **ChromaDB** - Embedded vector database

### Supported Embeddings

- **OpenAI** - `text-embedding-3-small` (1536 dimensions)
- **Gemini** - `gemini-embedding-001` (1536 dimensions)

### Sample Results

```markdown
| Backend | Embedding | Model | Dim | Bulk Create | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|-------------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 2.68s | 515.47ms | 6.48ms | 1.76ms | ✅ |
| astradb | openai | text-embedding-3-small | 1536 | 32.56s | 1.09s | 875.63ms | 1.44s | ✅ |
| milvus | openai | text-embedding-3-small | 1536 | 21.24s | 1.04s | 551.36ms | 180.25ms | ✅ |
| chroma | openai | text-embedding-3-small | 1536 | 36.08s | 900.75ms | 2.51s | 521.35ms | ✅ |
| pgvector | gemini | models/gemini-embedding-001 | 1536 | 31.50s | 65.29ms | 6.14ms | 1.78ms | ✅ |
| astradb | gemini | models/gemini-embedding-001 | 1536 | 1m 2.65s | 882.48ms | 818.93ms | 1.44s | ✅ |
| milvus | gemini | models/gemini-embedding-001 | 1536 | 50.26s | 835.50ms | 572.62ms | 224.16ms | ✅ |
| chroma | gemini | models/gemini-embedding-001 | 1536 | 1m 3.39s | 628.08ms | 3.16s | 394.21ms | ✅ |
```

### Requirements

**Environment Variables:**

```bash
# Embedding providers (at least one required)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Database backends (optional, script will skip if not configured)
PGVECTOR_CONNECTION_STRING=postgresql://...
ASTRADB_API_ENDPOINT=https://...
ASTRADB_APPLICATION_TOKEN=AstraCS:...
MILVUS_API_ENDPOINT=https://...
MILVUS_API_TOKEN=...
```

### Recommended Workflow

```bash
# Step 1: Quick verification (1-2 minutes)
python scripts/benchmark.py --num-docs 1 --backends pgvector --embedding-providers openai

# Step 2: Fast comparison with 10 docs (5-10 minutes)
python scripts/benchmark.py --num-docs 10

# Step 3: Production benchmark with 1000 docs (30-60 minutes)
python scripts/benchmark.py --num-docs 1000 --output benchmark.md
```

### Output

Results are saved to `benchmark.md` (or custom path) with:
- Performance summary table comparing all combinations
- Detailed metrics for each backend + embedding pair
- Query DSL operator test results
- Timestamps and configuration details

**Example output:**
```
📄 Markdown report saved to: benchmark.md
```

See [benchmarking documentation](docs/benchmarking.md) for more details.

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
engine.delete(article2.id, article3.id)

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
│                        VectorEngine                         │
│  (Unified API, automatic embedding, flexible input)         │
└───────────────────┬──────────────────┬──────────────────────┘
                    │                  │
        ┌───────────▼──────────┐  ┌───-▼─────────────────┐
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

- [x] **v1.0 Stable Release**
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
