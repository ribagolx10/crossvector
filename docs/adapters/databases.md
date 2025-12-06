# Database Adapters

Backend-specific features, capabilities, and configuration for vector databases.

## Overview

CrossVector supports 4 vector database backends:

| Backend | Nested Metadata | Metadata-Only Search | Requires Vector | License |
|---------|----------------|----------------------|-----------------|---------|
| **AstraDB** | ✅ Full | ✅ Yes | ❌ No | Proprietary |
| **ChromaDB** | ❌ Flattened | ✅ Yes | ❌ No | Apache 2.0 |
| **Milvus** | ✅ Full | ❌ No | ✅ Yes | Apache 2.0 |
| **PgVector** | ✅ Full | ✅ Yes | ❌ No | PostgreSQL |

---

## AstraDB

DataStax Astra DB - Serverless Cassandra with vector search.

### Features

- ✅ **Full nested metadata** - Complete JSON document support
- ✅ **Metadata-only search** - Filter without vector similarity
- ✅ **Universal operators** - All 8 operators supported
- ✅ **Scalable** - Serverless auto-scaling
- ✅ **Managed** - Fully hosted service

### Installation

```bash
pip install crossvector[astradb]
```

### Configuration

**Environment Variables:**

```bash
ASTRA_DB_APPLICATION_TOKEN="AstraCS:xxx"
ASTRA_DB_API_ENDPOINT="https://xxx.apps.astra.datastax.com"
ASTRA_DB_KEYSPACE="default_keyspace"  # Optional
```

**Programmatic:**

```python
from crossvector.dbs.astradb import AstraDBAdapter

db = AstraDBAdapter(
    token="AstraCS:xxx",
    api_endpoint="https://xxx.apps.astra.datastax.com",
    keyspace="default_keyspace"
)
```

### Schema

AstraDB uses special field names:

- `_id` - Document primary key
- `$vector` - Embedding vector
- All other fields are metadata

**Example document:**

```json
{
  "_id": "doc-123",
  "$vector": [0.1, 0.2, ...],
  "text": "Document content",
  "category": "tech",
  "author": {
    "name": "John",
    "role": "admin"
  }
}
```

### Nested Metadata

Full JSON document support:

```python
from crossvector.querydsl.q import Q

# Deep nesting
results = engine.search(
    "query",
    where=Q(author__profile__verified=True) & Q(post__stats__views__gte=1000)
)
```

### Capabilities

```python
engine = VectorEngine(db=AstraDBAdapter(), embedding=...)

# Metadata-only search
results = engine.search(
    query=None,
    where=Q(status="published")
)

# All operators
results = engine.search(
    "query",
    where=(
        Q(category="tech") &
        Q(score__gte=0.8) &
        Q(tags__in=["python", "ai"]) &
        ~Q(archived=True)
    )
)
```

### Performance

- **Collection limits:** 10M+ documents per collection
- **Throughput:** High (serverless auto-scaling)
- **Latency:** ~10-50ms typical
- **Cost:** Pay-per-request pricing

### Best Practices

```python
# Use metadata-only for fast filtering
results = engine.search(query=None, where={"status": {"$eq": "active"}})

# Leverage nested metadata
metadata = {
    "user": {"id": "user123", "tier": "premium"},
    "content": {"type": "article", "category": "tech"}
}

# Batch operations for efficiency
engine.bulk_create(docs, batch_size=100)
```

---

## ChromaDB

Open-source embedding database with Python-first API.

### Features

- ⚠️ **Flattened metadata** - No nested object support (auto-flattened)
- ✅ **Metadata-only search** - Filter without vector similarity
- ✅ **Multiple deployment modes** - Cloud, HTTP, or local persistence
- ✅ **Strict config validation** - Prevents conflicting settings
- ✅ **Explicit imports** - Clear dependency management
- ✅ **Lazy initialization** - Optimal resource usage
- ✅ **Common operators** - All 8 operators supported
- ✅ **In-memory/persistent** - Multiple storage backends
- ✅ **Open source** - Apache 2.0 license

### Installation

```bash
# Local/in-memory
pip install crossvector[chroma]

# ChromaDB Cloud
pip install crossvector[chroma-cloud]
```

### Configuration

**Environment Variables:**

```bash
# ChromaDB Cloud (priority 1)
CHROMA_API_KEY="your-api-key"
CHROMA_TENANT="tenant-name"
CHROMA_DATABASE="database-name"

# Self-hosted HTTP (priority 2, requires no CHROMA_PERSIST_DIR)
CHROMA_HOST="localhost"
CHROMA_PORT="8000"

# Local persistence (priority 3, requires no CHROMA_HOST)
CHROMA_PERSIST_DIR="./chroma_data"
```

**Important:** Cannot set both `CHROMA_HOST` and `CHROMA_PERSIST_DIR`. Choose one deployment mode:
- **Cloud**: Set `CHROMA_API_KEY`
- **HTTP**: Set `CHROMA_HOST` (not `CHROMA_PERSIST_DIR`)
- **Local**: Set `CHROMA_PERSIST_DIR` (not `CHROMA_HOST`)

**Programmatic:**

```python
from crossvector.dbs.chroma import ChromaAdapter

# Cloud mode
db = ChromaAdapter()  # Uses CHROMA_API_KEY from env

# HTTP mode
db = ChromaAdapter()  # Uses CHROMA_HOST from env

# Local mode
db = ChromaAdapter()  # Uses CHROMA_PERSIST_DIR from env
```

**Configuration Validation:**

CrossVector enforces strict configuration validation:

```python
# ✅ Valid: Cloud only
CHROMA_API_KEY="..."

# ✅ Valid: HTTP only
CHROMA_HOST="localhost"

# ✅ Valid: Local only
CHROMA_PERSIST_DIR="./data"

# ❌ Invalid: Conflicting settings
CHROMA_HOST="localhost"
CHROMA_PERSIST_DIR="./data"
# Raises: MissingConfigError with helpful message
```

### Schema

ChromaDB automatically flattens nested metadata:

**Input:**

```python
metadata = {
    "user": {
        "name": "John",
        "role": "admin"
    }
}
```

**Stored as:**

```python
{
    "user.name": "John",
    "user.role": "admin"
}
```

### Nested Metadata

Nested queries work via automatic flattening:

```python
from crossvector.querydsl.q import Q

# This works (auto-flattened)
results = engine.search(
    "query",
    where=Q(user__role="admin")
)

# Compiled to: {"user.role": {"$eq": "admin"}}
```

**Limitation:** Cannot query nested structures as objects.

### Capabilities

```python
engine = VectorEngine(db=ChromaDBAdapter(), embedding=...)

# Metadata-only search
results = engine.search(
    query=None,
    where=Q(category="tech")
)

# All operators
results = engine.search(
    "query",
    where=(
        Q(category="tech") &
        Q(score__gte=0.8) &
        Q(status__in=["active", "pending"])
    )
)

# Wrapper requirement
# Multiple conditions automatically wrapped in $and
```

### Performance

- **Collection limits:** 100K+ documents recommended
- **Throughput:** High (in-memory)
- **Latency:** <10ms (in-memory), 20-50ms (persistent)
- **Cost:** Free (self-hosted)

### Best Practices

```python
# Use flat metadata structure for best compatibility
metadata = {
    "category": "tech",
    "author_name": "John",  # Flat instead of author.name
    "author_role": "admin"
}

# Choose deployment mode explicitly
# Option 1: Cloud (managed)
CHROMA_API_KEY="..."

# Option 2: Self-hosted HTTP server
CHROMA_HOST="localhost"

# Option 3: Local persistence (development)
CHROMA_PERSIST_DIR="./chroma_data"

# Don't mix deployment modes - causes MissingConfigError
# ❌ Don't do: CHROMA_HOST + CHROMA_PERSIST_DIR

# Batch operations for efficiency
engine.bulk_create(docs, batch_size=100)

# Leverage lazy initialization
db = ChromaAdapter()  # Client created only when first used
```

---

## Milvus

High-performance distributed vector database.

### Features

- ✅ **Full nested metadata** - JSON field support (via dynamic fields)
- ✅ **Metadata-only search** - Query without vector via `query()` method
- ✅ **Common operators** - All 8 operators supported
- ✅ **High performance** - Distributed architecture
- ✅ **Scalable** - Horizontal scaling
- ✅ **Lazy initialization** - Optimal resource usage

### Installation

```bash
pip install crossvector[milvus]
```

### Configuration

**Environment Variables:**

```bash
MILVUS_HOST="localhost"
MILVUS_PORT="19530"
MILVUS_USER="username"  # Optional
MILVUS_PASSWORD="password"  # Optional
MILVUS_DB_NAME="default"  # Optional
```

**Programmatic:**

```python
from crossvector.dbs.milvus import MilvusAdapter

db = MilvusAdapter(
    host="localhost",
    port=19530,
    user="username",
    password="password",
    db_name="default"
)
```

### Schema

Milvus uses boolean expression filters:

```python
# Query compiles to Milvus expression
Q(category="tech") & Q(score__gt=0.8)
# => '(category == "tech") and (score > 0.8)'

Q(status__in=["active", "pending"])
# => 'status in ["active", "pending"]'
```

### Vector Requirement

**All queries require vector input:**

```python
# ✅ Correct
results = engine.search("query text", where=Q(category="tech"))

# ❌ Error: Milvus requires vector
results = engine.search(query=None, where=Q(category="tech"))
```

**Workaround for metadata-only:**

```python
if not engine.supports_metadata_only:
    # Use empty string to generate minimal vector
    results = engine.search("", where=Q(status="active"))
```

### Nested Metadata

Full support via JSON field:

```python
from crossvector.querydsl.q import Q

# Nested queries
results = engine.search(
    "query",
    where=Q(user__role="admin") & Q(post__stats__views__gte=1000)
)

# Compiles to: '(user["role"] == "admin") and (post["stats"]["views"] >= 1000)'
```

### Capabilities

```python
engine = VectorEngine(db=MilvusAdapter(), embedding=...)

# Vector required
results = engine.search(
    "query text",  # Must provide query
    where=Q(category="tech") & Q(score__gte=0.8)
)

# All operators
results = engine.search(
    "query",
    where=(
        Q(status="published") &
        Q(priority__in=[1, 2, 3]) &
        Q(score__gt=0.5) &
        ~Q(archived=True)
    )
)
```

### Performance

- **Collection limits:** Billions of vectors
- **Throughput:** Very high (distributed)
- **Latency:** <10ms (optimized indexes)
- **Cost:** Free (self-hosted)

### Best Practices

```python
# Always provide query vector
results = engine.search("query", where=filters)

# Use nested metadata
metadata = {
    "user": {"id": 123, "tier": "premium"},
    "content": {"type": "video", "duration": 600}
}

# Index metadata fields for performance
# (Configure in Milvus collection schema)

# Batch operations
engine.bulk_create(docs, batch_size=1000)
```

---

## PgVector

PostgreSQL extension for vector similarity search.

### Features

- ✅ **Full nested metadata** - JSONB support with `#>>` operator
- ✅ **Metadata-only search** - Filter without vector similarity
- ✅ **Common operators** - All 8 operators with numeric casting
- ✅ **ACID transactions** - Full PostgreSQL guarantees
- ✅ **Mature ecosystem** - PostgreSQL tooling

### Installation

```bash
pip install crossvector[pgvector]
```

### PostgreSQL Setup

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table (handled automatically by adapter)
```

### Configuration

**Environment Variables:**

```bash
VECTOR_COLLECTION_NAME="vectordb"
PGVECTOR_HOST="localhost"
PGVECTOR_PORT="5432"
PGVECTOR_USER="postgres"
PGVECTOR_PASSWORD="password"
```

**Programmatic:**

```python
from crossvector.dbs.pgvector import PgVectorAdapter

db = PgVectorAdapter(
    dbname="vectordb",
    host="localhost",
    port=5432,
    user="postgres",
    password="password"
)
```

### Schema

PgVector stores metadata as JSONB:

```sql
CREATE TABLE vector_db (
    id TEXT PRIMARY KEY,
    vector vector(1536),
    text TEXT,
    metadata JSONB,
    created_timestamp DOUBLE PRECISION,
    updated_timestamp DOUBLE PRECISION
);
```

### JSONB Features

#### Nested Metadata Access

Uses `#>>` operator for nested paths:

```python
from crossvector.querydsl.q import Q

# Simple field
Q(category="tech")
# => "metadata->>'category' = 'tech'"

# Nested field
Q(user__role="admin")
# => "metadata #>> '{user,role}' = 'admin'"

# Deep nesting
Q(post__stats__views__gte=1000)
# => "(metadata #>> '{post,stats,views}')::numeric >= 1000"
```

#### Numeric Casting

Automatic casting for numeric comparisons:

```python
# Text stored as string, but compared numerically
Q(score__gt=0.8)
# => "(metadata->>'score')::numeric > 0.8"

Q(price__lte=100)
# => "(metadata->>'price')::numeric <= 100"
```

### Capabilities

```python
engine = VectorEngine(db=PgVectorAdapter(), embedding=...)

# Metadata-only search
results = engine.search(
    query=None,
    where=Q(status="published")
)

# Nested metadata
results = engine.search(
    "query",
    where=Q(user__profile__verified=True) & Q(user__stats__posts__gte=10)
)

# Numeric comparisons (auto-cast)
results = engine.search(
    "query",
    where=Q(score__gte=0.8) & Q(price__lt=100)
)

# All operators
results = engine.search(
    "query",
    where=(
        Q(category="tech") &
        Q(level__in=["beginner", "intermediate"]) &
        Q(rating__gte=4.0) &
        ~Q(archived=True)
    )
)
```

### Performance

- **Collection limits:** Millions of vectors (PostgreSQL limits)
- **Throughput:** High (ACID overhead)
- **Latency:** 10-50ms typical
- **Cost:** Free (self-hosted PostgreSQL)

### Indexing

```sql
-- Create IVFFlat index for faster vector search
CREATE INDEX ON vector_db
USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

-- Create GIN index for metadata queries
CREATE INDEX ON vector_db USING GIN (metadata);

-- Create index on specific nested field
CREATE INDEX ON vector_db ((metadata->>'category'));
```

### Best Practices

```python
# Use nested metadata with JSONB
metadata = {
    "user": {"id": 123, "role": "admin"},
    "content": {"type": "article", "tags": ["python", "ai"]}
}

# Numeric fields work with string or number
metadata = {"score": "0.95"}  # Auto-cast in comparisons
metadata = {"score": 0.95}    # Direct numeric

# Index frequently queried fields
# CREATE INDEX ON vector_db ((metadata->>'category'));

# Batch operations with transactions
engine.bulk_create(docs, batch_size=500)

# Use metadata-only for fast filtering
results = engine.search(query=None, where={"status": {"$eq": "active"}})
```

---

## Comparison Matrix

### Feature Comparison

| Feature | AstraDB | ChromaDB | Milvus | PgVector |
|---------|---------|----------|---------|----------|
| **Nested Metadata** | ✅ Full | ❌ Flattened | ✅ Full | ✅ Full (JSONB) |
| **Metadata-Only Search** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Numeric Casting** | ✅ Yes | ⚠️ Limited | ✅ Yes | ✅ Auto |
| **Transaction Support** | ❌ No | ❌ No | ❌ No | ✅ ACID |
| **Horizontal Scaling** | ✅ Auto | ❌ No | ✅ Yes | ⚠️ Read replicas |
| **Managed Service** | ✅ Yes | ✅ Cloud | ⚠️ Zilliz | ❌ Self-host |
| **Open Source** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |

### Operator Support

All backends support the same 8 operators:

| Operator | AstraDB | ChromaDB | Milvus | PgVector |
|----------|---------|----------|---------|----------|
| `$eq` | ✅ | ✅ | ✅ | ✅ |
| `$ne` | ✅ | ✅ | ✅ | ✅ |
| `$gt` | ✅ | ✅ | ✅ | ✅ |
| `$gte` | ✅ | ✅ | ✅ | ✅ |
| `$lt` | ✅ | ✅ | ✅ | ✅ |
| `$lte` | ✅ | ✅ | ✅ | ✅ |
| `$in` | ✅ | ✅ | ✅ | ✅ |
| `$nin` | ✅ | ✅ | ✅ | ✅ |

### Use Case Recommendations

#### Choose AstraDB if

- ✅ Need managed serverless solution
- ✅ Want full nested metadata support
- ✅ Require high scalability
- ✅ Prefer pay-as-you-go pricing

#### Choose ChromaDB if

- ✅ Want simple setup (in-memory)
- ✅ Building prototype/MVP
- ✅ Don't need nested metadata
- ✅ Prefer open source

#### Choose Milvus if

- ✅ Need maximum performance
- ✅ Have large-scale deployment (billions of vectors)
- ✅ Want distributed architecture
- ✅ All queries include vector search

#### Choose PgVector if

- ✅ Already using PostgreSQL
- ✅ Need ACID transactions
- ✅ Want full SQL capabilities
- ✅ Prefer mature, stable ecosystem

---

## Switching Backends

Same code works across all backends:

```python
from crossvector import VectorEngine
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.querydsl.q import Q

# Create embedding adapter (same for all)
embedding = OpenAIEmbeddingAdapter()

# Choose backend (interchangeable)
if backend == "astradb":
    from crossvector.dbs.astradb import AstraDBAdapter
    db = AstraDBAdapter()
elif backend == "chroma":
    from crossvector.dbs.chroma import ChromaDBAdapter
    db = ChromaDBAdapter()
elif backend == "milvus":
    from crossvector.dbs.milvus import MilvusAdapter
    db = MilvusAdapter()
else:  # pgvector
    from crossvector.dbs.pgvector import PgVectorAdapter
    db = PgVectorAdapter()

# Same API for all backends
engine = VectorEngine(db=db, embedding=embedding)

# Same operations
doc = engine.create("Document text", category="tech")
results = engine.search("query", where=Q(category="tech"), limit=10)
```

**Only consideration:** Check `engine.supports_metadata_only` for Milvus.

---

## Next Steps

- [Embedding Adapters](embeddings.md) - Embedding providers
- [API Reference](../api.md) - Complete API documentation
- [Query DSL](../querydsl.md) - Advanced filtering
- [Configuration](../configuration.md) - Settings reference
