# Quick Start Guide

This guide will get you up and running with CrossVector in minutes.

## Prerequisites

1. Install CrossVector with your chosen backend and embedding provider:

```bash
pip install crossvector[pgvector,openai]
```

1. Set up environment variables (create a `.env` file):

```bash
OPENAI_API_KEY=sk-...
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DBNAME=vector_db
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=postgres
```

## Basic Usage

### Initialize the Engine

```python
from crossvector import VectorEngine
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Create engine instance
engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(model_name="text-embedding-3-small"),
    db=PgVectorAdapter(),
    collection_name="my_documents",
    store_text=True  # Store original text
)
```

### Create Documents

CrossVector accepts flexible input formats:

```python
# Method 1: Simple string
doc1 = engine.create("The quick brown fox jumps over the lazy dog")

# Method 2: Dict with metadata
doc2 = engine.create({
    "text": "Python is a programming language",
    "metadata": {"category": "tech", "level": "beginner"}
})

# Method 3: Dict with inline metadata
doc3 = engine.create(
    text="Machine learning basics",
    category="AI",
    difficulty="intermediate"
)

# Method 4: VectorDocument instance
from crossvector import VectorDocument
doc4 = engine.create(
    VectorDocument(
        id="custom-id-123",
        text="Advanced deep learning",
        metadata={"category": "AI", "level": "advanced"}
    )
)

print(f"Created: {doc1.id}, {doc2.id}, {doc3.id}, {doc4.id}")
```

### Search Documents

```python
# Simple text search
results = engine.search("programming tutorials", limit=5)

for doc in results:
    print(f"Score: {doc.metadata.get('score', 0):.3f}")
    print(f"Text: {doc.text}")
    print(f"Metadata: {doc.metadata}")
    print("---")
```

### Search with Filters

```python
from crossvector.querydsl.q import Q

# Search with metadata filters
results = engine.search(
    "python guide",
    where=Q(category="tech") & Q(level="beginner"),
    limit=10
)

# Range queries
results = engine.search(
    "articles",
    where=Q(views__gte=100) & Q(views__lte=1000)
)

# IN operator
results = engine.search(
    "content",
    where=Q(status__in=["published", "featured"])
)
```

### Get Document by ID

```python
# Retrieve specific document
doc = engine.get(doc1.id)
print(f"Retrieved: {doc.text}")

# Get by metadata (must return exactly one)
doc = engine.get(category="tech", level="beginner")
```

### Update Documents

```python
# Update document
doc1.metadata["updated"] = True
doc1.metadata["views"] = 150
updated = engine.update(doc1)

# Bulk update
updates = [
    {"id": doc2.id, "text": "Updated text", "metadata": {"featured": True}},
    {"id": doc3.id, "metadata": {"category": "ML"}},
]
engine.bulk_update(updates)
```

### Delete Documents

```python
# Delete single document
deleted_count = engine.delete(doc1.id)
print(f"Deleted {deleted_count} document(s)")

# Delete multiple documents
deleted_count = engine.delete([doc2.id, doc3.id])
print(f"Deleted {deleted_count} document(s)")
```

### Count Documents

```python
total = engine.count()
print(f"Total documents: {total}")
```

## Advanced Features

### Batch Operations

```python
# Bulk create
docs = [
    {"text": f"Document {i}", "metadata": {"index": i}}
    for i in range(100)
]
created = engine.bulk_create(docs, batch_size=50)
print(f"Created {len(created)} documents")

# Upsert (insert or update)
docs = [
    {"id": "doc-1", "text": "New or updated doc 1"},
    {"id": "doc-2", "text": "New or updated doc 2"},
]
upserted = engine.upsert(docs)
```

### Django-Style Operations

```python
# Get or create
doc, created = engine.get_or_create(
    text="Unique document",
    metadata={"key": "value"}
)
if created:
    print("Created new document")
else:
    print("Document already exists")

# Update or create
doc, created = engine.update_or_create(
    {"id": "doc-123"},
    text="Updated content",
    defaults={"metadata": {"status": "updated"}}
)
```

### Metadata-Only Search

```python
# Search by metadata without vector similarity
docs = engine.search(
    query=None,  # No vector search
    where=Q(category="tech") & Q(published=True),
    limit=50
)
```

### Complex Queries

```python
# Boolean combinations
high_quality = Q(rating__gte=4.5) & Q(reviews__gte=10)
featured = Q(featured=True)
results = engine.search(
    "products",
    where=high_quality | featured
)

# Negation
results = engine.search(
    "articles",
    where=~Q(status="archived")
)

# Nested metadata
results = engine.search(
    "documents",
    where=Q(user__role="admin") & Q(user__verified=True)
)
```

## Different Backend Examples

### AstraDB

```python
from crossvector.dbs.astradb import AstraDBAdapter

engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(),
    db=AstraDBAdapter(),  # Uses ASTRA_DB_* env vars
    collection_name="vectors"
)
```

### ChromaDB

```python
from crossvector.dbs.chroma import ChromaAdapter

# Cloud mode (uses CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE)
engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(),
    db=ChromaAdapter(),
    collection_name="vectors"
)

# Local mode (uses CHROMA_PERSIST_DIR)
# Set CHROMA_PERSIST_DIR=./chroma_data in .env
engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(),
    db=ChromaAdapter(),
    collection_name="vectors"
)
```

### Milvus

```python
from crossvector.dbs.milvus import MilvusAdapter

engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(),
    db=MilvusAdapter(),  # Uses MILVUS_API_ENDPOINT, MILVUS_API_KEY
    collection_name="vectors"
)
```

### Gemini Embeddings

```python
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

engine = VectorEngine(
    embedding=GeminiEmbeddingAdapter(
        model_name="gemini-embedding-001",
        dim=1536
    ),
    db=PgVectorAdapter(),
    collection_name="vectors"
)
```

## Error Handling

```python
from crossvector.exceptions import (
    DoesNotExist,
    MultipleObjectsReturned,
    InvalidFieldError,
)

# Handle missing documents
try:
    doc = engine.get("nonexistent-id")
except DoesNotExist as e:
    print(f"Document not found: {e.message}")

# Handle multiple results
try:
    doc = engine.get(status="active")  # Multiple matches
except MultipleObjectsReturned as e:
    print(f"Multiple documents found: {e.message}")

# Handle invalid queries
try:
    results = engine.search("query", where={"field": {"$regex": ".*"}})
except InvalidFieldError as e:
    print(f"Unsupported operator: {e.message}")
```

## Complete Example

Here's a full working example:

```python
from crossvector import VectorEngine
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter
from crossvector.querydsl.q import Q

# Initialize
engine = VectorEngine(
    embedding=OpenAIEmbeddingAdapter(),
    db=PgVectorAdapter(),
    collection_name="articles"
)

# Create sample articles
articles = [
    {
        "text": "Introduction to Python programming",
        "metadata": {"category": "tutorial", "level": "beginner", "views": 1500}
    },
    {
        "text": "Advanced machine learning techniques",
        "metadata": {"category": "tutorial", "level": "advanced", "views": 3200}
    },
    {
        "text": "Best practices for API design",
        "metadata": {"category": "guide", "level": "intermediate", "views": 2100}
    },
]

# Bulk insert
created_docs = engine.bulk_create(articles)
print(f"Created {len(created_docs)} articles")

# Search with filters
results = engine.search(
    "python tutorials for beginners",
    where=Q(category="tutorial") & Q(level__in=["beginner", "intermediate"]),
    limit=5
)

print(f"\nFound {len(results)} results:")
for doc in results:
    print(f"- {doc.text[:50]}... (views: {doc.metadata.get('views', 0)})")

# Update popular articles
for doc in results:
    if doc.metadata.get("views", 0) > 2000:
        doc.metadata["featured"] = True
        engine.update(doc)

# Count total
total = engine.count()
print(f"\nTotal articles: {total}")
```

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Query DSL](querydsl.md) - Advanced filtering and queries
- [Configuration](configuration.md) - Environment variables and settings
- [Database Adapters](adapters/databases.md) - Backend-specific features
