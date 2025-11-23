# Configuration

## Environment Variables

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

# Vector settings
VECTOR_METRIC=cosine                # Distance metric: cosine, dot_product, euclidean
VECTOR_STORE_TEXT=true              # Store original text in database (true/false)
```

## Configuration Options

### Vector Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VECTOR_METRIC` | string | `cosine` | Distance metric for similarity search. Options: `cosine`, `dot_product`, `euclidean` |
| `VECTOR_STORE_TEXT` | boolean | `true` | Whether to store original text in the database. Set to `false` to save storage space |

### Storage Optimization

If you're only using embeddings for search and don't need to retrieve the original text, you can disable text storage:

```python
from crossvector import VectorEngine

engine = VectorEngine(
    embedding_adapter=...,
    db_adapter=...,
    collection_name="my_docs",
    store_text=False  # Don't store text, only embeddings and metadata
)
```

This can significantly reduce storage requirements, especially for large text documents.
