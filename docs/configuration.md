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

# Primary key generation
PRIMARY_KEY_MODE=uuid               # Mode: uuid, hash_text, hash_vector, int64, auto
# PRIMARY_KEY_FACTORY=mymodule.custom_pk_generator  # Optional: custom PK factory function
```

## Configuration Options

### Vector Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VECTOR_METRIC` | string | `cosine` | Distance metric for similarity search. Options: `cosine`, `dot_product`, `euclidean` |
| `VECTOR_STORE_TEXT` | boolean | `true` | Whether to store original text in the database. Set to `false` to save storage space |

### Primary Key Generation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRIMARY_KEY_MODE` | string | `uuid` | Primary key generation mode. Options: `uuid` (random UUID), `hash_text` (SHA256 of text), `hash_vector` (SHA256 of vector), `int64` (sequential integer as string), `auto` (smart mode - hash text if available, else vector, else UUID) |
| `PRIMARY_KEY_FACTORY` | string | None | Optional: Dotted path to custom PK factory function (e.g., `mymodule.custom_pk_generator`). Function should accept `(text: str, vector: List[float], metadata: Dict[str, Any])` and return `str` |

**Examples:**

```python
# Use UUID (default)
PRIMARY_KEY_MODE=uuid

# Use SHA256 hash of text content
PRIMARY_KEY_MODE=hash_text

# Use sequential integers (returned as string: "1", "2", "3", ...)
PRIMARY_KEY_MODE=int64

# Use custom factory function
PRIMARY_KEY_MODE=uuid
PRIMARY_KEY_FACTORY=myapp.utils.generate_custom_id

# Custom factory example:
# File: myapp/utils.py
# def generate_custom_id(text: str, vector: List[float], metadata: Dict[str, Any]) -> str:
#     return f"doc_{metadata.get('category', 'default')}_{uuid.uuid4().hex[:8]}"
```

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
