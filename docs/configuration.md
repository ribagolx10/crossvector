# Configuration Guide

This guide covers all configuration options for CrossVector.

## Environment Variables

CrossVector uses environment variables for configuration. Create a `.env` file in your project root:

```bash
# .env file
# OpenAI Embedding Provider
OPENAI_API_KEY=sk-...

# Gemini Embedding Provider
GEMINI_API_KEY=AI...

# Optional: Override default embedding model
VECTOR_EMBEDDING_MODEL=text-embedding-3-small

# AstraDB Backend
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_API_ENDPOINT=https://...apps.astra.datastax.com

# ChromaDB Cloud Configuration
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...

# ChromaDB Self-hosted Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# ChromaDB Local Configuration
CHROMA_PERSIST_DIR=./chroma_data

# Milvus Configuration
MILVUS_API_ENDPOINT=https://...
MILVUS_API_KEY=...

# PgVector Configuration
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
VECTOR_COLLECTION_NAME=vector_db
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=postgres

# Vector Settings
VECTOR_METRIC=cosine
VECTOR_STORE_TEXT=true
VECTOR_DIM=1536
VECTOR_SEARCH_LIMIT=10

# Primary Key Configuration
PRIMARY_KEY_MODE=uuid

# Logging
LOG_LEVEL=INFO
```

## Settings Reference

### Embedding Provider Settings

#### OpenAI

```bash
OPENAI_API_KEY=sk-...              # Required: Your OpenAI API key
```

Supported models (defaults to `text-embedding-3-small`):

- `text-embedding-3-small` (1536 dims)
- `text-embedding-3-large` (3072 dims)
- `text-embedding-ada-002` (1536 dims, legacy)

#### Gemini

```bash
GEMINI_API_KEY=AI...               # Required: Your Gemini API key
# Note: Use GEMINI_API_KEY (not GOOGLE_API_KEY)
```

Supported models (defaults to `gemini-embedding-001`):

- `gemini-embedding-001` (768-3072 dims, recommended)
- `text-embedding-005` (768 dims)
- `text-embedding-004` (768 dims, legacy)

#### Shared Embedding Model (Optional)

```bash
# Override default model for all embedding adapters
VECTOR_EMBEDDING_MODEL=gemini-embedding-001
```

If not set, each adapter uses its own default:
- OpenAI: `text-embedding-3-small`
- Gemini: `gemini-embedding-001`

### Database Settings

#### AstraDB

```bash
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...  # Required: Application token
ASTRA_DB_API_ENDPOINT=https://...       # Required: API endpoint
# Note: Collection name uses VECTOR_COLLECTION_NAME (shared setting)
```

Get your credentials from [Astra Portal](https://astra.datastax.com/).

#### ChromaDB

**Cloud Mode:**

```bash
CHROMA_API_KEY=...      # Required for cloud
CHROMA_TENANT=...       # Required for cloud
CHROMA_DATABASE=...     # Required for cloud
```

**Self-Hosted HTTP Mode:**

```bash
CHROMA_HOST=localhost   # Required for self-hosted
CHROMA_PORT=8000        # Optional: Default 8000
# IMPORTANT: Must NOT set CHROMA_PERSIST_DIR when using HTTP mode
```

**Local Persistence Mode:**

```bash
CHROMA_PERSIST_DIR=./chroma_data  # Required for local
# IMPORTANT: Must NOT set CHROMA_HOST when using local mode
```

**Configuration Priority and Validation:**

ChromaDB adapter uses strict configuration validation with this priority:

1. **Cloud** (if `CHROMA_API_KEY` is set)
2. **HTTP** (if `CHROMA_HOST` is set AND `CHROMA_PERSIST_DIR` is NOT set)
3. **Local** (if `CHROMA_PERSIST_DIR` is set OR neither HTTP nor Cloud configured)

**Important:** Cannot set both `CHROMA_HOST` and `CHROMA_PERSIST_DIR` simultaneously. This will raise `MissingConfigError` with a helpful message explaining the conflict.

```python
# ✅ Valid configurations:
# Cloud only
CHROMA_API_KEY="..."

# HTTP only
CHROMA_HOST="localhost"

# Local only
CHROMA_PERSIST_DIR="./data"

# ❌ Invalid - raises MissingConfigError:
CHROMA_HOST="localhost"
CHROMA_PERSIST_DIR="./data"  # Conflict!
```

#### Milvus

```bash
MILVUS_API_ENDPOINT=https://...  # Required: Milvus/Zilliz endpoint
MILVUS_API_KEY=...               # Optional: API key for cloud
```

For local Milvus:

```bash
MILVUS_API_ENDPOINT=http://localhost:19530
```

#### PgVector

```bash
PGVECTOR_HOST=localhost      # Required: PostgreSQL host
PGVECTOR_PORT=5432          # Optional: Default 5432
VECTOR_COLLECTION_NAME=vector_db   # Required: Database name
PGVECTOR_USER=postgres      # Optional: Default postgres
PGVECTOR_PASSWORD=postgres  # Optional: Default postgres
```

**Important**: `VECTOR_COLLECTION_NAME` is required. CrossVector will attempt to create the database if it doesn't exist (requires CREATEDB privilege).

### Vector Settings

```bash
# Distance metric for vector similarity
VECTOR_METRIC=cosine
# Options: cosine, euclidean, dot_product

# Whether to store original text with vectors
VECTOR_STORE_TEXT=true
# Options: true, false

# Default embedding dimension (informational)
VECTOR_DIM=1536

# Default search result limit
VECTOR_SEARCH_LIMIT=10
```

### Primary Key Configuration

CrossVector supports multiple primary key generation strategies:

```bash
PRIMARY_KEY_MODE=uuid
```

Available modes:

| Mode | Description | Example |
|------|-------------|---------|
| `uuid` | Random UUID (default) | `f47ac10b-58cc-4372-a567-0e02b2c3d479` |
| `hash_text` | SHA256 hash of text | `9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08` |
| `hash_vector` | SHA256 hash of vector | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `int64` | Sequential integer | `1`, `2`, `3`, ... |
| `auto` | Hash text if available, else hash vector, else UUID | Varies |

**Custom PK Factory:**

You can provide a custom primary key generation function:

```bash
PRIMARY_KEY_FACTORY=mymodule.generate_custom_id
```

The function signature should be:

```python
def generate_custom_id(
    text: str | None,
    vector: List[float] | None,
    metadata: Dict[str, Any]
) -> str:
    """Generate custom primary key."""
    return f"custom-{text[:10]}"
```

### Logging

```bash
LOG_LEVEL=INFO
```

Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Programmatic Configuration

You can also configure settings programmatically:

```python
from crossvector.settings import settings

# Override settings
settings.VECTOR_STORE_TEXT = False
settings.VECTOR_SEARCH_LIMIT = 20
settings.PRIMARY_KEY_MODE = "hash_text"
settings.LOG_LEVEL = "DEBUG"

# Verify settings
print(f"Store text: {settings.VECTOR_STORE_TEXT}")
print(f"Search limit: {settings.VECTOR_SEARCH_LIMIT}")
print(f"PK mode: {settings.PRIMARY_KEY_MODE}")
```

## Runtime Configuration

Many settings can be overridden at runtime:

```python
from crossvector import VectorEngine

# Override store_text at engine level
engine = VectorEngine(
    embedding=embedding,
    db=db,
    collection_name="docs",
    store_text=False  # Override VECTOR_STORE_TEXT
)

# Override limit at search level
results = engine.search("query", limit=50)  # Override VECTOR_SEARCH_LIMIT
```

## Backend-Specific Configuration

### AstraDB

```python
from crossvector.dbs.astradb import AstraDBAdapter

# Default: uses env vars
db = AstraDBAdapter()

# Initialize with custom metric
engine = VectorEngine(
    embedding=embedding,
    db=db,
    collection_name="vectors"
)
```

### ChromaDB

```python
from crossvector.dbs.chroma import ChromaAdapter

# Cloud mode
db = ChromaAdapter()  # Uses CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE

# HTTP mode
db = ChromaAdapter()  # Uses CHROMA_HOST, CHROMA_PORT

# Local mode
db = ChromaAdapter()  # Uses CHROMA_PERSIST_DIR
```

### Milvus

```python
from crossvector.dbs.milvus import MilvusAdapter

# Uses MILVUS_API_ENDPOINT and MILVUS_API_KEY
db = MilvusAdapter()

engine = VectorEngine(
    embedding=embedding,
    db=db,
    collection_name="vectors"
)
```

### PgVector

```python
from crossvector.dbs.pgvector import PgVectorAdapter

# Uses PGVECTOR_* env vars
db = PgVectorAdapter()

engine = VectorEngine(
    embedding=embedding,
    db=db,
    collection_name="vectors"
)
```

## Configuration Best Practices

### 1. Use Environment Variables

Store sensitive data in `.env` and add it to `.gitignore`:

```bash
# .gitignore
.env
.env.local
.env.*.local
```

### 2. Separate Configurations

Use different `.env` files for different environments:

```bash
.env.development
.env.staging
.env.production
```

Load the appropriate file:

```python
from dotenv import load_dotenv
import os

env = os.getenv("APP_ENV", "development")
load_dotenv(f".env.{env}")
```

### 3. Validate Configuration

Check required settings on startup:

```python
from crossvector.settings import settings
from crossvector.exceptions import MissingConfigError

def validate_config():
    if not settings.OPENAI_API_KEY:
        raise MissingConfigError(
            "OPENAI_API_KEY is required",
            config_key="OPENAI_API_KEY",
            hint="Add OPENAI_API_KEY to your .env file"
        )
    if not settings.VECTOR_COLLECTION_NAME:
        raise MissingConfigError(
            "VECTOR_COLLECTION_NAME is required",
            config_key="VECTOR_COLLECTION_NAME"
        )

validate_config()
```

### 4. Use Type-Safe Settings

Access settings through the settings object for validation:

```python
from crossvector.settings import settings

# Good: Type-safe and validated
store_text = settings.VECTOR_STORE_TEXT

# Bad: String manipulation
store_text = os.getenv("VECTOR_STORE_TEXT") == "true"
```

### 5. Document Your Configuration

Create a `.env.example` file:

```bash
# .env.example
# Copy this file to .env and fill in your values

# OpenAI (required)
OPENAI_API_KEY=your-key-here

# PgVector (required)
VECTOR_COLLECTION_NAME=your-database-name
PGVECTOR_HOST=localhost
PGVECTOR_PASSWORD=your-password

# Optional settings
VECTOR_STORE_TEXT=true
LOG_LEVEL=INFO
```

## Troubleshooting

### Missing Configuration

```python
from crossvector.exceptions import MissingConfigError

try:
    db = PgVectorAdapter()
except MissingConfigError as e:
    print(f"Missing: {e.details['config_key']}")
    print(f"Hint: {e.details['hint']}")
```

### Invalid Configuration

```python
from crossvector.settings import settings

# Validate PRIMARY_KEY_MODE
valid_modes = {"uuid", "hash_text", "hash_vector", "int64", "auto"}
if settings.PRIMARY_KEY_MODE not in valid_modes:
    raise ValueError(f"Invalid PRIMARY_KEY_MODE: {settings.PRIMARY_KEY_MODE}")
```

### Connection Issues

Enable DEBUG logging to see connection details:

```bash
LOG_LEVEL=DEBUG
```

```python
from crossvector import VectorEngine
from crossvector.settings import settings

settings.LOG_LEVEL = "DEBUG"

# You'll see detailed connection logs
engine = VectorEngine(embedding=embedding, db=db)
```

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Quick Start](quickstart.md) - Build your first application
- [Database Adapters](adapters/databases.md) - Backend-specific features
