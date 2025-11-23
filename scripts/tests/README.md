# CrossVector Test Scripts

This directory contains test scripts to validate CrossVector adapters with real cloud APIs.

## Prerequisites

1. Install CrossVector with all dependencies:

```bash
uv sync --all-extras
```

1. Configure your `.env` file in the project root with your API credentials.

## Available Test Scripts

### Individual Database Tests

Each script tests a specific database adapter:

- **`tests/test_astradb.py`** - Test AstraDB adapter with real cloud API
- **`tests/test_chroma_cloud.py`** - Test ChromaDB Cloud adapter
- **`tests/test_milvus.py`** - Test Milvus cloud adapter
- **`tests/test_pgvector.py`** - Test PGVector (PostgreSQL) adapter

### Comprehensive Integration Test

- **`tests/test_integration.py`** - Tests the complete VectorEngine with all adapters
  - Tests all CRUD operations (Create, Read, Update, Delete)
  - Tests search functionality
  - Tests with both OpenAI and Gemini embeddings
  - Validates the unified API across all databases

## Running Tests

### Run Individual Tests

```bash
# Test AstraDB
uv run python scripts/tests/test_astradb.py

# Test ChromaDB
uv run python scripts/tests/test_chroma_cloud.py

# Test Milvus
uv run python scripts/tests/test_milvus.py

# Test PGVector
uv run python scripts/tests/test_pgvector.py
```

### Run Integration Test

```bash
uv run python scripts/tests/test_integration.py
```

## Required Environment Variables

### OpenAI (for embeddings)

```bash
OPENAI_API_KEY=sk-...
```

### Gemini (optional, for embeddings)

```bash
GOOGLE_API_KEY=...
```

### AstraDB

```bash
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_API_ENDPOINT=https://...
```

### ChromaDB Cloud

```bash
CHROMA_API_KEY=...
CHROMA_CLOUD_TENANT=...
CHROMA_CLOUD_DATABASE=...
```

### Milvus

```bash
MILVUS_API_ENDPOINT=https://...
MILVUS_USER=...
MILVUS_PASSWORD=...
```

### PGVector

```bash
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DBNAME=postgres
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=...
```

## Notes

- All test scripts create a test collection/table and clean up after themselves
- The integration test (`test_integration.py`) is the most comprehensive and tests all functionality
- Make sure you have valid credentials for the databases you want to test
- Some tests may fail if the database service is not available or credentials are invalid
