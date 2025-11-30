# Integration Tests for Vector Search

This directory contains integration tests for vector search operations across all supported backends.

## Test Structure

Each backend has its own test module:

- **`test_astradb.py`** - AstraDB Data API integration tests
- **`test_chroma.py`** - ChromaDB integration tests (local/cloud)
- **`test_milvus.py`** - Milvus integration tests
- **`test_pgvector.py`** - PostgreSQL pgvector extension tests

## What's Tested

### Common DSL Operators (All Backends)
All backends support these 8 universal operators:
- `$eq` - Equality
- `$ne` - Not equal
- `$gt` - Greater than
- `$gte` - Greater than or equal
- `$lt` - Less than
- `$lte` - Less than or equal
- `$in` - Value in list
- `$nin` - Value not in list

### Query Combinations
- AND combinations: `Q(field1=value) & Q(field2=value)`
- OR combinations: `Q(field1=value) | Q(field2=value)`
- Complex nested: `(Q(a=1) & Q(b=2)) | Q(c=3)`

### Backend-Specific Features

#### AstraDB
- ✅ Metadata-only search
- ✅ Nested metadata with dot notation
- ✅ Universal dict and Q object formats

#### ChromaDB
- ✅ Metadata-only search
- ⚠️ Flattened metadata (nested stored as dot keys)
- ✅ Requires `$and` wrapper for multiple fields

#### Milvus
- ❌ No metadata-only search (vector required)
- ✅ Nested metadata with JSON field access
- ✅ Boolean expression compilation

#### PgVector
- ✅ Metadata-only search
- ✅ Nested JSONB queries with `#>>` operator
- ✅ Numeric casting with `::numeric`
- ✅ Deep nested paths (e.g., `data__user__name`)

## Running Tests

### Run all search integration tests:
```bash
pytest tests/searches/
```

### Run specific backend:
```bash
pytest tests/searches/test_astradb.py
pytest tests/searches/test_chroma.py
pytest tests/searches/test_milvus.py
pytest tests/searches/test_pgvector.py
```

### Run with verbose output:
```bash
pytest tests/searches/ -v
```

### Run specific test:
```bash
pytest tests/searches/test_pgvector.py::TestPgVectorQueryDSL::test_nested_metadata_jsonb
```

## Requirements

These tests require:

1. **Environment Variables** - Set in `.env` file:
   ```bash
   # OpenAI (for embeddings)
   OPENAI_API_KEY=sk-...

   # AstraDB
   ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
   ASTRA_DB_API_ENDPOINT=https://...

   # ChromaDB Cloud (optional)
   CHROMA_API_KEY=...
   CHROMA_TENANT=...
   CHROMA_DATABASE=...

   # Milvus
   MILVUS_API_ENDPOINT=http://localhost:19530
   MILVUS_API_KEY=...  # if cloud

   # PgVector
   PGVECTOR_HOST=localhost
   PGVECTOR_PORT=5432
   PGVECTOR_DBNAME=vector_db
   PGVECTOR_USER=postgres
   PGVECTOR_PASSWORD=postgres
   ```

2. **Running Database Services** - Ensure backends are accessible:
   - AstraDB: Cloud service (requires token)
   - ChromaDB: Local or cloud
   - Milvus: Local Docker or cloud
   - PgVector: PostgreSQL with pgvector extension

3. **Python Dependencies**:
   ```bash
   pip install crossvector[all]
   # Or specific backends:
   pip install crossvector[astradb,chroma,milvus,pgvector,openai]
   ```

## Test Behavior

- **Auto-skip**: Tests automatically skip if backend is not configured
- **Cleanup**: Each test suite cleans up test data before/after execution
- **Isolation**: Tests use unique collection names to avoid conflicts
- **Module-scoped fixtures**: Reuse engine and sample docs across test class

## Adding New Tests

When adding search functionality tests:

1. Add test methods to appropriate backend test class
2. Use descriptive test names: `test_<feature>_<behavior>`
3. Include docstrings explaining what's tested
4. Verify with real backend before committing
5. Update this README if new features are tested

## Troubleshooting

### Tests Skip with "not configured"
- Check `.env` file has required credentials
- Verify environment variables are loaded (use `load_dotenv()`)

### Connection Errors
- Ensure backend services are running
- Check network connectivity
- Verify firewall rules for cloud services

### Test Failures
- Check backend API changes
- Verify Query DSL compilation for that backend
- Review backend-specific limitations in docs

## Related Documentation

- [Query DSL Guide](../../docs/querydsl.md)
- [Database Adapters](../../docs/adapters/databases.md)
- [Architecture](../../docs/architecture.md)
