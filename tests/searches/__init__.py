"""Integration tests for vector search operations across all backends.

This package contains backend-specific search tests that validate:
- Query DSL compilation and execution
- Common operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin)
- Nested metadata queries
- Metadata-only search capabilities
- Backend-specific features and limitations

Each test module corresponds to a specific vector database backend:
- test_astradb.py - AstraDB Data API tests
- test_chroma.py - ChromaDB tests (local and cloud)
- test_milvus.py - Milvus tests (requires vector for all searches)
- test_pgvector.py - PostgreSQL pgvector extension tests

These tests require real database connections and are skipped if:
- Required environment variables are not set
- Backend services are not accessible
- API keys/credentials are invalid

For local testing without real backends, see the unit tests in parent directories.
"""
