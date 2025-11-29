# CrossVector

## Cross-platform Vector Database Engine

A flexible, production-ready vector database engine with pluggable adapters for multiple vector databases,
(AstraDB, ChromaDB, Milvus, PGVector) and embedding providers (OpenAI, Gemini, and more).

Simplify your vector search infrastructure with a single, unified API across all major vector databases.

## Features

- 🔌 **Pluggable Architecture**: Easy adapter pattern for both databases and embeddings
- 🗄️ **Multiple Vector Databases**: AstraDB, ChromaDB, Milvus, PGVector
- 🤖 **Multiple Embedding Providers**: OpenAI, Gemini
- 🎯 **Smart Document Handling**: Auto-generated IDs (UUID/hash/int64/custom), optional text storage
- 📦 **Install Only What You Need**: Optional dependencies per adapter
- 🔒 **Type-Safe**: Full Pydantic validation
- 🔄 **Consistent API**: Same interface across all adapters

## Supported Vector Databases

| Database | Status | Features |
| ---------- | -------- | ---------- |
| **AstraDB** | ✅ Production | Cloud-native Cassandra, lazy initialization |
| **ChromaDB** | ✅ Production | Cloud/HTTP/Local modes, auto-fallback |
| **Milvus** | ✅ Production | Auto-indexing, schema validation |
| **PGVector** | ✅ Production | PostgreSQL extension, JSONB metadata |

## Supported Embedding Providers

| Provider | Status | Models |
| ---------- | -------- | -------- |
| **OpenAI** | ✅ Production | text-embedding-3-small, 3-large, ada-002 |
| **Gemini** | ✅ Production | text-embedding-004, gemini-embedding-001 |
