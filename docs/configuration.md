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
VECTOR_METRIC=cosine
```
