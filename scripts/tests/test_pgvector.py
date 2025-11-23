"""
Test script: Insert docs into PGVector (PostgreSQL) using OpenAI text-embedding-3-small
"""

from dotenv import load_dotenv

from crossvector.dbs.pgvector import PGVectorAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Load .env
load_dotenv()

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "pgvector is a vector extension for PostgreSQL.",
]

# 1. Get embeddings
embedder = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
embeddings = embedder.get_embeddings(texts)

# 2. Initialize PGVector
pgvector = PGVectorAdapter()
pgvector.drop_collection("test_vectors")
pgvector.initialize(table_name="test_vectors", embedding_dimension=embedder.embedding_dimension)

# 3. Insert docs
docs = [
    {"_id": str(i), "vector": emb, "text": text, "metadata": {"source": "test"}}
    for i, (emb, text) in enumerate(zip(embeddings, texts))
]
pgvector.upsert(docs)
print(f"Inserted {len(docs)} documents into PGVector.")

# 4. Search test
results = pgvector.search(embeddings[0], limit=2, fields={"text", "metadata"})
print("Search results:", results)

# Test get
doc = pgvector.get("0")
print("Retrieved document:", doc)

# Test count
count = pgvector.count()
print(f"Total documents: {count}")
