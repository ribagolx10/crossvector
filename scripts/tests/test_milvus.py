"""
Test script: Insert docs into Milvus using OpenAI text-embedding-3-small
"""

import time

from crossvector.dbs.milvus import MilvusDBAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Example docs
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Milvus is a cloud-native vector database.",
]

# 1. Get embeddings
embedder = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
embeddings = embedder.get_embeddings(texts)

# 2. Insert into Milvus
milvus = MilvusDBAdapter()
milvus.initialize(collection_name="test_vectors", embedding_dimension=embedder.embedding_dimension)
milvus.drop_collection("test_vectors")

docs = [
    {"_id": str(i), "vector": emb, "text": text, "metadata": {"source": "test"}}
    for i, (emb, text) in enumerate(zip(embeddings, texts))
]
milvus.upsert(docs)
print(f"Inserted {len(docs)} documents into Milvus.")


time.sleep(5)  # Wait for indexing

# 3. Search test
results = milvus.search(embeddings[0], limit=2, fields={"text", "metadata"})
print("Search results:", results)

# Test get
doc = milvus.get("0")
print("Retrieved document:", doc)

# Test count
count = milvus.count()
print(f"Total documents: {count}")
