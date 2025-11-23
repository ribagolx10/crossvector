"""
Test script: Insert docs into AstraDB using OpenAI text-embedding-3-small
"""

from dotenv import load_dotenv

from crossvector.dbs.astradb import AstraDBAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Load .env
load_dotenv()

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "AstraDB is a cloud-native vector database.",
]

embedder = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
embeddings = embedder.get_embeddings(texts)

# Delete collection 'test_vectors' if it already exists
adapter = AstraDBAdapter()
db = adapter.db
if "test_vectors" in db.list_collection_names():
    db.drop_collection("test_vectors")
    print("Dropped collection 'test_vectors'.")

# Create collection and insert data
adapter.initialize(collection_name="test_vectors", embedding_dimension=embedder.embedding_dimension)
docs = [
    {"_id": str(i), "vector": emb, "text": text, "metadata": {"source": "test"}}
    for i, (emb, text) in enumerate(zip(embeddings, texts))
]
adapter.upsert(docs)
print(f"Inserted {len(docs)} documents into AstraDB.")

# Search: find 2 nearest results
results = adapter.search(embeddings[0], limit=2, fields={"text", "metadata"})
print("Search results:", results)

# Test get
doc = adapter.get("0")
print("Retrieved document:", doc)

# Test count
count = adapter.count()
print(f"Total documents: {count}")
