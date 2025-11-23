"""
Test script: Insert docs into ChromaDB Cloud using OpenAI text-embedding-3-small
"""

from dotenv import load_dotenv

from crossvector.dbs.chroma import ChromaDBAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Load .env
load_dotenv()

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "ChromaDB is a cloud-native vector database.",
]

embedder = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
embeddings = embedder.get_embeddings(texts)

# Initialize ChromaDB adapter
adapter = ChromaDBAdapter()

# Initialize or get collection
try:
    adapter.initialize(collection_name="test_vectors", embedding_dimension=embedder.embedding_dimension)
    print("Created collection 'test_vectors'.")
except Exception as e:
    print(f"Collection may already exist: {e}")
    adapter.collection = adapter.client.get_collection(name="test_vectors")

# Insert docs
docs = [
    {"_id": str(i), "vector": emb, "text": text, "metadata": {"source": "test"}}
    for i, (emb, text) in enumerate(zip(embeddings, texts))
]
adapter.upsert(docs)
print(f"Inserted {len(docs)} documents into ChromaDB Cloud.")

# Search: find 2 nearest results
results = adapter.search(embeddings[0], limit=2, fields={"text", "metadata"})
print("Search results:", results)

# Test get
doc = adapter.get("0")
print("Retrieved document:", doc)

# Test count
count = adapter.count()
print(f"Total documents: {count}")
