# Quick Start

```python
from crossvector import VectorEngine, VectorDocument
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.astradb import AstraDBAdapter

# Initialize engine
engine = VectorEngine(
    embedding_adapter=OpenAIEmbeddingAdapter(model_name="text-embedding-3-small"),
    db_adapter=AstraDBAdapter(),
    collection_name="my_documents",
    store_text=True  # Optional: Set to False to save space
)

# Method 1: Create from texts (Recommended - Auto embedding)
result = engine.upsert_from_texts(
    texts=["The quick brown fox", "Artificial intelligence", "My article"],
    metadatas=[
        {"category": "animals"},
        {"category": "tech"},
        {
            "title": "Introduction to AI",
            "created_at": "2024-01-15T10:00:00Z",  # Your article timestamp
            "author": "John Doe"
        }
    ],
    pks=["doc1", "doc2", None]  # None = auto-generated
)
print(f"Inserted {len(result)} documents")

# Method 2: Upsert VectorDocument directly (if you have embeddings)
docs = [
    VectorDocument(
        id="doc3",
        text="Python programming",
        vector=[0.1]*1536,  # Pre-computed embedding
        metadata={"category": "tech"}
    )
]
result = engine.upsert(docs)

# Each document gets:
# - Auto-generated ID (SHA256 hash if not provided)
# - created_timestamp: Unix timestamp (float)
# - updated_timestamp: Unix timestamp (float)

# Search
results = engine.search(query="AI and ML", limit=5)
for doc in results:
    print(f"Score: {getattr(doc, 'score', 'N/A')}, Text: {doc.text}")

# Get document by ID
doc = engine.get("doc2")
print(f"Created at: {doc.created_timestamp}")  # Unix timestamp

# Count documents
count = engine.count()

# Delete documents
deleted = engine.delete("doc2")  # Single ID
deleted = engine.delete(["doc1", "doc2"])  # Multiple IDs
```
