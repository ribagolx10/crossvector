# Quick Start

```python
from crossvector import VectorEngine, Document, UpsertRequest, SearchRequest
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.astradb import AstraDBAdapter

# Initialize engine
engine = VectorEngine(
    embedding_adapter=OpenAIEmbeddingAdapter(model_name="text-embedding-3-small"),
    db_adapter=AstraDBAdapter(),
    collection_name="my_documents",
    store_text=True  # Optional: Set to False to save space
)

# Upsert documents with auto-generated features
docs = [
    # Auto-generated ID and timestamps
    Document(text="The quick brown fox", metadata={"category": "animals"}),

    # Manual ID with auto timestamps
    Document(id="doc2", text="Artificial intelligence", metadata={"category": "tech"}),

    # Preserve your own timestamps
    Document(
        text="My article",
        metadata={
            "title": "Introduction to AI",
            "created_at": "2024-01-15T10:00:00Z",  # Your article timestamp
            "author": "John Doe"
        }
    ),
]
result = engine.upsert(UpsertRequest(documents=docs))
print(f"Inserted {result['count']} documents")

# Each document gets:
# - Auto-generated ID (SHA256 hash if not provided)
# - created_timestamp: Unix timestamp (float)
# - updated_timestamp: Unix timestamp (float)

# Search
results = engine.search(SearchRequest(query="AI and ML", limit=5))
for doc in results:
    print(f"Score: {doc.get('score', 'N/A')}, Text: {doc.get('text')}")

# Get document by ID
doc = engine.get("doc2")
print(f"Created at: {doc.get('created_timestamp')}")  # Unix timestamp

# Count documents
count = engine.count()

# Delete documents
engine.delete_one("doc2")
```
