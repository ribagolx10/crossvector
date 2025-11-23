# Quick Start

```python
from crossvector import VectorEngine, Document, UpsertRequest, SearchRequest
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.astradb import AstraDBAdapter

# Initialize engine
engine = VectorEngine(
    embedding_adapter=OpenAIEmbeddingAdapter(model_name="text-embedding-3-small"),
    db_adapter=AstraDBAdapter(),
    collection_name="my_documents"
)

# Upsert documents
docs = [
    Document(id="doc1", text="The quick brown fox", metadata={"category": "animals"}),
    Document(id="doc2", text="Artificial intelligence", metadata={"category": "tech"}),
]
result = engine.upsert(UpsertRequest(documents=docs))
print(f"Inserted {result['count']} documents")

# Search
results = engine.search(SearchRequest(query="AI and ML", limit=5))
for doc in results:
    print(f"Score: {doc.get('$similarity', 'N/A')}, Text: {doc.get('text')}")

# Get document by ID
doc = engine.get("doc1")

# Count documents
count = engine.count()

# Delete documents
engine.delete_one("doc1")
engine.delete_many(["doc2", "doc3"])
```
