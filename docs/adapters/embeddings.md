# Embedding Adapters

Embedding provider integrations for generating vector representations.

## Overview

CrossVector supports multiple embedding providers:

| Provider | Models | Max Tokens | Dimensions | License |
|----------|--------|------------|------------|---------|
| **OpenAI** | text-embedding-3-small, 3-large, ada-002 | 8,191 | 1536/3072 | Proprietary |
| **Google Gemini** | text-embedding-004, embedding-001 | 2,048 | 768 | Proprietary |

---

## OpenAI Embeddings

OpenAI's text embedding models via official API.

### Features

- ✅ **High quality** - Industry-leading embeddings
- ✅ **Multiple models** - Small (fast) to large (accurate)
- ✅ **Flexible dimensions** - 1536 or 3072
- ✅ **Batch support** - Up to 2048 texts per request
- ✅ **Efficient** - Optimized for production

### Installation

```bash
pip install crossvector[openai]
```

### Configuration

**Environment Variables:**

```bash
OPENAI_API_KEY="sk-..."
# Optional: Override default model
VECTOR_EMBEDDING_MODEL="text-embedding-3-small"
```

**Programmatic:**

```python
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Uses default model (text-embedding-3-small)
embedding = OpenAIEmbeddingAdapter()

# Or specify model explicitly
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-large")
```

### Available Models

#### text-embedding-3-small

Best for most use cases - balanced performance and cost.

```python
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
```

**Specifications:**

- **Dimensions:** 1536 (default) or configurable
- **Max tokens:** 8,191
- **Performance:** ~62.3% on MTEB
- **Cost:** $0.02 / 1M tokens
- **Speed:** Fast

#### text-embedding-3-large

Highest quality embeddings for demanding applications.

```python
embedding = OpenAIEmbeddingAdapter(
    model_name="text-embedding-3-large",
    dimensions=3072
)
```

**Specifications:**

- **Dimensions:** 3072 (default) or configurable
- **Max tokens:** 8,191
- **Performance:** ~64.6% on MTEB
- **Cost:** $0.13 / 1M tokens
- **Speed:** Slower than small

#### text-embedding-ada-002 (Legacy)

Previous generation model, still supported.

```python
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-ada-002")
```

**Specifications:**

- **Dimensions:** 1536 (fixed)
- **Max tokens:** 8,191
- **Performance:** ~61.0% on MTEB
- **Cost:** $0.10 / 1M tokens
- **Status:** Legacy, use v3 models instead

### Usage

#### Basic Usage

```python
from crossvector import VectorEngine
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Create adapter
embedding = OpenAIEmbeddingAdapter()

# Use with engine
engine = VectorEngine(
    db=PgVectorAdapter(),
    embedding=embedding,
    collection_name="documents"
)

# Embeddings generated automatically
doc = engine.create("Python is a programming language")
print(doc.vector[:5])  # [0.123, 0.456, ...]
```

#### Batch Embeddings

```python
# Bulk operations use batch API automatically
docs = [
    "Document 1 text",
    "Document 2 text",
    "Document 3 text",
    # ... up to 2048 texts
]

created = engine.bulk_create(docs, batch_size=100)
# Embeddings generated in batches
```

#### Custom Dimensions

```python
# Smaller dimensions = faster, less accurate
embedding = OpenAIEmbeddingAdapter(
    model_name="text-embedding-3-small",
    dimensions=512  # Reduce from 1536
)

# Larger dimensions = slower, more accurate
embedding = OpenAIEmbeddingAdapter(
    model_name="text-embedding-3-large",
    dimensions=3072  # Full dimensions
)
```

### Direct Embedding Access

```python
embedding = OpenAIEmbeddingAdapter()

# Single text
vector = embedding.get_embeddings(["Hello world"])[0]
print(len(vector))  # 1536

# Multiple texts (batch)
texts = ["Text 1", "Text 2", "Text 3"]
vectors = embedding.get_embeddings(texts)
print(len(vectors))  # 3
```

### Error Handling

```python
from crossvector.exceptions import EmbeddingError

try:
    embedding = OpenAIEmbeddingAdapter(api_key="invalid")
    vectors = embedding.get_embeddings(["text"])
except EmbeddingError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

### Performance Tips

```python
# Use small model for speed
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

# Reduce dimensions for faster search
embedding = OpenAIEmbeddingAdapter(dimensions=512)

# Batch operations for efficiency
engine.bulk_create(docs, batch_size=100)

# Cache embeddings when possible
# (Store in VectorEngine with store_text=True)
```

### Cost Optimization

```python
# Choose model by use case
if use_case == "production_search":
    # Best balance
    model = "text-embedding-3-small"  # $0.02 / 1M tokens
elif use_case == "high_accuracy":
    # Maximum quality
    model = "text-embedding-3-large"  # $0.13 / 1M tokens
else:
    # Development/testing
    model = "text-embedding-3-small"

embedding = OpenAIEmbeddingAdapter(model_name=model)
```

---

## Google Gemini Embeddings

Google's embedding models via Gemini API.

### Features

- ✅ **High performance** - Latest generation models
- ✅ **Task-specific** - Optimize for retrieval, clustering, etc.
- ✅ **Efficient** - Lower cost than OpenAI
- ✅ **Flexible** - Multiple task types

### Installation

```bash
pip install crossvector[gemini]
```

### Configuration

**Environment Variables:**

```bash
GEMINI_API_KEY="your-key"
# Optional: Override default model
VECTOR_EMBEDDING_MODEL="gemini-embedding-001"
```

**Programmatic:**

```python
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

# Uses default model (gemini-embedding-001)
embedding = GeminiEmbeddingAdapter()

# Or specify model explicitly
embedding = GeminiEmbeddingAdapter(model_name="text-embedding-005")
```

### Available Models

#### gemini-embedding-001 (Recommended)

State-of-the-art model with flexible dimensions and multilingual support.

```python
embedding = GeminiEmbeddingAdapter(
    model_name="gemini-embedding-001",
    dim=1536,  # 768, 1536, or 3072
    task_type="retrieval_document"
)
```

**Specifications:**

- **Dimensions:** 768, 1536 (default), or 3072
- **Max tokens:** 2,048
- **Task types:** retrieval_document, retrieval_query, semantic_similarity, classification
- **Best performance:** Across English, multilingual, and code tasks

#### text-embedding-005

Specialized for English and code tasks.

```python
embedding = GeminiEmbeddingAdapter(model_name="text-embedding-005")
```

**Specifications:**

- **Dimensions:** 768 (fixed)
- **Max tokens:** 2,048
- **Best for:** English-only content

#### text-embedding-004 (Legacy)

```python
embedding = GeminiEmbeddingAdapter(model_name="text-embedding-004")
```

**Status:** Use gemini-embedding-001 or text-embedding-005 instead

### Task Types

Optimize embeddings for specific use cases:

```python
# For documents being stored
embedding = GeminiEmbeddingAdapter(task_type="retrieval_document")

# For search queries
embedding = GeminiEmbeddingAdapter(task_type="RETRIEVAL_QUERY")

# For semantic similarity
embedding = GeminiEmbeddingAdapter(task_type="SEMANTIC_SIMILARITY")

# For classification
embedding = GeminiEmbeddingAdapter(task_type="CLASSIFICATION")

# For clustering
embedding = GeminiEmbeddingAdapter(task_type="CLUSTERING")
```

**Recommended:** Use `RETRIEVAL_DOCUMENT` for storing and `RETRIEVAL_QUERY` for searching.

### Usage

#### Basic Usage

```python
from crossvector import VectorEngine
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Create adapter
embedding = GeminiEmbeddingAdapter(task_type="RETRIEVAL_DOCUMENT")

# Use with engine
engine = VectorEngine(
    db=PgVectorAdapter(),
    embedding=embedding,
    collection_name="documents"
)

# Create documents
doc = engine.create("Python programming tutorial")
print(len(doc.vector))  # 768
```

#### Task-Specific Embeddings

```python
# Store documents with RETRIEVAL_DOCUMENT
doc_embedding = GeminiEmbeddingAdapter(task_type="RETRIEVAL_DOCUMENT")
engine = VectorEngine(db=..., embedding=doc_embedding)

docs = [
    "Document 1 content",
    "Document 2 content",
]
engine.bulk_create(docs)

# Search with RETRIEVAL_QUERY
query_embedding = GeminiEmbeddingAdapter(task_type="RETRIEVAL_QUERY")
query_vector = query_embedding.get_embeddings(["search query"])[0]

# Manual vector search
results = engine.search(query_vector, limit=10)
```

#### Batch Embeddings

```python
embedding = GeminiEmbeddingAdapter()

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
vectors = embedding.get_embeddings(texts)

# Use in bulk operations
docs = [{"text": text} for text in texts]
created = engine.bulk_create(docs, batch_size=50)
```

### Error Handling

```python
from crossvector.exceptions import EmbeddingError

try:
    embedding = GeminiEmbeddingAdapter(api_key="invalid")
    vectors = embedding.get_embeddings(["text"])
except EmbeddingError as e:
    print(f"Error: {e.message}")
    print(f"Provider: {e.details['provider']}")
```

### Performance Tips

```python
# Use task-specific embeddings
embedding = GeminiEmbeddingAdapter(task_type="RETRIEVAL_DOCUMENT")

# Batch operations for efficiency
texts = ["Text 1", "Text 2", ..., "Text N"]
vectors = embedding.get_embeddings(texts)

# Cache embeddings
engine = VectorEngine(db=..., embedding=..., store_text=True)
```

---

## Comparison

### Model Comparison

| Model | Provider | Dimensions | Max Tokens | Quality | Cost | Speed |
|-------|----------|------------|------------|---------|------|-------|
| text-embedding-3-small | OpenAI | 1536 | 8,191 | ⭐⭐⭐⭐ | 💰 Low | ⚡⚡⚡ Fast |
| text-embedding-3-large | OpenAI | 3072 | 8,191 | ⭐⭐⭐⭐⭐ | 💰💰 Med | ⚡⚡ Med |
| text-embedding-004 | Gemini | 768 | 2,048 | ⭐⭐⭐⭐ | 💰 Low | ⚡⚡⚡ Fast |

### Cost Comparison

| Provider | Model | Cost (per 1M tokens) |
|----------|-------|----------------------|
| OpenAI | text-embedding-3-small | $0.02 |
| OpenAI | text-embedding-3-large | $0.13 |
| OpenAI | text-embedding-ada-002 | $0.10 |
| Gemini | text-embedding-004 | Lower than OpenAI |

### Use Case Recommendations

#### Choose OpenAI if

- ✅ Need highest quality embeddings
- ✅ Working with longer documents (8K tokens)
- ✅ Want flexible dimensions (512-3072)
- ✅ Prefer industry-standard solution

#### Choose Gemini if

- ✅ Want lower costs
- ✅ Need task-specific optimization
- ✅ Working with shorter texts (<2K tokens)
- ✅ Prefer Google ecosystem

---

## Custom Embedding Adapter

Create custom adapter for other providers:

```python
from crossvector.abc import EmbeddingAdapter
from typing import List

class CustomEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, api_key: str, model_name: str = "custom-model"):
        self.api_key = api_key
        self.model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        # Implement your embedding logic here
        # Should return list of vectors
        vectors = []
        for text in texts:
            vector = self._embed_text(text)
            vectors.append(vector)
        return vectors

    def _embed_text(self, text: str) -> List[float]:
        """Generate single embedding."""
        # Your implementation
        pass

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return 768  # Your model dimensions

# Use custom adapter
embedding = CustomEmbeddingAdapter(api_key="...")
engine = VectorEngine(db=..., embedding=embedding)
```

### Required Methods

- `get_embeddings(texts: List[str]) -> List[List[float]]` - Generate embeddings
- `dimensions` property - Return embedding dimensions

### Optional Methods

- `embed_query(text: str) -> List[float]` - Single text embedding
- `validate_dimensions(vector: List[float])` - Validate vector dimensions

---

## Switching Providers

Same API across all providers:

```python
from crossvector import VectorEngine

# Choose provider
if provider == "openai":
    from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
    embedding = OpenAIEmbeddingAdapter()
else:  # gemini
    from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
    embedding = GeminiEmbeddingAdapter()

# Same usage
engine = VectorEngine(db=..., embedding=embedding)
doc = engine.create("Text content")
results = engine.search("query", limit=10)
```

**Note:** Dimensions must match when switching providers with existing collections.

---

## Best Practices

### Production Deployment

```python
import os
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Use environment variables
embedding = OpenAIEmbeddingAdapter(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("OPENAI_MODEL", "text-embedding-3-small")
)

# Error handling
from crossvector.exceptions import EmbeddingError

try:
    vectors = embedding.get_embeddings(texts)
except EmbeddingError as e:
    logger.error(f"Embedding failed: {e.message}")
    # Fallback or retry logic
```

### Batch Processing

```python
# Process large datasets efficiently
def embed_documents(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        created = engine.bulk_create(batch, batch_size=batch_size)
        print(f"Processed {i+len(batch)}/{len(documents)}")
```

### Cost Management

```python
# Monitor usage
total_tokens = sum(len(text.split()) for text in texts)
estimated_cost = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens

print(f"Estimated cost: ${estimated_cost:.4f}")

# Use smaller model for testing
if os.getenv("ENV") == "development":
    embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
else:
    embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-large")
```

### Caching

```python
# Store text with vectors for caching
engine = VectorEngine(
    db=...,
    embedding=...,
    store_text=True  # Enable text storage
)

# Retrieve without re-embedding
doc = engine.get("doc-id")
print(doc.text)  # Original text available
print(doc.vector)  # Pre-computed vector
```

---

## Next Steps

- [Database Adapters](databases.md) - Backend features
- [API Reference](../api.md) - Complete API documentation
- [Configuration](../configuration.md) - Settings reference
- [Quick Start](../quickstart.md) - Get started guide
