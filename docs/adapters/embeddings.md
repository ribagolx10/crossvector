# Embedding Adapters

Embedding provider integrations for generating vector representations.

## Overview

CrossVector supports multiple embedding providers, with **Google Gemini** recommended for most users due to its free tier and performance.

### Comparison Matrix

| Feature | 🥇 Google Gemini | 🥈 OpenAI |
|---------|-----------------|-----------|
| **Best For** | **Free tier & Speed** | Quality & Ecosystem |
| **Free Tier** | ✅ **1,500 RPM (Generous)** | ❌ No (Paid only) |
| **Search Speed** | ⚡ **Fast (~200ms)** | ⚡ Fast (~400ms) |
| **Storage** | 📉 **Small (768 dims)** | 📈 Large (1536 dims) |
| **Models** | `text-embedding-004` | `text-embedding-3-small` |
| **Max Tokens** | 2,048 | 8,191 |
| **Cost** | Free / Low | $0.02 / 1M tokens |

---

## Google Gemini Embeddings (Recommended)

Google's state-of-the-art embedding models via Gemini API.

### Why Gemini?

- ✅ **Free Tier**: Up to 1,500 requests/minute for free.
- ✅ **Faster**: 1.5x faster search latency than OpenAI.
- ✅ **Storage Efficient**: 768 dimensions require 50% less storage than 1536.
- ✅ **Quality**: Excellent performance for search and retrieval.

### Installation

```bash
pip install crossvector[gemini]
```

### Configuration

**Environment Variables:**

```bash
GEMINI_API_KEY="AI..."  # Get at https://makersuite.google.com/app/apikey
# Optional: Override default model
VECTOR_EMBEDDING_MODEL="gemini-embedding-001"
```

**Programmatic:**

```python
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

# Uses default model (gemini-embedding-001)
embedding = GeminiEmbeddingAdapter()

# Or specify model explicitly
embedding = GeminiEmbeddingAdapter(model_name="models/embedding-001")
```

### Available Models

#### models/text-embedding-004 (Default)

Latest generation model, balanced for performance and quality.

- **Dimensions:** 768
- **Max tokens:** 2,048
- **Best for:** Most search and RAG applications.

```python
embedding = GeminiEmbeddingAdapter(model_name="models/text-embedding-004")
```

#### models/embedding-001

Previous generation, widely supported.

- **Dimensions:** 768
- **Max tokens:** 2,048
- **Task Types:** Supports specific task optimization.

```python
embedding = GeminiEmbeddingAdapter(
    model_name="models/embedding-001",
    task_type="retrieval_document"
)
```

### Task Types

Optimize embeddings for specific use cases (supported by `embedding-001`):

```python
# For storing documents
embedding = GeminiEmbeddingAdapter(task_type="RETRIEVAL_DOCUMENT")

# For search queries
embedding = GeminiEmbeddingAdapter(task_type="RETRIEVAL_QUERY")

# For semantic similarity
embedding = GeminiEmbeddingAdapter(task_type="SEMANTIC_SIMILARITY")
```

### Usage Examples

#### Basic Usage

```python
from crossvector import VectorEngine
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Initialize with Gemini
engine = VectorEngine(
    db=PgVectorAdapter(),
    embedding=GeminiEmbeddingAdapter(),
    collection_name="documents"
)

# Embeddings generated automatically
doc = engine.create("Gemini embeddings are fast!")
print(len(doc.vector))  # 768
```

#### Batch Processing

```python
embedding = GeminiEmbeddingAdapter()
texts = ["Text 1", "Text 2", "Text 3"]

# Generate batch embeddings
vectors = embedding.get_embeddings(texts)
print(f"Generated {len(vectors)} vectors")
```

---

## OpenAI Embeddings (Alternative)

OpenAI's industry-standard embedding models.

### When to use OpenAI?

- ✅ **Long Documents**: Supports up to 8,191 tokens per text.
- ✅ **High Dimensions**: Need 1536 or 3072 dimensions.
- ✅ **Ecosystem**: Already using OpenAI for LLMs.

### Installation

```bash
pip install crossvector[openai]
```

### Configuration

**Environment Variables:**

```bash
OPENAI_API_KEY="sk-..."
```

**Programmatic:**

```python
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Uses default model (text-embedding-3-small)
embedding = OpenAIEmbeddingAdapter()
```

### Available Models

- **text-embedding-3-small** (Default): 1536 dims, $0.02/1M tokens.
- **text-embedding-3-large**: 3072 dims, best quality, expensive.
- **text-embedding-ada-002**: Legacy model.

### Usage Example

```python
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Initialize
embedding = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

# Generate vector
vector = embedding.get_embeddings(["Hello OpenAI"])[0]
print(len(vector))  # 1536
```

---

## Switching Providers

CrossVector's unified API makes switching providers easy:

```python
from crossvector import VectorEngine

# Toggle provider based on config
USE_OPENAI = False

if USE_OPENAI:
    from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
    embedding = OpenAIEmbeddingAdapter()
else:
    from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
    embedding = GeminiEmbeddingAdapter()

# Engine works exactly the same
engine = VectorEngine(db=..., embedding=embedding)
```

**Note:** If you switch providers for an existing collection, you must re-index your data because the vector dimensions and semantic space will change (e.g., 768 vs 1536).

---

## Custom Embedding Adapter

You can implement your own adapter for any provider (HuggingFace, Cohere, etc.):

```python
from crossvector.abc import EmbeddingAdapter
from typing import List

class CustomEmbeddingAdapter(EmbeddingAdapter):
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Your custom logic here
        return [[0.1, 0.2] for _ in texts]

    @property
    def dimensions(self) -> int:
        return 2  # Return actual dimensions
```

---

## Error Handling

Handle embedding errors gracefully:

```python
from crossvector.exceptions import EmbeddingError

try:
    embedding.get_embeddings(["text"])
except EmbeddingError as e:
    print(f"Embedding failed: {e.message}")
    if "quota" in str(e).lower():
        print("Rate limit exceeded!")
```

## Next Steps

- [Database Adapters](databases.md) - Choose your vector database
- [Configuration](../configuration.md) - Setup API keys
- [Quick Start](../quickstart.md) - Get started guide
