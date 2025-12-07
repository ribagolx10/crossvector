# Embedding Adapters

Embedding provider integrations for generating vector representations.

## Overview

CrossVector supports multiple embedding providers, with **Google Gemini** recommended for most users due to its free tier and performance.

### Comparison Matrix

| Feature | Google Gemini | OpenAI |
|---------|-----------------|-----------|
| **Best For** | Free tier & Speed | Quality & Ecosystem |
| **Free Tier** | 1,500 RPM (Generous) | No (Paid only) |
| **Default Model** | `models/text-embedding-004` (768 dims) | `text-embedding-3-small` (1536 dims) |
| **Custom Dimensions** | Yes (models/gemini-embedding-001: 768, 1536, 3072) | No (fixed per model) |
| **Search Speed** | Fast (~200ms) | Fast (~400ms) |
| **Storage** | Small (768 dims default) | Large (1536+ dims) |
| **Max Tokens** | 2,048 | 8,191 |
| **Cost** | Free / Low | $0.02 / 1M tokens |

---

## Google Gemini Embeddings (Recommended)

Google's state-of-the-art embedding models via Gemini API.

### Why Gemini?

- **Free Tier**: Up to 1,500 requests/minute for free.
- **Faster**: 1.5x faster search latency than OpenAI.
- **Storage Efficient**: Default 768 dimensions (50% smaller than OpenAI's 1536).
- **Flexible**: Optional 1536 or 3072 dimensions if needed (models/gemini-embedding-001).
- **Quality**: Excellent performance for search and retrieval tasks.

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

#### models/text-embedding-004 (Default - Latest)

Latest Gemini embedding model, state-of-the-art performance.

- **Dimensions:** 768 (default)
- **Max tokens:** 2,048
- **Best for:** Most search, RAG, and semantic matching applications
- **Performance:** Fast, balanced quality

```python
embedding = GeminiEmbeddingAdapter(model_name="models/text-embedding-004")
```

#### models/gemini-embedding-001

State-of-the-art with flexible dimensions.

- **Dimensions:** 1536 (default), supports 768 or 3072
- **Max tokens:** 2,048
- **Best for:** Applications needing custom embedding dimensions
- **Quality:** Excellent across multilingual and code tasks

```python
# Default 1536 dimensions
embedding = GeminiEmbeddingAdapter(model_name="models/gemini-embedding-001")

# Custom dimensions
embedding = GeminiEmbeddingAdapter(
    model_name="models/gemini-embedding-001",
    dim=768  # 768, 1536, or 3072
)
```

#### Legacy Models

- `models/text-embedding-005`: English and code (768 dims)
- `models/text-multilingual-embedding-002`: Multilingual (768 dims)
- `models/text-embedding-004`: Previous generation (768 dims)

```python
embedding = GeminiEmbeddingAdapter(model_name="models/text-embedding-005")
```

### Usage Examples

#### Basic Usage

```python
from crossvector import VectorEngine
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Initialize with Gemini (default: models/text-embedding-004, 768 dims)
engine = VectorEngine(
    db=PgVectorAdapter(),
    embedding=GeminiEmbeddingAdapter(),
    collection_name="documents"
)

# Embeddings generated automatically (768 dims)
doc = engine.create("Gemini embeddings are fast!")
print(len(doc.vector))  # 768

# Or use gemini-embedding-001 with custom dimensions
engine = VectorEngine(
    db=PgVectorAdapter(),
    embedding=GeminiEmbeddingAdapter(
        model_name="models/gemini-embedding-001",
        dim=1536  # 768, 1536, or 3072
    ),
    collection_name="documents"
)
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

- **Long Documents**: Supports up to 8,191 tokens per text.
- **High Dimensions**: Need 1536 or 3072 dimensions.
- **Ecosystem**: Already using OpenAI for LLMs.

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
from typing import List, Optional

class CustomEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, model_name: str = "custom-model", dim: int = 384):
        super().__init__(model_name=model_name, dim=dim)
        # Your initialization logic

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        # Your custom embedding logic here
        # Must return list of vectors with length = self.dim
        return [[0.1, 0.2, ...] for _ in texts]

# Use your custom adapter
embedding = CustomEmbeddingAdapter(dim=384)
engine = VectorEngine(db=..., embedding=embedding)
```

**Important:** Your adapter must:
- Inherit from `EmbeddingAdapter`
- Implement `get_embeddings(texts: List[str]) -> List[List[float]]`
- Set `dim` to match the vector dimension your model produces

---

## Error Handling

Handle embedding errors gracefully:

```python
from crossvector.exceptions import SearchError, MissingConfigError

try:
    embedding = GeminiEmbeddingAdapter()
    vectors = embedding.get_embeddings(["text"])
except MissingConfigError as e:
    print(f"Missing configuration: {e.details['config_key']}")
    print(f"Hint: {e.details['hint']}")
except SearchError as e:
    print(f"Embedding failed: {e.message}")
    if "rate" in str(e).lower():
        print("Rate limit exceeded!")
```

## Next Steps

- [Database Adapters](databases.md) - Choose your vector database
- [Configuration](../configuration.md) - Setup API keys
- [Quick Start](../quickstart.md) - Get started guide
