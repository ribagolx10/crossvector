# Embedding Adapters

## OpenAI

```python
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
```

## Gemini

```python
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

# Default model (models/gemini-embedding-001)
adapter = GeminiEmbeddingAdapter()

# Custom model
adapter = GeminiEmbeddingAdapter(model_name="models/text-embedding-004")

# Dynamic dimensionality (only for supported models like gemini-embedding-001)
adapter = GeminiEmbeddingAdapter(
    model_name="models/gemini-embedding-001",
    output_dimensionality=128
)
```

## Creating a Custom Embedding Adapter

```python
from crossvector.abc import EmbeddingAdapter
from typing import List

class MyCustomEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Initialize your client

    @property
    def embedding_dimension(self) -> int:
        return 768  # Your model's dimension

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Your implementation
        pass
```
