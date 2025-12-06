# CrossVector Documentation

Welcome to CrossVector - a unified Python library for vector database operations with pluggable backends and embedding providers.

## What is CrossVector?

CrossVector provides a consistent, high-level API across multiple vector databases (AstraDB, ChromaDB, Milvus, PgVector) and embedding providers (OpenAI, Gemini). Write your code once, switch backends without rewriting your application.

## Key Features

- **🔌 Pluggable Architecture**: 4 vector databases, 2 embedding providers, lazy initialization
- **🎯 Unified API**: Consistent interface across all adapters with standardized error handling
- **🔍 Advanced Querying**: Type-safe Query DSL with Q objects
- **🚀 Performance**: Automatic batch embedding, bulk operations, lazy client initialization
- **🛡️ Type-Safe**: Full Pydantic v2 validation and structured exceptions
- **⚙️ Flexible Configuration**: Environment variables, explicit config validation, multiple PK strategies

## Quick Navigation

### Getting Started

- [Installation](installation.md) - Install CrossVector and dependencies
- [Quick Start](quickstart.md) - Your first CrossVector program
- [Configuration](configuration.md) - Environment variables and settings

### Core Concepts

- [API Reference](api.md) - Complete VectorEngine API
- [Schema](schema.md) - VectorDocument and data models
- [Query DSL](querydsl.md) - Advanced filtering with Q objects

### Adapters

- [Database Adapters](adapters/databases.md) - AstraDB, ChromaDB, Milvus, PgVector
- [Embedding Adapters](adapters/embeddings.md) - OpenAI, Gemini

### Development

- [Contributing](contributing.md) - How to contribute to CrossVector
- [Architecture](architecture.md) - System design and components

## Quick Example

> 💡 **Recommended**: Use Gemini for free tier and faster performance. [See why →](quickstart.md)

```python
from crossvector import VectorEngine
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

# Initialize with Gemini (free tier, 1536-dim vectors)
engine = VectorEngine(
    db=PgVectorAdapter(),
    embedding=GeminiEmbeddingAdapter(),
    collection_name="my_docs"
)

# Create and search
doc = engine.create("CrossVector makes vector databases easy")
results = engine.search("vector database library", limit=5)
```

**Why Gemini?**
- ✅ Free API tier (1,500 RPM)
- ✅ 1.5x faster search than OpenAI
- ✅ 50% smaller vectors (768 vs 1536 dims)

**With OpenAI?** [See alternative setup →](quickstart.md#using-openai-instead)

# Query with filters
from crossvector.querydsl.q import Q
results = engine.search(
    "machine learning",
    where=Q(category="tech") & Q(level="beginner")
)
```

## Backend Support Matrix

| Feature | AstraDB | ChromaDB | Milvus | PgVector |
|---------|---------|----------|--------|----------|
| Vector Search | ✅ | ✅ | ✅ | ✅ |
| Metadata-Only Search | ✅ | ✅ | ✅ | ✅ |
| Nested Metadata | ✅ | ✅* | ❌ | ✅ |
| Numeric Comparisons | ✅ | ✅ | ✅ | ✅ |
| Lazy Initialization | ✅ | ✅ | ✅ | ✅ |
| Config Validation | ✅ | ✅ | ✅ | ✅ |

*ChromaDB supports nested metadata via dot-notation when flattened.

## Status

**Current Version**: 0.1.0 (Beta)

⚠️ **Beta Status**: CrossVector is currently in beta. Do not use in production until version 1.0.

- API may change without notice
- Database schemas may evolve
- Features are still being tested

**Recommended for:**

- ✅ Prototyping and development
- ✅ Learning vector databases
- ❌ Production applications

## Support

- **GitHub**: [thewebscraping/crossvector](https://github.com/thewebscraping/crossvector)
- **Issues**: [Report bugs](https://github.com/thewebscraping/crossvector/issues)
- **Discussions**: [Ask questions](https://github.com/thewebscraping/crossvector/discussions)

## License

CrossVector is released under the MIT License. See [LICENSE](../LICENSE) for details.
