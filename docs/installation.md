# Installation

This guide covers all installation options for CrossVector.

## Requirements

- Python 3.11 or higher
- pip or poetry for package management

## Installation Options

### 1. Core Package Only (Minimal)

Install just the core library without any adapters:

```bash
pip install crossvector
```

This includes:

- Core `VectorEngine` class
- Pydantic schemas and validation
- Query DSL (Q objects)
- Base adapter interfaces

**Note**: You'll need to install specific adapters separately.

### 2. With Specific Backend + Embedding

Choose the backend and embedding provider you need:

```bash
# AstraDB + OpenAI
pip install crossvector[astradb,openai]

# ChromaDB + OpenAI
pip install crossvector[chromadb,openai]

# Milvus + Gemini
pip install crossvector[milvus,gemini]

# PgVector + OpenAI
pip install crossvector[pgvector,openai]
```

### 3. All Backends

Install all database adapters with your choice of embedding:

```bash
# All databases + OpenAI
pip install crossvector[all-dbs,openai]

# All databases + Gemini
pip install crossvector[all-dbs,gemini]
```

### 4. All Embedding Providers

Install all embedding providers with your choice of database:

```bash
# AstraDB + All embeddings
pip install crossvector[astradb,all-embeddings]

# ChromaDB + All embeddings
pip install crossvector[chromadb,all-embeddings]
```

### 5. Complete Installation

Install everything:

```bash
pip install crossvector[all]
```

This includes all backends and all embedding providers.

### 6. From Git Repository

Install directly from the GitHub repository:

```bash
# Latest main branch
pip install git+https://github.com/thewebscraping/crossvector.git

# With specific extras
pip install git+https://github.com/thewebscraping/crossvector.git#egg=crossvector[astradb,openai]

# Specific branch
pip install git+https://github.com/thewebscraping/crossvector.git@main#egg=crossvector[all]

# Specific tag/version
pip install git+https://github.com/thewebscraping/crossvector.git@v1.0.0#egg=crossvector
```

This is useful for:

- Testing development versions before release
- Contributing to the project
- Using features from a specific branch

## Optional Dependencies Reference

### Database Adapters

| Extra | Includes | Use Case |
|-------|----------|----------|
| `astradb` | `astrapy>=2.1.0` | AstraDB serverless |
| `chromadb` | `chromadb>=1.3.4` | ChromaDB cloud/local |
| `milvus` | `pymilvus>=2.6.4` | Milvus/Zilliz cloud |
| `pgvector` | `pgvector>=0.4.1`, `psycopg2-binary>=2.9.11` | PostgreSQL with pgvector extension |
| `all-dbs` | All of the above | All backends |

### Embedding Providers

| Extra | Includes | Use Case |
|-------|----------|----------|
| `openai` | `openai>=2.6.1` | OpenAI embeddings |
| `gemini` | `google-genai>=0.3.0` | Google Gemini embeddings |
| `all-embeddings` | All of the above | All providers |

### Development

| Extra | Includes | Use Case |
|-------|----------|----------|
| `dev` | pytest, mypy, ruff, mkdocs, etc. | Development and testing |

Install dev dependencies:

```bash
pip install crossvector[dev]
```

## Verify Installation

After installation, verify it works:

```python
import crossvector
print(crossvector.__version__)

# Check imports
from crossvector import VectorEngine, VectorDocument
from crossvector.querydsl.q import Q

print("CrossVector installed successfully!")
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade crossvector[your-extras]
```

**Important**: Pin to specific versions for reproducible environments:

```bash
pip install crossvector[astradb,openai]==1.0.0
```

## Troubleshooting

### Import Errors

If you get import errors for adapters:

```python
# This will fail if you didn't install the adapter
from crossvector.dbs.astradb import AstraDBAdapter
# ImportError: cannot import name 'AstraDBAdapter'
```

**Solution**: Install the required extra:

```bash
pip install crossvector[astradb]
```

### Dependency Conflicts

If you encounter dependency conflicts, try:

```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CrossVector
pip install crossvector[your-extras]
```

### Version Pinning

For reproducible environments, use a requirements.txt:

```txt
crossvector[astradb,openai]==1.0.0
# Or with specific dependencies
astrapy==2.1.0
openai==2.6.1
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Build your first application
- [Configuration](configuration.md) - Set up environment variables
- [API Reference](api.md) - Explore the full API
