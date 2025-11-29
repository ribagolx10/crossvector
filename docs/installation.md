# Installation

## Minimal (core only)

```bash
pip install crossvector
```

## With specific adapters

```bash
# AstraDB + OpenAI
pip install crossvector[astradb,openai]

# ChromaDB + OpenAI
pip install crossvector[chromadb,openai]

# All databases + OpenAI
pip install crossvector[all-dbs,openai]

# Everything
pip install crossvector[all]
```

## Configure

Set required environment variables for embeddings and databases:

```bash
export OPENAI_API_KEY=...       # OpenAI embeddings
export GOOGLE_API_KEY=...       # Gemini embeddings
export LOG_LEVEL=INFO           # Optional: control logging verbosity
```
