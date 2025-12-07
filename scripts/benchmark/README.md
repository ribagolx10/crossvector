# CrossVector Benchmark Suite

Comprehensive benchmarking tools for evaluating vector database adapter performance across multiple backends.

## Files

- **`run.py`** - Main benchmark runner for comprehensive performance testing
- **`fixtures.py`** - Fixture generation library with realistic, topic-specific content
- **`generate_fixtures.py`** - CLI tool to generate large-scale benchmark data
- **`fixtures.json`** - Pre-generated benchmark fixtures (default: 50 docs, 10 queries)
- **`benchmark.md`** - Latest benchmark results and performance metrics

## Quick Start

### Generate Fixtures

```bash
# Generate fixtures with OpenAI embeddings (auto-named: data/openai_100.json)
python -m scripts.benchmark.generate_fixtures --docs 100 --add-vectors

# Generate fixtures with Gemini embeddings (auto-named: data/gemini_100.json)
python -m scripts.benchmark.generate_fixtures --docs 100 --add-vectors --embedding-provider gemini

# Generate static fixtures without embeddings (auto-named: data/static_1000.json)
python -m scripts.benchmark.generate_fixtures --docs 1000

# Custom output path
python -m scripts.benchmark.generate_fixtures --docs 500 --output custom_fixtures.json --add-vectors
```

**Available options:**
- `--docs NUM` - Number of documents to generate (default: 10000)
- `--queries NUM` - Number of search queries (default: 1000)
- `--output PATH` - Output file path (default: auto-generated in data/ folder)
- `--seed NUM` - Random seed for reproducibility (default: 42)
- `--add-vectors` - Generate vectors using embedding provider
- `--embedding-provider {openai,gemini}` - Embedding provider (default: openai)

### Run Benchmarks

The benchmark system automatically manages fixtures for you:

```bash
# Auto-generate fixture if not exists: data/openai_10.json
python -m scripts.benchmark --num-docs 10 --skip-slow

# Use specific embedding provider (auto-generates: data/gemini_50.json)
python -m scripts.benchmark --num-docs 50 --embedding-providers gemini --skip-slow

# Use existing fixture file
python -m scripts.benchmark --use-fixtures data/openai_100.json --skip-slow

# Or run directly
python scripts/benchmark/run.py --num-docs 100 --backends pgvector --skip-slow
```

**How it works:**
1. If no `--use-fixtures` specified, looks for `data/{provider}_{num_docs}.json`
2. If fixture doesn't exist, auto-generates it with real embeddings
3. If fixture exists, loads and uses pre-computed vectors (no API calls)
4. Reuses same fixture for subsequent runs with same params

**Available options:**
- `--num-docs NUM` - Number of documents per test (default: 3)
- `--backends LIST` - Comma-separated backend list (default: all)
- `--embedding-providers LIST` - Embedding providers to test (default: all)
- `--use-fixtures PATH` - Load specific fixtures file
- `--add-vectors` - Generate vectors if fixture lacks them
- `--skip-slow` - Skip slow cloud backends (astradb, milvus)
- `--skip-slow` - Skip slow backends
- `--use-fixtures PATH` - Use pre-generated fixtures

## Fixture Structure

Generated fixtures (fixtures.json):

```json
{
  "documents": [
    {
      "id": "uuid",
      "text": "realistic content...",
      "vector": [0.123, -0.456, ...],  // Optional: pre-computed embeddings
      "metadata": {
        "author": {"name": "...", "expertise": "...", "contributions": 0},
        "publication": {"date": "...", "year": 2025, "version": "1.0"},
        "stats": {"views": 0, "likes": 0, "shares": 0, "comments": 0},
        "technical": {"frameworks": [...], "languages": [...], "complexity_score": 5},
        "tags": [...],
        "rating": 5,
        "difficulty": "intermediate"
      }
    }
  ],
  "queries": [
    "search query string...",
    "another query..."
  ]
}
```

## Optimized Benchmarking with Pre-computed Vectors

### Save Token Budget - Use Pre-computed Vectors

When using `--use-fixtures` with pre-computed vectors, **no embedding API calls are made**:

```bash
# Load fixtures with pre-computed vectors (0 API calls)
python scripts/benchmark/run.py --use-fixtures scripts/benchmark/fixtures.json --skip-slow

# Results: Saves OpenAI/Gemini embedding tokens completely!
# Vector search uses pre-computed embeddings in JSON fixtures
```

### How It Works

1. **Fixtures with vectors**: Load pre-computed embeddings from JSON (no API calls)
   ```bash
   python scripts/benchmark/run.py --use-fixtures fixtures.json
   ```

2. **Fixtures without vectors**: Auto-generate static vectors (based on seed 42)
   ```bash
   python scripts/benchmark/run.py --use-fixtures fixtures.json --add-vectors
   ```

3. **Batch vector search**: Supports searching with multiple pre-computed vectors
   - No external API calls needed
   - Focuses on database performance only
   - Consistent results across runs

### Benefits

- **Cost-effective**: No OpenAI/Gemini API calls for embedding
- **Fast testing**: Static vectors generated instantly
- **Reproducible**: Fixed seed ensures consistent vectors
- **Scalable**: Test with 1000+ documents for pennies

### Example Workflow

```bash
# Step 1: Generate large fixture file once (if not exists)
python -m scripts.benchmark.generate_fixtures --docs 5000 --output large_fixtures.json

# Step 2: Generate vectors for the fixtures (one-time)
python scripts/benchmark/run.py --use-fixtures large_fixtures.json --add-vectors --num-docs 5000

# Step 3: Run benchmarks repeatedly without any API costs
python scripts/benchmark/run.py --use-fixtures large_fixtures.json --num-docs 1000 --skip-slow
python scripts/benchmark/run.py --use-fixtures large_fixtures.json --num-docs 2000 --backends pgvector
```

## Features

### Dynamic Content Generation
- Topic-specific content for: AI/ML, Cloud Computing, Database Systems, DevOps/Infrastructure
- Realistic technical language and templates
- Reproducible generation with seed-based randomization

### Complex Metadata
- Nested author information
- Publication metadata with timestamps
- Usage statistics tracking
- Technical framework/language tagging
- Difficulty levels and content types

### Comprehensive Testing
- 8 metadata filter test cases
- Single-word, multi-word, and phrase-based queries
- Support for filtering, range queries, and complex combinations
- Backend caching behavior evaluation

## Performance Notes

- Fixtures file can grow large (1GB+ for 100k+ documents)
- Recommend generating fixtures on first run: `python -m scripts.benchmark.generate_fixtures`
- Generated fixtures are deterministic (same seed = same data)
- Pre-computed vectors are stored in fixtures.json and reduce overhead by ~90%
- Tests use static vectors (no embedding API calls) for reproducibility

## Results

Latest benchmark results saved in `benchmark.md` after each test run.
