# Benchmarking Guide

This guide explains how to use CrossVector's benchmarking tool to measure and compare performance across different database backends and embedding providers.

## Overview

The benchmark tool (`scripts/benchmark.py`) provides comprehensive performance testing for:
- **4 Database Backends**: PgVector, AstraDB, Milvus, ChromaDB
- **2 Embedding Providers**: OpenAI, Gemini
- **7 Operation Types**: Bulk create, individual create, vector search, metadata search, Query DSL operators, updates, deletes

## Quick Start

### Basic Usage

```bash
# Test all backends with both embeddings (10 documents)
python scripts/benchmark.py --num-docs 10

# Full benchmark with 1000 documents
python scripts/benchmark.py

# Test specific configuration
python scripts/benchmark.py --backends pgvector milvus --embedding-providers openai --num-docs 100
```

### Command Line Options

```bash
python scripts/benchmark.py [OPTIONS]

Options:
  --num-docs INT                    Number of documents to test (default: 1000)
  --backends NAME [NAME ...]        Specific backends: pgvector, astradb, milvus, chroma
  --embedding-providers NAME        Embedding providers: openai, gemini
  --skip-slow                       Skip slow cloud backends (astradb, milvus) for faster testing
  --search-limit INT                Number of results to return in search operations (default: 100)
  --collection-name STR             Custom collection name (default: auto-generate with UUID8)
  --timeout INT                     Timeout per backend test in seconds (default: 60)
  --output PATH                     Output file path (default: benchmark.md)
  --use-fixtures PATH               Path to pre-generated fixtures JSON file
  --add-vectors                     Generate and add vectors to fixture documents
```

## What Gets Measured

### 1. Upsert Performance
Measures throughput for batch document upsert with automatic embedding generation.

**Metrics:**
- Duration (seconds)
- Throughput (docs/sec)

### 2. Individual Create Performance
Tests single document creation with embedding generation.

**Metrics:**
- Average duration per document

### 3. Vector Search Performance
Semantic similarity search using embedded queries.

**Metrics:**
- Average query duration (10 queries tested)
- Queries per second

### 4. Metadata-Only Search
Filtering without vector similarity (if supported by backend).

**Metrics:**
- Average query duration
- Support status

### 5. Query DSL Operators
Tests all 10 Query DSL operators:
- Comparison: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`
- Array: `in`, `nin`
- Logical: `and` (`&`), `or` (`|`)

**Metrics:**
- Average operator query duration
- Number of operators successfully tested

### 6. Update Operations
Document update performance.

**Metrics:**
- Average update duration (100 updates tested)

### 7. Delete Operations
Batch deletion throughput.

**Metrics:**
- Duration
- Throughput (docs/sec)

## Prerequisites

### Required Environment Variables

**Embedding Providers** (at least one required):
```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Gemini
export GEMINI_API_KEY=AI...
```

**Database Backends** (optional, will skip if not configured):

```bash
# PgVector
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=5432
export PGVECTOR_DATABASE=vector_db
export PGVECTOR_USER=postgres
export PGVECTOR_PASSWORD=postgres
# Or use connection string:
export PGVECTOR_CONNECTION_STRING=postgresql://user:pass@host:port/db

# AstraDB
export ASTRADB_API_ENDPOINT=https://...apps.astra.datastax.com
export ASTRADB_APPLICATION_TOKEN=AstraCS:...

# Milvus
export MILVUS_API_ENDPOINT=https://...
export MILVUS_API_TOKEN=...

# ChromaDB (optional for cloud)
export CHROMA_HOST=api.trychroma.com
export CHROMA_API_KEY=ck-...
export CHROMA_TENANT=...
export CHROMA_DATABASE=Test
```

## Running Benchmarks

### Recommended Workflow

#### Step 1: Quick Verification (1-2 minutes)

Test that everything is configured correctly:

```bash
python scripts/benchmark.py --num-docs 1 --backends pgvector --embedding-providers openai
```

#### Step 2: Fast Comparison (5-10 minutes)

Compare all backends with small dataset:

```bash
python scripts/benchmark.py --num-docs 10
```

This runs **8 combinations** (4 backends × 2 embeddings) with 10 documents each.

#### Step 3: Production Benchmark (30-60 minutes)

Full performance test with larger dataset:

```bash
python scripts/benchmark.py --num-docs 1000 --output benchmark_full.md
```

**Note:** This will:
- Make ~1000+ API calls to embedding providers
- Take 30-60 minutes depending on network and API rate limits
- Cost approximately $0.10-0.20 in API fees

### Targeted Benchmarks

#### Test Specific Backend

```bash
# Only PgVector with both embeddings
python scripts/benchmark.py --backends pgvector --num-docs 100
```

#### Test Specific Embedding

```bash
# All backends with only OpenAI
python scripts/benchmark.py --embedding-providers openai --num-docs 100
```

#### Compare Two Backends

```bash
# PgVector vs Milvus
python scripts/benchmark.py --backends pgvector milvus --num-docs 100
```

## Understanding Results

### Output Format

Results are saved as a markdown file (default: `benchmark.md`) with:

1. **Performance Summary Table** - Quick comparison across all combinations
2. **Detailed Results** - Individual metrics for each backend+embedding pair
3. **Notes** - Configuration and methodology details

### Example Output

```markdown
## Performance Summary

| Backend | Embedding | Model | Dim | Upsert | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|--------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 7.06s | 21.26ms | 6.21ms | 22.63ms | OK |
| astradb | openai | text-embedding-3-small | 1536 | 18.89s | 23.86s | 1.11s | 15.15s | OK |
| milvus | openai | text-embedding-3-small | 1536 | 7.94s | 654.43ms | 569.52ms | 2.17s | OK |
| chroma | openai | text-embedding-3-small | 1536 | 17.08s | 654.76ms | 1.23s | 4.73s | OK |
| pgvector | gemini | models/gemini-embedding-001 | 1536 | 6.65s | 18.72ms | 6.40ms | 20.25ms | OK |
| astradb | gemini | models/gemini-embedding-001 | 1536 | 11.25s | 6.71s | 903.37ms | 15.05s | OK |
| milvus | gemini | models/gemini-embedding-001 | 1536 | 6.14s | 571.90ms | 561.38ms | 1.91s | OK |
| chroma | gemini | models/gemini-embedding-001 | 1536 | 18.93s | 417.28ms | 1.24s | 4.63s | OK |
```

### Interpreting Metrics

**Bulk Create:**
- Lower duration = better
- Higher throughput (docs/sec) = better
- Gemini typically slower due to API rate limits

**Search:**
- Lower average duration = better
- Milvus typically fastest for vector search
- Gemini often faster than OpenAI for search (smaller vectors)

**Updates & Deletes:**
- Lower duration = better
- PgVector typically fast for updates due to SQL efficiency

**Query DSL Operators:**
- Should test 10/10 operators successfully
- Duration typically <1ms for metadata operations

## Performance Tips

### For Better Results

1. **Stable Network**: Run benchmarks on stable network connection
2. **Isolated Environment**: Avoid running other heavy processes
3. **Warm-up**: First run may be slower due to cold starts
4. **Multiple Runs**: Run 2-3 times and use median values for important decisions

### API Rate Limits

Be aware of rate limits:
- **OpenAI**: 3,500 requests/min (Tier 2)
- **Gemini**: 1,500 requests/min (free tier)

For large benchmarks (--num-docs 1000+), the tool will automatically pace requests.

#### Avoiding quota errors

- If you see Gemini `RESOURCE_EXHAUSTED`, rerun with `--embedding-providers openai` or reduce `--num-docs`.
- To avoid embedding API calls entirely, provide fixtures with vectors (e.g., `--use-fixtures scripts/benchmark/data/openai_3.json --add-vectors`) or let the tool generate static vectors when configured.
- Keep long runs to a single backend to reduce concurrent calls (e.g., `--backends pgvector`).

## Comparing Before/After Changes

When optimizing performance:

```bash
# Before changes
python scripts/benchmark.py --num-docs 100 --output benchmark_before.md

# Make your changes to code

# After changes
python scripts/benchmark.py --num-docs 100 --output benchmark_after.md

# Compare the two markdown files
diff benchmark_before.md benchmark_after.md
```

Or use a markdown diff tool for better visualization.

## Troubleshooting

### Backend Not Available

If you see:
```
AstraDB not available: Missing ASTRADB_API_ENDPOINT
```

Solution: Set the required environment variables or the backend will be skipped.

### Embedding API Errors

If you see rate limit errors:
```
bulk_create failed: Rate limit exceeded
```

Solutions:
- Reduce `--num-docs`
- Wait and retry
- Check API quota/billing

### Slow Performance

If benchmarks are unexpectedly slow:
- Check network latency to database
- Verify database is not under load
- Check API rate limits aren't being hit
- Try reducing `--num-docs` for faster iterations

## Advanced Usage

### Custom Test Data

Modify `scripts/benchmark.py` to use custom test data:

```python
# In generate_documents() function
SAMPLE_TEXTS = [
    "Your custom text 1",
    "Your custom text 2",
    # ...
]
```

### Adding Custom Metrics

Extend `benchmark_backend()` method to add custom metrics:

```python
# In BenchmarkRunner.benchmark_backend()
# After existing benchmarks, add:

# Custom metric
print("\nCustom Metric...")
duration, result = benchmark_operation("custom", lambda: engine.custom_operation())
results["custom_metric"] = {"duration": duration}
```

## Cost Estimation

Approximate costs for running benchmarks:

| Documents | OpenAI Cost | Gemini Cost | Total Time |
|-----------|-------------|-------------|------------|
| 10        | $0.001      | Free        | 2-5 min    |
| 100       | $0.01       | Free        | 10-15 min  |
| 1000      | $0.10       | Free        | 30-60 min  |

**Note:** Costs are approximate and depend on:
- Embedding model used
- Document text length
- Current API pricing

For Gemini, the free tier typically covers benchmarking needs.

## Best Practices

1. **Start Small**: Always test with `--num-docs 10` first
2. **Document Results**: Save benchmark outputs with timestamps
3. **Consistent Environment**: Run on same machine/network for comparisons
4. **Version Control**: Commit benchmark results with code changes
5. **CI/CD Integration**: Consider running small benchmarks in CI for regression testing

## Examples

### Example 1: Quick Backend Comparison

```bash
# Compare PgVector and Milvus with 50 docs
python scripts/benchmark.py --backends pgvector milvus --num-docs 50
```

### Example 2: Embedding Provider Comparison

```bash
# Test which embedding is faster for your use case
python scripts/benchmark.py --backends pgvector --num-docs 200
```

### Example 3: Pre-Release Validation

```bash
# Full benchmark before major release
python scripts/benchmark.py --num-docs 1000 --output release_v1.0_benchmark.md
```

### Example 4: Query Performance Focus

```bash
# Test with more documents to stress search performance
python scripts/benchmark.py --backends milvus --num-docs 5000
```

## Contributing

Found a performance issue or want to add a new benchmark metric? See [Contributing Guide](contributing.md#benchmarking).
