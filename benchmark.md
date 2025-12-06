# CrossVector Benchmark Results

**Generated:** 2025-12-06 16:55:35

**Documents per test:** 3

---

## Performance Summary

**Tested backends:** pgvector, chroma

**Skipped backends:** astradb, milvus ⏭️

**Test Results:** 1/2 passed, 1 ❌ failed

| Backend | Embedding | Model | Dim | Bulk Create | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|-------------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 24.01ms | 6.35ms | 7.36ms | 0.75ms | ✅ |
| chroma | openai | - | - | - | - | - | - | ❌ Failed to deserialize the JSON body into the targe... |

---

## PGVECTOR + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 24.01ms
- **Throughput:** 124.94 docs/sec

### Individual Create

- **Average Duration:** 7.06ms
- **Sample Size:** 3 documents

### Vector Search

- **Average Duration:** 6.35ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.30ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 0.51ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 7.36ms
- **Sample Size:** 3 documents

### Delete Operations

- **Duration:** 0.75ms
- **Throughput:** 4011.13 docs/sec
- **Sample Size:** 3 documents

---

## UNKNOWN + UNKNOWN Details

❌ **Error:** Failed to deserialize the JSON body into the target type: name: invalid type: null, expected a strin

## Failed Tests ❌

### CHROMA + OPENAI

**Error:** Failed to deserialize the JSON body into the target type: name: invalid type: null, expected a strin

## Notes

- Tests use specified embedding providers with their default models
- Bulk operations create documents in batches
- Search operations retrieve 100 results per query
- Times are averaged over multiple runs for stability
- Different embedding providers may have different dimensions and performance characteristics
