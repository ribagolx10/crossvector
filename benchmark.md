# CrossVector Benchmark Results

**Generated:** 2025-12-06 13:22:59

**Documents per test:** 10

---

## Performance Summary

**Tested backends:** pgvector, chroma

**Skipped backends:** astradb, milvus ⏭️

| Backend | Embedding | Model | Dim | Bulk Create | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|-------------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 1.79s | 489.52ms | 8.15ms | 1.92ms | ✅ |
| chroma | openai | text-embedding-3-small | 1536 | 5.42s | 856.61ms | 3.27s | 403.46ms | ✅ |
| pgvector | gemini | models/text-embedding-004 | 768 | 2.90s | 227.91ms | 3.53ms | 0.54ms | ✅ |
| chroma | gemini | models/text-embedding-004 | 768 | 7.10s | 754.66ms | 3.35s | 580.94ms | ✅ |

---

## PGVECTOR + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 1.79s
- **Throughput:** 5.59 docs/sec

### Individual Create

- **Average Duration:** 474.29ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 489.52ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.53ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 0.84ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 8.15ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 1.92ms
- **Throughput:** 5197.40 docs/sec
- **Sample Size:** 10 documents

---

## CHROMA + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 5.42s
- **Throughput:** 1.85 docs/sec

### Individual Create

- **Average Duration:** 1.48s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 856.61ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 312.26ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 318.14ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 3.27s
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 403.46ms
- **Throughput:** 24.79 docs/sec
- **Sample Size:** 10 documents

---

## PGVECTOR + GEMINI Details

**Embedding:** gemini - models/text-embedding-004 (768 dimensions)

### Bulk Create

- **Duration:** 2.90s
- **Throughput:** 3.45 docs/sec

### Individual Create

- **Average Duration:** 251.68ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 227.91ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.41ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 3.45ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 3.53ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 0.54ms
- **Throughput:** 18452.72 docs/sec
- **Sample Size:** 10 documents

---

## CHROMA + GEMINI Details

**Embedding:** gemini - models/text-embedding-004 (768 dimensions)

### Bulk Create

- **Duration:** 7.10s
- **Throughput:** 1.41 docs/sec

### Individual Create

- **Average Duration:** 992.68ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 754.66ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 356.69ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 313.49ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 3.35s
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 580.94ms
- **Throughput:** 17.21 docs/sec
- **Sample Size:** 10 documents

---

## Notes

- Tests use specified embedding providers with their default models
- Bulk operations create documents in batches
- Search operations retrieve 10 results per query
- Times are averaged over multiple runs for stability
- Different embedding providers may have different dimensions and performance characteristics
