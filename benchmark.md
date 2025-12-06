# CrossVector Benchmark Results

**Generated:** 2025-12-06 13:30:02

**Documents per test:** 10

---

## Performance Summary

| Backend | Embedding | Model | Dim | Bulk Create | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|-------------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 1.06s | 532.51ms | 8.10ms | 0.59ms | ✅ |
| astradb | openai | text-embedding-3-small | 1536 | 4.47s | 1.02s | 795.77ms | 264.58ms | ✅ |
| milvus | openai | text-embedding-3-small | 1536 | 4.28s | 944.09ms | 544.77ms | 171.25ms | ✅ |
| chroma | openai | text-embedding-3-small | 1536 | 6.59s | 849.47ms | 2.33s | 406.67ms | ✅ |
| pgvector | gemini | models/text-embedding-004 | 768 | 3.04s | 234.79ms | 3.33ms | 0.83ms | ✅ |
| astradb | gemini | models/text-embedding-004 | 768 | 5.93s | 798.04ms | 809.51ms | 305.70ms | ✅ |
| milvus | gemini | models/text-embedding-004 | 768 | 5.78s | 743.93ms | 557.08ms | 171.39ms | ✅ |
| chroma | gemini | models/text-embedding-004 | 768 | 6.67s | 584.03ms | 1.94s | 402.35ms | ✅ |

---

## PGVECTOR + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 1.06s
- **Throughput:** 9.42 docs/sec

### Individual Create

- **Average Duration:** 476.04ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 532.51ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.64ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 0.83ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 8.10ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 0.59ms
- **Throughput:** 17091.70 docs/sec
- **Sample Size:** 10 documents

---

## ASTRADB + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 4.47s
- **Throughput:** 2.24 docs/sec

### Individual Create

- **Average Duration:** 1.02s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 1.02s
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 528.66ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 404.08ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 795.77ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 264.58ms
- **Throughput:** 37.80 docs/sec
- **Sample Size:** 10 documents

---

## MILVUS + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 4.28s
- **Throughput:** 2.33 docs/sec

### Individual Create

- **Average Duration:** 1.16s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 944.09ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 515.74ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 533.24ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 544.77ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 171.25ms
- **Throughput:** 58.39 docs/sec
- **Sample Size:** 10 documents

---

## CHROMA + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 6.59s
- **Throughput:** 1.52 docs/sec

### Individual Create

- **Average Duration:** 1.38s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 849.47ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 309.60ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 306.87ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 2.33s
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 406.67ms
- **Throughput:** 24.59 docs/sec
- **Sample Size:** 10 documents

---

## PGVECTOR + GEMINI Details

**Embedding:** gemini - models/text-embedding-004 (768 dimensions)

### Bulk Create

- **Duration:** 3.04s
- **Throughput:** 3.29 docs/sec

### Individual Create

- **Average Duration:** 246.80ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 234.79ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.51ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 0.51ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 3.33ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 0.83ms
- **Throughput:** 12035.31 docs/sec
- **Sample Size:** 10 documents

---

## ASTRADB + GEMINI Details

**Embedding:** gemini - models/text-embedding-004 (768 dimensions)

### Bulk Create

- **Duration:** 5.93s
- **Throughput:** 1.69 docs/sec

### Individual Create

- **Average Duration:** 819.77ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 798.04ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 531.29ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 389.24ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 809.51ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 305.70ms
- **Throughput:** 32.71 docs/sec
- **Sample Size:** 10 documents

---

## MILVUS + GEMINI Details

**Embedding:** gemini - models/text-embedding-004 (768 dimensions)

### Bulk Create

- **Duration:** 5.78s
- **Throughput:** 1.73 docs/sec

### Individual Create

- **Average Duration:** 932.02ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 743.93ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 511.85ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 514.27ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 557.08ms
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 171.39ms
- **Throughput:** 58.34 docs/sec
- **Sample Size:** 10 documents

---

## CHROMA + GEMINI Details

**Embedding:** gemini - models/text-embedding-004 (768 dimensions)

### Bulk Create

- **Duration:** 6.67s
- **Throughput:** 1.50 docs/sec

### Individual Create

- **Average Duration:** 1.03s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 584.03ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 317.23ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 491.17ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 1.94s
- **Sample Size:** 10 documents

### Delete Operations

- **Duration:** 402.35ms
- **Throughput:** 24.85 docs/sec
- **Sample Size:** 10 documents

---

## Notes

- Tests use specified embedding providers with their default models
- Bulk operations create documents in batches
- Search operations retrieve 10 results per query
- Times are averaged over multiple runs for stability
- Different embedding providers may have different dimensions and performance characteristics
