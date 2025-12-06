# CrossVector Benchmark Results

**Generated:** 2025-12-06 15:25:24

**Documents per test:** 100

---

## Performance Summary

| Backend | Embedding | Model | Dim | Bulk Create | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|-------------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 2.68s | 515.47ms | 6.48ms | 1.76ms | ✅ |
| astradb | openai | text-embedding-3-small | 1536 | 32.56s | 1.09s | 875.63ms | 1.44s | ✅ |
| milvus | openai | text-embedding-3-small | 1536 | 21.24s | 1.04s | 551.36ms | 180.25ms | ✅ |
| chroma | openai | text-embedding-3-small | 1536 | 36.08s | 900.75ms | 2.51s | 521.35ms | ✅ |
| pgvector | gemini | models/gemini-embedding-001 | 1536 | 31.50s | 65.29ms | 6.14ms | 1.78ms | ✅ |
| astradb | gemini | models/gemini-embedding-001 | 1536 | 1m 2.65s | 882.48ms | 818.93ms | 1.44s | ✅ |
| milvus | gemini | models/gemini-embedding-001 | 1536 | 50.26s | 835.50ms | 572.62ms | 224.16ms | ✅ |
| chroma | gemini | models/gemini-embedding-001 | 1536 | 1m 3.39s | 628.08ms | 3.16s | 394.21ms | ✅ |

---

## PGVECTOR + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 2.68s
- **Throughput:** 37.32 docs/sec

### Individual Create

- **Average Duration:** 461.16ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 515.47ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 1.88ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 1.96ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 6.48ms
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 1.76ms
- **Throughput:** 56918.22 docs/sec
- **Sample Size:** 100 documents

---

## ASTRADB + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 32.56s
- **Throughput:** 3.07 docs/sec

### Individual Create

- **Average Duration:** 1.05s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 1.09s
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 555.51ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 588.78ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 875.63ms
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 1.44s
- **Throughput:** 69.42 docs/sec
- **Sample Size:** 100 documents

---

## MILVUS + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 21.24s
- **Throughput:** 4.71 docs/sec

### Individual Create

- **Average Duration:** 1.15s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 1.04s
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 537.32ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 519.36ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 551.36ms
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 180.25ms
- **Throughput:** 554.79 docs/sec
- **Sample Size:** 100 documents

---

## CHROMA + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 36.08s
- **Throughput:** 2.77 docs/sec

### Individual Create

- **Average Duration:** 1.22s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 900.75ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 322.43ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 326.25ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 2.51s
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 521.35ms
- **Throughput:** 191.81 docs/sec
- **Sample Size:** 100 documents

---

## PGVECTOR + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Bulk Create

- **Duration:** 31.50s
- **Throughput:** 3.17 docs/sec

### Individual Create

- **Average Duration:** 90.84ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 65.29ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 2.50ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 1.96ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 6.14ms
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 1.78ms
- **Throughput:** 56178.73 docs/sec
- **Sample Size:** 100 documents

---

## ASTRADB + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Bulk Create

- **Duration:** 1m 2.65s
- **Throughput:** 1.60 docs/sec

### Individual Create

- **Average Duration:** 898.10ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 882.48ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 561.61ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 521.13ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 818.93ms
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 1.44s
- **Throughput:** 69.33 docs/sec
- **Sample Size:** 100 documents

---

## MILVUS + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Bulk Create

- **Duration:** 50.26s
- **Throughput:** 1.99 docs/sec

### Individual Create

- **Average Duration:** 1.06s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 835.50ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 516.13ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 514.58ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 572.62ms
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 224.16ms
- **Throughput:** 446.10 docs/sec
- **Sample Size:** 100 documents

---

## CHROMA + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Bulk Create

- **Duration:** 1m 3.39s
- **Throughput:** 1.58 docs/sec

### Individual Create

- **Average Duration:** 1.21s
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 628.08ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 345.02ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 313.67ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 3.16s
- **Sample Size:** 100 documents

### Delete Operations

- **Duration:** 394.21ms
- **Throughput:** 253.67 docs/sec
- **Sample Size:** 100 documents

---

## Notes

- Tests use specified embedding providers with their default models
- Bulk operations create documents in batches
- Search operations retrieve 10 results per query
- Times are averaged over multiple runs for stability
- Different embedding providers may have different dimensions and performance characteristics
