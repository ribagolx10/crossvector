# CrossVector Benchmark Results

**Generated:** 2025-12-06 22:12:50

**Documents per test:** 1000

---

## Performance Summary

**Tested backends:** pgvector, astradb, milvus, chroma

**Test Results:** 8/8 passed

| Backend | Embedding | Model | Dim | Upsert | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|--------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 7.06s | 21.26ms | 6.21ms | 22.63ms | ✅ |
| astradb | openai | text-embedding-3-small | 1536 | 18.89s | 23.86s | 1.11s | 15.15s | ✅ |
| milvus | openai | text-embedding-3-small | 1536 | 7.94s | 654.43ms | 569.52ms | 2.17s | ✅ |
| chroma | openai | text-embedding-3-small | 1536 | 17.08s | 654.76ms | 1.23s | 4.73s | ✅ |
| pgvector | gemini | models/gemini-embedding-001 | 1536 | 6.65s | 18.72ms | 6.40ms | 20.25ms | ✅ |
| astradb | gemini | models/gemini-embedding-001 | 1536 | 11.25s | 6.71s | 903.37ms | 15.05s | ✅ |
| milvus | gemini | models/gemini-embedding-001 | 1536 | 6.14s | 571.90ms | 561.38ms | 1.91s | ✅ |
| chroma | gemini | models/gemini-embedding-001 | 1536 | 18.93s | 417.28ms | 1.24s | 4.63s | ✅ |

---

## PGVECTOR + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Upsert

- **Duration:** 7.06s
- **Throughput:** 141.70 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 7.47ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 21.26ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.50ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 7.19ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 6.21ms
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 22.63ms
- **Throughput:** 44193.37 docs/sec
- **Sample Size:** 1000 documents

---

## ASTRADB + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Upsert

- **Duration:** 18.89s
- **Throughput:** 52.93 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 777.52ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 23.86s
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 303.96ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 1.28s
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 1.11s
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 15.15s
- **Throughput:** 66.01 docs/sec
- **Sample Size:** 1000 documents

---

## MILVUS + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Upsert

- **Duration:** 7.94s
- **Throughput:** 125.92 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 716.15ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 654.43ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 519.66ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 536.89ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 569.52ms
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 2.17s
- **Throughput:** 459.90 docs/sec
- **Sample Size:** 1000 documents

---

## CHROMA + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Upsert

- **Duration:** 17.08s
- **Throughput:** 58.54 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 786.42ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 654.76ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 326.76ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 652.08ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 1.23s
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 4.73s
- **Throughput:** 211.23 docs/sec
- **Sample Size:** 1000 documents

---

## PGVECTOR + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Upsert

- **Duration:** 6.65s
- **Throughput:** 150.40 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 7.61ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 18.72ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 10.67ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 10.68ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 6.40ms
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 20.25ms
- **Throughput:** 49383.68 docs/sec
- **Sample Size:** 1000 documents

---

## ASTRADB + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Upsert

- **Duration:** 11.25s
- **Throughput:** 88.86 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 597.59ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 6.71s
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 1.78s
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 1.78s
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 903.37ms
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 15.05s
- **Throughput:** 66.44 docs/sec
- **Sample Size:** 1000 documents

---

## MILVUS + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Upsert

- **Duration:** 6.14s
- **Throughput:** 162.86 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 717.84ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 571.90ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 541.96ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 567.30ms
- **Operators Tested:** 4/4
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 561.38ms
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 1.91s
- **Throughput:** 522.37 docs/sec
- **Sample Size:** 1000 documents

---

## CHROMA + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Upsert

- **Duration:** 18.93s
- **Throughput:** 52.81 docs/sec
- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)

### Individual Create

- **Average Duration:** 790.81ms
- **Sample Size:** 10 documents

### Vector Search

- **Average Duration:** 417.28ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 347.51ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 347.05ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 1.24s
- **Sample Size:** 1000 documents

### Delete Operations

- **Duration:** 4.63s
- **Throughput:** 216.00 docs/sec
- **Sample Size:** 1000 documents

---

## Notes

- Tests use specified embedding providers with their default models
- Upsert operations create new documents or update existing ones (can be run repeatedly)
- Search operations retrieve 100 results per query
- Times are averaged over multiple runs for stability
- Different embedding providers may have different dimensions and performance characteristics
