# CrossVector Benchmark Results

**Generated:** 2025-12-06 16:57:17

**Documents per test:** 3

---

## Performance Summary

**Tested backends:** pgvector, astradb, milvus, chroma

**Test Results:** 2/8 passed, 6 ❌ failed

| Backend | Embedding | Model | Dim | Bulk Create | Search (avg) | Update (avg) | Delete (batch) | Status |
|---------|-----------|-------|-----|-------------|--------------|--------------|----------------|--------|
| pgvector | openai | text-embedding-3-small | 1536 | 23.35ms | 5.31ms | 5.82ms | 0.94ms | ✅ |
| astradb | openai | - | - | - | - | - | - | ❌ Failed to initialize AstraDB collection: The used ... |
| milvus | openai | - | - | - | - | - | - | ❌ <ParamError: (code=1, message=`collection_name` va... |
| chroma | openai | - | - | - | - | - | - | ❌ Failed to deserialize the JSON body into the targe... |
| pgvector | gemini | models/gemini-embedding-001 | 1536 | 33.76ms | 5.61ms | 6.31ms | 1.25ms | ✅ |
| astradb | gemini | - | - | - | - | - | - | ❌ Failed to initialize AstraDB collection: The used ... |
| milvus | gemini | - | - | - | - | - | - | ❌ <ParamError: (code=1, message=`collection_name` va... |
| chroma | gemini | - | - | - | - | - | - | ❌ Failed to deserialize the JSON body into the targe... |

---

## PGVECTOR + OPENAI Details

**Embedding:** openai - text-embedding-3-small (1536 dimensions)

### Bulk Create

- **Duration:** 23.35ms
- **Throughput:** 128.46 docs/sec

### Individual Create

- **Average Duration:** 9.10ms
- **Sample Size:** 3 documents

### Vector Search

- **Average Duration:** 5.31ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.27ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 0.38ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 5.82ms
- **Sample Size:** 3 documents

### Delete Operations

- **Duration:** 0.94ms
- **Throughput:** 3205.02 docs/sec
- **Sample Size:** 3 documents

---

## UNKNOWN + UNKNOWN Details

❌ **Error:** Failed to initialize AstraDB collection: The used schema name is not supported: The command attempte

## UNKNOWN + UNKNOWN Details

❌ **Error:** <ParamError: (code=1, message=`collection_name` value None is illegal)>

## UNKNOWN + UNKNOWN Details

❌ **Error:** Failed to deserialize the JSON body into the target type: name: invalid type: null, expected a strin

## PGVECTOR + GEMINI Details

**Embedding:** gemini - models/gemini-embedding-001 (1536 dimensions)

### Bulk Create

- **Duration:** 33.76ms
- **Throughput:** 88.87 docs/sec

### Individual Create

- **Average Duration:** 7.26ms
- **Sample Size:** 3 documents

### Vector Search

- **Average Duration:** 5.61ms
- **Queries Tested:** 10

### Metadata-Only Search

- **Average Duration:** 0.33ms
- **Queries Tested:** 10

### Query DSL Operators (Q Objects)

- **Average Duration:** 0.37ms
- **Operators Tested:** 10/10
- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or

### Update Operations

- **Average Duration:** 6.31ms
- **Sample Size:** 3 documents

### Delete Operations

- **Duration:** 1.25ms
- **Throughput:** 2392.19 docs/sec
- **Sample Size:** 3 documents

---

## UNKNOWN + UNKNOWN Details

❌ **Error:** Failed to initialize AstraDB collection: The used schema name is not supported: The command attempte

## UNKNOWN + UNKNOWN Details

❌ **Error:** <ParamError: (code=1, message=`collection_name` value None is illegal)>

## UNKNOWN + UNKNOWN Details

❌ **Error:** Failed to deserialize the JSON body into the target type: name: invalid type: null, expected a strin

## Failed Tests ❌

### ASTRADB + OPENAI

**Error:** Failed to initialize AstraDB collection: The used schema name is not supported: The command attempte

### MILVUS + OPENAI

**Error:** <ParamError: (code=1, message=`collection_name` value None is illegal)>

### CHROMA + OPENAI

**Error:** Failed to deserialize the JSON body into the target type: name: invalid type: null, expected a strin

### ASTRADB + GEMINI

**Error:** Failed to initialize AstraDB collection: The used schema name is not supported: The command attempte

### MILVUS + GEMINI

**Error:** <ParamError: (code=1, message=`collection_name` value None is illegal)>

### CHROMA + GEMINI

**Error:** Failed to deserialize the JSON body into the target type: name: invalid type: null, expected a strin

## Notes

- Tests use specified embedding providers with their default models
- Bulk operations create documents in batches
- Search operations retrieve 100 results per query
- Times are averaged over multiple runs for stability
- Different embedding providers may have different dimensions and performance characteristics
