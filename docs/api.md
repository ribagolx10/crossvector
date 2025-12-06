# API Reference

Complete API reference for CrossVector.

## VectorEngine

The main class for interacting with vector databases.

### Constructor

```python
VectorEngine(
    db: VectorDBAdapter,
    embedding: EmbeddingAdapter,
    collection_name: str = "vector_db",
    store_text: bool = False
)
```

**Parameters:**

- `db` (VectorDBAdapter): Database adapter instance
- `embedding` (EmbeddingAdapter): Embedding adapter instance
- `collection_name` (str): Name of the collection to use
- `store_text` (bool): Whether to store original text with vectors

**Example:**

```python
from crossvector import VectorEngine
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.dbs.pgvector import PgVectorAdapter

engine = VectorEngine(
    db=PgVectorAdapter(),
    embedding=GeminiEmbeddingAdapter(),
    collection_name="documents",
    store_text=True
)
```

### Properties

#### `engine.db`

Access the database adapter instance.

```python
db = engine.db
print(db.__class__.__name__)  # "PgVectorAdapter"
```

#### `engine.adapter`

Alias for `engine.db`.

#### `engine.embedding`

Access the embedding adapter instance.

```python
emb = engine.embedding
print(emb.model_name)  # "models/text-embedding-004"
```

#### `engine.supports_metadata_only`

Check if the backend supports metadata-only search (without vector).

```python
if engine.supports_metadata_only:
    results = engine.search(query=None, where={"status": {"$eq": "active"}})
```

---

## Document Operations

### create()

Create a single document.

```python
create(
    doc: str | Dict[str, Any] | VectorDocument = None,
    *,
    text: str = None,
    vector: List[float] = None,
    metadata: Dict[str, Any] = None,
    **kwargs
) -> VectorDocument
```

**Parameters:**

- `doc`: Document input (str, dict, or VectorDocument)
- `text`: Text content (overrides doc.text)
- `vector`: Pre-computed vector (skips embedding)
- `metadata`: Metadata dict
- `**kwargs`: Additional metadata fields or id/pk

**Returns:** `VectorDocument` instance

**Raises:**

- `DocumentExistsError`: If document with ID already exists
- `InvalidFieldError`: If vector dimension mismatch

**Examples:**

```python
# String input
doc = engine.create("My document text")

# Dict input
doc = engine.create({"text": "Content", "metadata": {"key": "value"}})

# With metadata as kwargs
doc = engine.create(text="Content", category="tech", priority=1)

# With pre-computed vector
doc = engine.create(text="Content", vector=[0.1, 0.2, ...])

# VectorDocument instance
from crossvector import VectorDocument
doc = engine.create(VectorDocument(
    id="custom-id",
    text="Content",
    metadata={"key": "value"}
))
```

---

### bulk_create()

Create multiple documents in batch.

```python
bulk_create(
    docs: List[str | Dict | VectorDocument],
    batch_size: int = None,
    ignore_conflicts: bool = False,
    update_conflicts: bool = False,
    update_fields: List[str] = None
) -> List[VectorDocument]
```

**Parameters:**

- `docs`: List of documents to create
- `batch_size`: Number of documents per batch (backend-specific default)
- `ignore_conflicts`: Skip documents with conflicting IDs
- `update_conflicts`: Update existing documents on ID conflict
- `update_fields`: Fields to update on conflict (None = all fields)

**Returns:** List of created `VectorDocument` instances

**Examples:**

```python
# Simple batch
docs = [
    "Document 1",
    "Document 2",
    "Document 3",
]
created = engine.bulk_create(docs)

# With metadata
docs = [
    {"text": f"Doc {i}", "metadata": {"index": i}}
    for i in range(100)
]
created = engine.bulk_create(docs, batch_size=50)

# With conflict handling
docs = [
    {"id": "doc-1", "text": "First"},
    {"id": "doc-2", "text": "Second"},
]
created = engine.bulk_create(docs, update_conflicts=True)
```

---

### update()

Update a single document.

```python
update(
    doc: str | Dict[str, Any] | VectorDocument,
    **kwargs
) -> VectorDocument
```

**Parameters:**

- `doc`: Document with ID to update
- `**kwargs`: Fields to update

**Returns:** Updated `VectorDocument`

**Raises:**

- `DocumentNotFoundError`: If document doesn't exist
- `MissingFieldError`: If ID is missing

**Examples:**

```python
# Update VectorDocument
doc = engine.get("doc-id")
doc.metadata["updated"] = True
updated = engine.update(doc)

# Update with dict
updated = engine.update({"id": "doc-id", "text": "New text"})

# Update specific fields
updated = engine.update(doc, text="New text", metadata={"key": "new"})
```

---

### bulk_update()

Update multiple documents in batch.

```python
bulk_update(
    docs: List[Dict | VectorDocument],
    batch_size: int = None,
    update_fields: List[str] = None
) -> List[VectorDocument]
```

**Parameters:**

- `docs`: List of documents to update (must include ID)
- `batch_size`: Number of documents per batch
- `update_fields`: Specific fields to update (None = all)

**Returns:** List of updated `VectorDocument` instances

**Examples:**

```python
updates = [
    {"id": "doc-1", "metadata": {"status": "published"}},
    {"id": "doc-2", "metadata": {"status": "published"}},
]
updated = engine.bulk_update(updates)
```

---

### upsert()

Insert or update documents (upsert operation).

```python
upsert(
    docs: List[str | Dict | VectorDocument],
    batch_size: int = None
) -> List[VectorDocument]
```

**Parameters:**

- `docs`: List of documents to upsert
- `batch_size`: Number of documents per batch

**Returns:** List of upserted `VectorDocument` instances

**Examples:**

```python
docs = [
    {"id": "doc-1", "text": "Updated or new"},
    {"id": "doc-2", "text": "Updated or new"},
]
upserted = engine.upsert(docs)
```

---

### get()

Retrieve a single document by ID or metadata filter.

```python
get(*args, **kwargs) -> VectorDocument
```

**Parameters:**

- `*args`: Optional positional ID
- `**kwargs`: ID (as `id`, `pk`, or `_id`) or metadata filters

**Returns:** `VectorDocument` instance

**Raises:**

- `DoesNotExist`: If no document matches
- `MultipleObjectsReturned`: If multiple documents match
- `MissingFieldError`: If no ID or filters provided

**Examples:**

```python
# By ID (positional)
doc = engine.get("doc-id")

# By ID (keyword)
doc = engine.get(id="doc-id")
doc = engine.get(pk="doc-id")

# By metadata (must return exactly one)
doc = engine.get(category="tech", status="active")
```

---

### get_or_create()

Get existing document or create if it doesn't exist.

```python
get_or_create(
    doc: str | Dict | VectorDocument = None,
    *,
    text: str = None,
    metadata: Dict[str, Any] = None,
    **kwargs
) -> Tuple[VectorDocument, bool]
```

**Parameters:**

- `doc`: Document to get or create
- `text`: Text content
- `metadata`: Metadata dict
- `**kwargs`: Additional fields or ID

**Returns:** Tuple of `(VectorDocument, created: bool)`

**Examples:**

```python
# Get or create by metadata
doc, created = engine.get_or_create(
    text="Content",
    metadata={"key": "unique-value"}
)
if created:
    print("Created new document")
else:
    print("Document already exists")

# Get or create by ID
doc, created = engine.get_or_create(
    id="doc-123",
    text="Content"
)
```

---

### update_or_create()

Update existing document or create if it doesn't exist.

```python
update_or_create(
    lookup: Dict[str, Any],
    *,
    text: str = None,
    metadata: Dict[str, Any] = None,
    defaults: Dict[str, Any] = None,
    create_defaults: Dict[str, Any] = None,
    **kwargs
) -> Tuple[VectorDocument, bool]
```

**Parameters:**

- `lookup`: Dict with ID or metadata to find document
- `text`: Text to set
- `metadata`: Metadata to set
- `defaults`: Fields to use for both update and create
- `create_defaults`: Fields to use only when creating
- `**kwargs`: Additional fields

**Returns:** Tuple of `(VectorDocument, created: bool)`

**Examples:**

```python
# Update or create by ID
doc, created = engine.update_or_create(
    {"id": "doc-123"},
    text="Updated or new content",
    defaults={"metadata": {"updated": True}}
)

# Update or create by metadata
doc, created = engine.update_or_create(
    {"status": "draft", "author": "user123"},
    text="Content",
    defaults={"metadata": {"reviewed": False}},
    create_defaults={"metadata": {"created_by": "system"}}
)
```

---

### delete()

Delete documents by ID.

```python
delete(ids: str | List[str]) -> int
```

**Parameters:**

- `ids`: Single ID or list of IDs to delete

**Returns:** Number of documents deleted

**Examples:**

```python
# Delete single
deleted = engine.delete("doc-id")

# Delete multiple
deleted = engine.delete("doc-1", "doc-2", "doc-3")
```

---

## Search Operations

### search()

Perform vector similarity search with optional filters.

```python
search(
    query: str | List[float] = None,
    *,
    where: Dict[str, Any] | Q = None,
    limit: int = None,
    offset: int = 0,
    fields: Set[str] = None
) -> List[VectorDocument]
```

**Parameters:**

- `query`: Search query (str for text, List[float] for vector, None for metadata-only)
- `where`: Metadata filters (dict or Q object)
- `limit`: Maximum results to return (default: VECTOR_SEARCH_LIMIT)
- `offset`: Number of results to skip
- `fields`: Set of fields to return

**Returns:** List of `VectorDocument` instances ordered by similarity

**Raises:**

- `SearchError`: If neither query nor where filter provided
- `InvalidFieldError`: If unsupported operators used

**Examples:**

```python
# Simple text search
results = engine.search("python tutorials", limit=10)

# Vector search
vector = engine.embedding.get_embeddings(["query"])[0]
results = engine.search(vector, limit=5)

# Search with filters
from crossvector.querydsl.q import Q
results = engine.search(
    "machine learning",
    where=Q(category="tech") & Q(level="beginner"),
    limit=20
)

# Metadata-only search (no vector)
results = engine.search(
    query=None,
    where={"status": {"$eq": "published"}},
    limit=50
)

# With pagination
results = engine.search("query", limit=10, offset=20)

# Specific fields only
results = engine.search("query", fields={"text", "metadata"})
```

---

### count()

Count total documents in collection.

```python
count() -> int
```

**Returns:** Total document count

**Example:**

```python
total = engine.count()
print(f"Total documents: {total}")
```

---

## Collection Operations

### drop_collection()

Delete the entire collection.

```python
drop_collection(collection_name: str) -> bool
```

**Parameters:**

- `collection_name`: Name of collection to drop

**Returns:** True if successful

**Warning:** This permanently deletes all documents in the collection.

**Example:**

```python
engine.drop_collection("old_collection")
```

---

### clear_collection()

Delete all documents from the collection (keep collection structure).

```python
clear_collection() -> int
```

**Returns:** Number of documents deleted

**Warning:** This permanently deletes all documents.

**Example:**

```python
deleted = engine.clear_collection()
print(f"Deleted {deleted} documents")
```

---

## Query DSL

### Q Objects

Composable query filters.

```python
from crossvector.querydsl.q import Q

# Simple equality
Q(category="tech")

# Comparison operators
Q(score__gte=0.8)
Q(price__lt=100)
Q(age__lte=65)

# IN / NOT IN
Q(status__in=["active", "pending"])
Q(role__nin=["guest"])

# Boolean combinations
Q(category="tech") & Q(level="beginner")  # AND
Q(featured=True) | Q(score__gte=0.9)      # OR
~Q(archived=True)                          # NOT

# Nested metadata
Q(user__role="admin")
Q(info__verified=True)
```

**Supported Operators:**

| Lookup | Operator | Example |
|--------|----------|---------|
| `eq` (or no suffix) | `$eq` | `Q(status="active")` or `Q(status__eq="active")` |
| `ne` | `$ne` | `Q(status__ne="inactive")` |
| `gt` | `$gt` | `Q(score__gt=0.5)` |
| `gte` | `$gte` | `Q(score__gte=0.5)` |
| `lt` | `$lt` | `Q(price__lt=100)` |
| `lte` | `$lte` | `Q(age__lte=65)` |
| `in` | `$in` | `Q(role__in=["admin", "mod"])` |
| `nin` | `$nin` | `Q(status__nin=["banned"])` |

### Universal Filter Format

Alternative dict-based filter format:

```python
# Equality
where = {"category": {"$eq": "tech"}}

# Comparison
where = {"score": {"$gt": 0.8}, "price": {"$lte": 100}}

# IN
where = {"status": {"$in": ["active", "pending"]}}

# Nested
where = {"user.role": {"$eq": "admin"}}

# Multiple conditions (implicit AND)
where = {
    "category": {"$eq": "tech"},
    "level": {"$eq": "beginner"},
    "score": {"$gte": 0.5}
}
```

---

## Exceptions

### Base Exception

```python
from crossvector.exceptions import CrossVectorError

try:
    # Operation
    pass
except CrossVectorError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

### Document Exceptions

```python
from crossvector.exceptions import (
    DoesNotExist,
    MultipleObjectsReturned,
    DocumentExistsError,
    DocumentNotFoundError,
    MissingDocumentError,
)

# Document not found
try:
    doc = engine.get("nonexistent-id")
except DoesNotExist as e:
    print(f"Not found: {e.details}")

# Multiple results when expecting one
try:
    doc = engine.get(status="active")
except MultipleObjectsReturned as e:
    print(f"Multiple: {e.details}")

# Document already exists
try:
    engine.create({"id": "existing-id", "text": "..."})
except DocumentExistsError as e:
    print(f"Exists: {e.details['document_id']}")
```

### Field Exceptions

```python
from crossvector.exceptions import (
    MissingFieldError,
    InvalidFieldError,
)

# Missing required field
try:
    doc = VectorDocument(text="...")  # Missing vector
except MissingFieldError as e:
    print(f"Missing: {e.details['field']}")

# Invalid field or operator
try:
    results = engine.search("query", where={"field": {"$regex": ".*"}})
except InvalidFieldError as e:
    print(f"Invalid: {e.message}")
```

### Configuration Exceptions

```python
from crossvector.exceptions import MissingConfigError

try:
    db = PgVectorAdapter()  # Missing VECTOR_COLLECTION_NAME
except MissingConfigError as e:
    print(f"Config: {e.details['config_key']}")
    print(f"Hint: {e.details['hint']}")
```

### Collection Exceptions

```python
from crossvector.exceptions import (
    CollectionNotFoundError,
    CollectionExistsError,
    CollectionNotInitializedError,
)

# Collection not found
try:
    db.get_collection("nonexistent")
except CollectionNotFoundError as e:
    print(f"Collection: {e.details['collection_name']}")

# Collection already exists
try:
    db.add_collection("existing", 1536)
except CollectionExistsError as e:
    print(f"Exists: {e.details['collection_name']}")
```

---

## VectorDocument Schema

```python
from crossvector import VectorDocument

# Create document
doc = VectorDocument(
    id="doc-123",
    vector=[0.1, 0.2, ...],
    text="Document text",
    metadata={"key": "value"},
    created_timestamp=1234567890.0,
    updated_timestamp=1234567890.0
)

# Properties
doc.pk  # Primary key (alias for id)

# Methods
doc.to_vector(require=True, output_format="list")  # Get vector
doc.to_metadata(sanitize=True)  # Get metadata
doc.to_storage_dict(store_text=True, use_dollar_vector=False)  # For DB

# Class methods
VectorDocument.from_text("Text", category="tech")
VectorDocument.from_dict({"text": "...", "metadata": {...}})
VectorDocument.from_kwargs(vector=[...], text="...", metadata={...})
VectorDocument.from_any(input_data)  # Auto-detect format
```

---

## Type Definitions

```python
from crossvector.types import Doc, DocId, DocIds

# Doc: Flexible document input
Doc = Union[str, Dict[str, Any], VectorDocument]

# DocId: Single document ID
DocId = Union[str, int]

# DocIds: Single or multiple document IDs
DocIds = Union[DocId, List[DocId]]
```

---

## Next Steps

- [Query DSL Guide](querydsl.md) - Advanced filtering
- [Schema Reference](schema.md) - Data models
- [Database Adapters](adapters/databases.md) - Backend features
- [Examples](quickstart.md) - Practical examples
