# Schema and Data Models

Data structures and schemas used in CrossVector.

## VectorDocument

The primary data model representing a document with embeddings.

### Fields

```python
from crossvector import VectorDocument

doc = VectorDocument(
    id: str | int,                    # Primary key
    vector: List[float],               # Embedding vector
    text: str = None,                  # Original text (optional)
    metadata: Dict[str, Any] = None,   # Arbitrary metadata
    created_timestamp: float = None,   # Creation timestamp
    updated_timestamp: float = None    # Last update timestamp
)
```

#### `id` (required)

Primary key identifier. Can be string or integer depending on PK strategy.

```python
doc = VectorDocument(id="doc-123", ...)
doc = VectorDocument(id=42, ...)
```

#### `vector` (required)

Embedding vector as list of floats. Dimension must match embedding model.

```python
doc = VectorDocument(
    id="doc-1",
    vector=[0.1, 0.2, 0.3, ...]  # 1536 dims for text-embedding-3-small
)
```

#### `text` (optional)

Original text content. Required if `VectorEngine.store_text=True`.

```python
doc = VectorDocument(
    id="doc-1",
    vector=[...],
    text="This is the original document text"
)
```

#### `metadata` (optional)

Arbitrary metadata dictionary. Supports nested structures (backend-dependent).

```python
doc = VectorDocument(
    id="doc-1",
    vector=[...],
    metadata={
        "category": "tech",
        "tags": ["python", "ai"],
        "author": {
            "name": "John",
            "role": "admin"
        },
        "score": 0.95,
        "featured": True
    }
)
```

#### `created_timestamp` (optional)

Unix timestamp for document creation. Auto-populated on insert.

```python
import time
doc = VectorDocument(
    id="doc-1",
    vector=[...],
    created_timestamp=time.time()
)
```

#### `updated_timestamp` (optional)

Unix timestamp for last update. Auto-updated on modification.

```python
doc = VectorDocument(
    id="doc-1",
    vector=[...],
    updated_timestamp=time.time()
)
```

---

## Properties

### `pk`

Alias for `id` property.

```python
doc = VectorDocument(id="doc-123", vector=[...])
print(doc.pk)  # "doc-123"
```

---

## Methods

### Constructor Classmethods

#### `from_text()`

Create document from text string with optional metadata.

```python
VectorDocument.from_text(
    text: str,
    **kwargs
) -> VectorDocument
```

**Example:**

```python
doc = VectorDocument.from_text(
    "My document text",
    category="tech",
    priority=1
)
# Creates: VectorDocument(text="...", metadata={"category": "tech", "priority": 1})
```

#### `from_dict()`

Create document from dictionary.

```python
VectorDocument.from_dict(
    data: Dict[str, Any]
) -> VectorDocument
```

**Example:**

```python
doc = VectorDocument.from_dict({
    "id": "doc-123",
    "text": "Content",
    "metadata": {"key": "value"},
    "vector": [0.1, 0.2, ...]
})
```

#### `from_kwargs()`

Create document from keyword arguments.

```python
VectorDocument.from_kwargs(**kwargs) -> VectorDocument
```

**Example:**

```python
doc = VectorDocument.from_kwargs(
    id="doc-123",
    text="Content",
    vector=[...],
    metadata={"key": "value"}
)
```

#### `from_any()`

Auto-detect input format and create document.

```python
VectorDocument.from_any(
    doc: str | Dict | VectorDocument
) -> VectorDocument
```

**Examples:**

```python
# From string
doc = VectorDocument.from_any("Text content")

# From dict
doc = VectorDocument.from_any({"text": "Content", "metadata": {...}})

# From VectorDocument (returns copy)
doc = VectorDocument.from_any(existing_doc)
```

---

### Data Export Methods

#### `to_vector()`

Extract vector as list or numpy array.

```python
to_vector(
    require: bool = True,
    output_format: str = "list"
) -> List[float] | np.ndarray | None
```

**Parameters:**

- `require`: Raise error if vector missing
- `output_format`: `"list"` or `"numpy"`

**Examples:**

```python
vector = doc.to_vector()  # List[float]
vector = doc.to_vector(output_format="numpy")  # np.ndarray
vector = doc.to_vector(require=False)  # None if missing
```

#### `to_metadata()`

Extract metadata dictionary.

```python
to_metadata(sanitize: bool = True) -> Dict[str, Any]
```

**Parameters:**

- `sanitize`: Remove None values

**Example:**

```python
metadata = doc.to_metadata()
# {"category": "tech", "score": 0.95}

metadata = doc.to_metadata(sanitize=False)
# {"category": "tech", "score": 0.95, "optional": None}
```

#### `to_storage_dict()`

Convert to database storage format.

```python
to_storage_dict(
    store_text: bool = False,
    use_dollar_vector: bool = False
) -> Dict[str, Any]
```

**Parameters:**

- `store_text`: Include text field
- `use_dollar_vector`: Use `$vector` key (AstraDB format)

**Examples:**

```python
# Standard format
storage = doc.to_storage_dict()
# {"id": "doc-1", "vector": [...], "metadata": {...}}

# With text
storage = doc.to_storage_dict(store_text=True)
# {"id": "doc-1", "vector": [...], "text": "...", "metadata": {...}}

# AstraDB format
storage = doc.to_storage_dict(use_dollar_vector=True)
# {"_id": "doc-1", "$vector": [...], "metadata": {...}}
```

---

## Metadata Schema

Metadata can contain arbitrary JSON-serializable data. Different backends support different levels of nesting.

### Flat Metadata (All Backends)

```python
metadata = {
    "category": "tech",
    "author": "John Doe",
    "score": 0.95,
    "published": True,
    "tags": ["python", "ai"],
    "count": 42
}
```

### Nested Metadata (Backend Support)

| Backend | Nested Support | Query Format |
|---------|----------------|--------------|
| AstraDB | ✅ Full | `{"user.role": {"$eq": "admin"}}` |
| PgVector | ✅ Full | `{"user.role": {"$eq": "admin"}}` |
| ChromaDB | ❌ Flattened | `{"user.role": {"$eq": "admin"}}` (auto-flattened) |
| Milvus | ✅ Full | `{"user.role": {"$eq": "admin"}}` |

**Example with nested metadata:**

```python
doc = VectorDocument(
    id="doc-1",
    vector=[...],
    metadata={
        "user": {
            "name": "John",
            "role": "admin",
            "verified": True
        },
        "post": {
            "title": "My Post",
            "stats": {
                "views": 1000,
                "likes": 50
            }
        }
    }
)

# Query nested fields
from crossvector.querydsl.q import Q
results = engine.search(
    "query",
    where=Q(user__role="admin") & Q(post__stats__views__gte=500)
)
```

---

## Metadata Types

### Supported Types

CrossVector supports standard JSON types in metadata:

```python
metadata = {
    "string": "text value",
    "integer": 42,
    "float": 3.14,
    "boolean": True,
    "null": None,
    "array": [1, 2, 3],
    "object": {"nested": "value"}
}
```

### Type Casting (Backend-Specific)

Some backends require explicit type casting for numeric comparisons:

**PgVector** (automatic numeric casting):

```python
# Text stored as string, but compared numerically
metadata = {"price": "99.99"}  # Stored as text
where = {"price": {"$gt": 50}}  # Cast to numeric for comparison
```

**Other Backends**:

Store numbers as actual numeric types when using comparison operators:

```python
# Correct
metadata = {"price": 99.99, "count": 42}

# Incorrect for numeric comparisons
metadata = {"price": "99.99", "count": "42"}
```

---

## Primary Key Strategies

Configure primary key generation in `VectorEngine` settings.

### Strategy: `uuid`

Generate UUID v4 strings.

```python
from crossvector.settings import CrossVectorSettings

settings = CrossVectorSettings(PK_STRATEGY="uuid")
# Generated IDs: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

### Strategy: `hash_text`

Hash document text using SHA256.

```python
settings = CrossVectorSettings(PK_STRATEGY="hash_text")
# Generated IDs: "5f4dcc3b5aa765d61d8327deb882cf99"

doc = engine.create("Hello world")
# doc.id = hash("Hello world")
```

**Note:** Requires `text` field to be present.

### Strategy: `hash_vector`

Hash embedding vector using SHA256.

```python
settings = CrossVectorSettings(PK_STRATEGY="hash_vector")
# Generated IDs: "7b8e4d2a9c1f3e5d6a0b4c8e2f7d9a1b"

doc = engine.create(vector=[0.1, 0.2, ...])
# doc.id = hash(vector)
```

### Strategy: `int64`

Generate random 64-bit integers.

```python
settings = CrossVectorSettings(PK_STRATEGY="int64")
# Generated IDs: 7234567890123456789
```

### Strategy: `auto`

Use backend's native auto-generation (if supported).

```python
settings = CrossVectorSettings(PK_STRATEGY="auto")
# Backend-specific ID generation
```

### Strategy: `custom`

Provide custom ID factory function.

```python
from crossvector import VectorEngine
from crossvector.settings import CrossVectorSettings

def my_id_factory() -> str:
    return f"doc-{int(time.time())}"

settings = CrossVectorSettings(
    PK_STRATEGY="custom",
    PK_FACTORY=my_id_factory
)

engine = VectorEngine(db=..., embedding=..., settings=settings)
doc = engine.create("Text")
# doc.id = "doc-1234567890"
```

**Factory signature:**

```python
def pk_factory() -> str | int:
    """Generate unique primary key."""
    pass
```

---

## Input Formats

VectorEngine accepts multiple input formats.

### String Input

Creates document with text only. Embedding generated automatically.

```python
doc = engine.create("My document text")
# VectorDocument(id=auto, text="...", vector=auto, metadata={})
```

### Dict Input

Flexible dictionary with any combination of fields.

```python
# Minimal
doc = engine.create({"text": "Content"})

# With metadata
doc = engine.create({
    "text": "Content",
    "metadata": {"key": "value"}
})

# With ID
doc = engine.create({
    "id": "custom-id",
    "text": "Content",
    "metadata": {...}
})

# With pre-computed vector
doc = engine.create({
    "text": "Content",
    "vector": [0.1, 0.2, ...],
    "metadata": {...}
})
```

### VectorDocument Input

Direct VectorDocument instance.

```python
from crossvector import VectorDocument

doc = VectorDocument(
    id="doc-123",
    text="Content",
    metadata={"key": "value"}
)
created = engine.create(doc)
```

### Kwargs Input

Metadata fields as keyword arguments.

```python
doc = engine.create(
    text="Content",
    category="tech",
    priority=1,
    featured=True
)
# metadata = {"category": "tech", "priority": 1, "featured": True}
```

---

## Validation Rules

### Required Fields

- **For creation:** Either `text` or `vector` must be provided
- **For search:** `id` is required
- **For update:** `id` is required

### Field Constraints

```python
from pydantic import ValidationError

try:
    doc = VectorDocument(
        # Missing id
        vector=[0.1, 0.2],
        text="Content"
    )
except ValidationError as e:
    print(e)
```

### Vector Dimension

Vector dimension must match embedding model:

```python
# text-embedding-3-small: 1536 dimensions
doc = VectorDocument(
    id="doc-1",
    vector=[...],  # Must be length 1536
    text="Content"
)
```

**Raises:** `InvalidFieldError` if dimension mismatch

---

## Serialization

### JSON Serialization

VectorDocument can be serialized to JSON:

```python
import json

doc = VectorDocument(
    id="doc-1",
    text="Content",
    vector=[0.1, 0.2, 0.3],
    metadata={"key": "value"}
)

# To JSON string
json_str = json.dumps(doc.model_dump())

# From JSON string
data = json.loads(json_str)
doc = VectorDocument(**data)
```

### Database Format

Different backends expect different formats:

**Standard (PgVector, Milvus, ChromaDB):**

```python
{
    "id": "doc-1",
    "vector": [0.1, 0.2, ...],
    "text": "Content",
    "metadata": {"key": "value"}
}
```

**AstraDB:**

```python
{
    "_id": "doc-1",
    "$vector": [0.1, 0.2, ...],
    "text": "Content",
    "metadata": {"key": "value"}
}
```

Use `to_storage_dict()` to get correct format:

```python
storage = doc.to_storage_dict(
    store_text=engine.store_text,
    use_dollar_vector=(engine.db.__class__.__name__ == "AstraDBAdapter")
)
```

---

## Examples

### Basic Document Creation

```python
from crossvector import VectorDocument, VectorEngine

# Create with text
doc = VectorDocument.from_text(
    "Python is a programming language",
    category="tech",
    language="python"
)

# Store in database
engine = VectorEngine(db=..., embedding=...)
created = engine.create(doc)

print(created.id)  # Auto-generated
print(created.vector[:5])  # [0.123, 0.456, ...]
print(created.metadata)  # {"category": "tech", "language": "python"}
```

### Document with Nested Metadata

```python
doc = VectorDocument(
    id="post-123",
    text="My blog post about AI",
    vector=[...],
    metadata={
        "post": {
            "title": "Introduction to AI",
            "category": "technology",
            "tags": ["ai", "ml", "python"]
        },
        "author": {
            "name": "Jane Doe",
            "role": "contributor",
            "verified": True
        },
        "stats": {
            "views": 1500,
            "likes": 89,
            "shares": 12
        }
    }
)

# Query nested
from crossvector.querydsl.q import Q
results = engine.search(
    "AI tutorials",
    where=Q(author__verified=True) & Q(stats__views__gte=1000)
)
```

### Batch Document Creation

```python
docs = [
    {"text": f"Document {i}", "metadata": {"index": i, "batch": "A"}}
    for i in range(100)
]

created = engine.bulk_create(docs, batch_size=50)
print(f"Created {len(created)} documents")
```

---

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Query DSL](querydsl.md) - Advanced filtering
- [Configuration](configuration.md) - Settings and strategies
- [Database Adapters](adapters/databases.md) - Backend features
