# VectorDocument Schema

CrossVector uses a Pydantic `VectorDocument` class for type-safe VectorDocument handling with powerful auto-generation features.

## Features

### 1. Auto-Generated ID

If you don't provide an ID, CrossVector automatically generates one based on your `PRIMARY_KEY_MODE` setting:

```python
from crossvector import VectorDocument

# Without ID - auto-generated based on PRIMARY_KEY_MODE
doc = VectorDocument(text="Hello world")
print(doc.id)  
# Possible values depending on PRIMARY_KEY_MODE:
# - 'uuid' (default): Random UUID like "a1b2c3d4e5f6..."
# - 'hash_text': SHA256 hash of text like "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
# - 'hash_vector': SHA256 hash of vector
# - 'int64': Sequential integer as string like "1", "2", "3", ...
# - 'auto': Hash text if available, else hash vector, else UUID
# - Custom factory: Use PRIMARY_KEY_FACTORY setting for custom logic

# With explicit ID
doc = VectorDocument(id="my-custom-id", text="Hello world")
print(doc.id)  # "my-custom-id"
```

### 2. Auto-Generated Timestamps

Every VectorDocument automatically gets creation and update timestamps:

```python
doc = VectorDocument(text="Hello world")

print(doc.created_timestamp)  # 1732349789.123456 (Unix timestamp)
print(doc.updated_timestamp)  # 1732349789.123456 (Unix timestamp)

# Convert to datetime if needed
from datetime import datetime, timezone
created_dt = datetime.fromtimestamp(doc.created_timestamp, tz=timezone.utc)
print(created_dt)  # 2024-11-23 11:16:29.123456+00:00
```

**Why Float/Unix Timestamp?**

- **Compact**: Numbers are smaller than ISO 8601 strings
- **Efficient**: Easy to compare and sort
- **Universal**: Works across all programming languages
- **Smaller storage**: ~8 bytes vs ~32 bytes for strings

### 3. No Timestamp Conflicts

You can safely use your own `created_at` and `updated_at` fields:

```python
doc = VectorDocument(
    text="My article",
    metadata={
        "title": "Introduction to AI",
        "created_at": "2024-01-15T10:00:00Z",  # Your article's timestamp
        "updated_at": "2024-11-20T15:30:00Z",  # Your article's timestamp
        "author": "John Doe"
    }
)

# CrossVector timestamps (internal tracking)
print(doc.created_timestamp)  # 1732349789.123456
print(doc.updated_timestamp)  # 1732349789.123456

# Your timestamps (preserved in metadata)
print(doc.metadata["created_at"])  # "2024-01-15T10:00:00Z"
print(doc.metadata["updated_at"])  # "2024-11-20T15:30:00Z"
```

## VectorDocument Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | No (auto-generated) | Unique identifier. Auto-generated based on PRIMARY_KEY_MODE if not provided (uuid/hash_text/hash_vector/int64/auto/custom) |
| `text` | `str` | Yes | The text content of the VectorDocument |
| `metadata` | `Dict[str, Any]` | No (default: `{}`) | Associated metadata |
| `created_timestamp` | `float` | No (auto-generated) | Unix timestamp when created |
| `updated_timestamp` | `float` | No (auto-generated) | Unix timestamp when last updated |

## Examples

### Basic VectorDocument

```python
doc = VectorDocument(text="Hello world")
```

### VectorDocument with Metadata

```python
doc = VectorDocument(
    text="Python is awesome",
    metadata={
        "language": "en",
        "category": "programming",
        "tags": ["python", "tutorial"]
    }
)
```

### VectorDocument with Custom ID

```python
doc = VectorDocument(
    id="article-123",
    text="Full article content here...",
    metadata={
        "title": "Getting Started with Python",
        "author": "John Doe"
    }
)
```

### Preserving Created Timestamp

When updating a VectorDocument, you can preserve the original creation timestamp:

```python
# Original VectorDocument
doc1 = VectorDocument(id="article-1", text="Original content")
original_created = doc1.created_timestamp

# Later, update the VectorDocument
doc2 = VectorDocument(
    id="article-1",
    text="Updated content",
    created_timestamp=original_created  # Preserve original
)

print(doc2.created_timestamp)  # Same as original
print(doc2.updated_timestamp)  # New timestamp
```

## Serialization

```python
doc = VectorDocument(text="Hello", metadata={"key": "value"})

# To dict
doc_dict = doc.model_dump()
print(doc_dict)
# {
#     'id': 'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e',
#     'text': 'Hello',
#     'metadata': {'key': 'value'},
#     'created_timestamp': 1732349789.123456,
#     'updated_timestamp': 1732349789.123456
# }

# To JSON
import json
doc_json = json.dumps(doc_dict)
```
