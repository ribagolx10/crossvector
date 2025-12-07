# Query DSL Guide

Comprehensive guide to CrossVector's Query DSL for building filters.

## Overview

CrossVector provides two ways to build filters:

1. **Q Objects** - Composable, Pythonic query builders
2. **Universal Filters** - Dict-based filter format

Both compile to backend-specific filter syntax automatically.

---

## Q Objects

### Basic Usage

```python
from crossvector.querydsl.q import Q

# Simple equality
Q(category="tech")

# Comparison operators
Q(score__gte=0.8)
Q(price__lt=100)

# IN operator
Q(status__in=["active", "pending"])

# NOT IN operator
Q(role__nin=["guest", "banned"])
```

### Combining Filters

```python
# AND (using &)
Q(category="tech") & Q(level="beginner")

# OR (using |)
Q(featured=True) | Q(score__gte=0.9)

# NOT (using ~)
~Q(archived=True)

# Complex combinations
(Q(category="tech") & Q(level="beginner")) | Q(featured=True)
```

### Nested Metadata

Access nested fields using double underscore (`__`):

```python
# Nested object
Q(user__role="admin")
Q(author__verified=True)

# Deep nesting
Q(post__stats__views__gte=1000)
Q(config__settings__enabled=True)
```

**Backend Support:**

| Backend | Nested Metadata |
|---------|-----------------|
| AstraDB | Full support |
| PgVector | Full support |
| Milvus | Full support |
| ChromaDB | Via dot notation |

---

## Supported Operators

CrossVector supports 8 universal operators that work across all backends:

### Field Operators

These 8 operators work on field values and are compiled to backend-specific syntax:

| Operator | Usage | Example |
|----------|-------|---------|
| `eq` | Equal | `Q(status="active")` or `Q(status__eq="active")` |
| `ne` | Not equal | `Q(status__ne="deleted")` |
| `gt` | Greater than | `Q(score__gt=0.8)` |
| `gte` | Greater than or equal | `Q(score__gte=0.8)` |
| `lt` | Less than | `Q(price__lt=100)` |
| `lte` | Less than or equal | `Q(stock__lte=10)` |
| `in` | In array | `Q(status__in=["active", "pending"])` |
| `nin` | Not in array | `Q(status__nin=["deleted", "banned"])` |

### Boolean Operators

These are used to combine Q objects (not field operators):

| Operator | Symbol | Example |
|----------|--------|---------|
| AND | `&` | `Q(category="tech") & Q(level="beginner")` |
| OR | `\|` | `Q(featured=True) \| Q(score__gte=0.9)` |
| NOT | `~` | `~Q(archived=True)` |

**Filter format:** `{"status": {"$ne": "deleted"}}`

---

### Comparison Operators

#### `gt` - Greater Than

```python
Q(score__gt=0.5)
Q(age__gt=18)
```

**Filter format:** `{"score": {"$gt": 0.5}}`

**Note:** Requires numeric values. PgVector automatically casts text to numeric.

#### `gte` - Greater Than or Equal

```python
Q(score__gte=0.8)
Q(price__gte=100)
```

**Filter format:** `{"score": {"$gte": 0.8}}`

#### `lt` - Less Than

```python
Q(price__lt=100)
Q(age__lt=65)
```

**Filter format:** `{"price": {"$lt": 100}}`

#### `lte` - Less Than or Equal

```python
Q(stock__lte=10)
Q(temperature__lte=30)
```

**Filter format:** `{"stock": {"$lte": 10}}`

#### `in` - In Array

```python
Q(status__in=["active", "pending", "review"])
Q(category__in=["tech", "science"])
Q(priority__in=[1, 2, 3])
```

**Filter format:** `{"status": {"$in": ["active", "pending", "review"]}}`

#### `nin` - Not In Array

```python
Q(status__nin=["deleted", "banned"])
Q(role__nin=["guest"])
```

**Filter format:** `{"status": {"$nin": ["deleted", "banned"]}}`

---

## Universal Filter Format

Alternative dict-based filter syntax.

### Basic Filters

```python
# Equality
where = {"status": {"$eq": "active"}}

# Comparison
where = {"score": {"$gt": 0.8}}

# IN
where = {"role": {"$in": ["admin", "moderator"]}}
```

### Multiple Conditions

Multiple conditions in same dict are combined with AND:

```python
where = {
    "category": {"$eq": "tech"},
    "level": {"$eq": "beginner"},
    "score": {"$gte": 0.5}
}
# Equivalent to: category="tech" AND level="beginner" AND score>=0.5
```

### Nested Fields

Use dot notation for nested metadata:

```python
where = {
    "user.role": {"$eq": "admin"},
    "user.verified": {"$eq": True}
}
```

---

## Complete Examples

### Search with Filters

```python
from crossvector import VectorEngine
from crossvector.querydsl.q import Q

engine = VectorEngine(db=..., embedding=...)

# Q object filters
results = engine.search(
    "python tutorials",
    where=Q(category="tech") & Q(level="beginner"),
    limit=10
)

# Universal filter
results = engine.search(
    "machine learning",
    where={
        "category": {"$eq": "ai"},
        "difficulty": {"$lte": 3}
    },
    limit=20
)

# Complex Q filters
results = engine.search(
    "database design",
    where=(Q(featured=True) | Q(score__gte=0.9)) & ~Q(archived=True),
    limit=15
)
```

### Metadata-Only Search

Some backends support filtering without vector search:

```python
# AstraDB, PgVector, ChromaDB, Milvus support metadata-only search
results = engine.search(
    query=None,  # No vector search
    where=Q(status="published") & Q(category="tech"),
    limit=50
)

# Always check backend support
if engine.supports_metadata_only:
    results = engine.search(query=None, where={"status": {"$eq": "active"}})
```

### Get Document with Filters

```python
# Get by metadata (must return exactly one)
doc = engine.get(status="draft", author="user123")

# Using Q
doc = engine.get(Q(slug="my-post") & Q(published=False))

# Universal filter
doc = engine.get(**{"status": {"$eq": "draft"}, "author": {"$eq": "user123"}})
```

---

## Operator Examples by Type

### String Values

```python
# Equality
Q(category="tech")
Q(author__name="John Doe")

# NOT equality
Q(status__ne="deleted")

# IN
Q(language__in=["python", "javascript", "rust"])

# NOT IN
Q(category__nin=["spam", "nsfw"])
```

### Numeric Values

```python
# Comparison
Q(score__gt=0.5)
Q(price__gte=100)
Q(age__lt=65)
Q(stock__lte=10)

# Equality
Q(count=42)
Q(version__eq=3)

# IN
Q(priority__in=[1, 2, 3])
```

### Boolean Values

```python
# Equality
Q(featured=True)
Q(verified=False)

# NOT
~Q(deleted=True)
~Q(archived=True)
```

### Nested Objects

```python
# Single level
Q(user__role="admin")
Q(author__verified=True)

# Multiple levels
Q(post__meta__featured=True)
Q(config__db__host="localhost")

# With operators
Q(user__stats__posts__gte=10)
Q(author__rating__gt=4.5)
```

### Array Fields

```python
# Check if array contains value
Q(tags__in=["python"])  # Has "python" tag

# Check if field is one of values
Q(status__in=["active", "pending"])
```

---

## Backend-Specific Compilation

Different backends compile filters differently:

### AstraDB

Pass-through universal operators directly:

```python
Q(score__gte=0.8)
# Compiles to: {"score": {"$gte": 0.8}}

Q(user__role="admin")
# Compiles to: {"user.role": {"$eq": "admin"}}
```

### ChromaDB

Flattens nested metadata with dot notation:

```python
Q(user__role="admin")
# Compiles to: {"user.role": {"$eq": "admin"}}

Q(score__gte=0.8) & Q(category="tech")
# Compiles to: {"$and": [{"score": {"$gte": 0.8}}, {"category": {"$eq": "tech"}}]}
```

### Milvus

Boolean expression syntax:

```python
Q(category="tech")
# Compiles to: 'category == "tech"'

Q(score__gt=0.8) & Q(price__lt=100)
# Compiles to: '(score > 0.8) and (price < 100)'

Q(status__in=["active", "pending"])
# Compiles to: 'status in ["active", "pending"]'
```

### PgVector

PostgreSQL WHERE clause with JSONB operators:

```python
Q(category="tech")
# Compiles to: "metadata->>'category' = 'tech'"

Q(score__gt=0.8)
# Compiles to: "(metadata->>'score')::numeric > 0.8"

Q(user__role="admin")
# Compiles to: "metadata #>> '{user,role}' = 'admin'"
```

---

## Advanced Usage

### Dynamic Filter Building

```python
def build_filter(category=None, min_score=None, featured=None):
    filters = []

    if category:
        filters.append(Q(category=category))

    if min_score is not None:
        filters.append(Q(score__gte=min_score))

    if featured is not None:
        filters.append(Q(featured=featured))

    # Combine with AND
    if filters:
        result = filters[0]
        for f in filters[1:]:
            result = result & f
        return result

    return None

# Use in search
where = build_filter(category="tech", min_score=0.8, featured=True)
results = engine.search("query", where=where)
```

### Complex Queries

```python
# Featured OR high score, but not archived
where = (Q(featured=True) | Q(score__gte=0.9)) & ~Q(archived=True)

# Tech category with beginner or intermediate level
where = Q(category="tech") & Q(level__in=["beginner", "intermediate"])

# Published articles by verified authors
where = (
    Q(type="article") &
    Q(status="published") &
    Q(author__verified=True) &
    Q(author__rating__gte=4.0)
)

results = engine.search("query", where=where)
```

### Conditional Filters

```python
# Build filter based on user role
user_role = "admin"

if user_role == "admin":
    where = None  # See all documents
elif user_role == "moderator":
    where = ~Q(status="draft")  # See all except drafts
else:
    where = Q(status="published")  # Only published

results = engine.search("query", where=where)
```

---

## Error Handling

### Unsupported Operators

```python
from crossvector.exceptions import InvalidFieldError

try:
    # Regex not supported
    results = engine.search("query", where={"text": {"$regex": "pattern"}})
except InvalidFieldError as e:
    print(f"Error: {e.message}")
    print(f"Operator: {e.details['operator']}")
```

### Type Mismatches

```python
# Correct: numeric comparison with number
Q(score__gt=0.8)

# Incorrect: numeric comparison with string (backend-dependent)
Q(score__gt="0.8")  # May fail on some backends
```

**Best Practice:** Use correct types for comparisons:

```python
# Numbers
Q(score__gte=0.8, price__lt=100, count=42)

# Strings
Q(category="tech", status="active")

# Booleans
Q(featured=True, archived=False)
```

---

## Performance Tips

### Index-Friendly Queries

```python
# Good: Simple equality on indexed field
Q(category="tech")

# Good: Range on indexed numeric field
Q(created_at__gte=timestamp)

# Slower: Complex nested queries
Q(user__profile__settings__theme="dark")
```

### Limit Result Sets

```python
# Always use limit for large datasets
results = engine.search("query", where=where, limit=100)
```

### Pagination

```python
# Page 1
results = engine.search("query", limit=20, offset=0)

# Page 2
results = engine.search("query", limit=20, offset=20)

# Page 3
results = engine.search("query", limit=20, offset=40)
```

---

## Testing Queries

### Check Backend Support

```python
engine = VectorEngine(db=..., embedding=...)

# Check capabilities
print(f"Metadata-only: {engine.supports_metadata_only}")
print(f"Backend: {engine.db.__class__.__name__}")

# Test query
where = Q(category="tech") & Q(score__gte=0.8)
print(f"Filter: {where.to_dict()}")
```

### Debug Compiled Filters

```python
from crossvector.querydsl.q import Q

# Build query
q = Q(category="tech") & Q(level__in=["beginner", "intermediate"])

# See universal format
print(q.to_dict())
# {'$and': [{'category': {'$eq': 'tech'}}, {'level': {'$in': ['beginner', 'intermediate']}}]}

# Compile for specific backend
from crossvector.querydsl.compilers.pgvector import PgVectorWhereCompiler
compiler = PgVectorWhereCompiler()
compiled = compiler.compile(q.to_dict())
print(compiled)
# "metadata->>'category' = 'tech' AND metadata->>'level' IN ('beginner', 'intermediate')"
```

---

## Migration Guide

### From Dict Filters

**Before:**

```python
where = {
    "category": "tech",
    "score": {"$gte": 0.8}
}
```

**After (Q objects):**

```python
from crossvector.querydsl.q import Q
where = Q(category="tech") & Q(score__gte=0.8)
```

### From Raw Backend Queries

**Before (PgVector):**

```python
query = "metadata->>'category' = 'tech' AND (metadata->>'score')::numeric >= 0.8"
```

**After (Universal):**

```python
where = Q(category="tech") & Q(score__gte=0.8)
# Compiles automatically for any backend
```

---

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Schema Reference](schema.md) - Data models
- [Database Adapters](adapters/databases.md) - Backend-specific features
- [Examples](quickstart.md) - Practical examples
