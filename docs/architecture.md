# Architecture

System design and architecture of CrossVector.

## Overview

CrossVector is designed as a unified interface for multiple vector database backends, providing a consistent API regardless of the underlying database technology.

```
┌─────────────────────────────────────────────────────────────┐
│                        Application                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       VectorEngine                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • create(), search(), update(), delete()            │   │
│  │  • get_or_create(), update_or_create()               │   │
│  │  • bulk_create(), bulk_update()                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                    │                      │
         ┌──────────┴──────────┐           │
         ▼                     ▼           ▼
┌──────────────────┐  ┌─────────────────────────┐
│ EmbeddingAdapter │  │   VectorDBAdapter       │
│  • Gemini        │  │   • AstraDB             │
│  • OpenAI        │  │   • ChromaDB            │
│  • Custom...     │  │   • Milvus              │
└──────────────────┘  │   • PgVector            │
                      └─────────────────────────┘
                                  │
                                  ▼
                      ┌─────────────────────┐
                      │   WhereCompiler     │
                      │   • AstraDB         │
                      │   • ChromaDB        │
                      │   • Milvus          │
                      │   • PgVector        │
                      └─────────────────────┘
```

---

## Core Components

### VectorEngine

The main interface for all vector operations.

**Responsibilities:**

- Document CRUD operations
- Vector similarity search
- Metadata filtering
- Collection management
- Input normalization (str, dict, VectorDocument)
- Automatic embedding generation
- Primary key management

**Key Methods:**

```python
class VectorEngine:
    def create(self, doc, **kwargs) -> VectorDocument
    def bulk_create(self, docs, **kwargs) -> List[VectorDocument]
    def bulk_update(self, docs, **kwargs) -> List[VectorDocument]
    def upsert(self, docs, **kwargs) -> List[VectorDocument]
    def update(self, doc, **kwargs) -> VectorDocument
    def delete(self, *ids) -> int
    def get(self, *args, **kwargs) -> VectorDocument
    def search(self, query, where=None, limit=None) -> List[VectorDocument]
    def get_or_create(self, doc, **kwargs) -> Tuple[VectorDocument, bool]
    def update_or_create(self, lookup, **kwargs) -> Tuple[VectorDocument, bool]
    def count() -> int
```

**Input Normalization:**

VectorEngine accepts flexible input formats:

```python
# String
engine.create("text")

# Dict
engine.create({"text": "...", "metadata": {...}})

# VectorDocument
engine.create(VectorDocument(...))

# Kwargs
engine.create(text="...", category="tech")
```

All inputs are normalized to `VectorDocument` via `_normalize_document()`.

---

### VectorDBAdapter (Abstract Base)

Abstract interface for vector database backends with lazy initialization pattern.

**Base Initialization:**

```python
class VectorDBAdapter(ABC):
    def __init__(
        self,
        collection_name: str | None = None,
        dim: int | None = None,
        store_text: bool | None = None,
        logger: Logger = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with lazy client/collection initialization."""
        self._client: Any = None  # Initialized by ABC
        self._collection: Any = None  # Initialized by ABC
        self.collection_name = collection_name or api_settings.VECTOR_COLLECTION_NAME
        self.dim = dim or api_settings.VECTOR_DIM
        self.store_text = store_text or api_settings.VECTOR_STORE_TEXT
```

**Required Methods:**

```python
class VectorDBAdapter(ABC):
    @abstractmethod
    def initialize(self, collection_name, dim, metric, **kwargs) -> None

    @abstractmethod
    def add_collection(self, collection_name, dim, metric) -> Any

    @abstractmethod
    def get_collection(self, collection_name) -> Any

    @abstractmethod
    def get_or_create_collection(self, collection_name, dim, metric) -> Any

    @abstractmethod
    def drop_collection(self, collection_name) -> bool

    @abstractmethod
    def clear_collection(self) -> int

    @abstractmethod
    def create(self, doc: VectorDocument) -> VectorDocument

    @abstractmethod
    def bulk_create(self, docs: List[VectorDocument], **kwargs) -> List[VectorDocument]

    @abstractmethod
    def bulk_update(self, docs: List[VectorDocument], **kwargs) -> List[VectorDocument]

    @abstractmethod
    def upsert(self, docs: List[VectorDocument], **kwargs) -> List[VectorDocument]

    @abstractmethod
    def search(
        self,
        vector: List[float] | None,
        limit: int,
        offset: int,
        where: Dict[str, Any] | None,
        fields: Set[str] | None
    ) -> List[VectorDocument]

    @abstractmethod
    def get(self, *args, **kwargs) -> VectorDocument

    @abstractmethod
    def update(self, doc: VectorDocument, **kwargs) -> VectorDocument

    @abstractmethod
    def delete(self, *ids) -> int

    @abstractmethod
    def count(self) -> int
```

**Capabilities:**

```python
class VectorDBAdapter:
    use_dollar_vector: bool = False  # Use '$vector' vs 'vector' key
    supports_metadata_only: bool = False  # Search without vector
    where_compiler: BaseWhere = None  # Backend-specific filter compiler
```

**Lazy Initialization Pattern:**

All adapters use lazy initialization for optimal resource usage:

```python
@property
def client(self):
    """Lazily initialize and return the database client."""
    if self._client is None:
        # Validate configuration
        if not api_settings.REQUIRED_CONFIG:
            raise MissingConfigError(...)
        # Initialize client
        self._client = create_client(...)
    return self._client
```

---

### EmbeddingAdapter (Abstract Base)

Abstract interface for embedding providers.

**Required Methods:**

```python
class EmbeddingAdapter(ABC):
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]

    @property
    def dim(self) -> int
        """The dimension of embeddings generated by the model."""
```

**Implementation Example:**

```python
class GeminiEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, api_key, model_name="models/text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name
        self._dim = 768

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Implementation detail...
        return vectors

    @property
    def dim(self) -> int:
        return self._dim
```

---

### WhereCompiler

Compiles universal filter format to backend-specific syntax.

**Base Class:**

```python
class WhereCompiler(ABC):
    # Capability flags
    SUPPORTS_NESTED: bool = False
    REQUIRES_VECTOR: bool = False
    REQUIRES_AND_WRAPPER: bool = False

    # Operator mapping
    _OP_MAP: Dict[str, str] = {}

    @abstractmethod
    def compile(self, where: Dict[str, Any]) -> Any
```

**Universal Filter Format:**

```python
{
    "field": {"$eq": "value"},
    "score": {"$gte": 0.8},
    "tags": {"$in": ["python", "ai"]}
}
```

**Backend-Specific Output:**

| Backend | Output Format |
|---------|---------------|
| AstraDB | Dict (pass-through) |
| ChromaDB | Dict with `$and` wrapper |
| Milvus | Boolean expression string |
| PgVector | SQL WHERE clause |

**Example Compilation:**

```python
# Input
where = {"category": {"$eq": "tech"}, "score": {"$gte": 0.8}}

# AstraDB
{"category": {"$eq": "tech"}, "score": {"$gte": 0.8}}

# ChromaDB
{"$and": [{"category": {"$eq": "tech"}}, {"score": {"$gte": 0.8}}]}

# Milvus
'(category == "tech") and (score >= 0.8)'

# PgVector
"metadata->>'category' = 'tech' AND (metadata->>'score')::numeric >= 0.8"
```

---

## Query Processing Flow

### Document Creation

```
1. Application
   │
   ├─> engine.create("text", category="tech")
   │
2. VectorEngine
   │
   ├─> _normalize_document()  # Convert to VectorDocument
   ├─> _ensure_pk()            # Generate ID if missing
   ├─> embedding.get_embeddings([text])  # Generate vector
   ├─> _prepare_for_storage()  # Format for backend
   │
3. VectorDBAdapter
   │
   ├─> insert(collection_name, [document])
   │
4. Database
   │
   └─> Store document
```

### Vector Search

```
1. Application
   │
   ├─> engine.search("query", where=Q(category="tech"), limit=10)
   │
2. VectorEngine
   │
   ├─> embedding.get_embeddings(["query"])  # Generate query vector
   ├─> _compile_where(where)                # Compile filters
   │
3. WhereCompiler
   │
   ├─> compile({"category": {"$eq": "tech"}})
   │
4. VectorDBAdapter
   │
   ├─> search(collection_name, query_vector, where, limit)
   │
5. Database
   │
   ├─> Vector similarity search
   ├─> Apply metadata filters
   └─> Return results
```

---

## Data Flow

### VectorDocument Lifecycle

```
┌───────────────┐
│  Application  │
│  Input        │
└───────┬───────┘
        │ str, dict, VectorDocument
        ▼
┌───────────────────────┐
│  VectorEngine         │
│  _normalize_document  │
└───────┬───────────────┘
        │ VectorDocument (partial)
        ▼
┌───────────────────┐
│  Primary Key      │
│  Generation       │
└───────┬───────────┘
        │ VectorDocument (with ID)
        ▼
┌───────────────────┐
│  Embedding        │
│  Generation       │
└───────┬───────────┘
        │ VectorDocument (complete)
        ▼
┌───────────────────────┐
│  Storage Format       │
│  Conversion           │
└───────┬───────────────┘
        │ Dict (backend-specific)
        ▼
┌───────────────┐
│  Database     │
│  Storage      │
└───────────────┘
```

---

## Design Patterns

### Adapter Pattern

VectorDBAdapter and EmbeddingAdapter use the Adapter pattern to provide a unified interface to different backends.

```python
# Unified interface
engine = VectorEngine(
    db=PgVectorAdapter(),      # Can swap with AstraDBAdapter()
    embedding=GeminiEmbeddingAdapter()  # Can swap with OpenAIEmbeddingAdapter()
)

# Same API regardless of adapters
doc = engine.create("text")
results = engine.search("query")
```

### Strategy Pattern

WhereCompiler uses the Strategy pattern to compile filters differently based on backend.

```python
# Each backend has its own compilation strategy
class AstraDBWhereCompiler(WhereCompiler):
    def compile(self, where):
        return where  # Pass-through

class MilvusWhereCompiler(WhereCompiler):
    def compile(self, where):
        return self._to_boolean_expr(where)  # Boolean expression

class PgVectorWhereCompiler(WhereCompiler):
    def compile(self, where):
        return self._to_sql_where(where)  # SQL WHERE clause
```

### Factory Pattern

Primary key generation uses the Factory pattern with configurable strategies.

```python
# Configure factory strategy
settings = CrossVectorSettings(PK_STRATEGY="uuid")

# Or custom factory
settings = CrossVectorSettings(
    PK_STRATEGY="custom",
    PK_FACTORY=lambda: f"doc-{uuid.uuid4()}"
)
```

---

## Configuration System

### Settings Hierarchy

```
1. Default values (in CrossVectorSettings)
   ↓
2. Environment variables (GEMINI_API_KEY, VECTOR_COLLECTION_NAME, etc.)
   ↓
3. Programmatic config (passed to constructors)
```

### Settings Class

```python
class CrossVectorSettings(BaseSettings):
    # General
    VECTOR_SEARCH_LIMIT: int = 10
    PK_STRATEGY: str = "uuid"
    PK_FACTORY: Optional[Callable] = None

    # Gemini
    GEMINI_API_KEY: str
    VECTOR_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # PgVector
    VECTOR_COLLECTION_NAME: str
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432

    # ... other settings

    class Config:
        env_file = ".env"
        case_sensitive = True
```

---

## Error Handling

### Exception Hierarchy

```
CrossVectorError (base)
├── DocumentError
│   ├── DoesNotExist
│   ├── MultipleObjectsReturned
│   ├── DocumentExistsError
│   ├── DocumentNotFoundError
│   └── MissingDocumentError
├── FieldError
│   ├── MissingFieldError
│   └── InvalidFieldError
├── CollectionError
│   ├── CollectionNotFoundError
│   ├── CollectionExistsError
│   └── CollectionNotInitializedError
├── ConfigError
│   └── MissingConfigError
├── SearchError
└── EmbeddingError
```

### Structured Exceptions

All exceptions include:

```python
class CrossVectorError(Exception):
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
```

**Usage:**

```python
try:
    doc = engine.get("nonexistent")
except DoesNotExist as e:
    print(e.message)  # "Document does not exist"
    print(e.details)  # {"collection": "docs", "query": {...}}
```

---

## Extension Points

### Custom Database Adapter

Implement `VectorDBAdapter`:

```python
class CustomDBAdapter(VectorDBAdapter):
    supports_metadata_only = True
    where_compiler = CustomWhereCompiler()

    def initialize(self, collection_name, dim, metric, **kwargs):
        # Initialize client and collection
        pass

    def create(self, doc: VectorDocument) -> VectorDocument:
        # Insert document
        pass

    def search(self, vector, limit, offset, where, fields):
        # Perform search
        pass

    # ... implement all other abstract methods
```

### Custom Embedding Adapter

Implement `EmbeddingAdapter`:

```python
class CustomEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, model_name, dim=None):
        super().__init__(model_name, dim)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Implementation
        return vectors

    @property
    def dim(self) -> int:
        return self._dim
```

### Custom WhereCompiler

Extend `WhereCompiler`:

```python
class CustomWhereCompiler(WhereCompiler):
    SUPPORTS_NESTED = True
    _OP_MAP = {...}

    def compile(self, where):
        # Implementation
        return compiled_filter
```

### Custom PK Factory

Provide dotted path to callable for ID generation:

```python
# In mymodule.py
def custom_id_generator(text: str | None, vector: List[float] | None, metadata: Dict) -> str:
    """Generate custom ID from text, vector, or metadata."""
    if text:
        return f"doc-{text[:20]}-{uuid.uuid4()}"
    return f"doc-{uuid.uuid4()}"

# In .env or settings
PRIMARY_KEY_MODE=custom
PRIMARY_KEY_FACTORY=mymodule.custom_id_generator
```

**Or via code:**

```python
from crossvector.settings import settings

# Set custom factory
settings.PRIMARY_KEY_MODE = "custom"
settings.PRIMARY_KEY_FACTORY = "mymodule.custom_id_generator"
```

---

## Performance Considerations

### Batch Operations

Use bulk operations for efficiency:

```python
# Good: Batch insert
docs = [{"text": f"Doc {i}"} for i in range(1000)]
engine.bulk_create(docs, batch_size=100)

# Bad: Individual inserts
for doc in docs:
    engine.create(doc)
```

### Embedding Caching

Store text with vectors to avoid re-embedding:

```python
engine = VectorEngine(
    db=...,
    embedding=...,
    store_text=True  # Cache text with vectors
)

# Later: Retrieve without re-embedding
doc = engine.get("doc-id")
print(doc.text)   # Available
print(doc.vector)  # Pre-computed
```

### Query Optimization

```python
# Use metadata-only when possible
if engine.supports_metadata_only:
    results = engine.search(query=None, where=filters)

# Limit results
results = engine.search("query", limit=100)

# Use pagination
for page in range(10):
    results = engine.search("query", limit=20, offset=page*20)
```

---

## Error Handling

### Exception Hierarchy

CrossVector provides structured exceptions with detailed context:

```python
from crossvector.exceptions import (
    MissingConfigError,      # Configuration errors
    CollectionNotFoundError, # Collection operations
    DocumentNotFoundError,   # Document operations
    SearchError,            # Search failures
    ConnectionError,        # Connection failures
)
```

### Configuration Validation

Strict validation with helpful error messages:

```python
# ChromaDB config conflict
CHROMA_HOST="localhost"
CHROMA_PERSIST_DIR="./data"

# Raises MissingConfigError:
# "Cannot set both CHROMA_HOST and CHROMA_PERSIST_DIR.
#  Choose one deployment mode:
#  - For HTTP: Set CHROMA_HOST (unset CHROMA_PERSIST_DIR)
#  - For Local: Set CHROMA_PERSIST_DIR (unset CHROMA_HOST)"
```

### Lazy Initialization Errors

Errors are raised when client is first accessed:

```python
db = ChromaAdapter()  # No error yet

# Error raised here when client property accessed:
engine = VectorEngine(db=db, embedding=...)
# MissingConfigError if config invalid
```

### Error Context

All exceptions include contextual information:

```python
try:
    doc = engine.get(id="nonexistent")
except DocumentNotFoundError as e:
    print(e.document_id)  # "nonexistent"
    print(e.operation)    # "get"
    print(e.adapter)      # "ChromaAdapter"
```

---

## Testing Architecture

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_engine.py           # VectorEngine tests
├── test_openai_embeddings.py
├── test_gemini_embeddings.py
└── test_querydsl_operators.py

scripts/
└── tests/
    ├── test_astradb.py      # Real DB tests
    ├── test_chroma_cloud.py
    ├── test_milvus.py
    └── test_pgvector.py
```

### Test Fixtures

```python
@pytest.fixture
def engine():
    return VectorEngine(
        db=MockDBAdapter(),
        embedding=MockEmbeddingAdapter(),
        collection_name="test"
    )

@pytest.fixture
def test_documents():
    return [
        {"text": "Doc 1", "metadata": {"category": "tech"}},
        {"text": "Doc 2", "metadata": {"category": "science"}},
    ]
```

---

## Security Considerations

### API Key Management

```python
# Good: Environment variables
import os
api_key = os.getenv("GEMINI_API_KEY")

# Bad: Hard-coded
api_key = "sk-..."
```

### Input Validation

All inputs are validated:

```python
# VectorDocument validation
doc = VectorDocument(
    id="doc-1",
    vector=[...],  # Dimension checked
    text="...",
    metadata={...}  # Sanitized
)
```

### SQL Injection Prevention

PgVector uses parameterized queries:

```python
# Safe: Parameterized
cursor.execute(
    "SELECT * FROM docs WHERE metadata->>'category' = %s",
    (category,)
)
```

---

## Future Enhancements

### Planned Features

- **Reranking support** - Post-search result reranking
- **Hybrid search** - Combine vector + full-text search
- **Multi-vector** - Multiple vectors per document
- **Async operations** - Non-blocking API
- **Streaming** - Stream large result sets

### Extension Ideas

- More embedding providers (Cohere, Hugging Face, etc.)
- Additional backends (Qdrant, Weaviate, Pinecone)
- Query caching layer
- Result pagination helpers
- Admin UI for collection management

---

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Contributing](contributing.md) - Contribution guidelines
- [Database Adapters](adapters/databases.md) - Backend details
- [Embedding Adapters](adapters/embeddings.md) - Embedding providers
