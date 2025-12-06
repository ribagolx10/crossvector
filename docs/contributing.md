# Contributing to CrossVector

Thank you for your interest in contributing to CrossVector!

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- [uv](https://docs.astral.sh/uv/) (recommended for fast package management)

### Development Setup

1. **Clone the repository:**

```bash
git clone https://github.com/thewebscraping/crossvector.git
cd crossvector
```

2. **Install dependencies with uv:**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project with all dependencies (dev + all backends/embeddings)
uv pip install -e ".[dev,all]"

# Or install specific extras
uv pip install -e ".[dev,pgvector,gemini]"  # Just PgVector + Gemini
```

3. **Setup pre-commit hooks:**

```bash
# Install pre-commit hooks
pre-commit install

# (Optional) Run on all files to test
pre-commit run --all-files
```

4. **Configure environment:**

```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

---

## Development Workflow

### Code Style

CrossVector follows PEP 8 and uses:

- **[Ruff](https://docs.astral.sh/ruff/)** for fast linting and formatting (replaces Black, isort, flake8)
- **pre-commit** for automated code quality checks
- **mypy** for type checking (optional, can be enabled in `.pre-commit-config.yaml`)

**Automatic formatting with pre-commit:**

Pre-commit hooks will automatically run on every commit. To manually run:

```bash
# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

**Manual formatting and linting:**

```bash
# Format code with ruff
ruff format src/ tests/ scripts/

# Lint and auto-fix issues
ruff check src/ tests/ scripts/ --fix

# Type checking (optional)
mypy src/
```

### Type Hints

All code must include type hints:

```python
from typing import List, Dict, Any, Optional
from crossvector import VectorDocument

def process_documents(
    docs: List[VectorDocument],
    filters: Optional[Dict[str, Any]] = None
) -> List[VectorDocument]:
    """Process documents with optional filters."""
    pass
```

---

## Testing

### Running Tests

**All tests:**

```bash
pytest
```

**Specific test file:**

```bash
pytest tests/test_engine.py
```

**With coverage:**

```bash
pytest --cov=crossvector --cov-report=html
```

**Integration tests with real backends:**

```bash
# Run all integration tests
pytest scripts/tests/ -v

# Specific backend
pytest scripts/tests/test_pgvector.py -v
```

## Benchmarking

Before submitting performance-related changes, run benchmarks to measure impact:

```bash
# Quick benchmark (10 docs)
python scripts/benchmark.py --num-docs 10

# Full benchmark (1000 docs) - before and after your changes
python scripts/benchmark.py --output benchmark_before.md
# ... make your changes ...
python scripts/benchmark.py --output benchmark_after.md

# Compare specific backend
python scripts/benchmark.py --backends pgvector --num-docs 100
```

The benchmark tool tests:
- Bulk and individual create operations
- Vector search performance
- Metadata-only search
- Query DSL operators (10 operators)
- Update and delete operations

Results are saved as markdown reports for easy comparison. See [Benchmarking Guide](benchmarking.md) for details.

### Writing Tests

**Test structure:**

```python
import pytest
from crossvector import VectorEngine
from crossvector.dbs.pgvector import PgVectorAdapter
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

class TestVectorEngine:
    @pytest.fixture
    def engine(self):
        """Create test engine."""
        return VectorEngine(
            db=PgVectorAdapter(),
            embedding=GeminiEmbeddingAdapter(),
            collection_name="test_collection"
        )

    def test_create_document(self, engine):
        """Test document creation."""
        doc = engine.create("Test content")
        assert doc.id is not None
        assert doc.text == "Test content"
        assert len(doc.vector) == 1536

    def test_search(self, engine):
        """Test vector search."""
        engine.create("Python tutorial")
        results = engine.search("python", limit=10)
        assert len(results) > 0
```

**Use fixtures:**

```python
@pytest.fixture(scope="module")
def test_data():
    """Create test data."""
    return [
        {"text": "Document 1", "metadata": {"category": "tech"}},
        {"text": "Document 2", "metadata": {"category": "science"}},
    ]

def test_with_fixture(engine, test_data):
    """Test using fixture data."""
    created = engine.bulk_create(test_data)
    assert len(created) == 2
```

### Test Coverage

Aim for >90% code coverage. Check coverage:

```bash
pytest --cov=crossvector --cov-report=term-missing
```

---

## Adding Features

### New Database Adapter

1. **Create adapter class:**

```python
# src/crossvector/dbs/newdb.py
from crossvector.abc import VectorDBAdapter
from typing import List, Dict, Any, Optional
from crossvector import VectorDocument

class NewDBAdapter(VectorDBAdapter):
    """Adapter for NewDB vector database."""

    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self._client = None

    def add_collection(
        self,
        collection_name: str,
        dimension: int,
        **kwargs
    ) -> bool:
        """Create collection."""
        pass

    def insert(
        self,
        collection_name: str,
        documents: List[VectorDocument],
        **kwargs
    ) -> List[VectorDocument]:
        """Insert documents."""
        pass

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        where: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        **kwargs
    ) -> List[VectorDocument]:
        """Search documents."""
        pass

    # Implement other required methods...
```

1. **Create where compiler:**

```python
# src/crossvector/querydsl/compilers/newdb.py
from crossvector.querydsl.compilers.base import WhereCompiler
from typing import Dict, Any

class NewDBWhereCompiler(WhereCompiler):
    """Compile filters for NewDB."""

    SUPPORTS_NESTED = True
    REQUIRES_VECTOR = False

    _OP_MAP = {
        "$eq": "==",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
        "$in": "in",
        "$nin": "not in",
    }

    def compile(self, where: Dict[str, Any]) -> str:
        """Compile to NewDB filter format."""
        pass
```

1. **Add tests:**

```python
# tests/test_newdb.py
import pytest
from crossvector import VectorEngine
from crossvector.dbs.newdb import NewDBAdapter

class TestNewDB:
    @pytest.fixture
    def engine(self):
        return VectorEngine(
            db=NewDBAdapter(),
            embedding=...,
            collection_name="test"
        )

    def test_create(self, engine):
        """Test document creation."""
        pass

    def test_search(self, engine):
        """Test vector search."""
        pass
```

1. **Update documentation:**

- Add to `docs/adapters/databases.md`
- Update feature comparison tables
- Add configuration examples

### New Embedding Provider

1. **Create adapter class:**

```python
# src/crossvector/embeddings/newprovider.py
from crossvector.abc import EmbeddingAdapter
from typing import List

class NewProviderEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for NewProvider embeddings."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "default-model"
    ):
        self.api_key = api_key
        self.model_name = model_name
        self._dimensions = 768

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        # Implementation
        pass

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions
```

1. **Add tests:**

```python
# tests/test_newprovider_embeddings.py
import pytest
from crossvector.embeddings.newprovider import NewProviderEmbeddingAdapter

def test_embeddings():
    """Test embedding generation."""
    adapter = NewProviderEmbeddingAdapter(api_key="test")
    vectors = adapter.get_embeddings(["test text"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 768
```

1. **Update documentation:**

- Add to `docs/adapters/embeddings.md`
- Add configuration examples
- Update comparison tables

---

## Documentation

### Writing Documentation

Documentation is in `docs/` directory using Markdown:

```bash
docs/
├── index.md              # Main page
├── installation.md       # Installation guide
├── quickstart.md         # Quick start tutorial
├── api.md                # API reference
├── schema.md             # Data models
├── querydsl.md           # Query DSL guide
├── configuration.md      # Configuration reference
└── adapters/
    ├── databases.md      # Database adapters
    └── embeddings.md     # Embedding adapters
```

**Building docs:**

```bash
mkdocs serve  # Local preview at http://127.0.0.1:8000
mkdocs build  # Build static site
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Add type hints to examples
- Show both success and error cases
- Update all affected docs when changing features

---

## Pull Request Process

### Before Submitting

1. **Run tests:**

```bash
pytest
pytest scripts/tests/ -v  # Integration tests with real backends
```

2. **Format and lint code:**

```bash
# Let pre-commit handle it automatically
pre-commit run --all-files

# Or manually
ruff format src/ tests/ scripts/
ruff check src/ tests/ scripts/ --fix
```

1. **Update documentation:**

- Add/update docstrings
- Update relevant .md files
- Add examples if needed

1. **Update CHANGELOG.md:**

```markdown
## [Unreleased]

### Added
- New feature X with Y capability

### Changed
- Modified Z to improve performance

### Fixed
- Bug in A causing B
```

### Submitting PR

1. **Create feature branch:**

```bash
git checkout -b feature/my-new-feature
```

1. **Commit changes:**

```bash
git add .
git commit -m "feat: add new feature X"
```

Use conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements

1. **Push branch:**

```bash
git push origin feature/my-new-feature
```

1. **Create Pull Request:**

- Go to GitHub repository
- Click "New Pull Request"
- Fill in template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for feature
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Code Review

- Respond to reviewer feedback
- Make requested changes
- Re-request review after changes

---

## Release Process

### Version Numbering

Follow Semantic Versioning (SemVer):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

### Creating Release

1. **Update version:**

```bash
# pyproject.toml
[project]
version = "0.2.0"
```

1. **Update CHANGELOG.md:**

```markdown
## [0.2.0] - 2024-01-15

### Added
- Feature X
- Feature Y

### Changed
- Improved Z performance

### Fixed
- Bug in A
```

1. **Create release:**

```bash
git tag v0.2.0
git push origin v0.2.0
```

1. **Publish to PyPI:**

```bash
python -m build
twine upload dist/*
```

---

## Community

### Communication

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** Questions and general discussion
- **Pull Requests:** Code contributions

### Getting Help

- Check existing [documentation](https://thewebscraping.github.io/crossvector/)
- Search [issues](https://github.com/yourusername/crossvector/issues)
- Ask in [discussions](https://github.com/yourusername/crossvector/discussions)

### Reporting Bugs

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Error occurs

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- CrossVector version: 0.1.0
- Python version: 3.11
- OS: macOS 14
- Backend: PgVector

## Additional Context
Any other relevant information
```

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize community benefit

### Enforcement

Violations can be reported to maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

## Questions?

Feel free to ask questions in:

- GitHub Issues (for bugs)
- GitHub Discussions (for general questions)
- Pull Request comments (for specific code questions)

Thank you for contributing to CrossVector! 🎉
