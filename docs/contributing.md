# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development

```bash
# Clone repository
git clone https://github.com/thewebscraping/crossvector.git
cd crossvector

# Install with dev dependencies
pip install -e ".[all,dev]"

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific adapter tests
pytest tests/test_astradb.py
pytest tests/test_chromadb.py
```
