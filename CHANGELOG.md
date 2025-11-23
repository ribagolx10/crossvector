# CrossVector - Changelog

## [0.1.1] - 2025-11-23

- Bumped package version to **0.1.1**.
- Added beta warning and production‑risk notice in README.
- Switched timestamps to float Unix timestamps (`created_timestamp`, `updated_timestamp`).
- Introduced `VECTOR_STORE_TEXT` configuration option.
- Fixed integration tests for AstraDB, ChromaDB, Milvus, and PGVector (table name handling, dimension parameter, score field).
- Updated documentation (README, quickstart, schema, configuration) to reflect new features and usage.
- Adjusted `.markdownlint.yaml` to disable MD060 table‑column‑style warnings.
- Cleaned up imports and resolved lint errors (ruff E402).

## Recent Updates (2025-11-23)

### GitHub Organization Update

- **Changed GitHub organization** from `twofarm` to `thewebscraping`
  - Updated all URLs in:
    - `pyproject.toml`
    - `mkdocs.yml`
    - `README.md`
    - `docs/contributing.md`
  - Documentation site URL: `https://thewebscraping.github.io/crossvector/`
  - Repository URL: `https://github.com/thewebscraping/crossvector`

### Test Infrastructure

- **Created `scripts/tests/` directory** with comprehensive test scripts for real cloud APIs:
  - `tests/test_astradb.py` - Test AstraDB cloud adapter
  - `tests/test_chroma_cloud.py` - Test ChromaDB Cloud adapter
  - `tests/test_milvus.py` - Test Milvus cloud adapter
  - `tests/test_pgvector.py` - Test PGVector adapter
  - `tests/test_integration.py` - Comprehensive integration test for VectorEngine
  - `tests/__init__.py` - Package initialization
  - `tests/README.md` - Detailed documentation for running tests

### Documentation

- **Created comprehensive MkDocs documentation**:
  - `mkdocs.yml` - Documentation configuration with Material theme
  - `docs/index.md` - Project overview
  - `docs/installation.md` - Installation guide
  - `docs/quickstart.md` - Quick start guide
  - `docs/configuration.md` - Configuration guide
  - `docs/adapters/databases.md` - Database adapter documentation
  - `docs/adapters/embeddings.md` - Embedding adapter documentation
  - `docs/api.md` - API reference
  - `docs/contributing.md` - Contributing guide

### CI/CD Workflows

- **Created GitHub Actions workflows**:
  - `.github/workflows/ci.yml` - Test and lint on push/PR
  - `.github/workflows/publish.yml` - Publish to PyPI on release (using Trusted Publishing)
  - `.github/workflows/docs.yml` - Deploy documentation to GitHub Pages
  - `.github/workflows/test-build.yml` - Test package build before release

### Development Tools

- **Pre-commit hooks**: `.pre-commit-config.yaml` for code quality
- **Release helper**: `scripts/release.sh` for automated releases
- **Markdown linting**: `.markdownlint.yaml` configuration

### Code Updates

- **Fixed test issues**:
  - Updated all mock paths in `test_openai_embeddings.py` from `llm_scraper` to `crossvector`
  - Added proper settings mocking to prevent API key errors
  - All 43 tests now passing

- **Updated Pydantic settings**:
  - Migrated from deprecated `class Config` to `SettingsConfigDict`
  - Removed deprecation warnings

- **Updated README**:
  - Changed Gemini status from "Placeholder" to "Production"
  - Updated roadmap to show Gemini as completed
  - Fixed all references from `VectorStoreEngine` to `VectorEngine`

### Environment Setup

- Copied `.env` file from `llm_scraper` project for testing with real cloud credentials
- Added `site/` to `.gitignore` for documentation builds

## Testing the Changes

### Run All Unit Tests

```bash
uv run pytest
```

### Test with Real Cloud APIs

```bash
# Run comprehensive integration test
uv run python scripts/tests/test_integration.py

# Or test individual databases
uv run python scripts/tests/test_astradb.py
uv run python scripts/tests/test_chroma_cloud.py
uv run python scripts/tests/test_milvus.py
uv run python scripts/tests/test_pgvector.py
```

### Build and View Documentation

```bash
# Build documentation
uv run mkdocs build

# Serve documentation locally
uv run mkdocs serve
```

## Next Steps

1. **Push to GitHub**:

   ```bash
   git add .
   git commit -m "Update GitHub org to thewebscraping, add docs and test scripts"
   git push origin main
   ```

2. **Create GitHub Repository**:
   - Create repository at `https://github.com/thewebscraping/crossvector`
   - Enable GitHub Pages for documentation
   - Set up PyPI trusted publishing for releases

3. **Publish to PyPI**:
   - Create a GitHub release
   - The publish workflow will automatically publish to PyPI

4. **Verify Documentation**:
   - Check documentation at `https://thewebscraping.github.io/crossvector/`
   - Ensure all links are working
