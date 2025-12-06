# CrossVector - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-06

### Added

**Benchmarking System:**
- Created comprehensive `scripts/benchmark.py` tool for performance testing
- Support for 4 database backends (pgvector, astradb, milvus, chroma)
- Support for 2 embedding providers (OpenAI, Gemini)
- 7 operation types tested: bulk/individual create, vector/metadata search, Query DSL operators, update, delete
- `--skip-slow` flag to skip cloud backends for faster local testing
- Smart Query DSL optimization: 4 operators for slow backends, 10 for fast backends
- Detailed markdown reports with performance metrics
- Performance summary shows tested vs skipped backends clearly

**Engine Improvements:**
- Added `VectorEngine.drop_collection()` method for collection cleanup
- Better collection lifecycle management

**Documentation:**
- Added benchmarking section to README.md (102 lines)
- Created comprehensive `docs/benchmarking.md` guide (385 lines)
- Updated `docs/contributing.md` with benchmarking workflow
- Added usage examples and best practices
- Cost estimation and troubleshooting guides

**Testing:**
- Added 50+ new unit tests
- Test coverage for ABC adapters (82%)
- Test coverage for logger (100%)
- Extended engine tests
- Schema, utils, and Q object coverage tests
- Total: 365 tests passing (from ~300)

**Architecture:**
- Enhanced ABC base class with unified initialization
- Improved adapter architecture
- Better error reporting in benchmarks
- Truncated error messages in reports for readability

### Changed

- Collection name defaults now use `api_settings.VECTOR_COLLECTION_NAME` instead of class constant
- Improved Milvus metadata-only search support verification
- Updated all adapter documentation
- Modernized contributing.md with uv, pre-commit, ruff

### Removed

- Removed `scripts/e2e.py` (replaced with `pytest scripts/tests`)
- Removed `DEFAULT_COLLECTION_NAME` class constant from adapters

### Fixed

- Fixed Milvus tests to verify metadata-only search functionality
- Fixed collection name handling across all adapters
- Better error messages in benchmark reports
- Proper cleanup in benchmark tests

### Breaking Changes

- `DEFAULT_COLLECTION_NAME` class constant removed - use `api_settings.VECTOR_COLLECTION_NAME` in settings instead
- Stricter ChromaDB config validation (prevents conflicting settings)

### Performance

- Benchmark results show ~60% reduction in API calls for cloud backends with optimization
- Local testing with `--skip-slow`: ~2-3 minutes vs 10+ minutes
- PgVector: ~6-10 docs/sec bulk create, ~0.5ms metadata queries
- Gemini: 1.5x faster search vs OpenAI for same operations

### Documentation Updates

- Repository URLs and references updated
- Enhanced architecture diagrams
- Improved API documentation
- Fixed all broken links

## [0.1.3] - 2025-11-30

### Testing Infrastructure
- **Reorganized test structure** for better separation between unit and integration tests
  - Moved real backend integration tests from `tests/searches/` to `scripts/tests/`
  - Created `tests/mock/` with in-memory adapter for Query DSL unit testing
  - Added comprehensive integration tests for all 4 backends (AstraDB, ChromaDB, Milvus, PgVector)
  - Integration tests are opt-in and require real backend credentials

### Query DSL Improvements
- **Fixed Milvus operator mapping** - Changed `in`/`not in` to uppercase `IN`/`NOT IN` for compliance
- **Improved test coverage** for Query DSL with mock backend tests
- All backends now consistently support 8 universal operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`

### CI/CD
- **Updated GitHub Actions workflow** to run only unit tests (`pytest tests/`)
- Integration tests excluded from CI to avoid credential requirements
- Added `integration` pytest marker for manual integration test execution
- Fixed pytest fixture imports in mock tests

### Documentation
- **Updated README.md** with opt-in integration test documentation
  - Added `scripts/tests/` usage examples
  - Environment variable setup guide for all backends
  - Static collection naming conventions (`test_crossvector`)
- Documented test separation strategy and rationale

### Bug Fixes
- Fixed missing fixture imports causing 15 test errors in mock tests
- Removed unused variable assignments in CRUD test methods
- Resolved pre-commit hook failures (ruff formatting)

## [0.1.2] - 2025-11-23

### Refactor Design
- Major refactoring and architecture improvements
- Enhanced Query DSL design and implementation patterns
- Improved adapter interface consistency across backends

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
