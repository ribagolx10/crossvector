"""Benchmark script for CrossVector database adapters.

Measures performance across different operations:
- Bulk create
- Individual create
- Vector search
- Metadata search
- Query DSL operators
- Update operations
- Delete operations

Usage:
    # Quick test with 10 documents
    python scripts/benchmark.py --num-docs 10

    # Fast test (skip slow cloud backends)
    python scripts/benchmark.py --num-docs 10 --skip-slow

    # Full benchmark with 1000 documents (default)
    python scripts/benchmark.py

    # Test specific backends and embeddings
    python scripts/benchmark.py --backends pgvector milvus --embedding-providers openai

    # Custom output file
    python scripts/benchmark.py --output results/my_benchmark.md

IMPORTANT NOTES ON BENCHMARK RESULTS:
=====================================

Results vary significantly based on deployment environment:

1. **PgVector**: Benchmarks are run against LOCAL PostgreSQL instance
   - Provides optimal latency and consistent performance
   - Results NOT comparable with cloud-hosted PgVector
   - For fair comparison: deploy PgVector in same region/network as cloud backends

2. **Cloud Backends** (AstraDB, Milvus, ChromaDB): Results affected by:
   - Network latency and geographic region
   - Regional proximity between client and server
   - Network conditions and bandwidth availability
   - Server load and resource allocation

3. **For Fair Comparison**:
   - Run benchmarks in your actual production environment
   - Ensure all backends deployed in same region
   - Use consistent network conditions across all backends
   - Account for network latency when interpreting results

4. **Embedding Providers**: API-based providers (OpenAI, Gemini)
   - API call latency included in embedding generation time
   - Batch sizes and rate limits affect overall performance
   - Static vectors used during benchmark (skip embedding API calls for DB isolation)

RECOMMENDATION: Conduct benchmarks in YOUR production environment with real network
conditions to get accurate, meaningful results for your specific use case.
"""

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.exceptions import MissingConfigError
from crossvector.schema import VectorDocument

# Import fixture generation functions
from .generate_fixtures import (
    generate_benchmark_docs,
    generate_search_queries,
)

load_dotenv()

# Timing constants for collection lifecycle operations
SLEEP_BEFORE_DROP = 0.0  # Immediate drop
SLEEP_AFTER_DROP = 0.5  # Wait for backend cleanup after drop
SLEEP_BEFORE_INIT = 0.2  # Brief wait before reinitializing
SLEEP_AFTER_INIT = 0.3  # Wait for collection to fully initialize
SLEEP_CLEANUP_FINAL = 0.5  # Wait for final cleanup after drop

# Lorem ipsum text for consistent document generation
LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium. Totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem."


def generate_documents(num_docs: int) -> List[Dict[str, Any]]:
    """Generate test documents with consistent lorem ipsum content.

    All documents have the same base text (lorem ipsum) to ensure
    consistent configuration across all benchmark runs.
    """
    docs = []
    # Use a portion of lorem ipsum proportional to num_docs to create varied but consistent docs
    words = LOREM_IPSUM.split()
    for i in range(num_docs):
        # Create document text by cycling through lorem ipsum words
        start_idx = (i * 10) % len(words)
        chunk_size = max(20, 200 // max(1, num_docs // 100))  # Vary chunk size based on num_docs
        text_chunk = " ".join(words[start_idx : start_idx + chunk_size])
        if not text_chunk:
            text_chunk = LOREM_IPSUM[:200]

        docs.append(
            {
                "text": f"{text_chunk} [Document {i}]",
                "metadata": {
                    "doc_id": i,
                    "category": f"cat_{i % 5}",
                    "score": (i % 100) / 100.0,
                    "batch": i // 100,
                },
            }
        )
    return docs


def generate_fixture_vectors(num_docs: int, dim: int = 1536) -> List[List[float]]:
    """Generate static vectors for fixture documents.

    Creates reproducible vectors with fixed seed for consistent benchmarking
    without needing embedding API calls.

    Args:
        num_docs: Number of documents to generate vectors for
        dim: Vector dimension (default 1536 for OpenAI embeddings)

    Returns:
        List of vectors, one per document
    """
    import random

    vectors: List[List[float]] = []
    random.seed(42)  # Fixed seed for reproducibility

    for i in range(num_docs):
        # Create normalized vector with values in [-1, 1] range
        vector = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        vectors.append(vector)

    return vectors


def load_fixtures_from_file(
    fixtures_path: str, num_docs: Optional[int] = None, add_vectors: bool = False
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load test documents and queries from a fixtures JSON file.

    Documents can either be:
    - Simple format: {"text": "...", "metadata": {...}}
    - With vectors: {"text": "...", "metadata": {...}, "vector": [...]}

    Queries can be:
    - String format: "query text"
    - Object format: {"type": "...", "text": "query text"}

    Args:
        fixtures_path: Path to fixtures.json file
        num_docs: Optional limit on number of documents to load (None = use all)
        add_vectors: If True and documents lack vectors, generate and add them

    Returns:
        Tuple of (documents list, query strings list)

    Raises:
        FileNotFoundError: If fixtures file doesn't exist
        json.JSONDecodeError: If fixtures file is invalid JSON
    """
    fixtures_file = Path(fixtures_path)
    if not fixtures_file.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")

    with open(fixtures_file, "r") as f:
        fixtures = json.load(f)

    # Extract documents and queries
    documents = fixtures.get("documents", [])
    raw_queries = fixtures.get("queries", [])

    # Limit number of documents if requested
    if num_docs and num_docs > 0:
        documents = documents[:num_docs]

    # Extract query text (handle both string and object formats)
    queries = []
    for query in raw_queries:
        if isinstance(query, str):
            queries.append(query)
        elif isinstance(query, dict):
            # Extract 'text' field from query object
            queries.append(query.get("text", str(query)))

    # Log fixture info
    has_vectors = any("vector" in doc for doc in documents)

    # Generate and add vectors if requested and documents lack them
    if add_vectors and not has_vectors:
        print(f"Generating vectors for {len(documents)} documents...")
        generated_vectors = generate_fixture_vectors(len(documents))
        for i, doc in enumerate(documents):
            doc["vector"] = generated_vectors[i]
        has_vectors = True
        print("Added vectors to all documents")
    print(f"Loaded {len(documents)} documents from {fixtures_path}")
    if has_vectors:
        print("Documents include pre-computed vectors")
    else:
        print("   ℹ️  Documents will need vectors computed during benchmark")
    print(f"Loaded {len(queries)} search queries from {fixtures_path}")

    return documents, queries


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def benchmark_operation(name: str, operation: callable) -> Tuple[float, Any]:
    """Benchmark a single operation and return duration and result."""
    start = time.time()
    try:
        result = operation()
        duration = time.time() - start
        return duration, result
    except Exception as e:
        duration = time.time() - start
        print(f"{name} failed: {e}")
        return duration, None


class BenchmarkRunner:
    """Run benchmarks across different database backends and embedding providers."""

    def __init__(
        self,
        num_docs: int = 1000,
        backends: Optional[List[str]] = None,
        embedding_providers: Optional[List[str]] = None,
        skip_slow: bool = False,
        search_limit: int = 100,
        collection_name: Optional[str] = None,
    ):
        """Initialize benchmark runner.

        Args:
            num_docs: Number of documents to use in benchmarks
            backends: List of backend names to test (None = all available)
            embedding_providers: List of embedding providers to test (None = all available)
            skip_slow: If True, skip slow cloud backends (astradb, milvus)
            search_limit: Number of results to return in search operations
            collection_name: Custom collection name (None = auto-generate with UUID)
        """
        self.num_docs = num_docs
        self.search_limit = search_limit
        # Always ensure collection_name is set (auto-generate with UUID if not provided)
        self.collection_name = collection_name or f"benchmark_test_{str(time.time())[:8]}"
        self.results: Dict[str, Dict[str, Any]] = {}

        # Define available backends
        all_backends = {
            "pgvector": self._init_pgvector,
            "astradb": self._init_astradb,
            "milvus": self._init_milvus,
            "chroma": self._init_chroma,
        }

        # Skip slow backends if requested
        if skip_slow:
            print("⚡ Skipping slow cloud backends (astradb, milvus)")
            all_backends = {k: v for k, v in all_backends.items() if k not in ["astradb", "milvus"]}

        # Filter backends if specified
        if backends:
            self.backends = {k: v for k, v in all_backends.items() if k in backends}
        else:
            self.backends = all_backends

        # Define available embedding providers
        all_providers = {
            "openai": self._init_openai_embedding,
            "gemini": self._init_gemini_embedding,
        }

        # Filter providers if specified
        if embedding_providers:
            self.embedding_providers = {k: v for k, v in all_providers.items() if k in embedding_providers}
        else:
            self.embedding_providers = all_providers

        # Batch size for embedding precomputation to reduce API calls
        self.embedding_batch_size = 100

    def _generate_static_vectors(self, num_docs: int, dim: int = 1536) -> List[List[float]]:
        """Generate static random vectors for benchmarks without calling embedding APIs.

        This creates realistic vectors with proper dimensionality without the overhead
        of actual embedding API calls, making benchmarks focused on database performance
        rather than embedding latency.
        """
        import random

        vectors: List[List[float]] = []
        random.seed(42)  # Fixed seed for reproducibility

        for i in range(num_docs):
            # Create vector with values in [-1, 1] range (typical for normalized embeddings)
            vector = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            vectors.append(vector)

        return vectors

    def _precompute_query_vectors(self, queries: List[str], dim: int = 1536) -> Dict[str, List[float]]:
        """Precompute static vectors for search queries.

        Maps each query to a consistent vector using a deterministic seed,
        so the same queries always get the same vectors across runs.
        """
        import random

        query_vectors = {}
        random.seed(42)  # Same seed for consistency

        for query in queries:
            # Each query gets a unique but deterministic vector
            vector = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            query_vectors[query] = vector

        return query_vectors

    def _precompute_doc_embeddings(self, docs: List[Dict[str, Any]], embedding: Any) -> List[List[float]]:
        """Precompute embeddings (static vectors) without calling embedding APIs.

        Uses static randomly generated vectors matching the embedding dimension
        to measure database performance without embedding latency.
        """
        # Get the embedding dimension from the adapter
        dim = getattr(embedding, "dim", 1536)

        # Generate static vectors instead of calling embedding APIs
        vectors = self._generate_static_vectors(len(docs), dim)

        for doc, vec in zip(docs, vectors):
            doc["vector"] = vec

        return vectors

    def _init_openai_embedding(self) -> Optional[Any]:
        """Initialize OpenAI embedding adapter."""
        try:
            from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

            return OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
        except Exception as e:
            print(f"OpenAI embedding not available: {e}")
            return None

    def _init_gemini_embedding(self) -> Optional[Any]:
        """Initialize Gemini embedding adapter."""
        try:
            from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

            # Use 1536 dimensions to match OpenAI for fair comparison
            return GeminiEmbeddingAdapter(model_name="gemini-embedding-001", dim=1536)
        except Exception as e:
            print(f"Gemini embedding not available: {e}")
            return None

    def _init_pgvector(self, embedding: Any, collection_name: str = None) -> Optional[VectorEngine]:
        """Initialize PgVector engine."""
        try:
            from crossvector.dbs.pgvector import PgVectorAdapter

            return VectorEngine(
                db=PgVectorAdapter(),
                embedding=embedding,
                collection_name=collection_name,
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"PgVector not available: {e}")
            return None

    def _init_astradb(self, embedding: Any, collection_name: str = None) -> Optional[VectorEngine]:
        """Initialize AstraDB engine."""
        try:
            from crossvector.dbs.astradb import AstraDBAdapter

            return VectorEngine(
                db=AstraDBAdapter(),
                embedding=embedding,
                collection_name=collection_name,
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"AstraDB not available: {e}")
            return None

    def _init_milvus(self, embedding: Any, collection_name: str = None) -> Optional[VectorEngine]:
        """Initialize Milvus engine."""
        try:
            from crossvector.dbs.milvus import MilvusAdapter

            return VectorEngine(
                db=MilvusAdapter(),
                embedding=embedding,
                collection_name=collection_name,
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"Milvus not available: {e}")
            return None

    def _init_chroma(self, embedding: Any, collection_name: str = None) -> Optional[VectorEngine]:
        """Initialize ChromaDB engine."""
        try:
            from crossvector.dbs.chroma import ChromaAdapter

            return VectorEngine(
                db=ChromaAdapter(),
                embedding=embedding,
                collection_name=collection_name,
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"ChromaDB not available: {e}")
            return None

    def cleanup_collection(self, engine: VectorEngine, backend_name: str, collection_name: str = None) -> None:
        """Clean up test collection."""
        try:
            engine.drop_collection(collection_name or "benchmark_test")
            time.sleep(0.1)
            print(f"Cleaned up {backend_name} collection")
        except Exception as e:
            print(f"Cleanup warning for {backend_name}: {e}")

    def benchmark_backend(
        self,
        backend_name: str,
        init_func: callable,
        embedding_name: str,
        embedding: Any,
        pre_docs: List[VectorDocument] = None,
        query_vectors: Dict[str, List[float]] = None,
    ) -> Dict[str, Any]:
        """Run benchmarks for a specific backend with specific embedding provider.

        Args:
            backend_name: Name of the backend
            init_func: Function to initialize the engine
            embedding_name: Name of the embedding provider
            embedding: Embedding adapter instance
            pre_docs: Pre-generated documents (with static vectors attached)
            query_vectors: Pre-computed query vectors

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {backend_name.upper()} + {embedding_name.upper()}")
        print(f"{'=' * 60}")

        # Initialize engine
        engine = init_func(embedding, self.collection_name)
        if not engine:
            return {"error": "Backend not available"}

        results = {
            "backend": backend_name,
            "embedding": embedding_name,
            "embedding_model": embedding.model_name if hasattr(embedding, "model_name") else "unknown",
            "embedding_dim": embedding.dim,
            "num_docs": self.num_docs,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Collection name already set from main() with random UUID suffix
            collection_name = self.collection_name

            # Try to drop old collection (gracefully ignore if not exists)
            try:
                engine.drop_collection(collection_name)
                time.sleep(SLEEP_AFTER_DROP)  # Give backend time to clean up (especially cloud backends)
            except Exception:
                pass  # Collection may not exist yet

            # Reinitialize with fresh collection
            time.sleep(SLEEP_BEFORE_INIT)  # Brief wait before reinit
            engine = init_func(embedding, collection_name)
            if not engine:
                return {"error": "Failed to reinitialize engine"}

            time.sleep(SLEEP_AFTER_INIT)  # Give backend time to fully initialize collection

            # Use pre-generated documents with vectors (computed once globally)
            # Skip duplicate generation if pre_docs provided
            if not pre_docs:
                print(f"\nGenerating {self.num_docs} test documents...")
                test_docs = generate_documents(self.num_docs)
                self._precompute_doc_embeddings(test_docs, embedding)
            else:
                test_docs = pre_docs.copy()
                print(f"Using pre-generated {self.num_docs} documents (static vectors already attached)")

            # 1. Bulk Create Performance
            print(f"\nUpsert ({self.num_docs} docs)...")
            # Use conservative batch_size to satisfy provider limits (e.g., Chroma max batch 1000)
            duration, upserted_docs = benchmark_operation("upsert", lambda: engine.upsert(test_docs, batch_size=100))
            results["upsert"] = {
                "duration": duration,
                "docs_per_sec": self.num_docs / duration if duration > 0 else 0,
                "success": upserted_docs is not None,
            }
            print(f"Duration: {format_duration(duration)}")
            print(f"{results['upsert']['docs_per_sec']:.2f} docs/sec")

            # 2. Individual Create Performance (small sample)
            sample_size = min(10, self.num_docs)
            print(f"\nIndividual Create ({sample_size} docs)...")
            individual_times = []
            # Generate additional vectors for individual creates (from pre-computed static vectors)
            dim = getattr(embedding, "dim", 1536)
            extra_vectors = self._generate_static_vectors(sample_size, dim)

            for i in range(sample_size):
                doc_data = {
                    "text": f"Individual test document {i}",
                    "metadata": {"type": "individual", "idx": i},
                    "vector": extra_vectors[i],  # Use pre-generated static vector
                }
                duration, _ = benchmark_operation(f"create_{i}", lambda d=doc_data: engine.create(d))
                individual_times.append(duration)

            avg_create = sum(individual_times) / len(individual_times) if individual_times else 0
            results["individual_create"] = {
                "avg_duration": avg_create,
                "sample_size": sample_size,
            }
            print(f"Avg Duration: {format_duration(avg_create)}")

            # 3. Vector Search Performance
            print("\nVector Search (10 queries with pre-computed vectors)...")
            search_queries = [
                "programming languages",
                "machine learning",
                "database optimization",
                "cloud computing",
                "web development",
                "data science",
                "cybersecurity",
                "mobile apps",
                "devops practices",
                "software architecture",
            ]

            # Use pre-computed query vectors (computed once globally per embedding)
            if not query_vectors:
                dim = getattr(embedding, "dim", 1536)
                query_vectors = self._precompute_query_vectors(search_queries, dim)

            search_times = []
            for query in search_queries:
                # Use precomputed vector directly - NO embedding API call
                vector = query_vectors.get(query)
                if vector:
                    duration, _ = benchmark_operation(
                        f"search_{query[:20]}", lambda v=vector: engine.search(query=v, limit=self.search_limit)
                    )
                    search_times.append(duration)

            avg_search = sum(search_times) / len(search_times) if search_times else 0
            results["vector_search"] = {
                "avg_duration": avg_search,
                "queries": len(search_queries),
            }
            print(f"Avg Duration: {format_duration(avg_search)}")
            print(f"{len(search_queries) / sum(search_times) if sum(search_times) > 0 else 0:.2f} queries/sec")

            # 4. Metadata-Only Search (if supported)
            if engine.supports_metadata_only:
                print("\nMetadata Search (10 queries)...")
                metadata_times = []
                for i in range(10):
                    duration, _ = benchmark_operation(
                        f"metadata_search_{i}",
                        lambda idx=i: engine.search(
                            query=None, where={"category": {"$eq": f"cat_{idx % 5}"}}, limit=self.search_limit
                        ),
                    )
                    metadata_times.append(duration)

                avg_metadata = sum(metadata_times) / len(metadata_times) if metadata_times else 0
                results["metadata_search"] = {
                    "avg_duration": avg_metadata,
                    "queries": len(metadata_times),
                    "supported": True,
                }
                print(f"   Avg Duration: {format_duration(avg_metadata)}")
            else:
                results["metadata_search"] = {"supported": False}
                print("\nMetadata Search: Not supported")

            # 4.5. Query DSL Operators Test (using Q objects)
            print("\nQuery DSL Operators (Q objects)...")
            from crossvector.querydsl import Q

            # For slow backends (astradb, milvus), test fewer operators
            backend_name_lower = backend_name.lower()
            is_slow_backend = backend_name_lower in ["astradb", "milvus"]

            if is_slow_backend:
                # Test only key operators for slow backends
                operator_tests = [
                    ("eq", lambda: engine.search(query=None, where=Q(category="cat_0"), limit=self.search_limit)),
                    ("gt", lambda: engine.search(query=None, where=Q(score__gt=0.5), limit=self.search_limit)),
                    (
                        "in",
                        lambda: engine.search(
                            query=None, where=Q(category__in=["cat_0", "cat_1"]), limit=self.search_limit
                        ),
                    ),
                    (
                        "and",
                        lambda: engine.search(
                            query=None, where=Q(category="cat_0") & Q(score__gte=0.5), limit=self.search_limit
                        ),
                    ),
                ]
                print("Testing 4 key operators (slow backend optimization)")
            else:
                # Test all operators for fast backends
                operator_tests = [
                    ("eq", lambda: engine.search(query=None, where=Q(category="cat_0"), limit=self.search_limit)),
                    ("ne", lambda: engine.search(query=None, where=Q(category__ne="cat_0"), limit=self.search_limit)),
                    ("gt", lambda: engine.search(query=None, where=Q(score__gt=0.5), limit=self.search_limit)),
                    ("gte", lambda: engine.search(query=None, where=Q(score__gte=0.5), limit=self.search_limit)),
                    ("lt", lambda: engine.search(query=None, where=Q(score__lt=0.5), limit=self.search_limit)),
                    ("lte", lambda: engine.search(query=None, where=Q(score__lte=0.5), limit=self.search_limit)),
                    (
                        "in",
                        lambda: engine.search(
                            query=None, where=Q(category__in=["cat_0", "cat_1"]), limit=self.search_limit
                        ),
                    ),
                    (
                        "nin",
                        lambda: engine.search(
                            query=None, where=Q(category__nin=["cat_0", "cat_1"]), limit=self.search_limit
                        ),
                    ),
                    (
                        "and",
                        lambda: engine.search(
                            query=None, where=Q(category="cat_0") & Q(score__gte=0.5), limit=self.search_limit
                        ),
                    ),
                    (
                        "or",
                        lambda: engine.search(
                            query=None, where=Q(category="cat_0") | Q(category="cat_1"), limit=self.search_limit
                        ),
                    ),
                ]

            operator_times = []
            successful_operators = 0
            for op_name, op_func in operator_tests:
                try:
                    duration, _ = benchmark_operation(f"operator_{op_name}", op_func)
                    operator_times.append(duration)
                    successful_operators += 1
                except Exception as e:
                    print(f"  Operator {op_name} skipped: {e}")

            if operator_times:
                avg_operator = sum(operator_times) / len(operator_times)
                results["query_dsl_operators"] = {
                    "avg_duration": avg_operator,
                    "operators_tested": successful_operators,
                    "total_operators": len(operator_tests),
                }
                print(
                    f"Avg Duration: {format_duration(avg_operator)} ({successful_operators}/{len(operator_tests)} operators)"
                )
            else:
                results["query_dsl_operators"] = {"supported": False}

            # 5. Update Performance (use all docs)
            print(f"\nUpdate Operations ({self.num_docs} updates)...")
            update_sample = min(self.num_docs, len(upserted_docs) if upserted_docs else 0)
            if upserted_docs and update_sample > 0:
                update_times = []
                for i in range(update_sample):
                    doc = upserted_docs[i]
                    doc.metadata["updated"] = True
                    doc.metadata["update_idx"] = i
                    duration, _ = benchmark_operation(f"update_{i}", lambda d=doc: engine.update(d))
                    update_times.append(duration)

                avg_update = sum(update_times) / len(update_times) if update_times else 0
                results["update"] = {
                    "avg_duration": avg_update,
                    "sample_size": update_sample,
                }
                print(f"   Avg Duration: {format_duration(avg_update)}")
            else:
                results["update"] = {"error": "No documents to update"}

            # 6. Delete Performance (all docs, batched)
            print(f"\nDelete Operations ({self.num_docs} deletes)...")
            delete_sample = min(self.num_docs, len(upserted_docs) if upserted_docs else 0)
            if upserted_docs and delete_sample > 0:
                batch_size = 100
                delete_ids = [doc.id for doc in upserted_docs[:delete_sample]]
                total_duration = 0.0
                for i in range(0, len(delete_ids), batch_size):
                    batch_ids = delete_ids[i : i + batch_size]
                    duration, _ = benchmark_operation("batch_delete", lambda ids=batch_ids: engine.delete(*ids))
                    total_duration += duration

                results["delete"] = {
                    "duration": total_duration,
                    "sample_size": delete_sample,
                    "docs_per_sec": delete_sample / total_duration if total_duration > 0 else 0,
                }
                print(f"Duration: {format_duration(total_duration)}")
                print(f"{results['delete']['docs_per_sec']:.2f} docs/sec")
            else:
                results["delete"] = {"error": "No documents to delete"}

            # 7. Count operation
            remaining_count = engine.count()
            results["final_count"] = remaining_count
            print(f"\nFinal document count: {remaining_count}")

        except Exception as e:
            print(f"\nBenchmark failed: {e}")
            results["error"] = str(e)
        finally:
            # Cleanup - try to drop collection (gracefully ignore if fails)
            if "collection_name" in locals():
                try:
                    engine.drop_collection(collection_name)
                    time.sleep(SLEEP_CLEANUP_FINAL)  # Give backend time to clean up
                except Exception:
                    pass  # Silently ignore cleanup errors

        return results

    def batch_search(
        self, engine: VectorEngine, query_vectors: List[List[float]], search_limit: int = 5
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Perform batch vector searches without API calls.

        Args:
            engine: VectorEngine instance
            query_vectors: List of pre-computed query vectors
            search_limit: Number of results to return per query

        Returns:
            Tuple of (total_time, results)
        """
        start_time = time.time()
        all_results = []

        for vector in query_vectors:
            try:
                results = engine.search(query=vector, limit=search_limit)
                all_results.extend(results)
            except Exception as e:
                print(f"    Query failed: {e}")

        elapsed = time.time() - start_time
        return elapsed, all_results

    def run_all(
        self, pre_docs: List[VectorDocument] = None, query_vectors: Dict[str, List[float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for all backends with all embedding providers.

        Args:
            pre_docs: Pre-generated documents with static vectors (computed once in main)
            query_vectors: Pre-computed search query vectors keyed by embedding provider
        """
        print(f"\n{'=' * 60}")
        print("CrossVector Benchmark Suite")
        print(f"{'=' * 60}")
        print(f"Documents per test: {self.num_docs}")
        print(f"Backends: {', '.join(self.backends.keys())}")
        print(f"Embeddings: {', '.join(self.embedding_providers.keys())}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for emb_name, emb_init_func in self.embedding_providers.items():
            embedding = emb_init_func()
            if not embedding:
                continue

            # For this embedding provider, attach vectors to documents if needed
            if pre_docs:
                docs_with_vectors = copy.deepcopy(pre_docs)
                self._precompute_doc_embeddings(docs_with_vectors, embedding)
            else:
                docs_with_vectors = None

            # Get pre-computed query vectors for this embedding (if provided)
            emb_query_vectors = query_vectors.get(emb_name) if query_vectors else None

            for backend_name, init_func in self.backends.items():
                result_key = f"{backend_name}_{emb_name}"
                try:
                    self.results[result_key] = self.benchmark_backend(
                        backend_name,
                        init_func,
                        emb_name,
                        embedding,
                        pre_docs=docs_with_vectors,
                        query_vectors=emb_query_vectors,
                    )
                except Exception as e:
                    # Skip failed backends gracefully instead of crashing
                    error_msg = str(e)[:100]
                    print(f"\nSkipping {backend_name}_{emb_name}: {error_msg}...")
                    self.results[result_key] = {"error": error_msg}

        return self.results

    def generate_markdown_report(self, output_file: str = "benchmark.md") -> None:
        """Generate markdown report from benchmark results."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Header
            f.write("# CrossVector Benchmark Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Documents per test:** {self.num_docs}\n\n")
            f.write("---\n\n")

            # Summary table
            f.write("## Performance Summary\n\n")

            # Show which backends were tested/skipped/errored
            all_backends = ["pgvector", "astradb", "milvus", "chroma"]
            tested_backends = list(self.backends.keys())
            skipped_backends = [b for b in all_backends if b not in tested_backends]

            # Count errors and successes
            total_tests = len(self.results)
            error_tests = sum(1 for r in self.results.values() if "error" in r)
            success_tests = total_tests - error_tests

            f.write(f"**Tested backends:** {', '.join(tested_backends)}\n\n")
            if skipped_backends:
                f.write(f"**Skipped backends:** {', '.join(skipped_backends)} ⏭️\n\n")

            f.write(f"**Test Results:** {success_tests}/{total_tests} passed")
            if error_tests > 0:
                f.write(f", {error_tests} failed\n\n")
            else:
                f.write("\n\n")

            f.write(
                "| Backend | Embedding | Model | Dim | Upsert | Search (avg) | Update (avg) | Delete (batch) | Status |\n"
            )
            f.write(
                "|---------|-----------|-------|-----|--------|--------------|--------------|----------------|--------|\n"
            )

            for result_key, result in self.results.items():
                if "error" in result:
                    backend = result.get("backend", result_key.split("_")[0])
                    embedding = result.get("embedding", result_key.split("_")[1] if "_" in result_key else "unknown")
                    error_msg = result["error"][:50] + "..." if len(result["error"]) > 50 else result["error"]
                    f.write(f"| {backend} | {embedding} | - | - | - | - | - | - | ERROR: {error_msg} |\n")
                    continue

                backend = result.get("backend", "unknown")
                embedding = result.get("embedding", "unknown")
                model = result.get("embedding_model", "unknown")
                dim = result.get("embedding_dim", 0)
                upsert_entry = result.get("upsert", {})
                update_entry = result.get("update", {})
                delete_entry = result.get("delete", {})

                bulk_create = format_duration(upsert_entry.get("duration", 0))
                search = format_duration(result.get("vector_search", {}).get("avg_duration", 0))
                update = (
                    "N/A"
                    if isinstance(update_entry, dict) and "error" in update_entry
                    else format_duration(update_entry.get("avg_duration", 0))
                )
                delete = (
                    "N/A"
                    if isinstance(delete_entry, dict) and "error" in delete_entry
                    else format_duration(delete_entry.get("duration", 0))
                )

                status_icon = "OK"
                if (isinstance(update_entry, dict) and "error" in update_entry) or (
                    isinstance(delete_entry, dict) and "error" in delete_entry
                ):
                    status_icon = "WARNING"

                f.write(
                    f"| {backend} | {embedding} | {model} | {dim} | {bulk_create} | {search} | {update} | {delete} | {status_icon} |\n"
                )

            f.write("\n---\n\n")

            # Detailed results per backend
            for result_key, result in self.results.items():
                backend = result.get("backend", "unknown")
                embedding = result.get("embedding", "unknown")
                f.write(f"## {backend.upper()} + {embedding.upper()} Details\n\n")

                if "error" in result:
                    f.write(f"**Error:** {result['error']}\n\n")
                    continue

                # Embedding info
                model = result.get("embedding_model", "unknown")
                dim = result.get("embedding_dim", 0)
                f.write(f"**Embedding:** {embedding} - {model} ({dim} dimensions)\n\n")

                # Upsert
                if "upsert" in result:
                    upsert = result["upsert"]
                    f.write("### Upsert\n\n")
                    f.write(f"- **Duration:** {format_duration(upsert.get('duration', 0))}\n")
                    f.write(f"- **Throughput:** {upsert.get('docs_per_sec', 0):.2f} docs/sec\n")
                    f.write(
                        "- **Note:** Upsert creates new documents or updates existing ones (can be run repeatedly)\n\n"
                    )

                # Individual Create
                if "individual_create" in result:
                    ic = result["individual_create"]
                    f.write("### Individual Create\n\n")
                    f.write(f"- **Average Duration:** {format_duration(ic.get('avg_duration', 0))}\n")
                    f.write(f"- **Sample Size:** {ic.get('sample_size', 0)} documents\n\n")

                # Vector Search
                if "vector_search" in result:
                    vs = result["vector_search"]
                    f.write("### Vector Search\n\n")
                    f.write(f"- **Average Duration:** {format_duration(vs.get('avg_duration', 0))}\n")
                    f.write(f"- **Queries Tested:** {vs.get('queries', 0)}\n\n")

                # Metadata Search
                if "metadata_search" in result:
                    ms = result["metadata_search"]
                    if ms.get("supported"):
                        f.write("### Metadata-Only Search\n\n")
                        f.write(f"- **Average Duration:** {format_duration(ms.get('avg_duration', 0))}\n")
                        f.write(f"- **Queries Tested:** {ms.get('queries', 0)}\n\n")
                    else:
                        f.write("### Metadata-Only Search\n\n")
                        f.write("- **Status:** Not supported\n\n")

                # Query DSL Operators
                if "query_dsl_operators" in result:
                    qo = result["query_dsl_operators"]
                    if qo.get("supported") is not False:
                        f.write("### Query DSL Operators (Q Objects)\n\n")
                        f.write(f"- **Average Duration:** {format_duration(qo.get('avg_duration', 0))}\n")
                        f.write(
                            f"- **Operators Tested:** {qo.get('operators_tested', 0)}/{qo.get('total_operators', 0)}\n"
                        )
                        f.write("- **Operators:** eq, ne, gt, gte, lt, lte, in, nin, and, or\n\n")
                    else:
                        f.write("### Query DSL Operators\n\n")
                        f.write("- **Status:** Not supported\n\n")

                # Update
                if "update" in result:
                    if isinstance(result["update"], dict) and "error" in result["update"]:
                        f.write("### Update Operations\n\n")
                        f.write(f"- **Status:** Not run ({result['update']['error']})\n\n")
                    else:
                        up = result["update"]
                        f.write("### Update Operations\n\n")
                        f.write(f"- **Average Duration:** {format_duration(up.get('avg_duration', 0))}\n")
                        f.write(f"- **Sample Size:** {up.get('sample_size', 0)} documents\n\n")

                # Delete
                if "delete" in result:
                    if isinstance(result["delete"], dict) and "error" in result["delete"]:
                        f.write("### Delete Operations\n\n")
                        f.write(f"- **Status:** Not run ({result['delete']['error']})\n\n")
                    else:
                        dl = result["delete"]
                        f.write("### Delete Operations\n\n")
                        f.write(f"- **Duration:** {format_duration(dl.get('duration', 0))}\n")
                        f.write(f"- **Throughput:** {dl.get('docs_per_sec', 0):.2f} docs/sec\n")
                        f.write(f"- **Sample Size:** {dl.get('sample_size', 0)} documents\n\n")
                f.write("---\n\n")

            # Error Summary Section
            error_results = {k: v for k, v in self.results.items() if "error" in v}
            if error_results:
                f.write("## Failed Tests\n\n")
                for result_key, result in error_results.items():
                    backend = result.get("backend", result_key.split("_")[0])
                    embedding = result.get("embedding", result_key.split("_")[1] if "_" in result_key else "unknown")
                    error_msg = result["error"]
                    f.write(f"### {backend.upper()} + {embedding.upper()}\n\n")
                    f.write(f"**Error:** {error_msg}\n\n")

            f.write("## Notes\n\n")
            f.write("- Tests use specified embedding providers with their default models\n")
            f.write("- Upsert operations create new documents or update existing ones (can be run repeatedly)\n")
            f.write("- Search operations retrieve 100 results per query\n")
            f.write("- Times are averaged over multiple runs for stability\n")
            f.write("- Different embedding providers may have different dimensions and performance characteristics\n")

            f.write("\n## Important: Benchmark Results Interpretation\n\n")
            f.write("**PgVector Local vs Cloud Backends:**\n")
            f.write("- **PgVector results**: Benchmarked against LOCAL PostgreSQL instance\n")
            f.write("  - Provides optimal latency with minimal network overhead\n")
            f.write("  - Results are NOT directly comparable with cloud-hosted PgVector\n")
            f.write("- **Cloud Backends** (AstraDB, Milvus, ChromaDB): Performance affected by:\n")
            f.write("  - Network latency and geographic region\n")
            f.write("  - Regional proximity between client and server\n")
            f.write("  - Network conditions and bandwidth availability\n")
            f.write("  - Server load and resource allocation\n\n")
            f.write("**For Fair Comparison:**\n")
            f.write("- Deploy all backends in the SAME REGION and NETWORK ENVIRONMENT\n")
            f.write("- Conduct benchmarks in YOUR PRODUCTION ENVIRONMENT with real network conditions\n")
            f.write("- Account for network latency when interpreting and comparing results\n")
            f.write("- PgVector: Consider cloud-hosted options for fair comparison (e.g., AWS RDS, Azure Database)\n\n")
            f.write("**Recommendation:**\n")
            f.write("These results are specific to this test environment. For your use case, run benchmarks\n")
            f.write("in your actual production deployment with backends in the same region and network\n")
            f.write("conditions to get accurate, meaningful performance metrics.\n")

        print(f"\nMarkdown report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark CrossVector database adapters")
    parser.add_argument("--num-docs", type=int, default=1000, help="Number of documents to test with (default: 1000)")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["pgvector", "astradb", "milvus", "chroma"],
        help="Specific backends to test (default: all available)",
    )
    parser.add_argument(
        "--embedding-providers",
        nargs="+",
        choices=["openai", "gemini"],
        help="Specific embedding providers to test (default: all available)",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow cloud backends (astradb, milvus) for faster testing",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=100,
        help="Number of results to return in search operations (default: 100)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Custom collection name (default: auto-generate with UUID8)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per backend test in seconds (default: 60). Cloud backends may need 120-300 seconds",
    )
    parser.add_argument("--output", type=str, default="benchmark.md", help="Output markdown file path")
    parser.add_argument(
        "--use-fixtures",
        type=str,
        default=None,
        help="Path to pre-generated fixtures JSON file (replaces generated documents)",
    )
    parser.add_argument(
        "--add-vectors",
        action="store_true",
        help="Generate and add vectors to fixture documents (skips embedding API calls)",
    )

    args = parser.parse_args()

    # Auto-generate fixtures for all embedding providers that will be used
    # Determine which providers to generate fixtures for
    providers_to_use = args.embedding_providers if args.embedding_providers else ["openai", "gemini"]

    # Generate fixtures for each provider if they don't exist
    if not args.use_fixtures:
        for embedding_provider in providers_to_use:
            fixture_path = Path(f"scripts/benchmark/data/{embedding_provider}_{args.num_docs}.json")

            if not fixture_path.exists():
                print(f"\n{'=' * 60}")
                print(f"Fixture not found: {fixture_path}")
                print(f"Auto-generating fixtures with {embedding_provider.upper()} embeddings...")
                print(f"{'=' * 60}\n")

                # Generate fixtures directly using imported functions
                try:
                    # Generate documents and queries (same for all providers)
                    num_queries = min(args.num_docs // 10, 100)
                    print(f"Generating {args.num_docs:,} documents with nested metadata...")
                    docs = generate_benchmark_docs(args.num_docs, seed=42)
                    print(f"Generated {len(docs):,} documents\n")

                    print(f"Generating {num_queries:,} diverse search queries...")
                    queries = generate_search_queries(num_queries, seed=42)
                    print(f"Generated {len(queries):,} queries\n")

                    # Generate vectors using embedding provider
                    print(f"Generating vectors using {embedding_provider.upper()} embedding...")
                    if embedding_provider == "openai":
                        from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

                        embedding_adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
                    else:  # gemini
                        from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

                        embedding_adapter = GeminiEmbeddingAdapter(model_name="gemini-embedding-001", dim=1536)

                    print(f"Model: {embedding_adapter.model_name}")
                    print(f"Dimension: {embedding_adapter.dim}")

                    # Generate vectors in batches
                    batch_size = 500
                    total_docs = len(docs)
                    for i in range(0, total_docs, batch_size):
                        batch = docs[i : i + batch_size]
                        batch_texts = [doc["text"] for doc in batch]
                        print(
                            f"Processing batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch)} docs)..."
                        )
                        vectors = embedding_adapter.get_embeddings(batch_texts)
                        for doc, vector in zip(batch, vectors):
                            doc["vector"] = vector

                    print(f"Generated {total_docs:,} vectors using {embedding_provider.upper()}\n")

                    # Save to file
                    fixture_path.parent.mkdir(parents=True, exist_ok=True)
                    total_text_length = sum(len(doc["text"]) for doc in docs)
                    fixtures = {
                        "metadata": {
                            "version": "1.0",
                            "generated_at": datetime.now().isoformat(),
                            "total_documents": len(docs),
                            "total_queries": len(queries),
                            "total_text_size_bytes": total_text_length,
                            "average_text_length": round(total_text_length / len(docs), 1),
                            "categories": list(set(doc["category"] for doc in docs)),
                            "num_categories": len(set(doc["category"] for doc in docs)),
                        },
                        "documents": docs,
                        "queries": queries,
                    }

                    with open(fixture_path, "w") as f:
                        json.dump(fixtures, f, indent=2)

                    file_size_mb = fixture_path.stat().st_size / 1024 / 1024
                    print(f"Fixtures saved: {fixture_path} ({file_size_mb:.1f} MB)\n")

                except Exception as e:
                    print(f"Failed to generate fixtures for {embedding_provider}: {e}")
                    print("Will try to continue with other providers...\n")

    # Load test documents from fixtures (if using --use-fixtures)
    # Otherwise, fixtures were already generated above for each provider
    test_docs = None
    search_queries = None

    print(f"\n{'=' * 60}")
    if args.use_fixtures:
        fixture_path = Path(args.use_fixtures)
        if fixture_path.exists():
            print(f"Loading fixtures from {fixture_path}...")
            try:
                test_docs, search_queries = load_fixtures_from_file(
                    str(fixture_path), args.num_docs, add_vectors=args.add_vectors
                )
                print(f"{'=' * 60}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Failed to load fixtures: {e}")
                print(f"{'=' * 60}")
                return
        else:
            print(f"Fixtures file not found: {fixture_path}")
            print(f"{'=' * 60}")
            return
    else:
        # Fixtures already generated per provider above
        # We'll load them per-provider in the benchmark loop
        print("Fixtures generated for each embedding provider")
        print(f"{'=' * 60}")

    # Create BenchmarkRunner instance
    # Constructor will auto-generate random collection_name if not provided
    runner = BenchmarkRunner(
        num_docs=args.num_docs,
        backends=args.backends,
        embedding_providers=args.embedding_providers,
        skip_slow=args.skip_slow,
        search_limit=args.search_limit,
        collection_name=args.collection_name,
    )

    # Load fixtures per embedding provider (if not using --use-fixtures)
    fixtures_by_provider = {}
    if not args.use_fixtures:
        for emb_name in runner.embedding_providers.keys():
            fixture_path = Path(f"scripts/benchmark/data/{emb_name}_{args.num_docs}.json")
            if fixture_path.exists():
                try:
                    docs, queries = load_fixtures_from_file(str(fixture_path), args.num_docs, add_vectors=False)
                    fixtures_by_provider[emb_name] = {"docs": docs, "queries": queries}
                except Exception as e:
                    print(f"Failed to load fixtures for {emb_name}: {e}")
    else:
        # Use the same fixtures for all providers (from --use-fixtures)
        for emb_name in runner.embedding_providers.keys():
            fixtures_by_provider[emb_name] = {"docs": test_docs, "queries": search_queries}

    # Pre-compute query vectors for each embedding provider
    query_vectors_by_embedding = {}
    for emb_name in runner.embedding_providers.keys():
        embedding_init = runner.embedding_providers[emb_name]
        embedding = embedding_init()
        if embedding and emb_name in fixtures_by_provider:
            dim = getattr(embedding, "dim", 1536)
            queries_list = fixtures_by_provider[emb_name]["queries"]
            query_vectors_by_embedding[emb_name] = runner._precompute_query_vectors(queries_list, dim)
            print(f"Pre-computed query vectors for {emb_name} (dim={dim})")

    # Run benchmarks with fixtures per provider
    # Modified run_all to handle per-provider fixtures
    print(f"\n{'=' * 60}")
    print("CrossVector Benchmark Suite")
    print(f"{'=' * 60}")
    print(f"Documents per test: {args.num_docs}")
    print(f"Backends: {', '.join(runner.backends.keys())}")
    print(f"Embeddings: {', '.join(runner.embedding_providers.keys())}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for emb_name, emb_init_func in runner.embedding_providers.items():
        embedding = emb_init_func()
        if not embedding:
            continue

        # Get fixtures for this specific provider
        if emb_name in fixtures_by_provider:
            provider_docs = fixtures_by_provider[emb_name]["docs"]
            # Attach vectors to documents
            docs_with_vectors = copy.deepcopy(provider_docs)
            runner._precompute_doc_embeddings(docs_with_vectors, embedding)
        else:
            docs_with_vectors = None

        # Get pre-computed query vectors for this embedding
        emb_query_vectors = query_vectors_by_embedding.get(emb_name)

        for backend_name, init_func in runner.backends.items():
            result_key = f"{backend_name}_{emb_name}"
            try:
                runner.results[result_key] = runner.benchmark_backend(
                    backend_name,
                    init_func,
                    emb_name,
                    embedding,
                    pre_docs=docs_with_vectors,
                    query_vectors=emb_query_vectors,
                )
            except Exception as e:
                # Skip failed backends gracefully instead of crashing
                error_msg = str(e)[:100]
                print(f"\nSkipping {backend_name}_{emb_name}: {error_msg}...")
                runner.results[result_key] = {"error": error_msg}

    # Generate report
    runner.generate_markdown_report(output_file=args.output)

    print(f"\n{'=' * 60}")
    print("Benchmark completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
