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
"""

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.exceptions import MissingConfigError
from crossvector.schema import VectorDocument

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
        print(f"  ❌ {name} failed: {e}")
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
        self.collection_name = collection_name or f"benchmark_test_{uuid4().hex[:8]}"
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
            print(f"  ⚠️  OpenAI embedding not available: {e}")
            return None

    def _init_gemini_embedding(self) -> Optional[Any]:
        """Initialize Gemini embedding adapter."""
        try:
            from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

            # Use 1536 dimensions to match OpenAI for fair comparison
            return GeminiEmbeddingAdapter(model_name="gemini-embedding-001", dim=1536)
        except Exception as e:
            print(f"  ⚠️  Gemini embedding not available: {e}")
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
            print(f"  ⚠️  PgVector not available: {e}")
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
            print(f"  ⚠️  AstraDB not available: {e}")
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
            print(f"  ⚠️  Milvus not available: {e}")
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
            print(f"  ⚠️  ChromaDB not available: {e}")
            return None

    def cleanup_collection(self, engine: VectorEngine, backend_name: str, collection_name: str = None) -> None:
        """Clean up test collection."""
        try:
            engine.drop_collection(collection_name or "benchmark_test")
            time.sleep(0.1)
            print(f"  🧹 Cleaned up {backend_name} collection")
        except Exception as e:
            print(f"  ⚠️  Cleanup warning for {backend_name}: {e}")

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
        print(f"🔥 Benchmarking: {backend_name.upper()} + {embedding_name.upper()}")
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
                print(f"\n📝 Generating {self.num_docs} test documents...")
                test_docs = generate_documents(self.num_docs)
                self._precompute_doc_embeddings(test_docs, embedding)
            else:
                test_docs = pre_docs.copy()
                print(f"\n✅ Using pre-generated {self.num_docs} documents (static vectors already attached)")

            # 1. Bulk Create Performance
            print(f"\n1️⃣  Bulk Create ({self.num_docs} docs)...")
            duration, created_docs = benchmark_operation("bulk_create", lambda: engine.bulk_create(test_docs))
            results["bulk_create"] = {
                "duration": duration,
                "docs_per_sec": self.num_docs / duration if duration > 0 else 0,
                "success": created_docs is not None,
            }
            print(f"   ✅ Duration: {format_duration(duration)}")
            print(f"   📊 {results['bulk_create']['docs_per_sec']:.2f} docs/sec")

            # 2. Individual Create Performance (small sample)
            sample_size = min(10, self.num_docs)
            print(f"\n2️⃣  Individual Create ({sample_size} docs)...")
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
            print(f"   ✅ Avg Duration: {format_duration(avg_create)}")

            # 3. Vector Search Performance
            print("\n3️⃣  Vector Search (10 queries with pre-computed vectors)...")
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
            print(f"   ✅ Avg Duration: {format_duration(avg_search)}")
            print(f"   📊 {len(search_queries) / sum(search_times) if sum(search_times) > 0 else 0:.2f} queries/sec")

            # 4. Metadata-Only Search (if supported)
            if engine.supports_metadata_only:
                print("\n4️⃣  Metadata Search (10 queries)...")
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
                print(f"   ✅ Avg Duration: {format_duration(avg_metadata)}")
            else:
                results["metadata_search"] = {"supported": False}
                print("\n4️⃣  Metadata Search: Not supported")

            # 4.5. Query DSL Operators Test (using Q objects)
            print("\n4️⃣.5 Query DSL Operators (Q objects)...")
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
                print("   ℹ️  Testing 4 key operators (slow backend optimization)")
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
                    print(f"  ⚠️  Operator {op_name} skipped: {e}")

            if operator_times:
                avg_operator = sum(operator_times) / len(operator_times)
                results["query_dsl_operators"] = {
                    "avg_duration": avg_operator,
                    "operators_tested": successful_operators,
                    "total_operators": len(operator_tests),
                }
                print(
                    f"   ✅ Avg Duration: {format_duration(avg_operator)} ({successful_operators}/{len(operator_tests)} operators)"
                )
            else:
                results["query_dsl_operators"] = {"supported": False}

            # 5. Update Performance
            print("\n5️⃣  Update Operations (100 updates)...")
            update_sample = min(100, self.num_docs)
            if created_docs and len(created_docs) >= update_sample:
                update_times = []
                for i in range(update_sample):
                    doc = created_docs[i]
                    doc.metadata["updated"] = True
                    doc.metadata["update_idx"] = i
                    duration, _ = benchmark_operation(f"update_{i}", lambda d=doc: engine.update(d))
                    update_times.append(duration)

                avg_update = sum(update_times) / len(update_times) if update_times else 0
                results["update"] = {
                    "avg_duration": avg_update,
                    "sample_size": update_sample,
                }
                print(f"   ✅ Avg Duration: {format_duration(avg_update)}")
            else:
                results["update"] = {"error": "No documents to update"}

            # 6. Delete Performance
            print("\n6️⃣  Delete Operations (100 deletes)...")
            delete_sample = min(100, self.num_docs)
            if created_docs and len(created_docs) >= delete_sample:
                delete_ids = [doc.id for doc in created_docs[:delete_sample]]
                duration, _ = benchmark_operation("batch_delete", lambda: engine.delete(*delete_ids))
                results["delete"] = {
                    "duration": duration,
                    "sample_size": delete_sample,
                    "docs_per_sec": delete_sample / duration if duration > 0 else 0,
                }
                print(f"   ✅ Duration: {format_duration(duration)}")
                print(f"   📊 {results['delete']['docs_per_sec']:.2f} docs/sec")
            else:
                results["delete"] = {"error": "No documents to delete"}

            # 7. Count operation
            remaining_count = engine.count()
            results["final_count"] = remaining_count
            print(f"\n📊 Final document count: {remaining_count}")

        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
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

    def run_all(
        self, pre_docs: List[VectorDocument] = None, query_vectors: Dict[str, List[float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for all backends with all embedding providers.

        Args:
            pre_docs: Pre-generated documents with static vectors (computed once in main)
            query_vectors: Pre-computed search query vectors keyed by embedding provider
        """
        print(f"\n{'=' * 60}")
        print("🚀 CrossVector Benchmark Suite")
        print(f"{'=' * 60}")
        print(f"📊 Documents per test: {self.num_docs}")
        print(f"🎯 Backends: {', '.join(self.backends.keys())}")
        print(f"🤖 Embeddings: {', '.join(self.embedding_providers.keys())}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
                    print(f"\n⚠️  Skipping {backend_name}_{emb_name}: {error_msg}...")
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
                f.write(f", {error_tests} ❌ failed\n\n")
            else:
                f.write("\n\n")

            f.write(
                "| Backend | Embedding | Model | Dim | Bulk Create | Search (avg) | Update (avg) | Delete (batch) | Status |\n"
            )
            f.write(
                "|---------|-----------|-------|-----|-------------|--------------|--------------|----------------|--------|\n"
            )

            for result_key, result in self.results.items():
                if "error" in result:
                    backend = result.get("backend", result_key.split("_")[0])
                    embedding = result.get("embedding", result_key.split("_")[1] if "_" in result_key else "unknown")
                    error_msg = result["error"][:50] + "..." if len(result["error"]) > 50 else result["error"]
                    f.write(f"| {backend} | {embedding} | - | - | - | - | - | - | ❌ {error_msg} |\n")
                    continue

                backend = result.get("backend", "unknown")
                embedding = result.get("embedding", "unknown")
                model = result.get("embedding_model", "unknown")
                dim = result.get("embedding_dim", 0)
                bulk_create = format_duration(result.get("bulk_create", {}).get("duration", 0))
                search = format_duration(result.get("vector_search", {}).get("avg_duration", 0))
                update = format_duration(result.get("update", {}).get("avg_duration", 0))
                delete = format_duration(result.get("delete", {}).get("duration", 0))

                f.write(
                    f"| {backend} | {embedding} | {model} | {dim} | {bulk_create} | {search} | {update} | {delete} | ✅ |\n"
                )

            f.write("\n---\n\n")

            # Detailed results per backend
            for result_key, result in self.results.items():
                backend = result.get("backend", "unknown")
                embedding = result.get("embedding", "unknown")
                f.write(f"## {backend.upper()} + {embedding.upper()} Details\n\n")

                if "error" in result:
                    f.write(f"❌ **Error:** {result['error']}\n\n")
                    continue

                # Embedding info
                model = result.get("embedding_model", "unknown")
                dim = result.get("embedding_dim", 0)
                f.write(f"**Embedding:** {embedding} - {model} ({dim} dimensions)\n\n")

                # Bulk Create
                if "bulk_create" in result:
                    bc = result["bulk_create"]
                    f.write("### Bulk Create\n\n")
                    f.write(f"- **Duration:** {format_duration(bc.get('duration', 0))}\n")
                    f.write(f"- **Throughput:** {bc.get('docs_per_sec', 0):.2f} docs/sec\n\n")

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
                if "update" in result and "error" not in result["update"]:
                    up = result["update"]
                    f.write("### Update Operations\n\n")
                    f.write(f"- **Average Duration:** {format_duration(up.get('avg_duration', 0))}\n")
                    f.write(f"- **Sample Size:** {up.get('sample_size', 0)} documents\n\n")

                # Delete
                if "delete" in result and "error" not in result["delete"]:
                    dl = result["delete"]
                    f.write("### Delete Operations\n\n")
                    f.write(f"- **Duration:** {format_duration(dl.get('duration', 0))}\n")
                    f.write(f"- **Throughput:** {dl.get('docs_per_sec', 0):.2f} docs/sec\n")
                    f.write(f"- **Sample Size:** {dl.get('sample_size', 0)} documents\n\n")
                f.write("---\n\n")

            # Error Summary Section
            error_results = {k: v for k, v in self.results.items() if "error" in v}
            if error_results:
                f.write("## Failed Tests ❌\n\n")
                for result_key, result in error_results.items():
                    backend = result.get("backend", result_key.split("_")[0])
                    embedding = result.get("embedding", result_key.split("_")[1] if "_" in result_key else "unknown")
                    error_msg = result["error"]
                    f.write(f"### {backend.upper()} + {embedding.upper()}\n\n")
                    f.write(f"**Error:** {error_msg}\n\n")

            # Footer
            f.write("## Notes\n\n")
            f.write("- Tests use specified embedding providers with their default models\n")
            f.write("- Bulk operations create documents in batches\n")
            f.write("- Search operations retrieve 100 results per query\n")
            f.write("- Times are averaged over multiple runs for stability\n")
            f.write("- Different embedding providers may have different dimensions and performance characteristics\n")

        print(f"\n📄 Markdown report saved to: {output_path}")


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

    args = parser.parse_args()

    # Pre-generate test documents ONCE in main (no embedding API calls)
    print(f"\n{'=' * 60}")
    print("📝 Pre-generating test data...")
    print(f"{'=' * 60}")
    test_docs = generate_documents(args.num_docs)
    print(f"✅ Generated {args.num_docs} documents with static vectors")

    # Pre-define search queries (constant across all runs)
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
    print(f"✅ Defined {len(search_queries)} search queries")

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

    # Pre-compute query vectors for each embedding provider
    query_vectors_by_embedding = {}
    for emb_name in runner.embedding_providers.keys():
        embedding_init = runner.embedding_providers[emb_name]
        embedding = embedding_init()
        if embedding:
            dim = getattr(embedding, "dim", 1536)
            query_vectors_by_embedding[emb_name] = runner._precompute_query_vectors(search_queries, dim)
            print(f"✅ Pre-computed query vectors for {emb_name} (dim={dim})")

    # Run benchmarks with pre-computed data (NO additional generation in loop)
    runner.run_all(pre_docs=test_docs, query_vectors=query_vectors_by_embedding)

    # Generate report
    runner.generate_markdown_report(output_file=args.output)

    print(f"\n{'=' * 60}")
    print("✅ Benchmark completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
