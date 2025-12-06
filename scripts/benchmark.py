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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.exceptions import MissingConfigError

load_dotenv()

# Sample data generator
SAMPLE_TEXTS = [
    "Python programming language and software development",
    "Machine learning and artificial intelligence applications",
    "Web development with modern frameworks",
    "Database design and optimization techniques",
    "Cloud computing and distributed systems",
    "Data science and statistical analysis",
    "Cybersecurity and network protection",
    "Mobile application development",
    "DevOps and continuous integration",
    "Software architecture and design patterns",
]


def generate_documents(num_docs: int) -> List[Dict[str, Any]]:
    """Generate test documents with varied content."""
    docs = []
    for i in range(num_docs):
        text_idx = i % len(SAMPLE_TEXTS)
        docs.append(
            {
                "text": f"{SAMPLE_TEXTS[text_idx]} - Document {i}",
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
    ):
        """Initialize benchmark runner.

        Args:
            num_docs: Number of documents to use in benchmarks
            backends: List of backend names to test (None = all available)
            embedding_providers: List of embedding providers to test (None = all available)
            skip_slow: If True, skip slow cloud backends (astradb, milvus)
        """
        self.num_docs = num_docs
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
            return GeminiEmbeddingAdapter(model_name="text-embedding-004", dim=1536)
        except Exception as e:
            print(f"  ⚠️  Gemini embedding not available: {e}")
            return None

    def _init_pgvector(self, embedding: Any) -> Optional[VectorEngine]:
        """Initialize PgVector engine."""
        try:
            from crossvector.dbs.pgvector import PgVectorAdapter

            return VectorEngine(
                db=PgVectorAdapter(),
                embedding=embedding,
                collection_name="benchmark_test",
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"  ⚠️  PgVector not available: {e}")
            return None

    def _init_astradb(self, embedding: Any) -> Optional[VectorEngine]:
        """Initialize AstraDB engine."""
        try:
            from crossvector.dbs.astradb import AstraDBAdapter

            return VectorEngine(
                db=AstraDBAdapter(),
                embedding=embedding,
                collection_name="benchmark_test",
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"  ⚠️  AstraDB not available: {e}")
            return None

    def _init_milvus(self, embedding: Any) -> Optional[VectorEngine]:
        """Initialize Milvus engine."""
        try:
            from crossvector.dbs.milvus import MilvusAdapter

            return VectorEngine(
                db=MilvusAdapter(),
                embedding=embedding,
                collection_name="benchmark_test",
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"  ⚠️  Milvus not available: {e}")
            return None

    def _init_chroma(self, embedding: Any) -> Optional[VectorEngine]:
        """Initialize ChromaDB engine."""
        try:
            from crossvector.dbs.chroma import ChromaAdapter

            return VectorEngine(
                db=ChromaAdapter(),
                embedding=embedding,
                collection_name="benchmark_test",
                store_text=True,
            )
        except (ImportError, MissingConfigError) as e:
            print(f"  ⚠️  ChromaDB not available: {e}")
            return None

    def cleanup_collection(self, engine: VectorEngine, backend_name: str) -> None:
        """Clean up test collection."""
        try:
            engine.drop_collection("benchmark_test")
            print(f"  🧹 Cleaned up {backend_name} collection")
        except Exception as e:
            print(f"  ⚠️  Cleanup warning for {backend_name}: {e}")

    def benchmark_backend(
        self, backend_name: str, init_func: callable, embedding_name: str, embedding: Any
    ) -> Dict[str, Any]:
        """Run benchmarks for a specific backend with specific embedding provider.

        Args:
            backend_name: Name of the backend
            init_func: Function to initialize the engine
            embedding_name: Name of the embedding provider
            embedding: Embedding adapter instance

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'=' * 60}")
        print(f"🔥 Benchmarking: {backend_name.upper()} + {embedding_name.upper()}")
        print(f"{'=' * 60}")

        # Initialize engine
        engine = init_func(embedding)
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
            # Cleanup before starting
            self.cleanup_collection(engine, backend_name)

            # Re-initialize after cleanup
            engine = init_func(embedding)
            if not engine:
                return {"error": "Failed to reinitialize after cleanup"}

            # Generate test data
            print(f"\n📝 Generating {self.num_docs} test documents...")
            test_docs = generate_documents(self.num_docs)

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
            for i in range(sample_size):
                doc_data = {
                    "text": f"Individual test document {i}",
                    "metadata": {"type": "individual", "idx": i},
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
            print("\n3️⃣  Vector Search (10 queries x 10 results)...")
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
            search_times = []
            for query in search_queries:
                duration, _ = benchmark_operation(f"search_{query[:20]}", lambda q=query: engine.search(q, limit=10))
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
                        lambda: engine.search(query=None, where={"category": {"$eq": f"cat_{i % 5}"}}, limit=10),
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
                    ("eq", lambda: engine.search(query=None, where=Q(category="cat_0"), limit=10)),
                    ("gt", lambda: engine.search(query=None, where=Q(score__gt=0.5), limit=10)),
                    ("in", lambda: engine.search(query=None, where=Q(category__in=["cat_0", "cat_1"]), limit=10)),
                    ("and", lambda: engine.search(query=None, where=Q(category="cat_0") & Q(score__gte=0.5), limit=10)),
                ]
                print("   ℹ️  Testing 4 key operators (slow backend optimization)")
            else:
                # Test all operators for fast backends
                operator_tests = [
                    ("eq", lambda: engine.search(query=None, where=Q(category="cat_0"), limit=10)),
                    ("ne", lambda: engine.search(query=None, where=Q(category__ne="cat_0"), limit=10)),
                    ("gt", lambda: engine.search(query=None, where=Q(score__gt=0.5), limit=10)),
                    ("gte", lambda: engine.search(query=None, where=Q(score__gte=0.5), limit=10)),
                    ("lt", lambda: engine.search(query=None, where=Q(score__lt=0.5), limit=10)),
                    ("lte", lambda: engine.search(query=None, where=Q(score__lte=0.5), limit=10)),
                    ("in", lambda: engine.search(query=None, where=Q(category__in=["cat_0", "cat_1"]), limit=10)),
                    ("nin", lambda: engine.search(query=None, where=Q(category__nin=["cat_0", "cat_1"]), limit=10)),
                    ("and", lambda: engine.search(query=None, where=Q(category="cat_0") & Q(score__gte=0.5), limit=10)),
                    (
                        "or",
                        lambda: engine.search(query=None, where=Q(category="cat_0") | Q(category="cat_1"), limit=10),
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
                duration, _ = benchmark_operation("batch_delete", lambda: engine.delete(delete_ids))
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
            # Cleanup
            self.cleanup_collection(engine, backend_name)

        return results

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for all backends with all embedding providers."""
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

            for backend_name, init_func in self.backends.items():
                result_key = f"{backend_name}_{emb_name}"
                self.results[result_key] = self.benchmark_backend(backend_name, init_func, emb_name, embedding)

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

            # Show which backends were tested/skipped
            all_backends = ["pgvector", "astradb", "milvus", "chroma"]
            tested_backends = list(self.backends.keys())
            skipped_backends = [b for b in all_backends if b not in tested_backends]

            if skipped_backends:
                f.write(f"**Tested backends:** {', '.join(tested_backends)}\n\n")
                f.write(f"**Skipped backends:** {', '.join(skipped_backends)} ⏭️\n\n")

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

            # Footer
            f.write("## Notes\n\n")
            f.write("- Tests use specified embedding providers with their default models\n")
            f.write("- Bulk operations create documents in batches\n")
            f.write("- Search operations retrieve 10 results per query\n")
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
    parser.add_argument("--output", type=str, default="benchmark.md", help="Output markdown file path")

    args = parser.parse_args()

    # Run benchmarks
    runner = BenchmarkRunner(
        num_docs=args.num_docs,
        backends=args.backends,
        embedding_providers=args.embedding_providers,
        skip_slow=args.skip_slow,
    )
    runner.run_all()

    # Generate report
    runner.generate_markdown_report(output_file=args.output)

    print(f"\n{'=' * 60}")
    print("✅ Benchmark completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
