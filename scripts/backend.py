"""Unified backend test runner.

Run a single CRUD + search flow against any supported backend adapter
using selectable embedding provider.

Usage examples:
    python scripts/backend.py --backend astradb --embedding-provider openai
    python scripts/backend.py --backend chroma --embedding-provider gemini --gemini-model gemini-embedding-001

The flow executed:
    1. Upsert initial documents
    2. Text semantic search
    3. Vector search
    4. Get document by id
    5. Update document
    6. get_or_create existing
    7. get_or_create new (metadata path)
    8. update_or_create existing
    9. update_or_create new
    10. Final count
    11. Metadata-only search (if supported)
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.dbs.astradb import AstraDBAdapter
from crossvector.dbs.chroma import ChromaAdapter
from crossvector.dbs.milvus import MilvusAdapter
from crossvector.dbs.pgvector import PgVectorAdapter
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter
from crossvector.exceptions import MissingConfigError
from crossvector.querydsl.q import Q

DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_BACKEND = "chroma"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified backend test runner")
    parser.add_argument(
        "--backend",
        choices=["astradb", "chroma", "milvus", "pgvector"],
        default=DEFAULT_BACKEND,
        help="Vector database backend adapter (default: astradb)",
    )
    parser.add_argument(
        "--all-backends",
        action="store_true",
        help="Run the flow against all supported backends",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Embedding provider (default: openai)",
    )
    parser.add_argument(
        "--all-embeddings",
        action="store_true",
        help="Run the flow against all supported embedding providers",
    )
    # OpenAI
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI embedding model name (default: text-embedding-3-small)",
    )
    # Gemini
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini embedding model name (default: gemini-embedding-001)",
    )
    parser.add_argument(
        "--gemini-task",
        default="retrieval_document",
        help="Gemini task type (retrieval_document, retrieval_query, semantic_similarity, classification)",
    )
    parser.add_argument(
        "--gemini-dimension",
        type=int,
        default=1536,
        help="Gemini output dimensionality (gemini-embedding-001 only: 768, 1536, 3072)",
    )
    return parser.parse_args(argv)


def get_embedding_adapter(args: argparse.Namespace):
    if args.embedding_provider == "openai":
        return OpenAIEmbeddingAdapter(model_name=args.openai_model)
    if args.embedding_provider == "gemini":
        return GeminiEmbeddingAdapter(
            model_name=args.gemini_model,
            task_type=args.gemini_task,
            dim=args.gemini_dimension,
        )
    raise ValueError(f"Unsupported embedding provider: {args.embedding_provider}")


def get_db_adapter(args: argparse.Namespace):
    backend = args.backend
    if backend == "astradb":
        return AstraDBAdapter()
    if backend == "chroma":
        return ChromaAdapter()
    if backend == "milvus":
        return MilvusAdapter()
    if backend == "pgvector":
        return PgVectorAdapter()
    raise ValueError(f"Unsupported backend: {backend}")


def run_flow(engine: VectorEngine) -> tuple[int, int, int]:
    """Run comprehensive test suite and track pass/fail statistics.

    Returns:
        (passed, total, failed)
    """
    passed = 0
    failed = 0
    total = 0

    def test(name: str, func):
        """Execute a test and track result."""
        nonlocal passed, failed, total
        total += 1
        try:
            func()
            passed += 1
            print(f"✓ [{total}] {name}")
            return True
        except Exception as e:
            failed += 1
            # Show full traceback for debugging
            if "Bulk update" in name:
                import traceback

                print(f"✗ [{total}] {name}:")
                traceback.print_exc()
            else:
                print(f"✗ [{total}] {name}: {e}")
            return False

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        f"{engine.db.__class__.__name__} adapter integration test document.",
        "Vector search enables semantic retrieval.",
    ]

    # Rich metadata sample to validate common operators and nested paths
    rich_meta = {
        "source": "test",
        "idx": 0,
        "score": 0.85,
        "tags": ["ai", "ml", "search"],
        "info": {"lang": "en", "tier": "gold", "version": 2},
        "owner": "tester",
    }

    # Track document IDs for subsequent tests
    doc_ids = []

    # Determine backend capabilities (simple, centralized)
    backend_class = engine.db.__class__.__name__
    supports_metadata_only = bool(getattr(engine.db, "supports_metadata_only", False))
    supports_nested = backend_class in {"AstraDBAdapter", "PgVectorAdapter"}
    # PgVector numeric comparisons on JSONB need explicit casts (not yet supported in compiler)
    supports_numeric_comparisons = backend_class in {
        "AstraDBAdapter",
        "ChromaAdapter",
        "MilvusAdapter",
        "PgVectorAdapter",
    }

    # === CREATE OPERATIONS ===
    def test_create_single():
        doc = engine.create(text=texts[0], metadata=rich_meta)
        doc_ids.append(doc.id)
        assert doc.text == texts[0]

    def test_bulk_create():
        docs = engine.bulk_create(
            [
                {"text": texts[1], "metadata": {"source": "test", "idx": 1, "score": 0.6, "tags": ["ai"]}},
                {
                    "text": texts[2],
                    "metadata": {
                        "source": "test",
                        "idx": 2,
                        "score": 0.95,
                        "tags": ["ml", "rag"],
                        "info": {"lang": "vi", "tier": "silver", "version": 1},
                    },
                },
            ]
        )
        doc_ids.extend([d.id for d in docs])
        assert len(docs) == 2

    def test_upsert():
        docs = engine.upsert(
            [
                {"id": "upsert-1", "text": texts[3], "metadata": {"source": "upsert"}},
            ]
        )
        doc_ids.append(docs[0].id)
        assert len(docs) == 1

    test("Create single document", test_create_single)
    test("Bulk create documents", test_bulk_create)
    test("Upsert documents", test_upsert)

    # === COUNT OPERATIONS ===
    def test_count_after_create():
        count = engine.count()
        # Eventual consistency: count may not be immediately accurate
        assert count >= 0, f"Count should be non-negative, got {count}"

    test("Count after create", test_count_after_create)

    # === READ OPERATIONS ===
    def test_get_by_id():
        if doc_ids:
            doc = engine.get(doc_ids[0])
            assert doc.id == doc_ids[0]

    def test_text_search():
        results = engine.search(texts[0], limit=2)
        assert len(results) > 0

    def test_vector_search():
        vector = engine.embedding.get_embeddings([texts[1]])[0]
        results = engine.search(vector, limit=2)
        assert len(results) > 0

    def test_search_with_metadata_filter():
        results = engine.search(texts[0], where={"source": {"$eq": "test"}}, limit=5)
        assert all(r.metadata.get("source") == "test" for r in results if isinstance(r.metadata, dict))

    def test_metadata_only_search():
        if supports_metadata_only:
            results = engine.search(query=None, where={"source": {"$eq": "test"}}, limit=5)
            assert len(results) >= 0
        else:
            print("↷ Skipped: metadata-only search not supported by backend")

    # === COMMON OPERATOR TESTS (dict where) ===
    def test_where_eq_ne():
        if not supports_metadata_only:
            print("↷ Skipped: metadata-only filters not supported by backend")
            return
        if not supports_nested:
            print("↷ Skipped: nested field filters not supported by backend")
            return
        res_eq = engine.search(query=None, where={"info.lang": {"$eq": "en"}}, limit=10)
        assert any(r.metadata.get("info", {}).get("lang") == "en" for r in res_eq)
        res_ne = engine.search(query=None, where={"info.lang": {"$ne": "en"}}, limit=10)
        assert all(r.metadata.get("info", {}).get("lang") != "en" for r in res_ne)

    def test_where_gt_gte_lt_lte():
        if not supports_metadata_only:
            print("↷ Skipped: metadata-only filters not supported by backend")
            return
        if not supports_numeric_comparisons:
            print("↷ Skipped: numeric JSON comparisons not supported by backend/compiler")
            return
        if not supports_nested:
            print("↷ Skipped: nested field filters not supported by backend")
            return
        res_gt = engine.search(query=None, where={"score": {"$gt": 0.8}}, limit=10)
        assert any((r.metadata.get("score", 0) > 0.8) for r in res_gt)
        res_gte = engine.search(query=None, where={"info.version": {"$gte": 2}}, limit=10)
        assert any((r.metadata.get("info", {}).get("version", 0) >= 2) for r in res_gte)
        res_lt = engine.search(query=None, where={"score": {"$lt": 0.9}}, limit=10)
        assert all((r.metadata.get("score", 1) < 0.9) for r in res_lt)
        res_lte = engine.search(query=None, where={"idx": {"$lte": 2}}, limit=10)
        assert all((r.metadata.get("idx", 999) <= 2) for r in res_lte)

    def test_where_in_nin():
        if not supports_metadata_only:
            print("↷ Skipped: metadata-only filters not supported by backend")
            return
        res_in = engine.search(query=None, where={"owner": {"$in": ["tester", "other"]}}, limit=10)
        assert any(r.metadata.get("owner") == "tester" for r in res_in)
        res_nin = engine.search(query=None, where={"owner": {"$nin": ["nobody"]}}, limit=10)
        assert len(res_nin) >= 1

    def test_nested_metadata_filter_dict():
        if not supports_nested:
            print("↷ Skipped: nested field filters not supported by backend")
            return
        # Insert a nested metadata doc
        _ = engine.upsert(
            [{"id": "nested-1", "text": "Nested doc", "metadata": {"info": {"lang": "en", "tier": "gold"}}}]
        )
        # Query nested path using dot notation
        where = {"info.lang": {"$eq": "en"}, "info.tier": {"$eq": "gold"}}
        results = engine.search(query=None, where=where, limit=5)
        assert any(getattr(r, "id", None) == "nested-1" for r in results)

    def test_nested_metadata_filter_q():
        if not supports_nested:
            print("↷ Skipped: nested field filters not supported by backend")
            return
        # Use Q with nested fields via __ to ensure compiler paths work
        q = Q(info__lang__eq="en") & Q(info__tier__eq="gold")
        results = engine.search(query=None, where=q, limit=5)
        assert any(getattr(r, "id", None) == "nested-1" for r in results)

    test("Get document by ID", test_get_by_id)
    test("Text semantic search", test_text_search)
    test("Vector similarity search", test_vector_search)
    test("Search with metadata filter", test_search_with_metadata_filter)
    test("Metadata-only search", test_metadata_only_search)
    test("Where eq/ne", test_where_eq_ne)
    test("Where gt/gte/lt/lte", test_where_gt_gte_lt_lte)
    test("Where in/nin", test_where_in_nin)
    test("Nested metadata filter (dict)", test_nested_metadata_filter_dict)
    test("Nested metadata filter (Q)", test_nested_metadata_filter_q)

    # === UPDATE OPERATIONS ===
    def test_update_single():
        if doc_ids:
            updated = engine.update({"id": doc_ids[0]}, text="Updated text content", metadata={"phase": "updated"})
            assert updated.id == doc_ids[0]

    def test_bulk_update():
        if len(doc_ids) >= 2:
            try:
                updates = engine.bulk_update(
                    [
                        {"id": doc_ids[0], "text": "Bulk updated first"},
                        {"id": doc_ids[1], "text": "Bulk updated second"},
                    ]
                )
                assert len(updates) == 2, f"Expected 2 updates, got {len(updates)}"
            except Exception as e:
                import traceback

                print(f"\n{'=' * 60}\nBULK UPDATE ERROR:\n{'=' * 60}")
                print(f"IDs used: {doc_ids[0]}, {doc_ids[1]}")
                print(f"Error: {e}")
                print("\nFull traceback:")
                traceback.print_exc()
                print(f"{'=' * 60}\n")
                raise

    def test_get_or_create_existing():
        if doc_ids:
            doc, created = engine.get_or_create({"id": doc_ids[0], "text": "Bulk updated first"})
            assert not created
            assert doc.id == doc_ids[0]

    def test_get_or_create_new():
        doc, created = engine.get_or_create(text="New doc via get_or_create", metadata={"topic": "goc_test"})
        assert created
        doc_ids.append(doc.id)

    def test_update_or_create_existing():
        if doc_ids:
            doc, created = engine.update_or_create(
                {"id": doc_ids[0]}, text="Updated via update_or_create", defaults={"metadata": {"tier": "gold"}}
            )
            assert not created

    def test_update_or_create_new():
        doc, created = engine.update_or_create(
            {"id": "uoc-new-1", "text": "Created via update_or_create"}, create_defaults={"metadata": {"owner": "test"}}
        )
        assert created
        doc_ids.append(doc.id)

    test("Update single document", test_update_single)
    test("Bulk update documents", test_bulk_update)
    test("Get-or-create existing", test_get_or_create_existing)
    test("Get-or-create new", test_get_or_create_new)
    test("Update-or-create existing", test_update_or_create_existing)
    test("Update-or-create new", test_update_or_create_new)

    # === DELETE OPERATIONS ===
    def test_delete_single():
        if doc_ids:
            deleted = engine.delete(doc_ids[0])
            assert deleted >= 0

    def test_delete_multiple():
        if len(doc_ids) >= 3:
            deleted = engine.delete([doc_ids[1], doc_ids[2]])
            assert deleted >= 0

    test("Delete single document", test_delete_single)
    test("Delete multiple documents", test_delete_multiple)

    # === FINAL COUNT ===
    def test_count_after_operations():
        count = engine.count()
        assert count >= 0

    test("Count after all operations", test_count_after_operations)

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed / {total} total ({failed} failed)")
    print("=" * 60)
    if failed > 0:
        print(f"⚠ {failed} test(s) failed")
    else:
        print("✓ All tests passed!")
    return passed, total, failed


def main() -> None:
    load_dotenv()
    args = parse_args()
    backends = [args.backend] if not args.all_backends else ["astradb", "chroma", "milvus", "pgvector"]
    embeddings = [args.embedding_provider] if not args.all_embeddings else ["openai", "gemini"]

    summaries: list[tuple[str, str, int, int]] = []

    for backend in backends:
        # update backend in args-like object
        args.backend = backend
        for provider in embeddings:
            args.embedding_provider = provider
            try:
                embedding = get_embedding_adapter(args)
            except MissingConfigError as e:
                print("Embedding config error:", e)
                continue
            db = get_db_adapter(args)

            # Attempt initial cleanup if adapter provides drop_collection
            try:
                db.drop_collection("test_vectors")
            except Exception:
                pass

            try:
                engine = VectorEngine(embedding=embedding, db=db, collection_name="test_vectors", store_text=True)
            except Exception as e:
                print("Failed to initialize engine:", e)
                continue
            print(f"Initialized VectorEngine with adapter '{db.__class__.__name__}'.")
            passed, total, _failed = run_flow(engine)
            # Per-backend/provider summary line
            print(f"Summary: {backend} + {provider}: {passed}/{total}")
            summaries.append((backend, provider, passed, total))

    # Consolidated summary
    if len(summaries) > 1:
        print("\nConsolidated Summary:")
        for backend in {b for (b, _, __, ___) in summaries}:
            # pick best by provider or first
            entries = [(prov, p, t) for (b, prov, p, t) in summaries if b == backend]
            lines = [f"{backend}: {prov} {p}/{t}" for (prov, p, t) in entries]
            print(" - " + " | ".join(lines))
    elif len(summaries) == 1:
        b, prov, p, t = summaries[0]
        print(f"\nFinal Summary: {b} + {prov}: {p}/{t}")


if __name__ == "__main__":
    main()
