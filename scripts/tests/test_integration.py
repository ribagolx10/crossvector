"""
Integration test script: Test VectorEngine with real cloud APIs
This script tests the unified VectorEngine interface across different databases.
"""

from dotenv import load_dotenv

from crossvector import Document, SearchRequest, UpsertRequest, VectorEngine
from crossvector.dbs.astradb import AstraDBAdapter
from crossvector.dbs.chroma import ChromaDBAdapter
from crossvector.dbs.milvus import MilvusDBAdapter
from crossvector.dbs.pgvector import PGVectorAdapter
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Load .env
load_dotenv()

# Test data
test_docs = [
    Document(
        id="doc1",
        text="The quick brown fox jumps over the lazy dog.",
        metadata={"category": "animals", "source": "test"},
    ),
    Document(
        id="doc2",
        text="Artificial intelligence is transforming the world.",
        metadata={"category": "technology", "source": "test"},
    ),
    Document(
        id="doc3",
        text="Machine learning enables computers to learn from data.",
        metadata={"category": "technology", "source": "test"},
    ),
]


def test_engine(db_name: str, db_adapter, embedding_adapter, collection_name: str):
    """Test VectorEngine with a specific database adapter."""
    print(f"\n{'=' * 80}")
    print(f"Testing {db_name} with {embedding_adapter.model_name}")
    print(f"{'=' * 80}")

    # Initialize engine
    engine = VectorEngine(embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name=collection_name)

    # Clean up existing data (if collection exists, drop it)
    try:
        if hasattr(db_adapter, "db") and collection_name in db_adapter.db.list_collection_names():
            db_adapter.db.drop_collection(collection_name)
            print(f"✓ Dropped existing collection '{collection_name}'")
    except Exception as e:
        print(f"Note: Could not drop collection (may not exist): {e}")

    # Re-initialize after dropping
    engine = VectorEngine(embedding_adapter=embedding_adapter, db_adapter=db_adapter, collection_name=collection_name)

    # Test 1: Upsert documents
    print("\n1. Testing upsert...")
    result = engine.upsert(UpsertRequest(documents=test_docs))
    print(f"✓ Inserted {result['count']} documents")

    # Test 2: Count documents
    print("\n2. Testing count...")
    count = engine.count()
    print(f"✓ Total documents: {count}")
    assert count == len(test_docs), f"Expected {len(test_docs)} documents, got {count}"

    # Test 3: Get document by ID
    print("\n3. Testing get...")
    doc = engine.get("doc1")
    print(f"✓ Retrieved document: {doc.get('text', 'N/A')[:50]}...")
    assert doc is not None, "Document not found"

    # Test 4: Search
    print("\n4. Testing search...")
    results = engine.search(SearchRequest(query="AI and machine learning", limit=2))
    print(f"✓ Found {len(results)} results")
    for i, result in enumerate(results, 1):
        similarity = result.get("$similarity", "N/A")
        text = result.get("text", "N/A")[:50]
        print(f"  {i}. Similarity: {similarity:.4f if isinstance(similarity, float) else similarity}, Text: {text}...")

    # Test 5: Delete one
    print("\n5. Testing delete_one...")
    deleted = engine.delete_one("doc1")
    print(f"✓ Deleted {deleted} document(s)")

    # Verify deletion
    count_after_delete = engine.count()
    print(f"✓ Documents after deletion: {count_after_delete}")
    assert count_after_delete == len(test_docs) - 1, (
        f"Expected {len(test_docs) - 1} documents, got {count_after_delete}"
    )

    # Test 6: Delete many
    print("\n6. Testing delete_many...")
    deleted = engine.delete_many(["doc2", "doc3"])
    print(f"✓ Deleted {deleted} document(s)")

    # Verify all deleted
    final_count = engine.count()
    print(f"✓ Final document count: {final_count}")
    assert final_count == 0, f"Expected 0 documents, got {final_count}"

    print(f"\n✅ All tests passed for {db_name}!")


def main():
    """Run all integration tests."""
    print("Starting CrossVector Integration Tests with Real Cloud APIs")
    print("=" * 80)

    # Test with OpenAI embeddings
    openai_embedder = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")

    # Test AstraDB
    try:
        test_engine("AstraDB", AstraDBAdapter(), openai_embedder, "test_crossvector_integration")
    except Exception as e:
        print(f"\n❌ AstraDB test failed: {e}")

    # Test ChromaDB
    try:
        test_engine("ChromaDB Cloud", ChromaDBAdapter(), openai_embedder, "test_crossvector_integration")
    except Exception as e:
        print(f"\n❌ ChromaDB test failed: {e}")

    # Test Milvus
    try:
        test_engine("Milvus", MilvusDBAdapter(), openai_embedder, "test_crossvector_integration")
    except Exception as e:
        print(f"\n❌ Milvus test failed: {e}")

    # Test PGVector (if available)
    try:
        test_engine("PGVector", PGVectorAdapter(), openai_embedder, "test_crossvector_integration")
    except Exception as e:
        print(f"\n❌ PGVector test failed: {e}")

    # Test with Gemini embeddings (optional)
    try:
        print("\n\n" + "=" * 80)
        print("Testing with Gemini Embeddings")
        print("=" * 80)
        gemini_embedder = GeminiEmbeddingAdapter(model_name="models/text-embedding-004")

        test_engine("AstraDB with Gemini", AstraDBAdapter(), gemini_embedder, "test_crossvector_gemini")
    except Exception as e:
        print(f"\n❌ Gemini embedding test failed: {e}")

    print("\n" + "=" * 80)
    print("Integration tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
