"""
Integration test script: Test VectorEngine with real cloud APIs
This script tests the unified VectorEngine interface across different databases.
"""

import time

from dotenv import load_dotenv

from crossvector import VectorEngine
from crossvector.dbs.astradb import AstraDBAdapter
from crossvector.dbs.chroma import ChromaDBAdapter
from crossvector.dbs.milvus import MilvusDBAdapter
from crossvector.dbs.pgvector import PGVectorAdapter
from crossvector.embeddings.gemini import GeminiEmbeddingAdapter
from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

# Load .env
load_dotenv()

# Test data
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Machine learning enables computers to learn from data.",
]
test_metadatas = [
    {"category": "animals", "source": "test"},
    {"category": "technology", "source": "test"},
    {"category": "technology", "source": "test"},
]
test_pks = ["doc1", "doc2", "doc3"]


def test_engine(db_name: str, db, embedding, collection_name: str):
    """Test VectorEngine with a specific database adapter."""
    print(f"\n{'=' * 80}")
    print(f"Testing {db_name} with {embedding.model_name}")
    print(f"{'=' * 80}")

    engine = VectorEngine(embedding=embedding, db=db, collection_name=collection_name)

    # Clean up existing data (if collection exists, drop it)
    try:
        engine.drop_collection(collection_name)
        time.sleep(1)
        print(f"Dropped existing collection '{collection_name}'")
    except Exception as e:
        print(f"Note: Could not drop collection (may not exist): {e}")

    # Re-initialize after dropping
    engine = VectorEngine(embedding=embedding, db=db, collection_name=collection_name)

    # Test 1: Upsert VectorDocuments (with auto-embedding)
    print("\n1. Testing upsert...")
    docs = [{"id": test_pks[i], "text": test_texts[i], "metadata": test_metadatas[i]} for i in range(len(test_texts))]
    result = engine.upsert(docs)
    print(f"Inserted {len(result)} VectorDocuments")

    # Test 2: Count VectorDocuments
    print("\n2. Testing count...")
    count = engine.count()
    print(f"Total documents: {count}")
    assert count == len(test_texts), f"Expected {len(test_texts)} VectorDocuments, got {count}"

    # Test 3: Get document by ID
    print("\n3. Testing get...")
    doc = engine.get("doc1")
    print(f"Retrieved doc: {doc.text[:50] if doc.text else 'N/A'}...")
    assert doc is not None, "VectorDocument not found"

    # Test 4: Search
    print("\n4. Testing search...")
    results = engine.search(query="AI and machine learning", limit=2)
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        score = getattr(result, "score", "N/A")
        text = result.text if result.text else "N/A"
        if text != "N/A":
            text = text[:50]
        if isinstance(score, (int, float)):
            print(f"  {i}. Score: {score:.4f}, Text: {text}...")
        else:
            print(f"  {i}. Score: {score}, Text: {text}...")

    # Test 5: Delete one
    print("\n5. Testing delete...")
    deleted = engine.delete("doc1")
    print(f"Deleted {deleted} document(s)")

    # Verify deletion
    count_after_delete = engine.count()
    print(f"VectorDocuments after deletion: {count_after_delete}")
    assert count_after_delete == len(test_texts) - 1, (
        f"Expected {len(test_texts) - 1} VectorDocuments, got {count_after_delete}"
    )

    # Test 6: Delete many
    print("\n6. Testing delete...")
    deleted = engine.delete(["doc2", "doc3"])
    print(f"Deleted {deleted} document(s)")

    # Verify all deleted
    final_count = engine.count()
    print(f"Final document count: {final_count}")
    assert final_count == 0, f"Expected 0 VectorDocuments, got {final_count}"

    print(f"\nAll tests passed for {db_name}!")


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
        print(f"\nAstraDB test failed: {e}")

    # Test ChromaDB
    try:
        test_engine("ChromaDB Cloud", ChromaDBAdapter(), openai_embedder, "test_crossvector_integration")
    except Exception as e:
        print(f"\nChromaDB test failed: {e}")

    # Test Milvus
    try:
        test_engine("Milvus", MilvusDBAdapter(), openai_embedder, "test_crossvector_integration")
    except Exception as e:
        print(f"\nMilvus test failed: {e}")

    # Test PGVector (if available)
    try:
        test_engine("PGVector", PGVectorAdapter(), openai_embedder, "test_crossvector_integration")
    except Exception as e:
        print(f"\nPGVector test failed: {e}")

    # Test with Gemini embeddings (optional)
    try:
        print("\n\n" + "=" * 80)
        print("Testing with Gemini Embeddings")
        print("=" * 80)
        gemini_embedder = GeminiEmbeddingAdapter(model_name="models/text-embedding-004")

        test_engine("AstraDB with Gemini", AstraDBAdapter(), gemini_embedder, "test_crossvector_gemini")
    except Exception as e:
        print(f"\nGemini embedding test failed: {e}")

    print("\n" + "=" * 80)
    print("Integration tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
