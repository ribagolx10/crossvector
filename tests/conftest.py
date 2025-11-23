"""Pytest configuration and fixtures for vector store tests."""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Vector databases enable semantic search.",
        "Python is a great programming language.",
        "Machine learning requires large amounts of data.",
    ]


@pytest.fixture(scope="session")
def sample_documents(sample_texts):
    """Sample documents with IDs and metadata."""
    from crossvector import Document

    return [
        Document(id=f"doc_{i}", text=text, metadata={"index": i, "category": "test"})
        for i, text in enumerate(sample_texts)
    ]


@pytest.fixture
def mock_embeddings(sample_texts):
    """Mock embeddings for testing without API calls."""
    import random

    dimension = 1536  # OpenAI text-embedding-3-small dimension

    def generate_mock_embedding():
        # Generate random normalized vector
        vec = [random.gauss(0, 1) for _ in range(dimension)]
        magnitude = sum(x**2 for x in vec) ** 0.5
        return [x / magnitude for x in vec]

    return [generate_mock_embedding() for _ in sample_texts]


@pytest.fixture
def openai_api_key():
    """OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture
def astradb_credentials():
    """AstraDB credentials from environment."""
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    if not token or not endpoint:
        pytest.skip("AstraDB credentials not set")
    return {"token": token, "endpoint": endpoint}


@pytest.fixture
def chromadb_cloud_credentials():
    """ChromaDB Cloud credentials from environment."""
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_CLOUD_TENANT")
    database = os.getenv("CHROMA_CLOUD_DATABASE")
    if not api_key:
        pytest.skip("ChromaDB credentials not set")
    return {"api_key": api_key, "tenant": tenant, "database": database}


@pytest.fixture
def milvus_credentials():
    """Milvus credentials from environment."""
    endpoint = os.getenv("MILVUS_API_ENDPOINT")
    if not endpoint:
        pytest.skip("Milvus credentials not set")
    return {
        "endpoint": endpoint,
        "user": os.getenv("MILVUS_USER"),
        "password": os.getenv("MILVUS_PASSWORD"),
    }


@pytest.fixture
def pgvector_credentials():
    """PGVector credentials from environment."""
    host = os.getenv("PGVECTOR_HOST", "localhost")
    return {
        "host": host,
        "port": os.getenv("PGVECTOR_PORT", "5432"),
        "dbname": os.getenv("PGVECTOR_DBNAME", "postgres"),
        "user": os.getenv("PGVECTOR_USER", "postgres"),
        "password": os.getenv("PGVECTOR_PASSWORD", "postgres"),
    }


# Cleanup fixtures
@pytest.fixture
def cleanup_collection():
    """Fixture to clean up test collections after tests."""
    collections_to_cleanup = []

    def register_cleanup(adapter, collection_name):
        collections_to_cleanup.append((adapter, collection_name))

    yield register_cleanup

    # Cleanup after test
    for adapter, collection_name in collections_to_cleanup:
        try:
            if hasattr(adapter, "db"):
                # AstraDB
                if collection_name in adapter.db.list_collection_names():
                    adapter.db.drop_collection(collection_name)
            elif hasattr(adapter, "client"):
                # ChromaDB or Milvus
                if hasattr(adapter.client, "delete_collection"):
                    adapter.client.delete_collection(collection_name)
        except Exception as e:
            print(f"Cleanup warning: {e}")
