"""Test the flexible input handling for add/upsert methods."""

from unittest.mock import MagicMock

import pytest

from crossvector.engine import VectorEngine
from crossvector.schema import Document, UpsertRequest
from crossvector.utils import normalize_documents


@pytest.fixture
def mock_embedding_adapter():
    """Create a mock embedding adapter."""
    adapter = MagicMock()
    adapter.embedding_dimension = 1536
    # Return embeddings matching the number of input texts
    adapter.get_embeddings.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    return adapter


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = MagicMock()
    adapter.initialize.return_value = None
    adapter.upsert.return_value = None
    return adapter


def test_normalize_documents_single_document(mock_embedding_adapter, mock_db_adapter):
    """Test normalizing a single Document object."""
    doc = Document(text="Hello world")
    result = normalize_documents(doc)

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].text == "Hello world"


def test_normalize_documents_single_dict(mock_embedding_adapter, mock_db_adapter):
    """Test normalizing a single dict."""
    doc_dict = {"text": "Hello world", "metadata": {"source": "test"}}
    result = normalize_documents(doc_dict)

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].text == "Hello world"
    assert result[0].metadata["source"] == "test"


def test_normalize_documents_list_of_documents(mock_embedding_adapter, mock_db_adapter):
    """Test normalizing a list of Document objects."""
    docs = [Document(text="Doc 1"), Document(text="Doc 2")]
    result = normalize_documents(docs)

    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)
    assert result[0].text == "Doc 1"
    assert result[1].text == "Doc 2"


def test_normalize_documents_list_of_dicts(mock_embedding_adapter, mock_db_adapter):
    """Test normalizing a list of dicts."""
    docs = [{"text": "Doc 1"}, {"text": "Doc 2", "metadata": {"source": "test"}}]
    result = normalize_documents(docs)

    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)
    assert result[0].text == "Doc 1"
    assert result[1].text == "Doc 2"
    assert result[1].metadata["source"] == "test"


def test_add_with_single_document(mock_embedding_adapter, mock_db_adapter):
    """Test add method with a single Document object."""
    engine = VectorEngine(
        embedding_adapter=mock_embedding_adapter,
        db_adapter=mock_db_adapter,
        collection_name="test_collection",
    )

    doc = Document(text="Hello world")
    result = engine.add(doc)

    assert result.status == "success"
    assert result.count == 1
    mock_db_adapter.upsert.assert_called_once()


def test_add_with_single_dict(mock_embedding_adapter, mock_db_adapter):
    """Test add method with a single dict."""
    engine = VectorEngine(
        embedding_adapter=mock_embedding_adapter,
        db_adapter=mock_db_adapter,
        collection_name="test_collection",
    )

    doc_dict = {"text": "Hello world", "metadata": {"source": "test"}}
    result = engine.add(doc_dict)

    assert result.status == "success"
    assert result.count == 1
    mock_db_adapter.upsert.assert_called_once()


def test_add_with_list_of_documents(mock_embedding_adapter, mock_db_adapter):
    """Test add method with a list of Document objects."""
    engine = VectorEngine(
        embedding_adapter=mock_embedding_adapter,
        db_adapter=mock_db_adapter,
        collection_name="test_collection",
    )

    docs = [Document(text="Doc 1"), Document(text="Doc 2")]
    result = engine.add(docs)

    assert result.status == "success"
    assert result.count == 2
    mock_db_adapter.upsert.assert_called_once()


def test_add_with_list_of_dicts(mock_embedding_adapter, mock_db_adapter):
    """Test add method with a list of dicts."""
    engine = VectorEngine(
        embedding_adapter=mock_embedding_adapter,
        db_adapter=mock_db_adapter,
        collection_name="test_collection",
    )

    docs = [{"text": "Doc 1"}, {"text": "Doc 2"}]
    result = engine.add(docs)

    assert result.status == "success"
    assert result.count == 2
    mock_db_adapter.upsert.assert_called_once()


def test_add_with_upsert_request(mock_embedding_adapter, mock_db_adapter):
    """Test add method with an UpsertRequest object."""
    engine = VectorEngine(
        embedding_adapter=mock_embedding_adapter,
        db_adapter=mock_db_adapter,
        collection_name="test_collection",
    )

    request = UpsertRequest(documents=[Document(text="Hello")])
    result = engine.add(request)

    assert result.status == "success"
    assert result.count == 1
    mock_db_adapter.upsert.assert_called_once()


def test_upsert_with_various_inputs(mock_embedding_adapter, mock_db_adapter):
    """Test upsert method with various input types."""
    engine = VectorEngine(
        embedding_adapter=mock_embedding_adapter,
        db_adapter=mock_db_adapter,
        collection_name="test_collection",
    )

    # Test with Document
    result = engine.upsert(Document(text="Test 1"))
    assert result.status == "success"
    assert result.count == 1

    # Test with dict
    result = engine.upsert({"text": "Test 2"})
    assert result.status == "success"
    assert result.count == 1

    # Test with list of Documents
    result = engine.upsert([Document(text="Test 3"), Document(text="Test 4")])
    assert result.status == "success"
    assert result.count == 2

    # Test with list of dicts
    result = engine.upsert([{"text": "Test 5"}, {"text": "Test 6"}])
    assert result.status == "success"
    assert result.count == 2


def test_normalize_documents_invalid_input(mock_embedding_adapter, mock_db_adapter):
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="Invalid input type"):
        normalize_documents(12345)  # Invalid type

    with pytest.raises(ValueError, match="Invalid input type"):
        normalize_documents("not a valid input")  # Invalid type
