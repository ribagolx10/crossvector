"""Extended tests for abc.py to increase coverage."""

from unittest.mock import Mock

import pytest

from crossvector.abc import EmbeddingAdapter, VectorDBAdapter
from crossvector.logger import Logger
from crossvector.settings import settings


class ConcreteEmbedding(EmbeddingAdapter):
    """Concrete implementation of EmbeddingAdapter for testing."""

    def get_embeddings(self, texts):
        """Return dummy embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]


class ConcreteVectorDB(VectorDBAdapter):
    """Concrete implementation of VectorDBAdapter for testing."""

    def initialize(self, collection_name, dim, metric="cosine", **kwargs):
        pass

    def add_collection(self, collection_name, dim, metric="cosine"):
        pass

    def get_collection(self, collection_name):
        return {}

    def get_or_create_collection(self, collection_name, dim, metric="cosine"):
        return {}

    def drop_collection(self, collection_name):
        return True

    def clear_collection(self):
        return 0

    def create(self, doc):
        return doc

    def update(self, doc):
        return doc

    def delete(self, ids):
        return len(ids) if isinstance(ids, list) else 1

    def get(self, pk):
        return None

    def search(self, vector, limit=10, **kwargs):
        return []

    def bulk_create(self, docs, **kwargs):
        return docs

    def bulk_update(self, docs, **kwargs):
        return docs

    def count(self):
        return 0

    def upsert(self, docs, **kwargs):
        return docs


class TestEmbeddingAdapter:
    """Tests for EmbeddingAdapter abstract base class."""

    def test_embedding_adapter_init_with_all_params(self):
        """Test EmbeddingAdapter initialization with all parameters."""
        logger = Logger("test")
        adapter = ConcreteEmbedding(model_name="test-model", dim=512, logger=logger)

        assert adapter.model_name == "test-model"
        assert adapter.dim == 512
        assert adapter.logger == logger

    def test_embedding_adapter_init_with_default_dim(self):
        """Test EmbeddingAdapter initialization with default dim from settings."""
        adapter = ConcreteEmbedding(model_name="test-model")
        assert adapter.dim == settings.VECTOR_DIM

    def test_embedding_adapter_init_with_custom_logger(self):
        """Test EmbeddingAdapter initialization with custom logger."""
        custom_logger = Logger("custom")
        adapter = ConcreteEmbedding(model_name="test", logger=custom_logger)
        assert adapter.logger == custom_logger

    def test_embedding_adapter_init_creates_default_logger(self):
        """Test EmbeddingAdapter initialization creates logger if not provided."""
        adapter = ConcreteEmbedding(model_name="test")
        assert isinstance(adapter.logger, Logger)
        assert adapter.logger._logger.name == "ConcreteEmbedding"

    def test_embedding_adapter_logger_property(self):
        """Test logger property returns the internal logger."""
        adapter = ConcreteEmbedding(model_name="test")
        assert isinstance(adapter.logger, Logger)

    def test_embedding_adapter_dim_property(self):
        """Test dim property returns correct dimension."""
        adapter = ConcreteEmbedding(model_name="test", dim=768)
        assert adapter.dim == 768

    def test_embedding_adapter_get_embeddings_implementation(self):
        """Test get_embeddings works on concrete implementation."""
        adapter = ConcreteEmbedding(model_name="test")
        texts = ["hello", "world"]
        embeddings = adapter.get_embeddings(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]

    def test_embedding_adapter_abstract_method_not_callable(self):
        """Test that EmbeddingAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingAdapter(model_name="test")

    def test_embedding_adapter_with_kwargs(self):
        """Test EmbeddingAdapter initialization with extra kwargs."""
        adapter = ConcreteEmbedding(model_name="test", dim=512, extra_param="value", another="param")
        assert adapter.model_name == "test"
        assert adapter.dim == 512

    def test_embedding_adapter_model_name_stored(self):
        """Test that model_name is properly stored."""
        adapter = ConcreteEmbedding(model_name="gpt-4-embedding")
        assert adapter.model_name == "gpt-4-embedding"

    def test_embedding_adapter_non_logger_init_creates_logger(self):
        """Test that passing non-Logger object as logger parameter triggers creation."""
        adapter = ConcreteEmbedding(model_name="test", logger="not-a-logger")
        assert isinstance(adapter.logger, Logger)


class TestVectorDBAdapter:
    """Tests for VectorDBAdapter abstract base class."""

    def test_vector_db_adapter_init_with_all_params(self):
        """Test VectorDBAdapter initialization with all parameters."""
        logger = Logger("test")
        adapter = ConcreteVectorDB(collection_name="test_collection", dim=512, store_text=False, logger=logger)

        assert adapter.collection_name == "test_collection"
        assert adapter.dim == 512
        assert adapter.store_text is False
        assert adapter.logger == logger

    def test_vector_db_adapter_init_with_defaults(self):
        """Test VectorDBAdapter initialization with default values."""
        adapter = ConcreteVectorDB()
        assert adapter.collection_name == "vector_db"
        assert adapter.dim == settings.VECTOR_DIM
        assert adapter.store_text == settings.VECTOR_STORE_TEXT

    def test_vector_db_adapter_collection_name_default(self):
        """Test default collection name from class constant."""
        adapter = ConcreteVectorDB()
        assert adapter.collection_name == settings.VECTOR_COLLECTION_NAME

    def test_vector_db_adapter_custom_collection_name(self):
        """Test setting custom collection name."""
        adapter = ConcreteVectorDB(collection_name="my_custom_collection")
        assert adapter.collection_name == "my_custom_collection"

    def test_vector_db_adapter_dim_from_settings(self):
        """Test dim defaults to settings value."""
        adapter = ConcreteVectorDB()
        assert adapter.dim == settings.VECTOR_DIM

    def test_vector_db_adapter_dim_custom(self):
        """Test dim can be customized."""
        adapter = ConcreteVectorDB(dim=256)
        assert adapter.dim == 256

    def test_vector_db_adapter_store_text_true(self):
        """Test store_text set to True."""
        adapter = ConcreteVectorDB(store_text=True)
        assert adapter.store_text is True

    def test_vector_db_adapter_store_text_false(self):
        """Test store_text set to False."""
        adapter = ConcreteVectorDB(store_text=False)
        assert adapter.store_text is False

    def test_vector_db_adapter_store_text_none_uses_settings(self):
        """Test store_text None uses settings value."""
        adapter = ConcreteVectorDB(store_text=None)
        assert adapter.store_text == settings.VECTOR_STORE_TEXT

    def test_vector_db_adapter_logger_property(self):
        """Test logger property."""
        adapter = ConcreteVectorDB()
        assert isinstance(adapter.logger, Logger)

    def test_vector_db_adapter_custom_logger(self):
        """Test custom logger initialization."""
        custom_logger = Logger("custom")
        adapter = ConcreteVectorDB(logger=custom_logger)
        assert adapter.logger == custom_logger

    def test_vector_db_adapter_creates_default_logger(self):
        """Test default logger is created from class name."""
        adapter = ConcreteVectorDB()
        assert isinstance(adapter.logger, Logger)
        assert adapter.logger._logger.name == "ConcreteVectorDB"

    def test_vector_db_adapter_non_logger_init(self):
        """Test non-Logger object triggers logger creation."""
        adapter = ConcreteVectorDB(logger="not-a-logger")
        assert isinstance(adapter.logger, Logger)

    def test_vector_db_adapter_collection_property_getter(self):
        """Test collection property getter."""
        adapter = ConcreteVectorDB()
        # Should raise error or return None since we haven't set it
        result = adapter.collection
        assert result is None

    def test_vector_db_adapter_collection_property_setter(self):
        """Test collection property setter."""
        adapter = ConcreteVectorDB()
        mock_collection = Mock()
        adapter.collection = mock_collection
        assert adapter.collection == mock_collection

    def test_vector_db_adapter_abstract_methods(self):
        """Test that VectorDBAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorDBAdapter()

    def test_vector_db_adapter_with_kwargs(self):
        """Test VectorDBAdapter accepts extra kwargs."""
        adapter = ConcreteVectorDB(
            collection_name="test",
            dim=512,
            extra_param="value",
            another="param",
        )
        assert adapter.collection_name == "test"
        assert adapter.dim == 512

    def test_vector_db_adapter_use_dollar_vector_class_attr(self):
        """Test use_dollar_vector class attribute."""
        assert hasattr(ConcreteVectorDB, "use_dollar_vector")
        assert ConcreteVectorDB.use_dollar_vector is False

    def test_vector_db_adapter_supports_metadata_only_class_attr(self):
        """Test supports_metadata_only class attribute."""
        assert hasattr(ConcreteVectorDB, "supports_metadata_only")
        assert ConcreteVectorDB.supports_metadata_only is False

    def test_vector_db_adapter_where_compiler_class_attr(self):
        """Test where_compiler class attribute."""
        assert hasattr(ConcreteVectorDB, "where_compiler")


class TestEmbeddingAdapterInheritance:
    """Tests for EmbeddingAdapter inheritance patterns."""

    def test_embedding_adapter_subclass_inherits_properties(self):
        """Test that subclass inherits dim property."""
        adapter = ConcreteEmbedding(model_name="test", dim=1024)
        assert hasattr(adapter, "dim")
        assert adapter.dim == 1024

    def test_embedding_adapter_subclass_inherits_logger(self):
        """Test that subclass inherits logger property."""
        adapter = ConcreteEmbedding(model_name="test")
        assert hasattr(adapter, "logger")
        assert isinstance(adapter.logger, Logger)


class TestVectorDBAdapterInheritance:
    """Tests for VectorDBAdapter inheritance patterns."""

    def test_vector_db_adapter_subclass_inherits_properties(self):
        """Test that subclass inherits all properties."""
        adapter = ConcreteVectorDB(collection_name="test", dim=512, store_text=False)
        assert adapter.collection_name == "test"
        assert adapter.dim == 512
        assert adapter.store_text is False


class TestEmbeddingAdapterIntegration:
    """Integration tests for EmbeddingAdapter."""

    def test_embedding_adapter_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        adapter1 = ConcreteEmbedding(model_name="model1", dim=512)
        adapter2 = ConcreteEmbedding(model_name="model2", dim=768)

        assert adapter1.model_name == "model1"
        assert adapter2.model_name == "model2"
        assert adapter1.dim == 512
        assert adapter2.dim == 768


class TestVectorDBAdapterIntegration:
    """Integration tests for VectorDBAdapter."""

    def test_vector_db_adapter_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        adapter1 = ConcreteVectorDB(collection_name="collection1", dim=512)
        adapter2 = ConcreteVectorDB(collection_name="collection2", dim=768)

        assert adapter1.collection_name == "collection1"
        assert adapter2.collection_name == "collection2"
        assert adapter1.dim == 512
        assert adapter2.dim == 768
