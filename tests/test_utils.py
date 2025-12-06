"""Tests for utility functions."""

import hashlib
import uuid
from unittest.mock import Mock, patch

import pytest

from crossvector.utils import (
    apply_update_fields,
    chunk_iter,
    extract_pk,
    generate_pk,
    load_custom_pk_factory,
    normalize_metadatas,
    normalize_pks,
    normalize_texts,
    prepare_item_for_storage,
    validate_primary_key_mode,
)


class TestChunkIter:
    """Tests for chunk_iter function."""

    def test_chunk_iter_normal(self):
        seq = [1, 2, 3, 4, 5, 6, 7]
        chunks = list(chunk_iter(seq, 3))
        assert chunks == [[1, 2, 3], [4, 5, 6], [7]]

    def test_chunk_iter_exact_size(self):
        seq = [1, 2, 3, 4, 5, 6]
        chunks = list(chunk_iter(seq, 3))
        assert chunks == [[1, 2, 3], [4, 5, 6]]

    def test_chunk_iter_size_zero(self):
        seq = [1, 2, 3]
        chunks = list(chunk_iter(seq, 0))
        assert chunks == [[1, 2, 3]]

    def test_chunk_iter_size_negative(self):
        seq = [1, 2, 3]
        chunks = list(chunk_iter(seq, -1))
        assert chunks == [[1, 2, 3]]

    def test_chunk_iter_empty_sequence(self):
        chunks = list(chunk_iter([], 3))
        assert chunks == []


class TestExtractPk:
    """Tests for extract_pk function."""

    def test_extract_from_kwargs_id(self):
        pk = extract_pk(id="test-id")
        assert pk == "test-id"

    def test_extract_from_kwargs_underscore_id(self):
        pk = extract_pk(_id="test-id")
        assert pk == "test-id"

    def test_extract_from_kwargs_pk(self):
        pk = extract_pk(pk="test-pk")
        assert pk == "test-pk"

    def test_extract_from_doc_object(self):
        doc = Mock(id="doc-id")
        pk = extract_pk(doc)
        assert pk == "doc-id"

    def test_kwargs_override_doc(self):
        doc = Mock(id="doc-id")
        pk = extract_pk(doc, id="override-id")
        assert pk == "override-id"

    def test_extract_none(self):
        pk = extract_pk()
        assert pk is None

    def test_extract_from_doc_no_id(self):
        doc = Mock(spec=[])  # No id attribute
        pk = extract_pk(doc)
        assert pk is None


class TestLoadCustomPkFactory:
    """Tests for load_custom_pk_factory function."""

    def test_load_none(self):
        factory = load_custom_pk_factory(None)
        assert factory is None

    def test_load_empty_string(self):
        factory = load_custom_pk_factory("")
        assert factory is None

    def test_load_invalid_path(self):
        factory = load_custom_pk_factory("nonexistent.module.func")
        assert factory is None

    def test_load_valid_callable(self):
        # Use built-in function as test
        factory = load_custom_pk_factory("builtins.str")
        assert factory is not None
        assert callable(factory)

    def test_load_non_callable(self):
        # Try to load a non-callable attribute
        factory = load_custom_pk_factory("sys.version")
        assert factory is None


class TestGeneratePk:
    """Tests for generate_pk function."""

    @patch("crossvector.utils.settings")
    def test_generate_uuid_mode(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "uuid"
        mock_settings.PRIMARY_KEY_FACTORY = None
        pk = generate_pk("test text", [0.1, 0.2])
        assert len(pk) == 32  # UUID hex
        # Verify it's a valid UUID
        uuid.UUID(pk, version=4)

    @patch("crossvector.utils.settings")
    def test_generate_hash_text_mode(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "hash_text"
        mock_settings.PRIMARY_KEY_FACTORY = None
        pk = generate_pk("test text", [0.1, 0.2])
        expected = hashlib.sha256("test text".encode("utf-8")).hexdigest()
        assert pk == expected

    @patch("crossvector.utils.settings")
    def test_generate_hash_vector_mode(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "hash_vector"
        mock_settings.PRIMARY_KEY_FACTORY = None
        vector = [0.1, 0.2, 0.3]
        pk = generate_pk("test", vector)
        vec_str = "|".join(f"{x:.8f}" for x in vector)
        expected = hashlib.sha256(vec_str.encode("utf-8")).hexdigest()
        assert pk == expected

    @patch("crossvector.utils.settings")
    def test_generate_int64_mode(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "int64"
        mock_settings.PRIMARY_KEY_FACTORY = None
        pk1 = generate_pk("test", [0.1])
        pk2 = generate_pk("test", [0.1])
        # Should be sequential integers
        assert pk1.isdigit()
        assert pk2.isdigit()
        assert int(pk2) > int(pk1)

    @patch("crossvector.utils.settings")
    def test_generate_auto_mode_with_text(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "auto"
        mock_settings.PRIMARY_KEY_FACTORY = None
        pk = generate_pk("test text", None)
        expected = hashlib.sha256("test text".encode("utf-8")).hexdigest()
        assert pk == expected

    @patch("crossvector.utils.settings")
    def test_generate_auto_mode_with_vector_only(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "auto"
        mock_settings.PRIMARY_KEY_FACTORY = None
        vector = [0.1, 0.2]
        pk = generate_pk(None, vector)
        vec_str = "|".join(f"{x:.8f}" for x in vector)
        expected = hashlib.sha256(vec_str.encode("utf-8")).hexdigest()
        assert pk == expected

    @patch("crossvector.utils.settings")
    def test_generate_auto_mode_fallback_uuid(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "auto"
        mock_settings.PRIMARY_KEY_FACTORY = None
        pk = generate_pk(None, None)
        assert len(pk) == 32
        uuid.UUID(pk, version=4)

    @patch("crossvector.utils.settings")
    def test_generate_invalid_mode_fallback(self, mock_settings):
        mock_settings.PRIMARY_KEY_MODE = "invalid_mode"
        mock_settings.PRIMARY_KEY_FACTORY = None
        pk = generate_pk("test", [0.1])
        assert len(pk) == 32  # Falls back to UUID


class TestValidatePrimaryKeyMode:
    """Tests for validate_primary_key_mode function."""

    def test_validate_uuid(self):
        assert validate_primary_key_mode("uuid") == "uuid"

    def test_validate_hash_text(self):
        assert validate_primary_key_mode("hash_text") == "hash_text"

    def test_validate_hash_vector(self):
        assert validate_primary_key_mode("hash_vector") == "hash_vector"

    def test_validate_int64(self):
        assert validate_primary_key_mode("int64") == "int64"

    def test_validate_auto(self):
        assert validate_primary_key_mode("auto") == "auto"

    def test_validate_invalid(self):
        with pytest.raises(ValueError, match="Invalid PRIMARY_KEY_MODE"):
            validate_primary_key_mode("invalid")


class TestNormalizeTexts:
    """Tests for normalize_texts function."""

    def test_normalize_single_string(self):
        result = normalize_texts("single text")
        assert result == ["single text"]

    def test_normalize_list_of_strings(self):
        texts = ["text1", "text2", "text3"]
        result = normalize_texts(texts)
        assert result == texts


class TestNormalizeMetadatas:
    """Tests for normalize_metadatas function."""

    def test_normalize_none(self):
        result = normalize_metadatas(None, 3)
        assert result == [{}, {}, {}]

    def test_normalize_single_dict(self):
        meta = {"key": "value"}
        result = normalize_metadatas(meta, 3)
        assert result == [meta, meta, meta]

    def test_normalize_list_of_dicts(self):
        metas = [{"a": 1}, {"b": 2}]
        result = normalize_metadatas(metas, 2)
        assert result == metas


class TestNormalizePks:
    """Tests for normalize_pks function."""

    def test_normalize_none(self):
        result = normalize_pks(None, 2)
        assert result == [None, None]

    def test_normalize_single_string_raises_when_count_mismatch(self):
        """Single pk with count > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Single pk provided but count is 3"):
            normalize_pks("pk-123", 3)

    def test_normalize_list(self):
        pks = ["pk1", "pk2"]
        result = normalize_pks(pks, 2)
        assert result == pks


class TestPrepareItemForStorage:
    """Tests for prepare_item_for_storage function."""

    def test_prepare_with_store_text_true(self):
        item = {"_id": "123", "text": "hello", "vector": [0.1], "meta": "data"}
        result = prepare_item_for_storage(item, store_text=True)
        assert "text" in result
        assert result["text"] == "hello"

    def test_prepare_with_store_text_false(self):
        item = {"_id": "123", "text": "hello", "vector": [0.1], "meta": "data"}
        result = prepare_item_for_storage(item, store_text=False)
        assert "text" not in result

    def test_prepare_keeps_id_and_dollar_vector(self):
        """prepare_item_for_storage keeps _id and converts to $vector."""
        item = {"_id": "123", "vector": [0.1], "meta": "data"}
        result = prepare_item_for_storage(item, store_text=True)
        assert result["_id"] == "123"
        assert result["$vector"] == [0.1]
        assert result["meta"] == "data"
        assert "vector" not in result  # 'vector' converted to '$vector'


class TestApplyUpdateFields:
    """Tests for apply_update_fields function."""

    def test_apply_none_fields_returns_all(self):
        item = {"field1": "val1", "field2": "val2"}
        result = apply_update_fields(item, None)
        assert result == item

    def test_apply_specific_fields(self):
        item = {"field1": "val1", "field2": "val2", "field3": "val3"}
        result = apply_update_fields(item, ["field1", "field3"])
        assert result == {"field1": "val1", "field3": "val3"}

    def test_apply_empty_fields_list_returns_all_except_id(self):
        """Empty list means use all fields except _id."""
        item = {"_id": "123", "field1": "val1", "field2": "val2"}
        result = apply_update_fields(item, [])
        assert "_id" not in result
        assert result == {"field1": "val1", "field2": "val2"}

    def test_apply_fields_not_in_item(self):
        item = {"field1": "val1"}
        result = apply_update_fields(item, ["field1", "field2"])
        # Only field1 should be in result since field2 doesn't exist
        assert result == {"field1": "val1"}
