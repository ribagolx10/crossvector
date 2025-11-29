"""Utility functions for crossvector.

Shared helpers extracted from adapters to reduce duplication.
"""

from typing import Iterator, List, Sequence, Dict, Any, Union, Literal

from .schema import VectorDocument


def normalize_ids(ids: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(ids, (str, int)):
        return [ids]
    return list(ids or [])


def extract_id(data: Dict[str, Any]) -> str | None:
    """Extract primary key from kwargs/dict supporting _id, id, or pk fields."""
    return data.get("_id") or data.get("id") or data.get("pk")


# ---------------------------------------------------------------------------
# Adapter shared helpers
# ---------------------------------------------------------------------------
def prepare_item_for_storage(doc: Dict[str, Any] | VectorDocument, *, store_text: bool = True) -> Dict[str, Any]:
    """Normalize a raw document dict into a unified storage format.

    Maps id/_id, vector/$vector, optionally text, and keeps remaining fields flat.
    This assumes upstream caller will adapt to backend field naming if needed.
    """
    # Handle VectorDocument instances
    if isinstance(doc, VectorDocument):
        return doc.dump(store_text=store_text, use_dollar_vector=True)
    # Dict-like path
    item: Dict[str, Any] = {}
    _id = doc.get("_id") or doc.get("id")  # type: ignore[attr-defined]
    if _id:
        item["_id"] = _id
    vector = doc.get("$vector") or doc.get("vector")  # type: ignore[attr-defined]
    if vector is not None:
        item["$vector"] = vector
    if store_text and "text" in doc:  # type: ignore
        item["text"] = doc["text"]  # type: ignore
    for k, v in doc.items():  # type: ignore
        if k not in ("_id", "id", "$vector", "vector", "text"):
            item[k] = v
    return item


def chunk_iter(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield successive chunks from a sequence."""
    if size <= 0:
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def apply_update_fields(item: Dict[str, Any], update_fields: Sequence[str] | None) -> Dict[str, Any]:
    """Filter item to only the update fields provided (excluding _id)."""
    fields = update_fields or [k for k in item.keys() if k != "_id"]
    return {k: item[k] for k in fields if k in item and k != "_id"}


# ---------------------------------------------------------------------------
# Input normalization helpers for VectorEngine
# ---------------------------------------------------------------------------
def normalize_texts(texts: Union[str, List[str]]) -> List[str]:
    """
    Normalize text input to list of strings.

    Args:
        texts: Single text string or list of text strings

    Returns:
        List of text strings

    Examples:
        >>> normalize_texts("Hello")
        ["Hello"]

        >>> normalize_texts(["Text 1", "Text 2"])
        ["Text 1", "Text 2"]
    """
    return [texts] if isinstance(texts, str) else texts


def normalize_metadatas(
    metadatas: Union[Dict[str, Any], List[Dict[str, Any]], None],
    count: int,
) -> List[Dict[str, Any]]:
    """
    Normalize metadata input to list of dicts matching text count.

    Args:
        metadatas: Single metadata dict, list of metadata dicts, or None
        count: Number of texts/documents

    Returns:
        List of metadata dicts (empty dicts if None provided)

    Examples:
        >>> normalize_metadatas(None, 2)
        [{}, {}]

        >>> normalize_metadatas({"key": "value"}, 2)
        [{"key": "value"}, {"key": "value"}]

        >>> normalize_metadatas([{"a": 1}, {"b": 2}], 2)
        [{"a": 1}, {"b": 2}]
    """
    if metadatas is None:
        return [{}] * count
    elif isinstance(metadatas, dict):
        return [metadatas] * count
    else:
        return metadatas


def normalize_pks(
    pks: Union[str, int, List[str], List[int], None],
    count: int,
) -> List[str | int | None]:
    """
    Normalize primary key input to list matching text count.

    Args:
        pks: Single pk, list of pks, or None (for auto-generation)
        count: Number of texts/documents

    Returns:
        List of pks or None values

    Examples:
        >>> normalize_pks(None, 2)
        [None, None]

        >>> normalize_pks("doc1", 1)
        ["doc1"]

        >>> normalize_pks(["doc1", "doc2"], 2)
        ["doc1", "doc2"]
    """
    if pks is None:
        return [None] * count
    elif isinstance(pks, (str, int)):
        return [pks]
    else:
        return list(pks)


def validate_primary_key_mode(
    mode: str,
) -> Literal["uuid", "hash_text", "hash_vector", "int64", "auto"]:
    """
    Validate PRIMARY_KEY_MODE setting value.

    Args:
        mode: The primary key mode to validate

    Returns:
        The validated mode

    Raises:
        ValueError: If mode is not a valid option

    Examples:
        >>> validate_primary_key_mode("uuid")
        "uuid"

        >>> validate_primary_key_mode("invalid")
        ValueError: Invalid PRIMARY_KEY_MODE: 'invalid'. Must be one of: uuid, hash_text, hash_vector, int64, auto
    """
    valid_modes = {"uuid", "hash_text", "hash_vector", "int64", "auto"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid PRIMARY_KEY_MODE: '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}")
    return mode  # type: ignore
