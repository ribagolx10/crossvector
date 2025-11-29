"""Utility functions for crossvector.

Shared helpers extracted from adapters to reduce duplication.
"""

from typing import Iterator, List, Sequence, Dict, Any, Union, Literal, Optional, Callable
import hashlib
import importlib
import uuid
from .settings import settings


# ===========================================================================
# Core utilities
# ===========================================================================


def chunk_iter(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield successive chunks from a sequence."""
    if size <= 0:
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def extract_id(data: Dict[str, Any]) -> str | None:
    """Extract primary key from kwargs/dict supporting _id, id, or pk fields."""
    return data.get("_id") or data.get("id") or data.get("pk")


# ===========================================================================
# Primary key generation
# ===========================================================================


def load_custom_pk_factory(path: Optional[str]) -> Optional[Callable]:
    """Load a custom primary key factory function from module path."""
    if not path:
        return None
    try:
        module_path, attr = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        fn = getattr(module, attr)
        if callable(fn):
            return fn
    except Exception:
        return None
    return None


_int64_pk_counter = 0
_custom_pk_factory = load_custom_pk_factory(getattr(settings, "PRIMARY_KEY_FACTORY", None))


def generate_pk(text: str | None, vector: List[float] | None, metadata: Dict[str, Any] | None = None) -> str:
    """Generate a primary key based on PRIMARY_KEY_MODE setting.

    Modes:
        - uuid: Random UUID (default)
        - hash_text: SHA256 hash of text
        - hash_vector: SHA256 hash of vector
        - int64: Sequential integer
        - auto: Hash text if available, else hash vector, else UUID
    """
    global _int64_pk_counter
    mode = (getattr(settings, "PRIMARY_KEY_MODE", "uuid") or "uuid").lower()
    if _custom_pk_factory:
        try:
            return str(_custom_pk_factory(text, vector, metadata or {}))
        except Exception:
            pass
    if mode == "uuid":
        return uuid.uuid4().hex
    if mode == "hash_text" and text:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    if mode == "hash_vector" and vector:
        vec_bytes = ("|".join(f"{x:.8f}" for x in vector)).encode("utf-8")
        return hashlib.sha256(vec_bytes).hexdigest()
    if mode == "int64":
        _int64_pk_counter += 1
        return str(_int64_pk_counter)
    if mode == "auto":
        if text:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
        if vector:
            vec_bytes = ("|".join(f"{x:.8f}" for x in vector)).encode("utf-8")
            return hashlib.sha256(vec_bytes).hexdigest()
        return uuid.uuid4().hex
    return uuid.uuid4().hex


def validate_primary_key_mode(
    mode: str,
) -> Literal["uuid", "hash_text", "hash_vector", "int64", "auto"]:
    """Validate PRIMARY_KEY_MODE setting value.

    Args:
        mode: The primary key mode to validate

    Returns:
        The validated mode

    Raises:
        ValueError: If mode is not a valid option
    """
    valid_modes = {"uuid", "hash_text", "hash_vector", "int64", "auto"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid PRIMARY_KEY_MODE: '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}")
    return mode  # type: ignore


# ===========================================================================
# Input normalization helpers for VectorEngine
# ===========================================================================


def normalize_texts(texts: Union[str, List[str]]) -> List[str]:
    """Normalize text input to list of strings."""
    return [texts] if isinstance(texts, str) else texts


def normalize_metadatas(
    metadatas: Union[Dict[str, Any], List[Dict[str, Any]], None],
    count: int,
) -> List[Dict[str, Any]]:
    """Normalize metadata input to list of dicts matching text count."""
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
    """Normalize primary key input to list matching text count."""
    if pks is None:
        return [None] * count
    elif isinstance(pks, (str, int)):
        if count == 1:
            return [pks]
        else:
            raise ValueError(f"Single pk provided but count is {count}")
    else:
        pk_list = list(pks)
        # Pad with None if necessary
        if len(pk_list) < count:
            pk_list.extend([None] * (count - len(pk_list)))
        return pk_list[:count]  # Truncate if too long


# ===========================================================================
# Adapter shared helpers
# ===========================================================================


def prepare_item_for_storage(doc: Dict[str, Any] | Any, *, store_text: bool = True) -> Dict[str, Any]:
    """Normalize a raw Document dict into a unified storage format.

    Maps id/_id, vector/$vector, optionally text, and keeps remaining fields flat.
    This assumes upstream caller will adapt to backend field naming if needed.
    """
    # Handle objects that implement 'dump' (e.g., Document)
    if hasattr(doc, "dump") and callable(getattr(doc, "dump")):
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


def apply_update_fields(item: Dict[str, Any], update_fields: Sequence[str] | None) -> Dict[str, Any]:
    """Filter item to only the update fields provided (excluding _id)."""
    fields = update_fields or [k for k in item.keys() if k != "_id"]
    return {k: item[k] for k in fields if k in item and k != "_id"}
