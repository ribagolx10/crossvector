"""Pydantic schemas for vector store operations."""

import hashlib
import importlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel, Field, model_validator
from .settings import settings
import uuid

_int64_pk_counter = 0


def _load_custom_pk_factory(path: Optional[str]) -> Optional[Callable]:
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


_custom_pk_factory = _load_custom_pk_factory(settings.PRIMARY_KEY_FACTORY)


def generate_pk(text: Optional[str], vector: Optional[List[float]], metadata: Optional[Dict[str, Any]] = None) -> str:
    global _int64_pk_counter
    mode = (settings.PRIMARY_KEY_MODE or "uuid").lower()
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


class Document(BaseModel):
    """Schema for a document to be inserted into the vector store."""

    id: Optional[str] = Field(
        None, description="Unique identifier for the document. If not provided, it will be generated from text hash."
    )
    text: str = Field(..., description="The text content of the document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata.")
    created_timestamp: Optional[float] = Field(
        None, description="Unix timestamp (seconds since epoch) when document was created."
    )
    updated_timestamp: Optional[float] = Field(
        None, description="Unix timestamp (seconds since epoch) when document was last updated."
    )

    @property
    def pk(self) -> Union[str, int]:
        if self.id is None:
            raise ValueError("Document id not set")
        return self.id

    @model_validator(mode="after")
    def generate_id_and_timestamps(self) -> "Document":
        if not self.id:
            self.id = generate_pk(self.text, None, self.metadata)

        # Check for reserved fields in metadata
        reserved_fields = {
            "created_at",
            "updated_at",
            "cv_created_at",
            "cv_updated_at",
            "created_timestamp",
            "updated_timestamp",
        }
        conflicting_fields = reserved_fields.intersection(self.metadata.keys())
        if conflicting_fields:
            import warnings

            warnings.warn(
                f"Metadata contains reserved timestamp fields {conflicting_fields}. "
                f"These will be overridden with automatic timestamps.",
                UserWarning,
                stacklevel=2,
            )

        # Set timestamps (Unix timestamp in seconds with microseconds precision)
        current_timestamp = datetime.now(timezone.utc).timestamp()

        # If created_timestamp is not set, this is a new document
        if not self.created_timestamp:
            self.created_timestamp = current_timestamp

        # Always update updated_timestamp
        self.updated_timestamp = current_timestamp

        return self

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "Document":
        pk = kwargs.pop("_id", None) or kwargs.pop("id", None)
        text = kwargs.pop("text", None)
        if text is None:
            raise ValueError("'text' is required for Document.from_kwargs")
        metadata = kwargs.pop("metadata", None) or {}
        # Remaining kwargs merge into metadata (avoid overwriting existing keys)
        for k, v in kwargs.items():
            if k not in metadata:
                metadata[k] = v
        return cls(id=pk, text=text, metadata=metadata)

    def dump(self, include_timestamps: bool = False) -> Dict[str, Any]:
        base = {"id": self.pk, "text": self.text, "metadata": self.metadata}
        if include_timestamps:
            base["created_timestamp"] = self.created_timestamp
            base["updated_timestamp"] = self.updated_timestamp
        return base


class VectorDocument(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the vector document.")
    vector: List[float] = Field(..., description="Embedding vector.")
    text: Optional[str] = Field(None, description="Optional text content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata.")
    created_timestamp: Optional[float] = Field(None, description="Creation timestamp.")
    updated_timestamp: Optional[float] = Field(None, description="Last update timestamp.")

    @property
    def pk(self) -> str:
        if self.id is None:
            raise ValueError("VectorDocument id not set")
        return self.id

    @model_validator(mode="after")
    def assign_defaults(self) -> "VectorDocument":
        if not self.id:
            self.id = generate_pk(self.text, self.vector, self.metadata)
        current_timestamp = datetime.now(timezone.utc).timestamp()
        if not self.created_timestamp:
            self.created_timestamp = current_timestamp
        self.updated_timestamp = current_timestamp
        return self

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "VectorDocument":
        pk = kwargs.pop("_id", None) or kwargs.pop("id", None)
        vector = kwargs.pop("$vector", None) or kwargs.pop("vector", None)
        if vector is None:
            raise ValueError("'vector' or '$vector' is required for VectorDocument.from_kwargs")
        text = kwargs.pop("text", None)
        metadata = kwargs.pop("metadata", None) or {}
        for k, v in kwargs.items():
            if k not in metadata:
                metadata[k] = v
        return cls(id=pk, vector=vector, text=text, metadata=metadata)

    def dump(
        self, *, store_text: bool = True, use_dollar_vector: bool = True, include_timestamps: bool = False
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"_id": self.id}
        if use_dollar_vector:
            out["$vector"] = self.vector
        else:
            out["vector"] = self.vector
        if store_text and self.text is not None:
            out["text"] = self.text
        for k, v in self.metadata.items():
            out[k] = v
        if include_timestamps:
            out["created_timestamp"] = self.created_timestamp
            out["updated_timestamp"] = self.updated_timestamp
        return out

    def to_storage_dict(self, *, store_text: bool = True, use_dollar_vector: bool = True) -> Dict[str, Any]:
        """Prepare document for storage in database.

        This is a convenience method that calls dump() with common parameters.
        Use this in adapters to convert VectorDocument to storage format.

        Args:
            store_text: Whether to include text field in output
            use_dollar_vector: If True, use '$vector' key; otherwise use 'vector'

        Returns:
            Dictionary ready for database storage
        """
        return self.dump(store_text=store_text, use_dollar_vector=use_dollar_vector, include_timestamps=False)
