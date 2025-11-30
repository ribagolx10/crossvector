"""Pydantic schemas for vector store operations."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .exceptions import InvalidFieldError, MissingFieldError
from .utils import extract_pk, generate_pk


class VectorDocument(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the vector document.")
    vector: List[float] = Field([], description="Embedding vector.")
    text: Optional[str] = Field(None, description="Optional text content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata.")
    created_timestamp: Optional[float] = Field(None, description="Creation timestamp.")
    updated_timestamp: Optional[float] = Field(None, description="Last update timestamp.")

    @property
    def pk(self) -> str:
        if self.id is None:
            raise MissingFieldError("Document ID not set", field="id")
        return self.id

    @model_validator(mode="after")
    def assign_defaults(self) -> "VectorDocument":
        # Sync private attr
        self._vector = self.vector
        if not self.id:
            self.id = generate_pk(self.text, self.vector, self.metadata)
        current_timestamp = datetime.now(timezone.utc).timestamp()
        if not self.created_timestamp:
            self.created_timestamp = current_timestamp
        self.updated_timestamp = current_timestamp
        return self

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "VectorDocument":
        pk = extract_pk(None, **kwargs)
        # Remove pk fields so they don't leak into metadata
        for k in ("_id", "id", "pk"):
            kwargs.pop(k, None)
        vector = kwargs.pop("vector", None)
        if "$vector" in kwargs and kwargs["$vector"] is not None:
            vector = kwargs.pop("$vector")
        if vector is None:
            raise MissingFieldError("'vector' or '$vector' is required for document.from_kwargs", field="vector")
        text = kwargs.pop("text", None)
        metadata = kwargs.pop("metadata", None) or {}
        for k, v in kwargs.items():
            if k not in metadata:
                metadata[k] = v
        return cls(id=pk, vector=vector, text=text, metadata=metadata)

    @classmethod
    def from_text(cls, text: str, **kwargs: Any) -> "VectorDocument":
        """Create VectorDocument from text with optional metadata.

        Args:
            text: Text content
            **kwargs: Additional fields (id, metadata, or any metadata fields)

        Returns:
            VectorDocument with empty vector (to be filled by engine)

        Examples:
            doc = VectorDocument.from_text("Hello", source="api", user_id="123")
            doc = VectorDocument.from_text("Hello", metadata={"source": "api"})
        """
        pk = extract_pk(None, **kwargs)
        for k in ("_id", "id", "pk"):
            kwargs.pop(k, None)
        metadata = kwargs.pop("metadata", None) or {}
        # Remaining kwargs are metadata fields
        for k, v in kwargs.items():
            if k not in metadata:
                metadata[k] = v
        return cls(id=pk, text=text, vector=[], metadata=metadata)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "VectorDocument":
        """Create VectorDocument from dict, merging with kwargs.

        Args:
            data: Dictionary with document fields
            **kwargs: Additional fields to merge/override

        Returns:
            VectorDocument instance

        Examples:
            doc = VectorDocument.from_dict({"text": "Hello", "source": "api"})
            doc = VectorDocument.from_dict({"text": "Hello"}, user_id="123")
        """
        merged = data.copy()
        merged.update(kwargs)
        # Try from_kwargs if vector exists, otherwise construct minimal doc
        if "$vector" in merged or "vector" in merged:
            return cls.from_kwargs(**merged)
        # No vector - create with minimal fields
        pk = extract_pk(None, **merged)
        for _k in ("_id", "id", "pk"):
            merged.pop(_k, None)
        text = merged.pop("text", None)
        metadata = merged.pop("metadata", None) or {}
        for k, v in merged.items():
            if k not in metadata:
                metadata[k] = v
        return cls(id=pk, text=text, vector=[], metadata=metadata)

    @classmethod
    def from_any(
        cls, doc: Union["VectorDocument", Dict[str, Any], str, None] = None, **kwargs: Any
    ) -> "VectorDocument":
        """Create VectorDocument from any input type.

        Universal factory method that handles:
        - VectorDocument: returns as-is
        - str: treats as text, kwargs become metadata
        - dict: merges with kwargs
        - None: constructs from kwargs (requires 'text' key)

        Args:
            doc: Input data (VectorDocument, dict, text string, or None)
            **kwargs: Additional fields to merge/override

        Returns:
            VectorDocument instance

        Raises:
            TypeError: If doc type is not supported
            ValueError: If cannot construct document from inputs

        Examples:
            # From text string
            doc = VectorDocument.from_any("Hello", source="api")

            # From dict
            doc = VectorDocument.from_any({"text": "Hello"}, user_id="123")

            # From kwargs only
            doc = VectorDocument.from_any(text="Hello", source="api")

            # VectorDocument pass-through
            doc = VectorDocument.from_any(existing_doc)
        """
        # Already a VectorDocument - return as-is
        if isinstance(doc, cls):
            return doc

        # Text string - create from text with kwargs as metadata
        if isinstance(doc, str):
            return cls.from_text(doc, **kwargs)

        # Dict - merge with kwargs
        if isinstance(doc, dict):
            return cls.from_dict(doc, **kwargs)

        # None - construct from kwargs (detect pk/id/_id)
        if doc is None and kwargs:
            if "text" in kwargs:
                text = kwargs.pop("text")
                return cls.from_text(text, **kwargs)
            else:
                return cls.from_dict(kwargs)

        # Invalid input
        if doc is None:
            raise InvalidFieldError("Need doc or kwargs to create VectorDocument", field="doc")

        raise TypeError(f"Cannot create VectorDocument from type: {type(doc).__name__}")

    def dump(
        self, *, store_text: bool = True, use_dollar_vector: bool = False, include_timestamps: bool = False
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

    def to_storage_dict(self, *, store_text: bool = True, use_dollar_vector: bool = False) -> Dict[str, Any]:
        """Prepare VectorDocument for storage in database.

        This is a convenience method that calls dump() with common parameters.
        Use this in adapters to convert VectorDocument to storage format.

        Args:
            store_text: Whether to include text field in output
            use_dollar_vector: If True, use '$vector' key; otherwise use 'vector'

        Returns:
            Dictionary ready for database storage
        """
        return self.dump(store_text=store_text, use_dollar_vector=use_dollar_vector, include_timestamps=False)

    def copy_with(self, **kwargs: Any) -> "VectorDocument":
        """Create a copy with specified fields overridden.

        Only updates fields that are:
        - Explicitly provided in kwargs AND
        - Either current field is None OR new value is truthy

        Supports both 'vector' and '$vector' keys.
        Metadata is always merged (not replaced).

        Args:
            **kwargs: Fields to update (id, text, vector, $vector, metadata)

        Returns:
            New VectorDocument instance with updated fields

        Examples:
            doc = VectorDocument(id="1", text="Hello", vector=[0.1, 0.2])

            # Update text only
            new_doc = doc.copy_with(text="World")

            # Merge metadata
            new_doc = doc.copy_with(metadata={"source": "api"})

            # Update only if current is None
            new_doc = doc.copy_with(text="Default")  # only if doc.text is None
        """
        new_id = self.id
        new_text = self.text
        new_vector = self.vector
        new_metadata = self.metadata.copy()

        # Update id only if current is None
        if "id" in kwargs and self.id is None and kwargs["id"]:
            new_id = kwargs["id"]

        # Update text only if current is None
        if "text" in kwargs and self.text is None and kwargs["text"]:
            new_text = kwargs["text"]

        # Update vector only if current is empty (support both keys)
        vector_val = kwargs.get("vector") or kwargs.get("$vector")
        if vector_val and (not self.vector):
            new_vector = vector_val

        # Merge metadata (always merge, never replace)
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            new_metadata.update(kwargs["metadata"])

        return VectorDocument(
            id=new_id,
            text=new_text,
            vector=new_vector or [],
            metadata=new_metadata,
        )

    # ------------------------------------------------------------------
    # New helper serialization methods
    # ------------------------------------------------------------------
    def to_vector(
        self,
        *,
        require: bool = False,
        output_format: Literal["dict", "json", "str", "list"] = "list",
    ) -> Any:
        """Return the underlying embedding vector.

        Args:
            require: If True, raise MissingFieldError when vector is empty.
            output_format: Desired format of output.
                - 'list' (default): Python list of floats
                - 'dict': {'vector': [...]} wrapper
                - 'json': JSON string
                - 'str': String representation

        Returns:
            Vector in requested format.

        Raises:
            MissingFieldError: If require=True and vector is empty.
        """
        if require and not self.vector:
            raise MissingFieldError("Vector is required", field="vector")
        vec = list(self.vector)
        if output_format == "list":
            return vec
        if output_format == "dict":
            return {"vector": vec}
        if output_format == "json":
            import json

            return json.dumps(vec, ensure_ascii=False)
        if output_format == "str":
            return str(vec)
        return vec  # fallback

    def to_metadata(
        self,
        *,
        exclude: set[str] | None = None,
        sanitize: bool = False,
        max_str_len: int | None = None,
        output_format: Literal["dict", "json", "str"] = "dict",
    ) -> Any:
        """Serialize metadata for adapter/storage use.

        Produces a dict of metadata excluding reserved keys. Optionally sanitizes
        complex values (list, dict, set, tuple, custom objects) into JSON strings.

        Args:
            exclude: Additional keys to exclude from output.
            sanitize: If True, convert non-primitive values to JSON (fallback to str).
            max_str_len: If provided, truncate very long JSON/string values to this length.
            output_format: Output format selection.
                - 'dict' (default): Python dict
                - 'json': JSON string
                - 'str': String representation (repr)

        Returns:
            Metadata in requested format.

        Notes:
            Reserved keys automatically excluded: id, _id, pk, vector, $vector, text,
            created_timestamp, updated_timestamp.
        """
        reserved = {"id", "_id", "pk", "vector", "$vector", "text", "created_timestamp", "updated_timestamp"}
        if exclude:
            reserved |= set(exclude)
        out: Dict[str, Any] = {}
        for k, v in self.metadata.items():
            if k in reserved:
                continue
            if not sanitize or isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
                continue
            # Sanitize complex types
            try:
                import json  # local import to avoid cost if unused

                serialized = json.dumps(v, ensure_ascii=False)
            except Exception:
                serialized = str(v)
            if max_str_len is not None and isinstance(serialized, str) and len(serialized) > max_str_len:
                serialized = serialized[:max_str_len] + "…"
            out[k] = serialized
        if output_format == "dict":
            return out
        if output_format == "json":
            import json

            return json.dumps(out, ensure_ascii=False)
        if output_format == "str":
            return str(out)
        return out
