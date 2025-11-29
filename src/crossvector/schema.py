"""Pydantic schemas for vector store operations."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator, PrivateAttr
from .utils import generate_pk


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
            raise ValueError("Document ID not set")
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
        pk = kwargs.pop("_id", None) or kwargs.pop("id", None)
        vector = kwargs.pop("$vector", None) or kwargs.pop("vector", None)
        if vector is None:
            raise ValueError("'vector' or '$vector' is required for document.from_kwargs")
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
