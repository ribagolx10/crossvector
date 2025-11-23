"""Pydantic schemas for vector store operations."""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def generate_id_and_timestamps(self) -> "Document":
        # Generate ID if missing
        if not self.id:
            # Generate SHA256 hash of the text
            self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

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


class UpsertRequest(BaseModel):
    """Request model for upserting documents."""

    documents: List[Document]


class SearchRequest(BaseModel):
    """Request model for performing a search."""

    query: str
    limit: int = 5
    fields: Set[str] = Field(
        default={"text", "metadata"},
        description="Fields to return in search results.",
    )


class VectorRequest(BaseModel):
    """
    A discriminated union for all possible vector store operations.
    The 'operation' field determines which model to use.
    """

    operation: Literal["upsert", "search"]
    params: UpsertRequest | SearchRequest
