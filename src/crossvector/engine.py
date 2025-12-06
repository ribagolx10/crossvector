"""
Main engine for orchestrating vector store operations.

This module provides the `VectorEngine`, a high-level class that uses
pluggable adapters for embedding and database operations. It provides
a convenient wrapper around the ABC interface with automatic embedding
generation and flexible input handling.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    from crossvector.querydsl.q import Q

from crossvector.settings import settings

from .abc import EmbeddingAdapter, VectorDBAdapter
from .exceptions import CrossVectorError, InvalidFieldError, MismatchError, MissingFieldError
from .logger import Logger
from .schema import VectorDocument
from .types import Doc
from .utils import extract_pk


class VectorEngine:
    """High-level orchestrator for vector database operations with automatic embedding.

    VectorEngine provides a unified, flexible interface for working with vector databases.
    It handles automatic embedding generation, flexible document input formats, and provides
    both single-document and batch operations following Django-style semantics.

    Key Features:
        - Flexible input: accepts str, dict, or VectorDocument for all operations
        - Automatic embedding generation for text without vectors
        - Batch operations with optimized bulk embedding
        - Django-style get_or_create and update_or_create patterns
        - Pluggable database and embedding adapters

    Attributes:
        collection_name: Active collection name
        store_text: Whether to store original text alongside vectors
        db: Database adapter instance
        embedding: Embedding adapter instance
    """

    def __init__(
        self,
        db: VectorDBAdapter,
        embedding: EmbeddingAdapter,
        collection_name: str = settings.VECTOR_COLLECTION_NAME,
        store_text: bool = settings.VECTOR_STORE_TEXT,
    ) -> None:
        """Initialize VectorEngine with database and embedding adapters.

        Args:
            db: Database adapter implementing VectorDBAdapter interface
            embedding: Embedding adapter implementing EmbeddingAdapter interface
            collection_name: Name of the collection to use (default from settings)
            store_text: Whether to store original text with vectors (default from settings)

        Note:
            Automatically initializes the underlying collection if the adapter supports it.
        """
        self._db = db
        self._embedding = embedding
        self.collection_name = collection_name
        self.store_text = store_text
        self.logger = Logger(self.__class__.__name__)
        # Initialize underlying collection if adapter supports it
        try:
            self._db.initialize(
                collection_name=collection_name,
                dim=self._embedding.dim,
                metric="cosine",
                store_text=store_text,
            )
        except AttributeError:
            # Optional for adapters that don't need explicit initialization
            pass
        self.logger.message(
            "VectorEngine initialized: db=%s embedding=%s store_text=%s",
            db.__class__.__name__,
            embedding.__class__.__name__,
            store_text,
        )

    @property
    def db(self) -> VectorDBAdapter:
        """Access the database adapter instance."""
        return self._db

    @property
    def adapter(self) -> VectorDBAdapter:
        """Access the database adapter (alias for db property)."""
        return self._db

    @property
    def embedding(self) -> EmbeddingAdapter:
        """Access the embedding adapter instance."""
        return self._embedding

    @property
    def supports_vector_search(self) -> bool:
        """Whether underlying adapter supports vector similarity search.

        Falls back to True if adapter does not explicitly define the flag to avoid
        accidental disabling of functionality for simple/mock adapters.
        """
        return bool(getattr(self.db, "supports_vector_search", True))

    @property
    def supports_metadata_only(self) -> bool:
        """Whether underlying adapter supports metadata-only search (no vector).

        Defaults to True for adapters that do not specify the capability.
        """
        return bool(getattr(self.db, "supports_metadata_only", True))

    # ------------------------------------------------------------------
    # Internal normalization helpers
    # ------------------------------------------------------------------
    def _doc_rebuild(
        self,
        doc: Optional[Doc] = None,
        *,
        text: str | None = None,
        vector: List[float] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> VectorDocument:
        """Normalize flexible document inputs into a VectorDocument.

        Accepts multiple input formats (str, dict, VectorDocument) and optional
        builder parameters, merging them into a single VectorDocument. Does not
        generate embeddings - that's left to the caller.

        Args:
            doc: Flexible document input (str | dict | VectorDocument | None)
            text: Optional text content (overrides doc.text if provided)
            vector: Optional vector embedding (overrides doc.vector if provided)
            metadata: Optional metadata dict (merged with doc.metadata if provided)
            **kwargs: Additional metadata fields or id/_id/pk for primary key

        Returns:
            Normalized VectorDocument instance
        """
        if isinstance(doc, VectorDocument):
            return doc

        base: Dict[str, Any] = {}
        if isinstance(doc, dict):
            base.update(doc)
        elif isinstance(doc, str):
            base["text"] = doc
        if text is not None:
            base["text"] = text
        if vector is not None:
            base["vector"] = vector
        if metadata is not None:
            base["metadata"] = metadata

        # Normalize id/_id/pk
        pk = extract_pk(None, **base, **kwargs)
        if pk is not None:
            base["id"] = pk

        return VectorDocument.from_any(base, **kwargs)

    def _doc_prepare_many(self, docs: list[Doc]) -> list[VectorDocument]:
        """Normalize flexible docs and batch-generate missing embeddings.

        Processes a list of flexible document inputs, normalizes each to VectorDocument,
        identifies documents missing vectors, and generates embeddings in a single
        batch call (avoiding N separate embedding API calls).

        Args:
            docs: List of flexible document inputs (str | dict | VectorDocument)

        Returns:
            List of normalized VectorDocuments with all vectors populated

        Note:
            Documents without text or vector are logged and skipped.
        """
        normalized: list[VectorDocument] = []
        to_embed_indices: list[int] = []
        texts_to_embed: list[str] = []
        for item in docs:
            doc_obj = self._doc_rebuild(item)
            # Skip invalid docs (no vector & no text)
            if (not doc_obj.vector) and (not doc_obj.text):
                self.logger.warning("Skipping doc without text/vector id=%s", doc_obj.id)
                continue
            if (not doc_obj.vector) and doc_obj.text:
                to_embed_indices.append(len(normalized))
                texts_to_embed.append(doc_obj.text)
            normalized.append(doc_obj)
        if texts_to_embed:
            embeddings = self.embedding.get_embeddings(texts_to_embed)
            for local_idx, emb in zip(to_embed_indices, embeddings):
                normalized[local_idx].vector = emb
        return normalized

    # ------------------------------------------------------------------
    # Single document operations
    # ------------------------------------------------------------------
    def create(
        self,
        doc: Optional[Doc] = None,
        *,
        text: str | None = None,
        vector: List[float] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> VectorDocument:
        """Create a new vector document with automatic embedding generation.

        Accepts flexible input formats and automatically generates embeddings for
        text content when no vector is provided. Supports explicit primary keys
        or automatic generation.

        Args:
            doc: Document input (str | dict | VectorDocument | None)
            text: Optional text content (overrides doc text)
            vector: Optional precomputed vector (skips embedding if provided)
            metadata: Optional metadata dict
            **kwargs: Additional metadata fields or id/_id/pk for primary key

        Returns:
            Created VectorDocument with populated vector and id

        Raises:
            InvalidFieldError: If neither text nor vector is provided

        Examples:
            >>> engine.create("Hello world")
            >>> engine.create({"text": "Hello", "source": "api"})
            >>> engine.create(text="Hello", metadata={"lang": "en"})
            >>> engine.create({"id": "doc123", "text": "Hello"})
        """
        # Normalize using a single helper
        doc = self._doc_rebuild(doc, text=text, vector=vector, metadata=metadata, **kwargs)

        # Auto-embed if needed
        if (not doc.vector) and doc.text:
            doc.vector = self.embedding.get_embeddings([doc.text])[0]
        if not doc.vector:
            raise InvalidFieldError("Document requires vector or text", field="vector", operation="create")

        self.logger.message("Create pk=%s", doc.id)
        return self.db.create(doc)

    def get(self, *args, **kwargs: Any) -> VectorDocument:
        """Retrieve a single document by primary key or metadata filter.

        Supports Django-style lookup patterns. Primary key takes precedence
        over metadata filters.

        Args:
            *args: Positional arguments (typically primary key as first arg)
            **kwargs: Keyword arguments for metadata filtering

        Returns:
            Matching VectorDocument

        Raises:
            DoesNotExist: If document not found
            MultipleObjectsReturned: If multiple documents match the filter

        Examples:
            >>> engine.get("doc123")
            >>> engine.get(source="api", category="tech")
        """
        return self.db.get(*args, **kwargs)

    def update(
        self,
        doc: Doc,
        *,
        text: str | None = None,
        vector: List[float] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> VectorDocument:
        """Update an existing document with automatic embedding regeneration.

        Updates the specified document, regenerating embeddings if text changes
        without a corresponding vector update. Requires explicit primary key.

        Args:
            doc: Document to update (str | dict | VectorDocument) - must include id
            text: Optional new text content
            vector: Optional new vector (skips embedding if provided with text)
            metadata: Optional metadata updates
            **kwargs: Additional metadata fields

        Returns:
            Updated VectorDocument

        Raises:
            MissingFieldError: If id is missing
            DocumentNotFoundError: If document not found

        Examples:
            >>> engine.update({"id": "doc123", "text": "New content"})
            >>> engine.update("doc123", text="Updated text", category="news")
        """
        # Normalize using a single helper
        doc = self._doc_rebuild(doc, text=text, vector=vector, metadata=metadata, **kwargs)

        # Ensure we have an id
        if doc.id is None:
            raise MissingFieldError("Cannot update without id", field="id", operation="update")

        # Auto-embed if text present and vector missing
        if (not doc.vector) and doc.text:
            doc.vector = self.embedding.get_embeddings([doc.text])[0]

        self.logger.message("Update pk=%s", doc.id)
        return self.db.update(doc, **kwargs)

    def delete(self, *args) -> int:
        """Delete one or more documents by primary key.

        Args:
            *args: One or more document IDs to delete

        Returns:
            Number of documents successfully deleted

        Examples:
            >>> engine.delete("doc123")
            >>> engine.delete("doc1", "doc2", "doc3")
        """
        if not args:
            return 0
        self.logger.message("Delete ids=%s", args)
        return self.db.delete(*args)

    def count(self) -> int:
        """Count total documents in the collection.

        Returns:
            Total number of documents
        """
        return self.db.count()

    def get_or_create(
        self,
        doc: Optional[Doc] = None,
        *,
        text: str | None = None,
        vector: List[float] | None = None,
        metadata: Dict[str, Any] | None = None,
        defaults: Dict[str, Any] | None = None,
        score_threshold: float = 0.9,
        priority: list[str] = ["pk", "vector", "metadata"],
        **kwargs,
    ) -> tuple[VectorDocument, bool]:
        """
        Get existing document or create new one, with multi-step lookup (pk, vector, metadata).

        Attempts lookup in order of priority:
            1. PK (id)
            2. Vector similarity (if supported)
            3. Metadata-only (if supported)
        If not found, creates new document with defaults.

        Args:
            doc: Document input (str | dict | VectorDocument | None)
            text: Optional text content
            vector: Optional vector (used for similarity search)
            metadata: Optional metadata for lookup/creation
            defaults: Additional fields applied only when creating
            score_threshold: Minimum similarity score for vector match
            priority: List of lookup steps ("pk", "vector", "metadata")
            **kwargs: Extra metadata or id/_id/pk fields

        Returns:
            Tuple of (document, created) where created is True if new document

        Raises:
            MismatchError: If provided text mismatches existing text for same ID

        Backend compatibility:
            - Not all backends support vector or metadata-only search. Steps are skipped if unsupported.
            - Example: AstraDB only supports PK and metadata, not vector similarity.

        Examples:
            >>> doc, created = engine.get_or_create("Hello", source="api")
            >>> doc, created = engine.get_or_create({"id": "doc123", "text": "Hello"})
            >>> doc, created = engine.get_or_create(
            ...     text="Hello",
            ...     metadata={"lang": "en"},
            ...     defaults={"priority": "high"}
            ... )
        """
        doc = self._doc_rebuild(doc, text=text, vector=vector, metadata=metadata, **kwargs)

        for step in priority:
            if step == "pk" and doc.id:
                try:
                    existing = self.get(doc.id)
                    if text and existing.text != text:
                        raise MismatchError(
                            "Text mismatch for PK",
                            provided_text=text,
                            existing_text=existing.text,
                            document_id=existing.id,
                        )
                    return existing, False
                except Exception:
                    continue

            elif step == "vector" and vector:
                if not self.supports_vector_search:
                    continue
                candidates = self.search(query=vector, limit=1, fields={"text", "metadata"})
                if candidates:
                    candidate = candidates[0]
                    score_val = candidate.metadata.get("score", 1.0) if isinstance(candidate.metadata, dict) else 1.0
                    if isinstance(score_val, (int, float)) and score_val >= score_threshold:
                        return candidate, False

            elif step == "metadata" and doc.metadata:
                if not self.supports_metadata_only:
                    continue
                # Wrap primitive metadata values into universal operator form {field: {"$eq": value}}
                lookup_metadata = {}
                for mk, mv in doc.metadata.items():
                    if isinstance(mv, dict):
                        lookup_metadata[mk] = mv
                    else:
                        lookup_metadata[mk] = {"$eq": mv}
                results = self.db.search(vector=None, limit=1, where=lookup_metadata)
                if results:
                    existing = results[0]
                    if text and existing.text != text:
                        raise MismatchError(
                            "Text mismatch for metadata lookup",
                            provided_text=text,
                            existing_text=existing.text,
                            document_id=existing.id,
                        )
                    return existing, False

        # Create new doc
        if defaults:
            doc = doc.copy_with(**defaults)
        return self.create(doc), True

    def update_or_create(
        self,
        doc: Optional[Doc] = None,
        *,
        text: str | None = None,
        vector: List[float] | None = None,
        metadata: Dict[str, Any] | None = None,
        defaults: Dict[str, Any] | None = None,
        create_defaults: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[VectorDocument, bool]:
        """
        Update an existing document by ID, or create new one if not found.

        Cross-engine safe: only updates by PK. Creation applies both defaults and create_defaults.
        Checks for text mismatch after update.

        Args:
            doc: Document input (must include id field)
            text: Optional text content
            vector: Optional vector embedding
            metadata: Optional metadata dict
            defaults: Fields applied to both update and create paths
            create_defaults: Fields applied only when creating (not updating)
            **kwargs: Additional metadata fields

        Returns:
            Tuple of (document, created) where created is False for update, True for create

        Raises:
            MissingFieldError: If no ID provided in doc or kwargs
            MismatchError: If text mismatch after update

        Backend compatibility:
            - Only PK-based update is supported for all backends.
            - Creation path applies all defaults and create_defaults.

        Examples:
            >>> doc, created = engine.update_or_create(
            ...     {"id": "doc123", "text": "Updated"},
            ...     defaults={"updated_at": "2024-01-01"}
            ... )
            >>> doc, created = engine.update_or_create(
            ...     {"id": "doc456", "text": "New"},
            ...     create_defaults={"created_at": "2024-01-01"}
            ... )
        """
        # Normalize doc and extract ID
        doc = self._doc_rebuild(doc, text=text, vector=vector, metadata=metadata, **kwargs)

        if not doc.id:
            raise MissingFieldError("Cannot update_or_create without id", field="id", operation="update_or_create")

        # Try update path
        try:
            # Apply defaults to both update and create paths
            if defaults:
                doc = doc.copy_with(**defaults)
            updated_doc = self.update(doc)
            return updated_doc, False
        except CrossVectorError:
            # Document not found → fallback to create
            pass

        # Merge defaults + create_defaults for creation
        merged_defaults: Dict[str, Any] = {}
        if defaults:
            merged_defaults.update(defaults)
        if create_defaults:
            merged_defaults.update(create_defaults)
        if merged_defaults:
            doc = doc.copy_with(**merged_defaults)

        # Ensure vector exists before create
        if not doc.vector and doc.text:
            doc.vector = self.embedding.get_embeddings([doc.text])[0]

        # Create new document
        new_doc = self.create(doc)
        return new_doc, bool(create_defaults)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------
    def bulk_create(
        self,
        docs: List[Doc],
        ignore_conflicts: bool = False,
        update_conflicts: bool = False,
    ) -> List[VectorDocument]:
        """Create multiple documents in batch with optimized embedding generation.

        Normalizes all inputs and generates embeddings in a single batch call
        for better performance. Supports conflict handling strategies.

        Args:
            docs: List of documents (str | dict | VectorDocument)
            ignore_conflicts: If True, skip documents with conflicting IDs
            update_conflicts: If True, update existing documents on ID conflict

        Returns:
            List of created VectorDocuments

        Examples:
            >>> engine.bulk_create(["Hello", "World", "Test"])
            >>> engine.bulk_create([
            ...     {"id": "doc1", "text": "First"},
            ...     {"id": "doc2", "text": "Second"}
            ... ])
        """
        prepared = self._doc_prepare_many(docs)
        self.logger.message("Bulk create count=%d", len(prepared))
        return self.db.bulk_create(
            prepared,
            ignore_conflicts=ignore_conflicts,
            update_conflicts=update_conflicts,
        )

    def bulk_update(
        self,
        docs: List[Doc],
        batch_size: int | None = None,
        ignore_conflicts: bool = False,
    ) -> List[VectorDocument]:
        """Update multiple existing documents in batch.

        Updates documents by ID with automatic embedding regeneration for
        changed text. All documents must have explicit IDs.

        Args:
            docs: List of documents to update (each must include id)
            batch_size: Optional batch size for chunked updates
            ignore_conflicts: If True, skip documents that don't exist

        Returns:
            List of updated VectorDocuments

        Raises:
            ValueError: If any document lacks an ID (when ignore_conflicts=False)

        Examples:
            >>> engine.bulk_update([
            ...     {"id": "doc1", "text": "Updated first"},
            ...     {"id": "doc2", "text": "Updated second"}
            ... ])
        """
        prepared = self._doc_prepare_many(docs)
        self.logger.message("Bulk update count=%d", len(prepared))
        return self.db.bulk_update(
            prepared,
            batch_size=batch_size,
            ignore_conflicts=ignore_conflicts,
        )

    def upsert(self, docs: list[Doc], batch_size: int | None = None) -> list[VectorDocument]:
        """Insert or update multiple documents in batch (upsert operation).

        Creates new documents or updates existing ones based on ID presence.
        Optimizes embedding generation with single batch call.

        Args:
            docs: List of documents (str | dict | VectorDocument)
            batch_size: Optional batch size for chunked operations

        Returns:
            List of upserted VectorDocuments

        Examples:
            >>> engine.upsert([
            ...     {"id": "doc1", "text": "First"},
            ...     {"id": "doc2", "text": "Second"}
            ... ])
        """
        prepared = self._doc_prepare_many(docs)
        self.logger.message("Upsert count=%d", len(prepared))
        return self.db.upsert(prepared, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Search and query operations
    # ------------------------------------------------------------------
    def search(
        self,
        query: Union[str, List[float], None] = None,
        limit: int | None = None,
        offset: int = 0,
        where: Union[Dict[str, Any], "Q", None] = None,
        fields: Set[str] | None = None,
    ) -> List[VectorDocument]:
        """Search for similar documents by text query, vector, or metadata/Q object.

        Performs semantic search using vector similarity. Automatically generates
        embeddings for text queries. Supports metadata filtering and field projection.
        Accepts universal dict or Q object for `where` argument.

        Args:
            query: Search query (str for text, List[float] for vector, None for metadata-only)
            limit: Maximum number of results (default from settings)
            offset: Number of results to skip
            where: Metadata filter conditions (dict or Q object)
            fields: Set of fields to include in results

        Returns:
            List of matching VectorDocuments, ordered by similarity

        Examples:
            >>> results = engine.search("machine learning", limit=5)
            >>> results = engine.search(
            ...     "AI trends",
            ...     where={"category": "tech", "year": 2024}
            ... )
            >>> results = engine.search(where=Q(category="tech", year=2024))
            >>> results = engine.search(where={"status": "active"})  # metadata-only
        """
        vector = None
        if isinstance(query, str):
            vector = self.embedding.get_embeddings([query])[0]
        elif isinstance(query, list):
            vector = query
        else:
            # Metadata-only search (query is None)
            if not self.supports_metadata_only:
                raise InvalidFieldError(
                    "Adapter does not support metadata-only search; vector is required",
                    field="vector",
                    operation="search",
                )

        # Use default limit from settings if not provided
        if limit is None:
            limit = settings.VECTOR_SEARCH_LIMIT

        return self.db.search(vector=vector, limit=limit, offset=offset, where=where, fields=fields)

    # ------------------------------------------------------------------
    # Collection management operations
    # ------------------------------------------------------------------
    def get_collection(self, collection_name: str | None = None) -> Any:
        """Get a collection by name.

        Args:
            collection_name: Name of collection (defaults to engine's active collection)

        Returns:
            Collection object (adapter-specific type)
        """
        return self.db.get_collection(collection_name or self.collection_name)

    def add_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        """Create a new collection with specified configuration.

        Args:
            collection_name: Name for the new collection
            dimension: Vector dimension size
            metric: Distance metric ("cosine", "euclidean", or "dot_product")
        """
        self.logger.message("Add collection name=%s dimension=%d metric=%s", collection_name, dimension, metric)
        self.db.add_collection(collection_name, dimension, metric)

    def get_or_create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> Any:
        """Get existing collection or create if it doesn't exist.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension (used if creating)
            metric: Distance metric (used if creating)

        Returns:
            Collection object (adapter-specific type)
        """
        return self.db.get_or_create_collection(collection_name, dimension, metric)

    def drop_collection(self, collection_name: str | None = None) -> bool:
        """Drop/delete a collection from the database.

        Args:
            collection_name: Name of collection to drop (defaults to engine's active collection)

        Returns:
            True if successful

        Examples:
            >>> engine.drop_collection("old_collection")
            >>> engine.drop_collection()  # Drops current collection
        """
        target_name = collection_name or self.collection_name
        self.logger.message("Drop collection name=%s", target_name)
        return self.db.drop_collection(target_name)
