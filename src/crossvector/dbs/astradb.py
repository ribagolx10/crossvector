"""Concrete adapter for AstraDB vector database.

This module provides the AstraDB implementation of the VectorDBAdapter interface,
enabling vector storage and retrieval using DataStax Astra DB's vector search capabilities.

Key Features:
    - Lazy client/database initialization
    - Full CRUD operations with VectorDocument models
    - Batch operations for bulk create/update/upsert
    - Configurable vector metrics (cosine, euclidean, dot_product)
    - Automatic collection management and schema creation
"""

import math
from typing import Any, Dict, List, Set

from astrapy import DataAPIClient
from astrapy.constants import DOC
from astrapy.constants import VectorMetric as AstraVectorMetric
from astrapy.data.collection import Collection
from astrapy.database import Database
from astrapy.info import CollectionDefinition, CollectionVectorOptions

from crossvector.abc import VectorDBAdapter
from crossvector.constants import VECTOR_METRIC_MAP, VectorMetric
from crossvector.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    CollectionNotInitializedError,
    DocumentExistsError,
    DocumentNotFoundError,
    DoesNotExist,
    MissingConfigError,
    MissingDocumentError,
    MissingFieldError,
    MultipleObjectsReturned,
    SearchError,
)
from crossvector.querydsl.compilers.astradb import AstraDBWhereCompiler, astradb_where
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.utils import (
    apply_update_fields,
    chunk_iter,
    extract_pk,
    prepare_item_for_storage,
)


class AstraDBAdapter(VectorDBAdapter):
    """Vector database adapter for DataStax Astra DB.

    Provides a high-level interface for vector operations using Astra DB's
    vector search capabilities. Implements lazy connection initialization
    and follows the standard VectorDBAdapter interface.

    Attributes:
        collection_name: Name of the active collection
        dim: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
        collection: Active AstraDB collection instance
    """

    _db: Database | None = None
    use_dollar_vector: bool = True
    where_compiler: AstraDBWhereCompiler = astradb_where
    supports_metadata_only: bool = True  # Allow metadata-only filtering without vector

    @property
    def client(self) -> DataAPIClient:
        """Lazily initialize and return the AstraDB DataAPIClient.

        Returns:
            Initialized DataAPIClient instance

        Raises:
            MissingConfigError: If ASTRA_DB_APPLICATION_TOKEN is not configured
        """
        if self._client is None:
            if not api_settings.ASTRA_DB_APPLICATION_TOKEN:
                raise MissingConfigError(
                    "ASTRA_DB_APPLICATION_TOKEN is not set. Please configure it in your .env file.",
                    config_key="ASTRA_DB_APPLICATION_TOKEN",
                    env_file=".env",
                )
            self._client = DataAPIClient(token=api_settings.ASTRA_DB_APPLICATION_TOKEN)
            self.logger.message("AstraDB DataAPIClient initialized.")
        return self._client

    @property
    def db(self) -> Database:
        """Lazily initialize and return the AstraDB database instance.

        Returns:
            Initialized Database instance

        Raises:
            MissingConfigError: If ASTRA_DB_API_ENDPOINT is not configured
        """
        if self._db is None:
            if not api_settings.ASTRA_DB_API_ENDPOINT:
                raise MissingConfigError(
                    "ASTRA_DB_API_ENDPOINT is not set. Please configure it in your .env file.",
                    config_key="ASTRA_DB_API_ENDPOINT",
                    env_file=".env",
                )
            self._db = self.client.get_database(api_endpoint=api_settings.ASTRA_DB_API_ENDPOINT)
            self.logger.message("AstraDB database connection established.")
        return self._db

    @property
    def collection(self) -> Collection[DOC] | None:
        """Return the active AstraDB collection instance.

        Returns:
            Collection instance or None if not initialized
        """
        return self._collection

    @collection.setter
    def collection(self, value: Collection[DOC] | None) -> None:
        """Set the collection object."""
        self._collection = value

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def initialize(
        self,
        collection_name: str,
        dim: int,
        metric: str | None = None,
        store_text: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the database and ensure the collection is ready.

        Args:
            collection_name: Name of the collection to use/create
            dim: Dimension of the vector embeddings
            metric: Distance metric ('cosine', 'euclidean', 'dot_product')
            store_text: Whether to store original text content
            **kwargs: Additional configuration options
        """
        self.store_text = store_text if store_text is not None else api_settings.VECTOR_STORE_TEXT
        if metric is None:
            metric = api_settings.VECTOR_METRIC or VectorMetric.COSINE
        self.get_or_create_collection(collection_name, dim, metric)
        self.logger.message(
            f"AstraDB initialized: collection='{collection_name}', "
            f"dimension={dim}, metric={metric}, store_text={self.store_text}"
        )

    def add_collection(self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE) -> Collection[DOC]:
        """Create a new AstraDB collection.

        Args:
            collection_name: Name of the collection to create
            dim: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            AstraDB Collection instance

        Raises:
            ValueError: If collection already exists
        """
        existing_collections = self.db.list_collection_names()
        if collection_name in existing_collections:
            raise CollectionExistsError("Collection already exists", collection_name=collection_name)

        self.collection_name = collection_name
        self.dim = dim
        if not hasattr(self, "store_text"):
            self.store_text = True

        vector_metric = VECTOR_METRIC_MAP.get(metric.lower(), AstraVectorMetric.COSINE)
        self.collection = self.db.create_collection(
            collection_name,
            definition=CollectionDefinition(
                vector=CollectionVectorOptions(
                    dimension=dim,
                    metric=vector_metric,
                ),
            ),
        )
        self.logger.message(f"AstraDB collection '{collection_name}' created successfully.")
        return self.collection

    def get_collection(self, collection_name: str) -> Collection[DOC]:
        """Get an existing AstraDB collection.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            AstraDB Collection instance

        Raises:
            ValueError: If collection doesn't exist
        """
        existing_collections = self.db.list_collection_names()
        if collection_name not in existing_collections:
            raise CollectionNotFoundError("Collection does not exist", collection_name=collection_name)

        self.collection = self.db.get_collection(collection_name)
        self.collection_name = collection_name
        self.logger.message(f"AstraDB collection '{collection_name}' retrieved.")
        return self.collection

    def get_or_create_collection(
        self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE
    ) -> Collection[DOC]:
        """Get or create the underlying AstraDB collection.

        Ensures the collection exists with proper vector configuration.
        If the collection doesn't exist, it will be created with the specified
        embedding dimension and distance metric.

        Args:
            collection_name: Name of the collection
            dim: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            AstraDB Collection instance

        Raises:
            CollectionExistsError: If collection already exists
            CollectionNotFoundError: If collection does not exist
            CollectionNotInitializedError: If collection is not initialized
            MissingConfigError: If configuration is missing
            SearchError: If collection initialization fails
        """
        try:
            self.collection_name = collection_name
            self.dim = dim
            if not hasattr(self, "store_text"):
                self.store_text = True

            existing_collections = self.db.list_collection_names()

            if collection_name in existing_collections:
                self.collection = self.db.get_collection(collection_name)
                self.logger.message(f"AstraDB collection '{collection_name}' retrieved.")
            else:
                vector_metric = VECTOR_METRIC_MAP.get(metric.lower(), AstraVectorMetric.COSINE)
                self.logger.message(f"Creating AstraDB collection '{collection_name}'...")
                self.collection = self.db.create_collection(
                    collection_name,
                    definition=CollectionDefinition(
                        vector=CollectionVectorOptions(
                            dimension=dim,
                            metric=vector_metric,
                        ),
                    ),
                )
                self.logger.message(f"AstraDB collection '{collection_name}' created successfully.")

            return self.collection
        except CollectionExistsError as e:
            self.logger.error(f"Collection already exists: {e}", exc_info=True)
            raise
        except CollectionNotFoundError as e:
            self.logger.error(f"Collection does not exist: {e}", exc_info=True)
            raise
        except CollectionNotInitializedError as e:
            self.logger.error(f"Collection not initialized: {e}", exc_info=True)
            raise
        except MissingConfigError as e:
            self.logger.error(f"Missing configuration: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize AstraDB collection: {e}", exc_info=True)
            raise SearchError(f"Failed to initialize AstraDB collection: {e}")

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the specified collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful
        """
        self.db.drop_collection(collection_name)
        self.logger.message(f"AstraDB collection '{collection_name}' dropped.")
        return True

    def clear_collection(self) -> int:
        """Delete all documents from the collection.

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="clear_collection", adapter="AstraDB"
            )
        result = self.collection.delete_many({})
        self.logger.message(f"Cleared {result.deleted_count} documents from collection.")
        return result.deleted_count

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="count", adapter="AstraDB")
        count = self.collection.count_documents({}, upper_bound=10000)
        return count

    # ------------------------------------------------------------------
    # Search Operations
    # ------------------------------------------------------------------

    def _compute_similarity(self, query_vector: List[float], result_vector: List[float]) -> float | None:
        """Private helper to compute cosine similarity between two vectors.

        Returns None on any error or if vectors are empty / length mismatch.
        """
        try:
            if not query_vector or not result_vector:
                return None
            # Length guard (Astra should enforce uniform dimensions)
            if len(query_vector) != len(result_vector):
                return None
            # Compute norms
            q_norm_sq = 0.0
            r_norm_sq = 0.0
            dot = 0.0
            for q, r in zip(query_vector, result_vector):
                q_norm_sq += q * q
                r_norm_sq += r * r
                dot += q * r
            if q_norm_sq <= 0.0 or r_norm_sq <= 0.0:
                return None
            similarity = dot / (math.sqrt(q_norm_sq) * math.sqrt(r_norm_sq))
            # Clamp minor floating drift
            if similarity > 1.0:
                similarity = 1.0
            elif similarity < -1.0:
                similarity = -1.0
            return similarity
        except Exception:
            return None

    def search(
        self,
        vector: List[float] | None = None,
        limit: int | None = None,
        offset: int = 0,
        where: Dict[str, Any] | None = None,
        fields: Set[str] | None = None,
    ) -> List[VectorDocument]:
        """Perform vector similarity search.

        Args:
            vector: Query vector embedding
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            where: Optional metadata filter conditions
            fields: Optional set of field names to include in results

        Returns:
            List of VectorDocument instances ordered by similarity

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="search", adapter="AstraDB")

        if limit is None:
            limit = api_settings.VECTOR_SEARCH_LIMIT

        # Build filter query
        if where is not None:
            # Compiler handles both Q objects and dicts
            where = self.where_compiler.to_where(where)

        try:
            # Projection rules:
            # - If performing vector search, include '$vector' to allow VectorDocument creation.
            # - If metadata-only search, we can exclude vector for efficiency and inject empty list.
            projection = None
            if fields and vector is not None:
                projection = {field: 1 for field in fields}

            # AstraDB doesn't have native skip, so we fetch limit+offset and slice
            fetch_limit = limit + offset

            if vector is None:
                # Metadata-only search (no vector sorting)
                results = list(
                    self.collection.find(
                        filter=where,
                        limit=fetch_limit,
                        projection=projection,
                    )
                )
            else:
                # Vector search with sorting
                results = list(
                    self.collection.find(
                        filter=where,
                        sort={"$vector": vector},
                        limit=fetch_limit,
                        projection=projection,
                    )
                )

            # Apply offset by slicing
            results = results[offset:]

            # Build map of full documents to restore metadata fields lost due to projection.
            # We only need this when we didn't explicitly request fields beyond vector/text.
            # Always fetch full docs so we can reconstruct metadata accurately (small result sets).
            ids: List[str] = [r.get("_id") for r in results if r.get("_id")]
            full_docs: List[Dict[str, Any]] = []
            full_map: Dict[str, Dict[str, Any]] = {}
            if ids:
                try:
                    full_docs = list(self.collection.find({"_id": {"$in": ids}}, limit=len(ids)))
                    for fd in full_docs:
                        _id = fd.get("_id")
                        if _id:
                            full_map[_id] = fd
                except Exception:
                    # Non-critical; fallback to projected docs
                    pass

            prepared_docs: List[VectorDocument] = []
            reserved: Set[str] = {"_id", "id", "pk", "text", "$vector", "vector"}

            # Query vector reference for similarity scoring (None for metadata-only searches)
            query_vector = vector if vector is not None else None

            for doc in results:
                _id = doc.get("_id")
                # Normalize vector field
                if "$vector" in doc:
                    vec = doc["$vector"]
                elif "vector" in doc:
                    vec = doc["vector"]
                else:
                    vec = []  # metadata-only path (no vector returned)
                    if query_vector:
                        self.logger.warning("AstraDB search missing vector for id=%s; similarity skipped", _id)

                # Restore full document metadata if available
                full_doc = full_map.get(_id, {}) if _id else {}
                metadata_block: Dict[str, Any] = {}

                # Merge explicit 'metadata' nested dict first (if adapter later stores it)
                if isinstance(full_doc.get("metadata"), dict):
                    for k, v in full_doc["metadata"].items():
                        metadata_block[k] = v

                # Include top-level metadata keys not in reserved set
                for k, v in full_doc.items():
                    if k not in reserved and k != "metadata":
                        metadata_block.setdefault(k, v)

                # Similarity score injection using helper
                if query_vector and vec and isinstance(vec, list):
                    similarity = self._compute_similarity(query_vector, vec)
                    if similarity is not None and "score" not in metadata_block:
                        metadata_block["score"] = similarity
                        self.logger.message("AstraDB similarity computed id=%s score=%.6f", _id, similarity)

                doc_dict: Dict[str, Any] = {"_id": _id, "vector": vec, "metadata": metadata_block}

                # Prefer text from projected doc, fallback to full_doc
                text_val = doc.get("text") or full_doc.get("text")
                if text_val is not None:
                    doc_dict["text"] = text_val

                prepared_docs.append(VectorDocument.from_kwargs(**doc_dict))

            self.logger.message(f"Vector search returned {len(prepared_docs)} results.")
            return prepared_docs
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}", exc_info=True)
            raise SearchError(f"AstraDB search failed: {e}") from e

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def get(self, *args, **kwargs) -> VectorDocument:
        """Retrieve a single document by its ID or metadata filter.

        Args:
            *args: Optional positional pk
            **kwargs: Metadata fields for filtering (e.g., name="value", status="active")
                     Special keys: pk/id/_id for direct lookup

        Returns:
            VectorDocument instance

        Raises:
            CollectionNotInitializedError: If collection is not initialized
            DoesNotExist: If no document matches
            MultipleObjectsReturned: If multiple documents match
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="get", adapter="AstraDB")

        pk = args[0] if args else None
        doc_id = pk or extract_pk(None, **kwargs) if not pk else pk

        # Priority 1: Direct pk lookup
        if doc_id:
            results = list(
                self.collection.find(
                    {"_id": doc_id},
                    limit=2,
                    projection={"$vector": 1, "_id": 1, "text": 1, "metadata": 1},
                )
            )
            if not results:
                raise DoesNotExist(f"Document with ID '{doc_id}' not found")
            if len(results) > 1:
                raise MultipleObjectsReturned(f"Multiple documents found with ID '{doc_id}'")
            raw = results[0]
            if "$vector" in raw:
                raw["vector"] = raw["$vector"]
            else:
                raw["vector"] = []
            return VectorDocument.from_kwargs(**raw)

        # Priority 2: Search by metadata kwargs using search method
        metadata_kwargs = {k: v for k, v in kwargs.items() if k not in ("pk", "id", "_id")}
        if not metadata_kwargs:
            raise MissingFieldError(
                "Either pk/id/_id or metadata filter kwargs required", field="id or filter", operation="get"
            )

        results = self.search(vector=None, where=metadata_kwargs, limit=2)
        if not results:
            raise DoesNotExist("No document found matching metadata filter")
        if len(results) > 1:
            raise MultipleObjectsReturned("Multiple documents found matching metadata filter")
        return results[0]

    def create(self, doc: VectorDocument) -> VectorDocument:
        """Create and persist a single document.

        Args:
            doc: VectorDocument instance to create

        Returns:
            Created VectorDocument instance

        Raises:
            CollectionNotInitializedError: If collection is not initialized
            DocumentExistsError: If document with same ID already exists
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="create", adapter="AstraDB")

        stored = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        # Conflict check
        if self.collection.find_one({"_id": doc.pk}):
            raise DocumentExistsError("Document already exists", document_id=doc.pk)

        self.collection.insert_one(stored)
        self.logger.message(f"Created document with id '{doc.pk}'.")
        return doc

    def update(self, doc: VectorDocument, **kwargs) -> VectorDocument:
        """Update a single document by ID.

        Strict update semantics: raises error if document doesn't exist.

        Args:
            doc: VectorDocument to update (must include id/pk)

        Returns:
            Updated VectorDocument instance

        Raises:
            CollectionNotInitializedError: If collection is not initialized
            MissingFieldError: If ID is missing
            DocumentNotFoundError: If document not found
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="update", adapter="AstraDB")

        pk = doc.id or extract_pk(None, **kwargs)
        if not pk:
            raise MissingFieldError("'id', '_id', or 'pk' is required for update", field="id", operation="update")

        existing = self.collection.find_one({"_id": pk})
        if not existing:
            raise DocumentNotFoundError("Document not found", document_id=pk, operation="update")

        prepared = prepare_item_for_storage(
            doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector),
            store_text=self.store_text,
        )
        update_doc = {k: v for k, v in prepared.items() if k not in ("_id", "$vector")}
        if "$vector" in prepared:
            update_doc["$vector"] = prepared["$vector"]

        if update_doc:
            self.collection.update_one({"_id": pk}, {"$set": update_doc})
            self.logger.message(f"Updated document with id '{pk}'.")

        refreshed = self.collection.find_one({"_id": pk}, projection={"$vector": 1, "_id": 1, "text": 1, "metadata": 1})
        # Convert $vector to vector for VectorDocument
        if "$vector" in refreshed:
            refreshed["vector"] = refreshed.pop("$vector")
        return VectorDocument.from_kwargs(**refreshed)

    def delete(self, *args) -> int:
        """Delete documents by ID.

        Args:
            *args: One or more document IDs to delete

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="delete", adapter="AstraDB")

        if not args:
            return 0

        if len(args) == 1:
            result = self.collection.delete_one({"_id": args[0]})
            deleted = result.deleted_count
        else:
            result = self.collection.delete_many({"_id": {"$in": args}})
            deleted = result.deleted_count

        self.logger.message(f"Deleted {deleted} documents.")
        return deleted

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    def bulk_create(
        self,
        docs: List[VectorDocument],
        batch_size: int = None,
        ignore_conflicts: bool = False,
        update_conflicts: bool = False,
        update_fields: List[str] = None,
    ) -> List[VectorDocument]:
        """Bulk create multiple documents.

        Args:
            docs: List of VectorDocument instances to create
            batch_size: Number of documents per batch (optional)
            ignore_conflicts: If True, skip conflicting documents
            update_conflicts: If True, update conflicting documents
            update_fields: Fields to update on conflict (if update_conflicts=True)

        Returns:
            List of successfully created VectorDocument instances

        Raises:
            CollectionNotInitializedError: If collection is not initialized
            DocumentExistsError: If conflict occurs and ignore_conflicts=False
        """
        if not self.collection:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="bulk_create", adapter="AstraDB"
            )
        if not docs:
            return []

        items_to_insert: List[Dict[str, Any]] = []
        created_docs: List[VectorDocument] = []

        for doc in docs:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            pk = doc.pk

            # Check conflict by _id
            existing = self.collection.find_one({"_id": pk})

            if existing:
                if ignore_conflicts:
                    continue
                if update_conflicts:
                    update_doc = apply_update_fields(item, update_fields)
                    if update_doc:
                        self.collection.update_one({"_id": existing["_id"]}, {"$set": update_doc})
                    continue
                raise DocumentExistsError(
                    "Document already exists", document_id=item.get("_id"), operation="bulk_create"
                )

            items_to_insert.append(item)
            created_docs.append(doc)

        if items_to_insert:
            if batch_size and batch_size > 0:
                for chunk in chunk_iter(items_to_insert, batch_size):
                    self.collection.insert_many(list(chunk))
            else:
                self.collection.insert_many(items_to_insert)

        self.logger.message(f"Bulk created {len(created_docs)} documents.")
        return created_docs

    def bulk_update(
        self,
        docs: List[VectorDocument],
        batch_size: int = None,
        ignore_conflicts: bool = False,
        update_fields: List[str] = None,
    ) -> List[VectorDocument]:
        """Bulk update existing documents by ID.

        Args:
            docs: List of VectorDocument instances to update
            batch_size: Number of updates per batch (optional)
            ignore_conflicts: If True, skip missing documents
            update_fields: Specific fields to update (None = all fields)

        Returns:
            List of successfully updated VectorDocument instances

        Raises:
            CollectionNotInitializedError: If collection is not initialized
            MissingDocumentError: If any document is missing and ignore_conflicts=False
        """
        if not self.collection:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="bulk_update", adapter="AstraDB"
            )
        if not docs:
            return []

        # Collect and validate primary keys
        pk_to_doc: Dict[str, VectorDocument] = {}
        for doc in docs:
            pk = doc.pk
            if not pk:
                if ignore_conflicts:
                    continue
                raise MissingDocumentError("Document missing ID", missing_ids=["<no_id>"], operation="bulk_update")
            pk_to_doc[pk] = doc

        if not pk_to_doc:
            return []

        # Single query to fetch existing documents (avoid N+1)
        pks = list(pk_to_doc.keys())
        existing_docs = list(self.collection.find({"_id": {"$in": pks}}))
        existing_map = {d["_id"]: d for d in existing_docs}

        # Detect missing
        missing = [pk for pk in pks if pk not in existing_map]
        if missing and not ignore_conflicts:
            raise MissingDocumentError("Missing documents for update", missing_ids=missing, operation="bulk_update")

        # Perform per-document updates (AstraDB has no multi-update with per-doc different bodies)
        updated_docs: List[VectorDocument] = []
        for pk, doc in pk_to_doc.items():
            if pk not in existing_map:
                continue  # skipped due to ignore_conflicts

            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(item, update_fields)
            if not update_doc:
                continue

            # Prepare $set payload (preserve vector field if present)
            set_payload: Dict[str, Any] = {}
            for k, v in update_doc.items():
                if k == "$vector":
                    set_payload["$vector"] = v
                elif k != "_id":
                    set_payload[k] = v

            if set_payload:
                self.collection.update_one({"_id": pk}, {"$set": set_payload})
                updated_docs.append(doc)

        self.logger.message(f"Bulk updated {len(updated_docs)} documents.")
        return updated_docs

    def upsert(self, docs: List[VectorDocument], batch_size: int = None) -> List[VectorDocument]:
        """Insert or update multiple documents.

        Args:
            docs: List of VectorDocument instances to upsert
            batch_size: Number of documents per batch (optional)

        Returns:
            List of upserted VectorDocument instances

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="upsert", adapter="AstraDB")
        if not docs:
            return []

        # Collect all IDs for single fetch (avoid N+1 lookups)
        ids = [doc.pk for doc in docs if doc.pk]
        existing_docs = list(self.collection.find({"_id": {"$in": ids}})) if ids else []
        existing_map = {d["_id"]: d for d in existing_docs}

        to_insert: List[Dict[str, Any]] = []
        updated: List[VectorDocument] = []
        inserted: List[VectorDocument] = []

        for doc in docs:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            pk = doc.pk
            if pk in existing_map:
                # Build $set update excluding _id
                set_payload: Dict[str, Any] = {}
                for k, v in item.items():
                    if k == "_id":
                        continue
                    set_payload[k] = v
                if set_payload:
                    self.collection.update_one({"_id": pk}, {"$set": set_payload})
                updated.append(doc)
            else:
                to_insert.append(item)
                inserted.append(doc)

        # Batch insert new documents
        if to_insert:
            if batch_size and batch_size > 0:
                for chunk in chunk_iter(to_insert, batch_size):
                    self.collection.insert_many(list(chunk))
            else:
                self.collection.insert_many(to_insert)

        total = len(updated) + len(inserted)
        self.logger.message(f"Upserted {total} documents (created={len(inserted)}, updated={len(updated)}).")
        return updated + inserted
