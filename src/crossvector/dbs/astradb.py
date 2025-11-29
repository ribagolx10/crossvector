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

import logging
import os
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from astrapy import DataAPIClient
from astrapy.constants import DOC
from astrapy.constants import VectorMetric as AstraVectorMetric
from astrapy.data.collection import Collection
from astrapy.database import Database
from astrapy.info import CollectionDefinition, CollectionVectorOptions

from crossvector.abc import VectorDBAdapter
from crossvector.constants import VECTOR_METRIC_MAP, VectorMetric
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.utils import (
    apply_update_fields,
    chunk_iter,
    extract_id,
    normalize_ids,
    prepare_item_for_storage,
)

log = logging.getLogger(__name__)


class AstraDBAdapter(VectorDBAdapter):
    """Vector database adapter for DataStax Astra DB.

    Provides a high-level interface for vector operations using Astra DB's
    vector search capabilities. Implements lazy connection initialization
    and follows the standard VectorDBAdapter interface.

    Attributes:
        collection_name: Name of the active collection
        embedding_dimension: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
        collection: Active AstraDB collection instance
    """

    use_dollar_vector: bool = True

    def __init__(self, **kwargs: Any):
        """Initialize the AstraDB adapter with lazy client setup.

        Args:
            **kwargs: Additional configuration options (currently unused)
        """
        self._client: DataAPIClient | None = None
        self._db: Database | None = None
        self.collection: Collection | None = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None
        self.store_text: bool = True
        log.info("AstraDBAdapter initialized.")

    @property
    def client(self) -> DataAPIClient:
        """Lazily initialize and return the AstraDB DataAPIClient.

        Returns:
            Initialized DataAPIClient instance

        Raises:
            ValueError: If ASTRA_DB_APPLICATION_TOKEN is not configured
        """
        if self._client is None:
            if not api_settings.ASTRA_DB_APPLICATION_TOKEN:
                raise ValueError("ASTRA_DB_APPLICATION_TOKEN is not set. Please configure it in your .env file.")
            self._client = DataAPIClient(token=api_settings.ASTRA_DB_APPLICATION_TOKEN)
            log.info("AstraDB DataAPIClient initialized.")
        return self._client

    @property
    def db(self) -> Database:
        """Lazily initialize and return the AstraDB database instance.

        Returns:
            Initialized Database instance

        Raises:
            ValueError: If ASTRA_DB_API_ENDPOINT is not configured
        """
        if self._db is None:
            if not api_settings.ASTRA_DB_API_ENDPOINT:
                raise ValueError("ASTRA_DB_API_ENDPOINT is not set. Please configure it in your .env file.")
            self._db = self.client.get_database(api_endpoint=api_settings.ASTRA_DB_API_ENDPOINT)
            log.info("AstraDB database connection established.")
        return self._db

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def initialize(
        self,
        collection_name: str,
        embedding_dimension: int,
        metric: str | None = None,
        store_text: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the database and ensure the collection is ready.

        Args:
            collection_name: Name of the collection to use/create
            embedding_dimension: Dimension of the vector embeddings
            metric: Distance metric ('cosine', 'euclidean', 'dot_product')
            store_text: Whether to store original text content
            **kwargs: Additional configuration options
        """
        self.store_text = store_text if store_text is not None else api_settings.VECTOR_STORE_TEXT
        if metric is None:
            metric = os.getenv("VECTOR_METRIC", VectorMetric.COSINE)
        self.get_or_create_collection(collection_name, embedding_dimension, metric)
        log.info(
            f"AstraDB initialized: collection='{collection_name}', "
            f"dimension={embedding_dimension}, metric={metric}, store_text={self.store_text}"
        )

    def add_collection(
        self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE
    ) -> Collection[DOC]:
        """Create a new AstraDB collection.

        Args:
            collection_name: Name of the collection to create
            embedding_dimension: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            AstraDB Collection instance

        Raises:
            ValueError: If collection already exists
        """
        existing_collections = self.db.list_collection_names()
        if collection_name in existing_collections:
            raise ValueError(f"Collection '{collection_name}' already exists.")

        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        if not hasattr(self, "store_text"):
            self.store_text = True

        vector_metric = VECTOR_METRIC_MAP.get(metric.lower(), AstraVectorMetric.COSINE)
        self.collection = self.db.create_collection(
            collection_name,
            definition=CollectionDefinition(
                vector=CollectionVectorOptions(
                    dimension=embedding_dimension,
                    metric=vector_metric,
                ),
            ),
        )
        log.info(f"AstraDB collection '{collection_name}' created successfully.")
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
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        self.collection = self.db.get_collection(collection_name)
        self.collection_name = collection_name
        log.info(f"AstraDB collection '{collection_name}' retrieved.")
        return self.collection

    def get_or_create_collection(
        self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE
    ) -> Collection[DOC]:
        """Get or create the underlying AstraDB collection.

        Ensures the collection exists with proper vector configuration.
        If the collection doesn't exist, it will be created with the specified
        embedding dimension and distance metric.

        Args:
            collection_name: Name of the collection
            embedding_dimension: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            AstraDB Collection instance

        Raises:
            Exception: If collection initialization fails
        """
        try:
            self.collection_name = collection_name
            self.embedding_dimension = embedding_dimension
            if not hasattr(self, "store_text"):
                self.store_text = True

            existing_collections = self.db.list_collection_names()

            if collection_name in existing_collections:
                self.collection = self.db.get_collection(collection_name)
                log.info(f"AstraDB collection '{collection_name}' retrieved.")
            else:
                vector_metric = VECTOR_METRIC_MAP.get(metric.lower(), AstraVectorMetric.COSINE)
                log.info(f"Creating AstraDB collection '{collection_name}'...")
                self.collection = self.db.create_collection(
                    collection_name,
                    definition=CollectionDefinition(
                        vector=CollectionVectorOptions(
                            dimension=embedding_dimension,
                            metric=vector_metric,
                        ),
                    ),
                )
                log.info(f"AstraDB collection '{collection_name}' created successfully.")

            return self.collection
        except Exception as e:
            log.error(f"Failed to initialize AstraDB collection: {e}", exc_info=True)
            raise

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the specified collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful
        """
        self.db.drop_collection(collection_name)
        log.info(f"AstraDB collection '{collection_name}' dropped.")
        return True

    def clear_collection(self) -> int:
        """Delete all documents from the collection.

        Returns:
            Number of documents deleted

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        result = self.collection.delete_many({})
        log.info(f"Cleared {result.deleted_count} documents from collection.")
        return result.deleted_count

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        count = self.collection.count_documents({}, upper_bound=10000)
        return count

    # ------------------------------------------------------------------
    # Search Operations
    # ------------------------------------------------------------------

    def search(
        self,
        vector: List[float],
        limit: int,
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
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        try:
            # Construct projection to exclude vector by default
            projection = {"$vector": 0}
            if fields:
                projection = {field: 1 for field in fields}

            # Build filter query
            filter_query = where if where else {}

            # AstraDB doesn't have native skip, so we fetch limit+offset and slice
            fetch_limit = limit + offset
            results = list(
                self.collection.find(
                    filter=filter_query,
                    sort={"$vector": vector},
                    limit=fetch_limit,
                    projection=projection,
                )
            )

            # Apply offset by slicing
            results = results[offset:]

            # Convert to VectorDocument instances
            documents = [VectorDocument.from_kwargs(**doc) for doc in results]
            log.info(f"Vector search returned {len(documents)} results.")
            return documents
        except Exception as e:
            log.error(f"Vector search failed: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def get(self, pk: Any = None, **kwargs) -> VectorDocument:
        """Retrieve a single document by its ID.

        Args:
            pk: Primary key value (positional)
            **kwargs: Alternative way to specify id via _id/id/pk keys

        Returns:
            VectorDocument instance

        Raises:
            ConnectionError: If collection is not initialized
            ValueError: If document ID is missing or document not found
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        doc_id = pk or extract_id(kwargs)
        if not doc_id:
            raise ValueError("Document ID is required (provide pk or id/_id/pk in kwargs)")

        doc = self.collection.find_one({"_id": doc_id})
        if not doc:
            raise ValueError(f"Document with ID '{doc_id}' not found")

        return VectorDocument.from_kwargs(**doc)

    def create(self, **kwargs: Any) -> VectorDocument:
        """Create and persist a single document.

        Expected kwargs:
            vector/$vector: List[float] - Vector embedding (required)
            text: str - Original text content (optional)
            metadata: dict - Additional metadata (optional)
            id/_id/pk: str - Explicit document ID (optional, auto-generated if missing)

        Args:
            **kwargs: Document fields as keyword arguments

        Returns:
            Created VectorDocument instance

        Raises:
            ConnectionError: If collection is not initialized
            ValueError: If document with same ID already exists
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        doc = VectorDocument.from_kwargs(**kwargs)
        stored = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        # Conflict check
        if self.collection.find_one({"_id": doc.pk}):
            raise ValueError(f"Conflict: document with id '{doc.pk}' already exists.")

        self.collection.insert_one(stored)
        log.info(f"Created document with id '{doc.pk}'.")
        return doc

    def get_or_create(self, defaults: Dict[str, Any] | None = None, **kwargs) -> Tuple[VectorDocument, bool]:
        """Get a document by ID or create it if not found.

        Args:
            defaults: Default values to use when creating new document
            **kwargs: Lookup fields and values (must include id/_id/pk)

        Returns:
            Tuple of (document, created) where created is True if new document was created

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        lookup_id = extract_id(kwargs)
        if lookup_id:
            try:
                found = self.get(lookup_id)
                return found, False
            except ValueError:
                pass

        # Create new document with merged defaults
        merged = {**(defaults or {}), **kwargs}
        new_doc = self.create(**merged)
        return new_doc, True

    def update(self, **kwargs) -> VectorDocument:
        """Update a single document by ID.

        Strict update semantics: raises error if document doesn't exist.

        Args:
            **kwargs: Must include id/_id/pk, plus fields to update

        Returns:
            Updated VectorDocument instance

        Raises:
            ConnectionError: If collection is not initialized
            ValueError: If ID is missing or document not found
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        id_val = extract_id(kwargs)
        if not id_val:
            raise ValueError("'id', '_id', or 'pk' is required for update")

        existing = self.collection.find_one({"_id": id_val})
        if not existing:
            raise ValueError(f"Document with ID '{id_val}' not found")

        prepared = prepare_item_for_storage(kwargs, store_text=self.store_text)
        update_doc = {k: v for k, v in prepared.items() if k not in ("_id", "$vector")}
        if "$vector" in prepared:
            update_doc["$vector"] = prepared["$vector"]

        if update_doc:
            self.collection.update_one({"_id": id_val}, {"$set": update_doc})
            log.info(f"Updated document with id '{id_val}'.")

        refreshed = self.collection.find_one({"_id": id_val})
        return VectorDocument.from_kwargs(**refreshed)

    def update_or_create(
        self, defaults: Dict[str, Any] | None = None, create_defaults: Dict[str, Any] | None = None, **kwargs
    ) -> Tuple[VectorDocument, bool]:
        """Update document if exists, otherwise create with merged defaults.

        Args:
            defaults: Default values for both update and create
            create_defaults: Default values used only when creating (overrides defaults)
            **kwargs: Fields to update or use for creation

        Returns:
            Tuple of (document, created) where created is True if new document was created

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        lookup_id = extract_id(kwargs)
        if lookup_id:
            try:
                updated = self.update(**kwargs)
                return updated, False
            except ValueError:
                pass

        # Create new document
        merged = {**(create_defaults or defaults or {}), **kwargs}
        new_doc = self.create(**merged)
        return new_doc, True

    def delete(self, ids: Union[str, Sequence[str]]) -> int:
        """Delete document(s) by ID.

        Args:
            ids: Single document ID or list of IDs to delete

        Returns:
            Number of documents deleted

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        pks = normalize_ids(ids)
        if not pks:
            return 0

        if len(pks) == 1:
            result = self.collection.delete_one({"_id": pks[0]})
            deleted = result.deleted_count
        else:
            result = self.collection.delete_many({"_id": {"$in": pks}})
            deleted = result.deleted_count

        log.info(f"Deleted {deleted} document(s).")
        return deleted

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    def bulk_create(
        self,
        documents: List[VectorDocument],
        batch_size: int = None,
        ignore_conflicts: bool = False,
        update_conflicts: bool = False,
        update_fields: List[str] = None,
    ) -> List[VectorDocument]:
        """Bulk create multiple documents.

        Args:
            documents: List of VectorDocument instances to create
            batch_size: Number of documents per batch (optional)
            ignore_conflicts: If True, skip conflicting documents
            update_conflicts: If True, update conflicting documents
            update_fields: Fields to update on conflict (if update_conflicts=True)

        Returns:
            List of successfully created VectorDocument instances

        Raises:
            ConnectionError: If collection is not initialized
            ValueError: If conflict occurs and ignore_conflicts=False
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        if not documents:
            return []

        items_to_insert: List[Dict[str, Any]] = []
        created_docs: List[VectorDocument] = []

        for doc in documents:
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
                raise ValueError(f"Conflict on unique fields for document _id={item.get('_id')}")

            items_to_insert.append(item)
            created_docs.append(doc)

        if items_to_insert:
            if batch_size and batch_size > 0:
                for chunk in chunk_iter(items_to_insert, batch_size):
                    self.collection.insert_many(list(chunk))
            else:
                self.collection.insert_many(items_to_insert)

        log.info(f"Bulk created {len(created_docs)} document(s).")
        return created_docs

    def bulk_update(
        self,
        documents: List[VectorDocument],
        batch_size: int = None,
        ignore_conflicts: bool = False,
        update_fields: List[str] = None,
    ) -> List[VectorDocument]:
        """Bulk update existing documents by ID.

        Args:
            documents: List of VectorDocument instances to update
            batch_size: Number of updates per batch (optional)
            ignore_conflicts: If True, skip missing documents
            update_fields: Specific fields to update (None = all fields)

        Returns:
            List of successfully updated VectorDocument instances

        Raises:
            ConnectionError: If collection is not initialized
            ValueError: If any document is missing and ignore_conflicts=False
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        if not documents:
            return []

        updated_docs: List[VectorDocument] = []
        missing: List[str] = []
        batch_ops: List[Dict[str, Any]] = []

        for doc in documents:
            pk = doc.pk
            if not pk:
                if ignore_conflicts:
                    continue
                missing.append("<no_id>")
                continue

            existing = self.collection.find_one({"_id": pk})
            if not existing:
                if ignore_conflicts:
                    continue
                missing.append(pk)
                continue

            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(item, update_fields)
            if not update_doc:
                continue

            batch_ops.append({"filter": {"_id": pk}, "update": {"$set": update_doc}})
            updated_docs.append(doc)

            # Flush batch if size reached
            if batch_size and batch_size > 0 and len(batch_ops) >= batch_size:
                for op in batch_ops:
                    self.collection.update_one(op["filter"], op["update"])
                batch_ops.clear()

        # Flush remaining operations
        for op in batch_ops:
            self.collection.update_one(op["filter"], op["update"])

        if missing:
            raise ValueError(f"Missing documents for update: {missing}")

        log.info(f"Bulk updated {len(updated_docs)} document(s).")
        return updated_docs

    def upsert(self, documents: List[VectorDocument], batch_size: int = None) -> List[VectorDocument]:
        """Insert or update multiple documents.

        Args:
            documents: List of VectorDocument instances to upsert
            batch_size: Number of documents per batch (optional)

        Returns:
            List of upserted VectorDocument instances

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        items = [
            doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            for doc in documents
        ]

        if batch_size and batch_size > 0:
            for chunk in chunk_iter(items, batch_size):
                self.collection.insert_many(list(chunk))
        else:
            self.collection.insert_many(items)

        log.info(f"Upserted {len(documents)} document(s).")
        return documents
