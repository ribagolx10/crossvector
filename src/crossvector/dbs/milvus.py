"""Concrete adapter for Milvus vector database.

This module provides the Milvus implementation of the VectorDBAdapter interface,
enabling vector storage and retrieval using Milvus's high-performance vector search engine.

Key Features:
    - Lazy client initialization with cloud/self-hosted support
    - Full CRUD operations with VectorDocument models
    - Batch operations for bulk create/update/upsert
    - Configurable vector metrics (cosine, euclidean, dot_product)
    - Dynamic schema creation with PRIMARY_KEY_MODE support
    - Automatic index creation and management
"""

import logging
import os
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from pymilvus import DataType, MilvusClient

from crossvector.abc import VectorDBAdapter
from crossvector.constants import VECTOR_METRIC_MAP, VectorMetric
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.utils import (
    apply_update_fields,
    extract_id,
    normalize_pks,
    prepare_item_for_storage,
)

log = logging.getLogger(__name__)


class MilvusDBAdapter(VectorDBAdapter):
    """Vector database adapter for Milvus.

    Provides a high-level interface for vector operations using Milvus's
    distributed vector database capabilities. Supports both cloud and
    self-hosted deployments with automatic schema and index management.

    Attributes:
        collection_name: Name of the active collection
        embedding_dimension: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
    """

    use_dollar_vector: bool = False

    def __init__(self, **kwargs: Any):
        """Initialize the Milvus adapter with lazy client setup.

        Args:
            **kwargs: Additional configuration options (currently unused)
        """
        self._client: MilvusClient | None = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None
        log.info("MilvusDBAdapter initialized.")

    @property
    def client(self) -> MilvusClient:
        """Lazily initialize and return the Milvus client.

        Returns:
            Initialized MilvusClient instance

        Raises:
            ValueError: If MILVUS_API_ENDPOINT is not configured
        """
        if self._client is None:
            uri = os.getenv("MILVUS_API_ENDPOINT")
            if not uri:
                raise ValueError("MILVUS_API_ENDPOINT is not set. Please configure it in your .env file.")
            user = os.getenv("MILVUS_USER")
            password = os.getenv("MILVUS_PASSWORD")
            token = None
            if user and password:
                token = f"{user}:{password}"
            self._client = MilvusClient(uri=uri, token=token)
            log.info(f"MilvusClient initialized with uri={uri}")
        return self._client

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
            f"Milvus initialized: collection='{collection_name}', "
            f"dimension={embedding_dimension}, metric={metric}, store_text={self.store_text}"
        )

    def _get_collection_info(self, collection_name: str) -> Dict[str, Any] | None:
        """Get collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection info dict or None if doesn't exist
        """
        try:
            return self.client.describe_collection(collection_name=collection_name)
        except Exception:
            return None

    def _get_index_info(self, collection_name: str) -> List[Dict[str, Any]] | None:
        """Get index information for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of index info dicts or None if doesn't exist
        """
        try:
            return self.client.describe_index(collection_name=collection_name)
        except Exception:
            return None

    def _build_schema(self, embedding_dimension: int) -> Any:
        """Build Milvus schema with dynamic PK type based on PRIMARY_KEY_MODE.

        Args:
            embedding_dimension: Dimension of vector embeddings

        Returns:
            Milvus schema object
        """
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        mode = (api_settings.PRIMARY_KEY_MODE or "uuid").lower()

        if mode == "int64":
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        else:
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=255, is_primary=True)

        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dimension)

        if self.store_text:
            # Max length for VARCHAR in Milvus is 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        return schema

    def _build_index_params(self, embedding_dimension: int, metric: str = VectorMetric.COSINE) -> Any:
        """Build Milvus index parameters.

        Args:
            embedding_dimension: Dimension of vector embeddings
            metric: Distance metric for vector search

        Returns:
            Milvus index parameters object
        """
        index_params = self.client.prepare_index_params()

        # Primary key index: only needed for VARCHAR (TRIE). INT64 primary uses default.
        mode = (api_settings.PRIMARY_KEY_MODE or "uuid").lower()
        if mode != "int64":
            index_params.add_index(field_name="id", index_type="TRIE")

        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type=metric.upper(), params={"nlist": 1024}
        )
        return index_params

    def add_collection(self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE) -> None:
        """Create a new Milvus collection.

        Args:
            collection_name: Name of the collection to create
            embedding_dimension: Vector embedding dimension
            metric: Distance metric for vector search

        Raises:
            ValueError: If collection already exists
        """
        info = self._get_collection_info(collection_name)
        if info:
            raise ValueError(f"Collection '{collection_name}' already exists.")

        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        if not hasattr(self, "store_text"):
            self.store_text = True

        metric_key = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        schema = self._build_schema(embedding_dimension)
        self.client.create_collection(collection_name=collection_name, schema=schema)
        index_params = self._build_index_params(embedding_dimension, metric_key)
        self.client.create_index(collection_name=collection_name, index_params=index_params)
        log.info(f"Milvus collection '{collection_name}' created with schema and index.")

    def get_collection(self, collection_name: str) -> None:
        """Get an existing Milvus collection.

        Args:
            collection_name: Name of the collection to retrieve

        Raises:
            ValueError: If collection doesn't exist
        """
        info = self._get_collection_info(collection_name)
        if not info:
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        self.collection_name = collection_name
        log.info(f"Milvus collection '{collection_name}' retrieved.")

    def get_or_create_collection(
        self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE
    ) -> None:
        """Get or create the underlying Milvus collection.

        Ensures the collection exists with proper vector configuration.
        If the collection schema doesn't match requirements (PK type, fields),
        it will be dropped and recreated.

        Args:
            collection_name: Name of the collection
            embedding_dimension: Vector embedding dimension
            metric: Distance metric for vector search

        Raises:
            Exception: If collection initialization fails
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        if not hasattr(self, "store_text"):
            self.store_text = True

        metric_key = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        info = self._get_collection_info(collection_name)
        index_info = self._get_index_info(collection_name)

        # Check schema compatibility
        has_vector_index = False
        if index_info:
            for idx in index_info:
                if idx.get("field_name") == "vector":
                    has_vector_index = True
                    break

        need_create = False
        if info:
            fields = info.get("fields", [])
            field_names = [f["name"] for f in fields]

            # Check if required fields exist
            if "id" not in field_names or "vector" not in field_names:
                self.client.drop_collection(collection_name=collection_name)
                log.info(f"Milvus collection '{collection_name}' dropped due to wrong schema.")
                need_create = True
            elif self.store_text and "text" not in field_names:
                # If we want to store text but the collection doesn't have it, recreate
                self.client.drop_collection(collection_name=collection_name)
                log.info(f"Milvus collection '{collection_name}' dropped to add 'text' field.")
                need_create = True
            elif not has_vector_index:
                # Index missing/wrong
                self.client.drop_collection(collection_name=collection_name)
                need_create = True
            else:
                # Validate PK data type vs PRIMARY_KEY_MODE
                mode = (api_settings.PRIMARY_KEY_MODE or "uuid").lower()
                id_field = next((f for f in fields if f["name"] == "id"), None)
                dtype = id_field.get("type") if id_field else None
                want_int64 = mode == "int64"
                is_int64 = dtype == DataType.INT64 if dtype is not None else False
                is_varchar = dtype == DataType.VARCHAR if dtype is not None else False

                if (want_int64 and not is_int64) or ((not want_int64) and not is_varchar):
                    self.client.drop_collection(collection_name=collection_name)
                    log.info("Milvus collection dropped to align PK type with PRIMARY_KEY_MODE.")
                    need_create = True
                else:
                    log.info(f"Milvus collection '{collection_name}' already exists with correct schema.")
        else:
            need_create = True

        if need_create:
            schema = self._build_schema(embedding_dimension)
            self.client.create_collection(collection_name=collection_name, schema=schema)
            index_params = self._build_index_params(embedding_dimension, metric_key)
            self.client.create_index(collection_name=collection_name, index_params=index_params)
            log.info(f"Milvus collection '{collection_name}' created with schema and index.")

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the specified collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful
        """
        self.client.drop_collection(collection_name=collection_name)
        log.info(f"Milvus collection '{collection_name}' dropped.")
        return True

    def clear_collection(self) -> int:
        """Delete all documents from the collection.

        Returns:
            Number of documents deleted

        Raises:
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        count = self.count()
        if count == 0:
            return 0

        # Delete using filter to match all non-empty IDs
        mode = (api_settings.PRIMARY_KEY_MODE or "uuid").lower()
        if mode == "int64":
            self.client.delete(collection_name=self.collection_name, filter="id >= 0")
        else:
            self.client.delete(collection_name=self.collection_name, filter="id != ''")

        log.info(f"Cleared {count} documents from collection.")
        return count

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        info = self.client.describe_collection(collection_name=self.collection_name)
        return info.get("num_entities", 0)

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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        self.client.load_collection(collection_name=self.collection_name)

        output_fields = ["metadata"]
        if self.store_text:
            if fields is None or "text" in fields:
                output_fields.append("text")

        # Build metadata filter expression if where is provided
        filter_expr = None
        if where:
            # Convert dict to Milvus filter expression: metadata["key"] == "value"
            conditions = [f'metadata["{k}"] == "{v}"' for k, v in where.items()]
            filter_expr = " and ".join(conditions)

        # Milvus fetch with offset: get limit+offset
        fetch_limit = limit + offset
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=fetch_limit,
            output_fields=output_fields,
            filter=filter_expr,
        )

        # MilvusClient returns list of lists, apply offset
        hits = results[0][offset:] if results else []

        # Convert to VectorDocument instances
        vector_docs = []
        for hit in hits:
            doc_dict = {"_id": hit.get("id"), "metadata": hit.get("metadata", {})}
            if "text" in hit:
                doc_dict["text"] = hit["text"]
            vector_docs.append(VectorDocument.from_kwargs(**doc_dict))

        log.info(f"Vector search returned {len(vector_docs)} results.")
        return vector_docs

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
            ValueError: If collection_name not set or document ID missing/not found
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        doc_id = pk or extract_id(kwargs)
        if not doc_id:
            raise ValueError("Document ID is required (provide pk or id/_id/pk in kwargs)")

        results = self.client.get(collection_name=self.collection_name, ids=[doc_id])
        if not results:
            raise ValueError(f"Document with ID '{doc_id}' not found")

        return VectorDocument.from_kwargs(**results[0])

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
            ValueError: If collection not set, vector missing, or document ID conflicts
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        doc = VectorDocument.from_kwargs(**kwargs)
        item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        pk = doc.pk
        vector = item.get("vector")
        if vector is None:
            raise ValueError("Vector required for create in Milvus.")

        # Conflict check
        existing = self.client.get(collection_name=self.collection_name, ids=[pk])
        if existing:
            raise ValueError(f"Conflict: document with id '{pk}' already exists.")

        text_val = item.get("text") if self.store_text else None
        if text_val and len(text_val) > 65535:
            text_val = text_val[:65535]
            doc.text = text_val  # keep returned model consistent

        metadata = {k: v for k, v in item.items() if k not in ("_id", "vector", "$vector", "text")}

        data: Dict[str, Any] = {"id": pk, "vector": vector, "metadata": metadata}
        if self.store_text and text_val is not None:
            data["text"] = text_val

        self.client.upsert(collection_name=self.collection_name, data=[data])
        log.info(f"Created document with id '{pk}'.")
        return doc

    def get_or_create(self, defaults: Dict[str, Any] | None = None, **kwargs) -> Tuple[VectorDocument, bool]:
        """Get a document by ID or create it if not found.

        Args:
            defaults: Default values to use when creating new document
            **kwargs: Lookup fields and values (must include id/_id/pk)

        Returns:
            Tuple of (document, created) where created is True if new document was created

        Raises:
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

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
            ValueError: If collection not set, ID missing, or document not found
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        id_val = extract_id(kwargs)
        if not id_val:
            raise ValueError("'id', '_id', or 'pk' is required for update")

        # Get existing document
        existing = self.client.get(collection_name=self.collection_name, ids=[id_val])
        if not existing:
            raise ValueError(f"Document with ID '{id_val}' not found")

        existing_doc = existing[0]
        prepared = prepare_item_for_storage(kwargs, store_text=self.store_text)

        # Build replacement doc using existing + updates
        vector = prepared.get("$vector") or prepared.get("vector") or existing_doc.get("vector")
        text_val = existing_doc.get("text", "")
        if self.store_text and "text" in prepared:
            text_val = prepared["text"]
            if len(text_val) > 65535:
                text_val = text_val[:65535]

        metadata = existing_doc.get("metadata", {})
        for k, v in prepared.items():
            if k not in ("_id", "$vector", "text"):
                metadata[k] = v

        data: Dict[str, Any] = {"id": id_val, "vector": vector, "metadata": metadata}
        if self.store_text:
            data["text"] = text_val

        self.client.upsert(collection_name=self.collection_name, data=[data])
        log.info(f"Updated document with id '{id_val}'.")

        # Return refreshed document
        refreshed = self.client.get(collection_name=self.collection_name, ids=[id_val])
        return VectorDocument.from_kwargs(**refreshed[0])

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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        pks = normalize_pks(ids)
        if not pks:
            return 0

        self.client.delete(collection_name=self.collection_name, ids=pks)
        log.info(f"Deleted {len(pks)} document(s).")
        return len(pks)

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
            ValueError: If collection not set, vector missing, or conflict occurs
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        if not documents:
            return []

        dataset: List[Dict[str, Any]] = []
        created_docs: List[VectorDocument] = []

        for doc in documents:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            pk = doc.pk

            # Conflict detection
            existing = self.client.get(collection_name=self.collection_name, ids=[pk])
            if existing:
                if ignore_conflicts:
                    continue
                if update_conflicts:
                    # Perform update instead
                    update_doc = apply_update_fields(item, update_fields)
                    # Build merged document
                    vector = update_doc.get("$vector") or update_doc.get("vector") or existing[0].get("vector")
                    text_val = update_doc.get("text", existing[0].get("text", ""))
                    if len(text_val) > 65535:
                        text_val = text_val[:65535]
                    metadata = existing[0].get("metadata", {})
                    for k, v in update_doc.items():
                        if k not in ("_id", "$vector", "text"):
                            metadata[k] = v
                    data = {"id": pk, "vector": vector, "metadata": metadata}
                    if self.store_text:
                        data["text"] = text_val
                    self.client.upsert(collection_name=self.collection_name, data=[data])
                    continue
                raise ValueError(f"Conflict on id '{pk}' during bulk_create.")

            vector = item.get("vector")
            if vector is None:
                raise ValueError("Vector required for bulk_create in Milvus.")

            data: Dict[str, Any] = {"id": pk, "vector": vector}
            if self.store_text and "text" in item:
                text_val = item.get("text", "")
                if len(text_val) > 65535:
                    text_val = text_val[:65535]
                data["text"] = text_val

            metadata = {k: v for k, v in item.items() if k not in ("_id", "vector", "$vector", "text")}
            data["metadata"] = metadata

            dataset.append(data)
            created_docs.append(doc)

        if dataset:
            if batch_size and batch_size > 0:
                for i in range(0, len(dataset), batch_size):
                    self.client.upsert(collection_name=self.collection_name, data=dataset[i : i + batch_size])
            else:
                self.client.upsert(collection_name=self.collection_name, data=dataset)

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
            ValueError: If collection not set or document missing (when ignore_conflicts=False)
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        if not documents:
            return []

        dataset: List[Dict[str, Any]] = []
        updated_docs: List[VectorDocument] = []
        missing: List[str] = []

        for doc in documents:
            pk = doc.pk
            if not pk:
                if ignore_conflicts:
                    continue
                missing.append("<no_id>")
                continue

            existing = self.client.get(collection_name=self.collection_name, ids=[pk])
            if not existing:
                if ignore_conflicts:
                    continue
                missing.append(pk)
                continue

            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(item, update_fields)

            # Build replacement doc using existing + updates
            vector = update_doc.get("$vector") or update_doc.get("vector") or existing[0].get("vector")
            text_val = existing[0].get("text", "")
            if self.store_text and "text" in update_doc:
                text_val = update_doc["text"]
                if len(text_val) > 65535:
                    text_val = text_val[:65535]

            metadata = existing[0].get("metadata", {})
            for k, v in update_doc.items():
                if k not in ("_id", "$vector", "text"):
                    metadata[k] = v

            data: Dict[str, Any] = {"id": pk, "vector": vector, "metadata": metadata}
            if self.store_text:
                data["text"] = text_val

            dataset.append(data)
            updated_docs.append(doc)

        if missing:
            raise ValueError(f"Missing documents for update: {missing}")

        if dataset:
            if batch_size and batch_size > 0:
                for i in range(0, len(dataset), batch_size):
                    self.client.upsert(collection_name=self.collection_name, data=dataset[i : i + batch_size])
            else:
                self.client.upsert(collection_name=self.collection_name, data=dataset)

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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        if not documents:
            return []

        data = []
        for doc in documents:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            doc_id = doc.pk
            vector = item.get("vector")

            doc_data: Dict[str, Any] = {"id": doc_id, "vector": vector}
            if self.store_text:
                text = item.get("text", "")
                if len(text) > 65535:
                    text = text[:65535]
                doc_data["text"] = text

            metadata = {k: v for k, v in item.items() if k not in ("_id", "vector", "$vector", "text")}
            doc_data["metadata"] = metadata
            data.append(doc_data)

        if batch_size and batch_size > 0:
            for i in range(0, len(data), batch_size):
                self.client.insert(collection_name=self.collection_name, data=data[i : i + batch_size])
        else:
            self.client.insert(collection_name=self.collection_name, data=data)

        log.info(f"Upserted {len(documents)} document(s).")
        return documents
