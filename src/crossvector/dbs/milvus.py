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

from typing import Any, Dict, List, Set

from pymilvus import Collection, DataType, MilvusClient

from crossvector.abc import VectorDBAdapter
from crossvector.constants import VECTOR_METRIC_MAP, VectorMetric
from crossvector.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    CollectionNotInitializedError,
    DocumentExistsError,
    DocumentNotFoundError,
    DoesNotExist,
    InvalidFieldError,
    MissingConfigError,
    MissingDocumentError,
    MissingFieldError,
    MultipleObjectsReturned,
)
from crossvector.querydsl.compilers.milvus import MilvusWhereCompiler, milvus_where
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.utils import (
    apply_update_fields,
    extract_pk,
    prepare_item_for_storage,
)


class MilvusAdapter(VectorDBAdapter):
    """Vector database adapter for Milvus.

    Provides a high-level interface for vector operations using Milvus's
    distributed vector database capabilities. Supports both cloud and
    self-hosted deployments with automatic schema and index management.

    Attributes:
        collection_name: Name of the active collection
        dim: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
    """

    use_dollar_vector: bool = False
    where_compiler: MilvusWhereCompiler = milvus_where
    # Capability flags: Milvus supports metadata-only search via query()
    supports_metadata_only: bool = True

    @property
    def client(self) -> MilvusClient:
        """Lazily initialize and return the Milvus client.

        Returns:
            Initialized MilvusClient instance

        Raises:
            ValueError: If MILVUS_API_ENDPOINT is not configured
        """
        if self._client is None:
            uri = api_settings.MILVUS_API_ENDPOINT
            if not uri:
                raise MissingConfigError(
                    "MILVUS_API_ENDPOINT is not set. Please configure it in your .env file.",
                    config_key="MILVUS_API_ENDPOINT",
                    env_file=".env",
                )
            if api_settings.MILVUS_API_KEY:
                self._client = MilvusClient(uri=uri, token=api_settings.MILVUS_API_KEY)
            else:
                self._client = MilvusClient(uri=uri)
            self.logger.message(f"MilvusClient initialized with uri={uri}")
        return self._client

    @property
    def collection(self) -> Collection | None:
        """Return the active Milvus collection instance.

        Returns:
            Collection instance or None if not initialized
        """
        return self._collection

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
            f"Milvus initialized: collection='{collection_name}', "
            f"dimension={dim}, metric={metric}, store_text={self.store_text}"
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

    def _build_schema(self, dim: int) -> Any:
        """Build Milvus schema with dynamic PK type based on PRIMARY_KEY_MODE.

        Args:
            dim: Dimension of vector embeddings

        Returns:
            Milvus schema object
        """
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        mode = (api_settings.PRIMARY_KEY_MODE or "uuid").lower()

        if mode == "int64":
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        else:
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=255, is_primary=True)

        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

        if self.store_text:
            # Max length for VARCHAR in Milvus is 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        return schema

    def _build_index_params(self, dim: int, metric: str = VectorMetric.COSINE) -> Any:
        """Build Milvus index parameters.

        Args:
            dim: Dimension of vector embeddings
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

    def add_collection(self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE) -> None:
        """Create a new Milvus collection.

        Args:
            collection_name: Name of the collection to create
            dim: Vector embedding dimension
            metric: Distance metric for vector search

        Raises:
            CollectionExistsError: If collection already exists
        """
        info = self._get_collection_info(collection_name)
        if info:
            raise CollectionExistsError("Collection already exists", collection_name=collection_name)

        self.collection_name = collection_name
        self.dim = dim
        if not hasattr(self, "store_text"):
            self.store_text = True

        metric_key = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        schema = self._build_schema(dim)
        self.client.create_collection(collection_name=collection_name, schema=schema)
        index_params = self._build_index_params(dim, metric_key)
        self.client.create_index(collection_name=collection_name, index_params=index_params)
        self.logger.message(f"Milvus collection '{collection_name}' created with schema and index.")

    def get_collection(self, collection_name: str) -> None:
        """Get an existing Milvus collection.

        Args:
            collection_name: Name of the collection to retrieve

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        info = self._get_collection_info(collection_name)
        if not info:
            raise CollectionNotFoundError("Collection does not exist", collection_name=collection_name)

        self.collection_name = collection_name
        self.logger.message(f"Milvus collection '{collection_name}' retrieved.")

    def get_or_create_collection(self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE) -> None:
        """Get or create the underlying Milvus collection.

        Ensures the collection exists with proper vector configuration.
        If the collection schema doesn't match requirements (PK type, fields),
        it will be dropped and recreated.

        Args:
            collection_name: Name of the collection
            dim: Vector embedding dimension
            metric: Distance metric for vector search

        Raises:
            Exception: If collection initialization fails
        """
        self.collection_name = collection_name
        self.dim = dim
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
                self.logger.message(f"Milvus collection '{collection_name}' dropped due to wrong schema.")
                need_create = True
            elif self.store_text and "text" not in field_names:
                # If we want to store text but the collection doesn't have it, recreate
                self.client.drop_collection(collection_name=collection_name)
                self.logger.message(f"Milvus collection '{collection_name}' dropped to add 'text' field.")
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
                    self.logger.message("Milvus collection dropped to align PK type with PRIMARY_KEY_MODE.")
                    need_create = True
                else:
                    self.logger.message(f"Milvus collection '{collection_name}' already exists with correct schema.")
        else:
            need_create = True

        if need_create:
            schema = self._build_schema(dim)
            self.client.create_collection(collection_name=collection_name, schema=schema)
            index_params = self._build_index_params(dim, metric_key)
            self.client.create_index(collection_name=collection_name, index_params=index_params)
            self.logger.message(f"Milvus collection '{collection_name}' created with schema and index.")

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the specified collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful
        """
        self.client.drop_collection(collection_name=collection_name)
        self.logger.message(f"Milvus collection '{collection_name}' dropped.")
        return True

    def clear_collection(self) -> int:
        """Delete all documents from the collection.

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection_name:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="clear_collection", adapter="Milvus"
            )

        count = self.count()
        if count == 0:
            return 0

        # Delete using filter to match all non-empty IDs
        mode = (api_settings.PRIMARY_KEY_MODE or "uuid").lower()
        if mode == "int64":
            self.client.delete(collection_name=self.collection_name, filter="id >= 0")
        else:
            self.client.delete(collection_name=self.collection_name, filter="id != ''")

        self.logger.message(f"Cleared {count} documents from collection.")
        return count

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="count", adapter="Milvus")
        info = self.client.describe_collection(collection_name=self.collection_name)
        return info.get("num_entities", 0)

    # ------------------------------------------------------------------
    # Search Operations
    # ------------------------------------------------------------------

    def search(
        self,
        vector: List[float] | None = None,
        limit: int | None = None,
        offset: int = 0,
        where: Dict[str, Any] | None = None,
        fields: Set[str] | None = None,
    ) -> List[VectorDocument]:
        """Perform vector similarity search or metadata-only query.

        Args:
            vector: Query vector embedding (optional for metadata-only search)
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            where: Optional metadata filter conditions
            fields: Optional set of field names to include in results

        Returns:
            List of VectorDocument instances (ordered by similarity if vector provided)

        Raises:
            CollectionNotInitializedError: If collection is not initialized

        Note:
            - With vector: Uses similarity search with optional metadata filtering
            - Without vector: Uses metadata-only query (requires where filter)
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="search", adapter="Milvus")

        if limit is None:
            limit = api_settings.VECTOR_SEARCH_LIMIT

        self.client.load_collection(collection_name=self.collection_name)

        output_fields = ["metadata"]
        if self.store_text:
            if fields is None or "text" in fields:
                output_fields.append("text")

        # Build metadata filter expression if where is provided
        if where is not None:
            # Compiler handles both Q objects and dicts
            where = self.where_compiler.to_where(where)

        # Milvus fetch with offset: get limit+offset
        fetch_limit = limit + offset

        if vector is None:
            # Metadata-only query returns a flat list
            results = self.client.query(
                collection_name=self.collection_name,
                limit=fetch_limit,
                output_fields=output_fields,
                filter=where,
            )
            # Query returns flat list, apply offset directly
            hits = results[offset:] if results else []
        else:
            # Vector search path returns list of lists
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vector],
                limit=fetch_limit,
                output_fields=output_fields,
                filter=where,
                anns_field="vector",
            )
            # Search returns list of lists, extract first result set and apply offset
            hits = results[0][offset:] if results else []

        # Convert to VectorDocument instances
        vector_docs = []
        for hit in hits:
            # Milvus search returns distance; embed a synthetic score if absent
            score_val = 1.0
            distance = hit.get("distance") or hit.get("score")
            if isinstance(distance, (int, float)):
                # Use inverse distance heuristic if cosine metric (distance range assumption) else raw
                score_val = 1.0 - distance if 0.0 <= distance <= 1.0 else float(distance)
            metadata_block = hit.get("metadata", {}) or {}
            if isinstance(metadata_block, dict) and "score" not in metadata_block:
                metadata_block["score"] = score_val
            doc_dict = {"_id": hit.get("id"), "vector": [], "metadata": metadata_block}
            if "text" in hit:
                doc_dict["text"] = hit["text"]
            vector_docs.append(VectorDocument.from_kwargs(**doc_dict))

        self.logger.message(f"Vector search returned {len(vector_docs)} results.")
        return vector_docs

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
            MissingFieldError: If neither pk nor metadata filters provided
            DoesNotExist: If no document matches
            MultipleObjectsReturned: If multiple documents match
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="get", adapter="Milvus")

        pk = args[0] if args else None
        doc_id = pk or extract_pk(None, **kwargs) if not pk else pk

        # Priority 1: Direct pk lookup
        if doc_id:
            results = self.client.get(collection_name=self.collection_name, ids=[doc_id, doc_id])
            if not results:
                raise DoesNotExist(f"Document with ID '{doc_id}' not found")
            if len(results) > 1:
                raise MultipleObjectsReturned(f"Multiple documents found with ID '{doc_id}'")
            return VectorDocument.from_kwargs(**results[0])

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
            InvalidFieldError: If vector missing
            DocumentExistsError: If document ID conflicts
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="create", adapter="Milvus")

        item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        pk = doc.pk
        vector = item.get("vector")
        if vector is None:
            raise InvalidFieldError("Vector is required", field="vector", operation="create")

        # Load collection before conflict check
        self.client.load_collection(collection_name=self.collection_name)

        # Conflict check
        existing = self.client.get(collection_name=self.collection_name, ids=[pk])
        if existing:
            raise DocumentExistsError("Document already exists", document_id=pk)

        text_val = item.get("text") if self.store_text else None
        if text_val and len(text_val) > 65535:
            text_val = text_val[:65535]
            doc.text = text_val  # keep returned model consistent

        metadata = {k: v for k, v in item.items() if k not in ("_id", "vector", "$vector", "text")}

        data: Dict[str, Any] = {"id": pk, "vector": vector, "metadata": metadata}
        if self.store_text and text_val is not None:
            data["text"] = text_val

        self.client.upsert(collection_name=self.collection_name, data=[data])
        self.logger.message(f"Created document with id '{pk}'.\")")
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
            MissingFieldError: If ID missing
            DocumentNotFoundError: If document not found
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="update", adapter="Milvus")

        pk = doc.id or extract_pk(None, **kwargs)
        if not pk:
            raise MissingFieldError("'id', '_id', or 'pk' is required for update", field="id", operation="update")

        # Get existing document
        existing = self.client.get(collection_name=self.collection_name, ids=[pk])
        if not existing:
            raise DocumentNotFoundError("Document not found", document_id=pk, operation="update")

        existing_doc = existing[0]
        prepared = prepare_item_for_storage(
            doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector),
            store_text=self.store_text,
        )

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

        data: Dict[str, Any] = {"id": pk, "vector": vector, "metadata": metadata}
        if self.store_text:
            data["text"] = text_val

        self.client.upsert(collection_name=self.collection_name, data=[data])
        self.logger.message(f"Updated document with id '{pk}'.")

        # Return refreshed document
        refreshed = self.client.get(collection_name=self.collection_name, ids=[pk])
        return VectorDocument.from_kwargs(**refreshed[0])

    def delete(self, *args) -> int:
        """Delete documents by ID.

        Args:
            *args: One or more document IDs to delete

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="delete", adapter="Milvus")

        if not args:
            return 0

        self.client.delete(collection_name=self.collection_name, ids=list(args))
        self.logger.message(f"Deleted {len(args)} documents.")
        return len(args)

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
            InvalidFieldError: If vector missing
            DocumentExistsError: If conflict occurs and ignore_conflicts/update_conflicts False
        """
        if not self.collection_name:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="bulk_create", adapter="Milvus"
            )
        if not docs:
            return []

        # Load collection before any operations
        self.client.load_collection(collection_name=self.collection_name)

        dataset: List[Dict[str, Any]] = []
        created_docs: List[VectorDocument] = []

        for doc in docs:
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
                raise DocumentExistsError("Document already exists", document_id=pk, operation="bulk_create")

            vector = item.get("vector")
            if vector is None:
                raise InvalidFieldError("Vector is required", field="vector", operation="bulk_create")

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
            MissingDocumentError: If any document missing and ignore_conflicts=False
        """
        if not self.collection_name:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="bulk_update", adapter="Milvus"
            )
        if not docs:
            return []

        # Collect all PKs and validate
        doc_map: Dict[str, VectorDocument] = {}
        for doc in docs:
            pk = doc.pk
            if not pk:
                if not ignore_conflicts:
                    raise MissingDocumentError("Document missing ID", missing_ids=["<no_id>"], operation="bulk_update")
                continue
            doc_map[pk] = doc

        if not doc_map:
            return []

        # Fetch all existing documents in ONE query
        pks = list(doc_map.keys())
        existing_docs = self.client.get(collection_name=self.collection_name, ids=pks)
        existing_map = {doc["id"]: doc for doc in existing_docs} if existing_docs else {}

        # Check for missing documents
        missing = [pk for pk in pks if pk not in existing_map]
        if missing:
            if not ignore_conflicts:
                raise MissingDocumentError("Missing documents for update", missing_ids=missing, operation="bulk_update")
            # Remove missing from processing
            for pk in missing:
                doc_map.pop(pk, None)

        # Per-document upsert with optional batching (safer, avoids unintended insert of missing docs)
        updated_docs: List[VectorDocument] = []
        batch_buffer: List[Dict[str, Any]] = []

        for pk, doc in doc_map.items():
            existing = existing_map[pk]
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(item, update_fields)
            if not update_doc:
                continue

            # Merge existing + updated fields
            vector = update_doc.get("$vector") or update_doc.get("vector") or existing.get("vector")
            text_val = existing.get("text", "")
            if self.store_text and "text" in update_doc:
                text_val = update_doc["text"]
                if len(text_val) > 65535:
                    text_val = text_val[:65535]

            # Merge metadata
            metadata = dict(existing.get("metadata", {}))
            for k, v in update_doc.items():
                if k not in ("_id", "$vector", "text"):
                    metadata[k] = v

            data: Dict[str, Any] = {"id": pk, "vector": vector, "metadata": metadata}
            if self.store_text:
                data["text"] = text_val

            if batch_size and batch_size > 0:
                batch_buffer.append(data)
                if len(batch_buffer) >= batch_size:
                    self.client.upsert(collection_name=self.collection_name, data=batch_buffer)
                    batch_buffer.clear()
            else:
                self.client.upsert(collection_name=self.collection_name, data=[data])

            updated_docs.append(doc)

        # Flush remaining batch
        if batch_buffer:
            self.client.upsert(collection_name=self.collection_name, data=batch_buffer)
        # Load collection to make data queryable
        self.client.load_collection(collection_name=self.collection_name)

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
            InvalidFieldError: If any document is missing a vector
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="upsert", adapter="Milvus")
        if not docs:
            return []

        data = []
        for doc in docs:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            doc_id = doc.pk
            vector = item.get("vector")

            if vector is None:
                raise InvalidFieldError("Vector is required", field="vector", operation="upsert")

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
                self.client.upsert(collection_name=self.collection_name, data=data[i : i + batch_size])
        else:
            self.client.upsert(collection_name=self.collection_name, data=data)
        # Load collection to make data queryable
        self.client.load_collection(collection_name=self.collection_name)

        self.logger.message(f"Upserted {len(docs)} documents.")
        return docs
