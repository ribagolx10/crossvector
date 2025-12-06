"""Concrete adapter for ChromaDB vector database.

This module provides the ChromaDB implementation of the VectorDBAdapter interface,
enabling vector storage and retrieval using ChromaDB's native vector search capabilities.

Key Features:
    - Flexible client initialization (Cloud, HTTP, Local)
    - Lazy client/collection initialization
    - Full CRUD operations with VectorDocument models
    - Batch operations for bulk create/update/upsert
    - Configurable vector metrics (cosine, euclidean, dot_product)
    - Automatic collection management and schema creation
"""

from typing import Any, Dict, List, Set, Union

from chromadb import Client, CloudClient, Collection, HttpClient
from chromadb.config import Settings

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
    SearchError,
)
from crossvector.querydsl.compilers.chroma import ChromaWhereCompiler, chroma_where
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.types import DocIds
from crossvector.utils import (
    apply_update_fields,
    extract_pk,
    flatten_metadata,
    prepare_item_for_storage,
    unflatten_metadata,
)


class ChromaAdapter(VectorDBAdapter):
    """Vector database adapter for ChromaDB.

    Provides a high-level interface for vector operations using ChromaDB's
    vector search capabilities. Supports multiple deployment modes (cloud,
    self-hosted, local) with automatic client initialization.

    Attributes:
        collection_name: Name of the active collection
        dim: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
        metric: Distance metric for vector search
    """

    use_dollar_vector: bool = False
    where_compiler: ChromaWhereCompiler = chroma_where
    # Capability flags
    supports_metadata_only: bool = True  # Chroma supports metadata-only filtering without vector

    @property
    def client(self) -> Union[Client, CloudClient, HttpClient]:
        """Lazily initialize and return the ChromaDB client.

        Selects client based on configuration priority:
        1. CloudClient (if CHROMA_API_KEY present)
        2. HttpClient (if CHROMA_HOST present, requires CHROMA_HOST and no CHROMA_PERSIST_DIR)
        3. Local persistence client (requires CHROMA_PERSIST_DIR or neither)

        Returns:
            Initialized ChromaDB client instance

        Raises:
            MissingConfigError: If configuration is missing or conflicting
            ConnectionError: If client initialization fails
        """
        if self._client is None:
            # 1) Try CloudClient if cloud API key present
            if api_settings.CHROMA_API_KEY:
                try:
                    self._client = CloudClient(
                        tenant=api_settings.CHROMA_TENANT,
                        database=api_settings.CHROMA_DATABASE,
                        api_key=api_settings.CHROMA_API_KEY,
                    )
                    self.logger.message("ChromaDB CloudClient initialized.")
                    return self._client
                except Exception as exc:
                    raise ConnectionError(
                        "Failed to initialize ChromaDB CloudClient",
                        adapter="ChromaDB",
                        original_error=str(exc),
                    ) from exc

            # 2) Try HttpClient (self-hosted server) if host/port provided
            if api_settings.CHROMA_HOST:
                # Validate: cannot specify both CHROMA_HOST and CHROMA_PERSIST_DIR
                if api_settings.CHROMA_PERSIST_DIR:
                    raise MissingConfigError(
                        "Cannot specify both CHROMA_HOST and CHROMA_PERSIST_DIR. "
                        "Choose one: either CHROMA_HOST (for remote server) or CHROMA_PERSIST_DIR (for local storage).",
                        config_key="CHROMA_HOST/CHROMA_PERSIST_DIR",
                        adapter="ChromaDB",
                        hint="Set either CHROMA_HOST or CHROMA_PERSIST_DIR, not both.",
                    )

                try:
                    if api_settings.CHROMA_PORT:
                        self._client = HttpClient(host=api_settings.CHROMA_HOST, port=int(api_settings.CHROMA_PORT))
                    else:
                        self._client = HttpClient(host=api_settings.CHROMA_HOST)

                    self.logger.message(
                        f"ChromaDB HttpClient initialized (host={api_settings.CHROMA_HOST}, port={api_settings.CHROMA_PORT})."
                    )
                    return self._client
                except Exception as e:
                    raise ConnectionError(
                        "Failed to initialize ChromaDB HttpClient",
                        adapter="ChromaDB",
                        original_error=str(e),
                    ) from e

            # 3) Local persistence client
            persist_dir = api_settings.CHROMA_PERSIST_DIR
            settings_obj = Settings(persist_directory=persist_dir) if persist_dir else Settings()
            try:
                self._client = Client(settings_obj)
                self.logger.message(f"ChromaDB local client initialized. Persist dir: {persist_dir}")
            except Exception as e:
                raise ConnectionError(
                    "Failed to initialize local ChromaDB client",
                    adapter="ChromaDB",
                    original_error=str(e),
                ) from e
        return self._client

    @property
    def collection(self) -> Collection:
        """Lazily return the cached ChromaDB collection instance.

        Returns:
            Active ChromaDB collection instance (may be None if not yet initialized)
        """
        return self._collection

    @collection.setter
    def collection(self, value: Collection | None) -> None:
        """Set the collection instance."""
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
            f"ChromaDB initialized: collection='{collection_name}', "
            f"dimension={dim}, metric={metric}, store_text={self.store_text}"
        )

    def add_collection(self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE) -> Collection:
        """Create a new ChromaDB collection.

        Args:
            collection_name: Name of the collection to create
            dim: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            ChromaDB Collection instance

        Raises:
            CollectionExistsError: If collection already exists
            MissingConfigError: If required configuration is missing
            SearchError: If collection creation fails
        """
        try:
            self.client.get_collection(collection_name)
            raise CollectionExistsError("Collection already exists", collection_name=collection_name)
        except Exception as e:
            if "already exists" in str(e).lower():
                raise CollectionExistsError("Collection already exists", collection_name=collection_name) from e

        self.collection_name = collection_name
        self.dim = dim
        if not hasattr(self, "store_text"):
            self.store_text = True

        self.metric = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.metric},
            embedding_function=None,
        )
        self.logger.message(f"ChromaDB collection '{collection_name}' created.")
        return self.collection

    def get_collection(self, collection_name: str) -> Collection:
        """Get an existing ChromaDB collection.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            ChromaDB Collection instance

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            MissingConfigError: If required configuration is missing
            SearchError: If collection retrieval fails
        """
        try:
            self.collection = self.client.get_collection(collection_name)
            self.collection_name = collection_name
            self.logger.message(f"ChromaDB collection '{collection_name}' retrieved.")
            return self.collection
        except Exception as e:
            raise CollectionNotFoundError("Collection does not exist", collection_name=collection_name) from e

    def get_or_create_collection(self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE) -> Collection:
        """Get or create the underlying ChromaDB collection.

        Ensures the collection exists with proper vector configuration.
        If the collection doesn't exist, it will be created with the specified
        distance metric.

        Args:
            collection_name: Name of the collection
            dim: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            ChromaDB Collection instance

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            CollectionExistsError: If collection already exists
            CollectionNotInitializedError: If collection is not initialized
            MissingConfigError: If required configuration is missing
            SearchError: If collection creation or retrieval fails
        """
        self.collection_name = collection_name
        self.dim = dim
        if not hasattr(self, "store_text"):
            self.store_text = True

        self.metric = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        if self.collection is not None and getattr(self.collection, "name", None) == collection_name:
            return self.collection

        try:
            self.collection = self.client.get_collection(collection_name)
            self.logger.message(f"ChromaDB collection '{collection_name}' retrieved.")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.metric},
                embedding_function=None,
            )
            self.logger.message(f"ChromaDB collection '{collection_name}' created.")
        return self.collection

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the specified collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful
        """
        self.client.delete_collection(collection_name)
        self.logger.message(f"ChromaDB collection '{collection_name}' dropped.")
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
                "Collection is not initialized", operation="clear_collection", adapter="ChromaDB"
            )
        count = self.collection.count()
        if count == 0:
            return 0
        results = self.collection.get(limit=count, include=[])
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        self.logger.message(f"Cleared {len(ids)} documents from collection.")
        return len(ids)

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="count", adapter="ChromaDB")
        return self.collection.count()

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
            SearchError: If neither vector nor where filter provided
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="search", adapter="ChromaDB")

        if limit is None:
            limit = api_settings.VECTOR_SEARCH_LIMIT

        # Always compile where via where_compiler; avoid method-specific tweaks
        if where is not None:
            where = self.where_compiler.to_where(where)

        # Metadata-only search not directly supported by ChromaDB
        # Use get with where filter if no vector provided
        if vector is None:
            if not where:
                raise SearchError(
                    "Either vector or where filter required for search", reason="both vector and where are missing"
                )
            # Use collection.get with where filter
            include = ["metadatas"]
            if self.store_text and (fields is None or "text" in fields):
                include.append("documents")

            fetch_limit = limit + offset
            results = self.collection.get(where=where, limit=fetch_limit, include=include)

            # Build initial document list
            ids = results["ids"] if results.get("ids") else []
            metadatas = results["metadatas"] if results.get("metadatas") else []
            documents = results["documents"] if results.get("documents") else [None] * len(ids)

            vector_docs = []
            for id_, meta, doc in zip(ids, metadatas, documents):
                # Unflatten metadata before returning
                metadata_val = unflatten_metadata(meta or {})

                # Build document without requiring vector (metadata-only fetch)
                vector_docs.append(
                    VectorDocument(
                        id=id_,
                        vector=[],  # unknown/omitted
                        text=doc if doc is not None else None,
                        metadata=metadata_val,
                    )
                )

            # Apply offset and limit after client filtering
            vector_docs = vector_docs[offset : offset + limit]
            self.logger.message(f"Search returned {len(vector_docs)} results.")
            return vector_docs

        # Vector search path
        include = ["metadatas", "distances"]
        if self.store_text:
            if fields is None or "text" in fields:
                include.append("documents")

        # ChromaDB fetch with offset: get limit+offset and slice later
        fetch_limit = limit + offset
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=fetch_limit,
            where=where,
            include=include,
        )

        # ChromaDB returns lists of lists (one per query)
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else [None] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [None] * len(ids)

        # Apply offset by slicing
        ids = ids[offset:]
        distances = distances[offset:]
        metadatas = metadatas[offset:]
        documents = documents[offset:]

        # Convert to VectorDocument instances
        vector_docs = []
        for id_, dist, meta, doc in zip(ids, distances, metadatas, documents):
            # Unflatten metadata before returning
            metadata_val = unflatten_metadata(meta or {})
            if isinstance(dist, (int, float)):
                metadata_val["score"] = 1 - dist

            vector_docs.append(
                VectorDocument(
                    id=id_,
                    vector=[],  # similarity search result can omit vector unless included explicitly
                    text=doc if doc is not None else None,
                    metadata=metadata_val,
                )
            )

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
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="get", adapter="ChromaDB")

        pk = args[0] if args else None
        doc_id = pk or extract_pk(None, **kwargs) if not pk else pk

        # Priority 1: Direct pk lookup
        if doc_id:
            results = self.collection.get(ids=[doc_id], limit=2, include=["embeddings", "metadatas", "documents"])
            if not results["ids"]:
                raise DoesNotExist(f"Document with ID '{doc_id}' not found")
            if len(results["ids"]) > 1:
                raise MultipleObjectsReturned(f"Multiple documents found with ID '{doc_id}'")
            embeddings = results.get("embeddings")
            if embeddings is None:
                embeddings = []
            metadatas = results.get("metadatas")
            if metadatas is None:
                metadatas = []
            documents = results.get("documents")
            if documents is None:
                documents = []
            vector_val = embeddings[0] if len(embeddings) > 0 else []
            metadata_val = metadatas[0] if len(metadatas) > 0 else {}
            text_val = documents[0] if len(documents) > 0 else None
            # Unflatten metadata to restore nested structure
            metadata_val = unflatten_metadata(metadata_val) if metadata_val else {}
            return VectorDocument(
                id=results["ids"][0],
                vector=vector_val
                if isinstance(vector_val, list)
                else list(vector_val)
                if vector_val is not None
                else [],
                text=text_val,
                metadata=metadata_val,
            )

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
            InvalidFieldError: If vector is missing
            DocumentExistsError: If document with same ID already exists
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="create", adapter="ChromaDB")

        pk = doc.pk
        try:
            vector = doc.to_vector(require=True, output_format="list")
        except MissingFieldError:
            raise InvalidFieldError("Vector is required", field="vector", operation="create")

        # Conflict check
        existing = self.collection.get(ids=[pk])
        if existing.get("ids"):
            raise DocumentExistsError("Document already exists", document_id=pk)

        text = doc.text if (self.store_text and doc.text is not None) else None
        metadata = doc.to_metadata(sanitize=True, output_format="dict")

        # Flatten nested metadata for Chroma
        metadata = flatten_metadata(metadata)

        self.collection.add(ids=[pk], embeddings=[vector], metadatas=[metadata], documents=[text] if text else None)
        self.logger.message(f"Created document with id '{pk}'.")
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
            raise CollectionNotInitializedError("Collection is not initialized", operation="update", adapter="ChromaDB")

        pk = doc.id or extract_pk(None, **kwargs)
        if not pk:
            raise MissingFieldError("'id', '_id', or 'pk' is required for update", field="id", operation="update")

        # Get existing document
        existing = self.collection.get(ids=[pk], include=["embeddings", "metadatas", "documents"])
        if not existing["ids"]:
            raise DocumentNotFoundError("Document not found", document_id=pk, operation="update")

        prepared = prepare_item_for_storage(
            doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector),
            store_text=self.store_text,
        )
        vector = prepared.get("$vector") or prepared.get("vector") or existing["embeddings"][0]
        text = prepared.get("text") if self.store_text else (existing.get("documents", [None])[0])

        # Start from existing metadata, overlay new fields
        existing_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        # Unflatten existing to allow nested updates
        metadata = unflatten_metadata(existing_meta)

        for k, v in prepared.items():
            if k not in ("_id", "$vector", "text"):
                metadata[k] = v

        # Flatten back for Chroma
        metadata = flatten_metadata(metadata)

        self.collection.update(ids=[pk], embeddings=[vector], metadatas=[metadata], documents=[text] if text else None)
        self.logger.message(f"Updated document with id '{pk}'.")

        # Return refreshed document
        refreshed = self.collection.get(ids=[pk], include=["embeddings", "metadatas", "documents"])
        doc_data = {
            "_id": refreshed["ids"][0],
            "vector": refreshed["embeddings"][0],
            "metadata": refreshed["metadatas"][0] if refreshed["metadatas"] else {},
        }
        if refreshed.get("documents"):
            doc_data["text"] = refreshed["documents"][0]

        return VectorDocument.from_kwargs(**doc_data)

    def delete(self, ids: DocIds) -> int:
        """Delete document(s) by ID.

        Args:
            ids: Single document ID or list of IDs to delete

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection:
            raise CollectionNotInitializedError("Collection is not initialized", operation="delete", adapter="ChromaDB")

        # Convert single ID to list
        if isinstance(ids, (str, int)):
            pks = [ids]
        else:
            pks = list(ids) if ids else []

        if not pks:
            return 0

        self.collection.delete(ids=pks)
        self.logger.message(f"Deleted {len(pks)} document(s).")
        return len(pks)

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
            DocumentExistsError: If conflict occurs and ignore_conflicts=False
        """
        if not self.collection:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="bulk_create", adapter="ChromaDB"
            )
        if not docs:
            return []

        to_add_ids: List[str] = []
        to_add_vectors: List[List[float]] = []
        to_add_metadatas: List[Dict[str, Any]] = []
        to_add_texts: List[str | None] = []
        created_docs: List[VectorDocument] = []

        for doc in docs:
            pk = doc.pk
            try:
                vector = doc.to_vector(require=True, output_format="list")
            except MissingFieldError:
                raise InvalidFieldError("Vector is required", field="vector", operation="bulk_create")

            existing = self.collection.get(ids=[pk])
            if existing.get("ids"):
                if ignore_conflicts:
                    continue
                if update_conflicts:
                    # Build update payload using helpers
                    base_dict = doc.to_storage_dict(
                        store_text=self.store_text, use_dollar_vector=self.use_dollar_vector
                    )
                    update_doc = apply_update_fields(base_dict, update_fields)
                    if not update_doc:
                        continue
                    vector_update = update_doc.get("$vector") or update_doc.get("vector") or vector
                    tmp_meta_doc = VectorDocument(
                        id=pk,
                        vector=[],
                        metadata={k: v for k, v in update_doc.items() if k not in {"_id", "$vector", "vector", "text"}},
                    )
                    meta_update = tmp_meta_doc.to_metadata(sanitize=True, output_format="dict")
                    text_update = update_doc.get("text") if self.store_text else None
                    self.collection.update(
                        ids=[pk],
                        embeddings=[vector_update],
                        metadatas=[meta_update] if meta_update else None,
                        documents=[text_update] if text_update else None,
                    )
                    continue
                raise DocumentExistsError("Document already exists", document_id=pk, operation="bulk_create")

            metadata = doc.to_metadata(
                exclude={"created_timestamp", "updated_timestamp"}, sanitize=True, output_format="dict"
            )
            # Flatten nested metadata for Chroma
            metadata = flatten_metadata(metadata)

            text_val = doc.text if (self.store_text and doc.text is not None) else None

            to_add_ids.append(pk)
            to_add_vectors.append(vector)
            to_add_metadatas.append(metadata)
            to_add_texts.append(text_val)
            created_docs.append(doc)

        if not to_add_ids:
            return []

        # ChromaDB batch insert with optional chunking
        if batch_size and batch_size > 0:
            for i in range(0, len(to_add_ids), batch_size):
                slice_ids = to_add_ids[i : i + batch_size]
                slice_vecs = to_add_vectors[i : i + batch_size]
                slice_meta = to_add_metadatas[i : i + batch_size]
                slice_docs = [t for t in to_add_texts[i : i + batch_size]] if self.store_text else None
                self.collection.add(
                    ids=slice_ids,
                    embeddings=slice_vecs,
                    metadatas=slice_meta,
                    documents=slice_docs,
                )
        else:
            self.collection.add(
                ids=to_add_ids,
                embeddings=to_add_vectors,
                metadatas=to_add_metadatas,
                documents=to_add_texts if self.store_text else None,
            )

        self.logger.message(f"Bulk created {len(created_docs)} document(s).")
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
                "Collection is not initialized", operation="bulk_update", adapter="ChromaDB"
            )
        if not docs:
            return []

        # Collect pks and map docs (avoid N+1 lookups)
        pk_doc_map: Dict[str, VectorDocument] = {}
        missing: List[str] = []
        for doc in docs:
            pk = doc.pk
            if not pk:
                if ignore_conflicts:
                    continue
                missing.append("<no_id>")
                continue
            pk_doc_map[pk] = doc

        if not pk_doc_map:
            if missing and not ignore_conflicts:
                raise MissingDocumentError("Missing documents for update", missing_ids=missing, operation="bulk_update")
            return []

        all_pks = list(pk_doc_map.keys())
        existing_batch = self.collection.get(ids=all_pks, include=["embeddings", "metadatas", "documents"])
        # Chroma returns only existing ids; build position map
        existing_ids = existing_batch.get("ids", []) or []
        id_index: Dict[str, int] = {pid: idx for idx, pid in enumerate(existing_ids)}

        # Determine missing ids
        for pk in all_pks:
            if pk not in id_index and not ignore_conflicts:
                missing.append(pk)

        if missing and not ignore_conflicts:
            raise MissingDocumentError("Missing documents for update", missing_ids=missing, operation="bulk_update")

        # Build update payloads
        update_ids: List[str] = []
        update_vectors: List[List[float]] = []
        update_metadatas: List[Dict[str, Any] | None] = []
        update_texts: List[str | None] = []
        updated_docs: List[VectorDocument] = []

        # Get embeddings list without truthiness check to avoid array comparison issues
        embeddings_list = existing_batch.get("embeddings", [])
        if embeddings_list is None:
            embeddings_list = []
        metadatas_list = existing_batch.get("metadatas", [])
        if metadatas_list is None:
            metadatas_list = []
        documents_list = existing_batch.get("documents", [])
        if documents_list is None:
            documents_list = []

        for pk, doc in pk_doc_map.items():
            if pk not in id_index:
                continue
            idx = id_index[pk]
            existing_embedding = embeddings_list[idx] if idx < len(embeddings_list) else None
            existing_metadata = metadatas_list[idx] if idx < len(metadatas_list) else {}
            existing_text = documents_list[idx] if idx < len(documents_list) else None

            base_dict = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(base_dict, update_fields)
            if not update_doc:
                continue

            # Use explicit None checks to avoid array truthiness issues
            vector_update = update_doc.get("$vector")
            if vector_update is None:
                vector_update = update_doc.get("vector")
            if vector_update is None:
                vector_update = existing_embedding
            # Unflatten existing metadata first
            metadata_merge = unflatten_metadata(dict(existing_metadata))
            for k, v in update_doc.items():
                if k not in {"_id", "$vector", "text", "vector"}:
                    metadata_merge[k] = v
            # Sanitize merged metadata via helper
            tmp_meta_doc = VectorDocument(id=pk, vector=[], metadata=metadata_merge)
            meta_result = tmp_meta_doc.to_metadata(sanitize=True, output_format="dict")
            # Flatten metadata for Chroma
            meta_result = flatten_metadata(meta_result) if meta_result else {}
            # Keep metadata if it exists and is a non-empty dict
            meta_update = (
                meta_result
                if (meta_result is not None and isinstance(meta_result, dict) and len(meta_result) > 0)
                else None
            )
            text_update = update_doc.get("text") if self.store_text else None
            if text_update is None and self.store_text:
                text_update = existing_text

            update_ids.append(pk)
            update_vectors.append(vector_update)
            update_metadatas.append(meta_update)
            update_texts.append(text_update if self.store_text else None)
            updated_docs.append(doc)

        if not updated_docs:
            self.logger.message("Bulk updated 0 document(s).")
            return []

        # Perform batched updates to reduce round-trips
        if batch_size and batch_size > 0:
            for i in range(0, len(update_ids), batch_size):
                slice_ids = update_ids[i : i + batch_size]
                slice_vectors = update_vectors[i : i + batch_size]
                slice_meta = update_metadatas[i : i + batch_size]
                slice_docs = update_texts[i : i + batch_size] if self.store_text else None
                self.collection.update(
                    ids=slice_ids,
                    embeddings=slice_vectors,
                    metadatas=slice_meta,
                    documents=slice_docs,
                )
        else:
            self.collection.update(
                ids=update_ids,
                embeddings=update_vectors,
                metadatas=update_metadatas,
                documents=update_texts if self.store_text else None,
            )

        self.logger.message(f"Bulk updated {len(updated_docs)} document(s). (single fetch, batched writes)")
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
            raise CollectionNotInitializedError("Collection is not initialized", operation="upsert", adapter="ChromaDB")
        if not docs:
            return []

        ids = []
        vectors = []
        metadatas = []
        texts = []

        for doc in docs:
            ids.append(doc.pk)
            try:
                vectors.append(doc.to_vector(require=True, output_format="list"))
            except MissingFieldError:
                vectors.append([])  # keep alignment; Chroma may reject empty but let upstream handle
            metadata = doc.to_metadata(
                exclude={"created_timestamp", "updated_timestamp"}, sanitize=True, output_format="dict"
            )
            # Flatten metadata for Chroma
            metadata = flatten_metadata(metadata)
            metadatas.append(metadata)
            texts.append(doc.text if (self.store_text and doc.text is not None) else None)

        # Use Chroma's native upsert API to insert or update
        if batch_size and batch_size > 0:
            for i in range(0, len(ids), batch_size):
                slice_ids = ids[i : i + batch_size]
                slice_vecs = vectors[i : i + batch_size]
                slice_meta = metadatas[i : i + batch_size]
                slice_docs = texts[i : i + batch_size] if self.store_text else None
                self.collection.upsert(
                    ids=slice_ids,
                    embeddings=slice_vecs,
                    metadatas=slice_meta,
                    documents=slice_docs,
                )
        else:
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=texts if self.store_text else None,
            )

        self.logger.message(f"Upserted {len(docs)} document(s).")
        return docs
