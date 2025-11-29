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

from typing import Any, Dict, List, Set

import chromadb
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
    MissingDocumentError,
    MissingFieldError,
    MultipleObjectsReturned,
    SearchError,
)
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.types import DocIds
from crossvector.utils import (
    apply_update_fields,
    extract_pk,
    normalize_pks,
    prepare_item_for_storage,
)


class ChromaDBAdapter(VectorDBAdapter):
    """Vector database adapter for ChromaDB.

    Provides a high-level interface for vector operations using ChromaDB's
    vector search capabilities. Supports multiple deployment modes (cloud,
    self-hosted, local) with automatic client initialization.

    Attributes:
        collection_name: Name of the active collection
        embedding_dimension: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
        metric: Distance metric for vector search
    """

    use_dollar_vector: bool = False

    def __init__(self, **kwargs: Any):
        """Initialize the ChromaDB adapter with lazy client setup.

        Args:
            **kwargs: Additional configuration options (currently unused)
        """
        super(ChromaDBAdapter, self).__init__(**kwargs)
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None

    @property
    def client(self) -> chromadb.Client:
        """Lazily initialize and return the ChromaDB client.

        Attempts initialization in order:
        1. CloudClient (if CHROMA_CLOUD_API_KEY present)
        2. HttpClient (if CHROMA_HTTP_HOST present)
        3. Local persistence client (fallback)

        Returns:
            Initialized ChromaDB client instance

        Raises:
            MissingConfigError: If required configuration is missing
            ConnectionError: If client initialization fails
        """
        if self._client is None:
            # 1) Try CloudClient if cloud API key present
            if api_settings.CHROMA_API_KEY:
                try:
                    self._client = chromadb.CloudClient(
                        tenant=api_settings.CHROMA_CLOUD_TENANT,
                        database=api_settings.CHROMA_CLOUD_DATABASE,
                        api_key=api_settings.CHROMA_API_KEY,
                    )
                    self.logger.message("ChromaDB CloudClient initialized.")
                    return self._client
                except Exception:
                    try:
                        # Fallback: top-level CloudClient
                        CloudClient = getattr(chromadb, "CloudClient", None)
                        if CloudClient:
                            self._client = CloudClient(
                                tenant=api_settings.CHROMA_CLOUD_TENANT,
                                database=api_settings.CHROMA_CLOUD_DATABASE,
                                api_key=api_settings.CHROMA_API_KEY,
                            )
                            self.logger.message("ChromaDB CloudClient (top-level) initialized.")
                            return self._client
                    except Exception as exc:
                        self.logger.error(
                            f"Failed to initialize ChromaDB CloudClient, falling back. {exc}", exc_info=True
                        )
                        raise ConnectionError("Failed to initialize cloud ChromaDB client", adapter="ChromaDB")

            # 2) Try HttpClient (self-hosted server) if host/port provided
            if api_settings.CHROMA_HTTP_HOST:
                try:
                    HttpClient = getattr(chromadb, "HttpClient", None)
                    if HttpClient:
                        if api_settings.CHROMA_HTTP_PORT:
                            self._client = HttpClient(
                                host=api_settings.CHROMA_HTTP_HOST, port=int(api_settings.CHROMA_HTTP_PORT)
                            )
                        else:
                            self._client = HttpClient(host=api_settings.CHROMA_HTTP_HOST)

                        self.logger.message(
                            f"ChromaDB HttpClient initialized (host={api_settings.CHROMA_HTTP_HOST}, port={api_settings.CHROMA_HTTP_PORT})."
                        )
                        return self._client
                except Exception as e:
                    self.logger.error(f"Failed to initialize ChromaDB HttpClient; falling back. {e}", exc_info=True)
                    raise ConnectionError("Failed to initialize self-hosted ChromaDB client", adapter="ChromaDB")

            # 3) Fallback: local persistence client
            persist_dir = api_settings.CHROMA_PERSIST_DIR
            settings_obj = Settings(persist_directory=persist_dir) if persist_dir else Settings()
            try:
                self._client = chromadb.Client(settings_obj)
                self.logger.message(f"ChromaDB local client initialized. Persist dir: {persist_dir}")
            except Exception as e:
                self.logger.error(f"Failed to initialize local ChromaDB client: {e}", exc_info=True)
                raise ConnectionError("Failed to initialize local ChromaDB client", adapter="ChromaDB")
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Lazily initialize and return the ChromaDB collection.

        Returns:
            Active ChromaDB collection instance

        Raises:
            ValueError: If collection_name or embedding_dimension not set
        """
        if not self.collection_name or not self.embedding_dimension:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="property_access", adapter="ChromaDB"
            )
        return self.get_collection(self.collection_name, self.embedding_dimension)

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
            metric = api_settings.VECTOR_METRIC or VectorMetric.COSINE
        self.get_or_create_collection(collection_name, embedding_dimension, metric)
        self.logger.message(
            f"ChromaDB initialized: collection='{collection_name}', "
            f"dimension={embedding_dimension}, metric={metric}, store_text={self.store_text}"
        )

    def add_collection(
        self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE
    ) -> chromadb.Collection:
        """Create a new ChromaDB collection.

        Args:
            collection_name: Name of the collection to create
            embedding_dimension: Vector embedding dimension
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
        self.embedding_dimension = embedding_dimension
        if not hasattr(self, "store_text"):
            self.store_text = True

        self.metric = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        self._collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.metric},
            embedding_function=None,
        )
        self.logger.message(f"ChromaDB collection '{collection_name}' created.")
        return self._collection

    def get_collection(self, collection_name: str) -> chromadb.Collection:
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
            self._collection = self.client.get_collection(collection_name)
            self.collection_name = collection_name
            self.logger.message(f"ChromaDB collection '{collection_name}' retrieved.")
            return self._collection
        except Exception as e:
            raise CollectionNotFoundError("Collection does not exist", collection_name=collection_name) from e

    def get_or_create_collection(
        self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE
    ) -> chromadb.Collection:
        """Get or create the underlying ChromaDB collection.

        Ensures the collection exists with proper vector configuration.
        If the collection doesn't exist, it will be created with the specified
        distance metric.

        Args:
            collection_name: Name of the collection
            embedding_dimension: Vector embedding dimension
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
        self.embedding_dimension = embedding_dimension
        if not hasattr(self, "store_text"):
            self.store_text = True

        self.metric = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        if self._collection is not None and getattr(self._collection, "name", None) == collection_name:
            return self._collection

        try:
            self._collection = self.client.get_collection(collection_name)
            self.logger.message(f"ChromaDB collection '{collection_name}' retrieved.")
        except Exception:
            self._collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.metric},
                embedding_function=None,
            )
            self.logger.message(f"ChromaDB collection '{collection_name}' created.")
        return self._collection

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
            results = self.collection.get(where=where, limit=limit + offset, include=include)
            # Apply offset
            ids = results["ids"][offset:] if results.get("ids") else []
            metadatas = results["metadatas"][offset:] if results.get("metadatas") else []
            documents = results["documents"][offset:] if results.get("documents") else [None] * len(ids)
            vector_docs = []
            for id_, meta, doc in zip(ids, metadatas, documents):
                doc_dict = {"_id": id_, "metadata": meta or {}}
                if doc is not None:
                    doc_dict["text"] = doc
                vector_docs.append(VectorDocument.from_kwargs(**doc_dict))
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
            doc_dict = {"_id": id_, "metadata": meta or {}}
            if doc is not None:
                doc_dict["text"] = doc
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
            doc_data = {
                "_id": results["ids"][0],
                "vector": results["embeddings"][0] if results.get("embeddings") else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
            }
            if results.get("documents"):
                doc_data["text"] = results["documents"][0]
            return VectorDocument.from_kwargs(**doc_data)

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

        stored = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        pk = doc.pk
        vector = stored.get("$vector") or stored.get("vector")
        if vector is None:
            raise InvalidFieldError("Vector is required", field="vector", operation="create")

        # Conflict check
        existing = self.collection.get(ids=[pk])
        if existing.get("ids"):
            raise DocumentExistsError("Document already exists", document_id=pk)

        text = stored.get("text") if self.store_text else None
        metadata = {k: v for k, v in stored.items() if k not in ("_id", "$vector", "text")}

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
        metadata = existing["metadatas"][0] if existing["metadatas"] else {}
        for k, v in prepared.items():
            if k not in ("_id", "$vector", "text"):
                metadata[k] = v

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

        pks = normalize_pks(ids)
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
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            pk = doc.pk
            vector = item.get("$vector") or item.get("vector")
            if vector is None:
                raise InvalidFieldError("Vector is required", field="vector", operation="bulk_create")

            # Conflict detection (id only)
            existing = self.collection.get(ids=[pk])
            if existing.get("ids"):
                if ignore_conflicts:
                    continue
                if update_conflicts:
                    # Perform update instead
                    update_doc = apply_update_fields(item, update_fields)
                    meta_update = {k: v for k, v in update_doc.items() if k not in ("_id", "$vector", "text")}
                    vector_update = update_doc.get("$vector") or update_doc.get("vector") or vector
                    text_update = update_doc.get("text") if self.store_text else None
                    self.collection.update(
                        ids=[pk],
                        embeddings=[vector_update],
                        metadatas=[meta_update] if meta_update else None,
                        documents=[text_update] if text_update else None,
                    )
                    continue
                raise DocumentExistsError("Document already exists", document_id=pk, operation="bulk_create")

            metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "text")}
            text_val = item.get("text") if self.store_text else None

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

        embeddings_list = existing_batch.get("embeddings", []) or []
        metadatas_list = existing_batch.get("metadatas", []) or []
        documents_list = existing_batch.get("documents", []) or []

        for pk, doc in pk_doc_map.items():
            if pk not in id_index:
                # skipped due to ignore_conflicts
                continue
            idx = id_index[pk]
            existing_embedding = embeddings_list[idx] if idx < len(embeddings_list) else None
            existing_metadata = metadatas_list[idx] if idx < len(metadatas_list) else {}
            existing_text = documents_list[idx] if idx < len(documents_list) else None

            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(item, update_fields)
            if not update_doc:
                continue

            vector_update = update_doc.get("$vector") or update_doc.get("vector") or existing_embedding
            # Merge metadata: preserve existing then overlay update fields (excluding reserved keys)
            metadata_merge = dict(existing_metadata)
            for k, v in update_doc.items():
                if k not in ("_id", "$vector", "text", "vector"):
                    metadata_merge[k] = v
            meta_update = metadata_merge if metadata_merge else None
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
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            ids.append(doc.pk)
            vectors.append(item.get("$vector") or item.get("vector"))
            metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "text")}
            metadatas.append(metadata)
            texts.append(item.get("text") if self.store_text else None)

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
