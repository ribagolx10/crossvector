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

import logging
import os
from typing import Any, Dict, List, Sequence, Set, Union, Tuple

import chromadb
from chromadb.config import Settings

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
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None
        log.info("ChromaDBAdapter initialized.")

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
            Exception: If all initialization attempts fail
        """
        if self._client is None:
            # 1) Try CloudClient if cloud API key present
            cloud_api_key = os.getenv("CHROMA_CLOUD_API_KEY") or os.getenv("CHROMA_API_KEY")
            cloud_tenant = os.getenv("CHROMA_CLOUD_TENANT") or os.getenv("CHROMA_TENANT")
            cloud_database = os.getenv("CHROMA_CLOUD_DATABASE") or os.getenv("CHROMA_DATABASE")
            if cloud_api_key:
                try:
                    self._client = chromadb.CloudClient(
                        tenant=cloud_tenant, database=cloud_database, api_key=cloud_api_key
                    )
                    log.info("ChromaDB CloudClient initialized.")
                    return self._client
                except Exception:
                    try:
                        # Fallback: top-level CloudClient
                        CloudClient = getattr(chromadb, "CloudClient", None)
                        if CloudClient:
                            self._client = CloudClient(
                                tenant=cloud_tenant, database=cloud_database, api_key=cloud_api_key
                            )
                            log.info("ChromaDB CloudClient (top-level) initialized.")
                            return self._client
                    except Exception:
                        log.exception("Failed to initialize ChromaDB CloudClient; falling back.")

            # 2) Try HttpClient (self-hosted server) if host/port provided
            http_host = os.getenv("CHROMA_HTTP_HOST") or os.getenv("CHROMA_SERVER_HOST")
            http_port = os.getenv("CHROMA_HTTP_PORT") or os.getenv("CHROMA_SERVER_PORT")
            if http_host:
                try:
                    HttpClient = getattr(chromadb, "HttpClient", None)
                    if HttpClient:
                        if http_port:
                            self._client = HttpClient(host=http_host, port=int(http_port))
                        else:
                            self._client = HttpClient(host=http_host)
                        log.info(f"ChromaDB HttpClient initialized (host={http_host}, port={http_port}).")
                        return self._client
                except Exception:
                    log.exception("Failed to initialize ChromaDB HttpClient; falling back.")

            # 3) Fallback: local persistence client
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", None)
            settings = Settings(persist_directory=persist_dir) if persist_dir else Settings()
            try:
                self._client = chromadb.Client(settings)
                log.info(f"ChromaDB local client initialized. Persist dir: {persist_dir}")
            except Exception:
                log.exception("Failed to initialize local ChromaDB client.")
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
            raise ValueError("Collection name and embedding dimension must be set. Call initialize().")
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
            metric = os.getenv("VECTOR_METRIC", VectorMetric.COSINE)
        self.get_or_create_collection(collection_name, embedding_dimension, metric)
        log.info(
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
            ValueError: If collection already exists
        """
        try:
            self.client.get_collection(collection_name)
            raise ValueError(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            if "already exists" in str(e).lower():
                raise ValueError(f"Collection '{collection_name}' already exists.") from e

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
        log.info(f"ChromaDB collection '{collection_name}' created.")
        return self._collection

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        """Get an existing ChromaDB collection.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            ChromaDB Collection instance

        Raises:
            ValueError: If collection doesn't exist
        """
        try:
            self._collection = self.client.get_collection(collection_name)
            self.collection_name = collection_name
            log.info(f"ChromaDB collection '{collection_name}' retrieved.")
            return self._collection
        except Exception as e:
            raise ValueError(f"Collection '{collection_name}' does not exist.") from e

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
            log.info(f"ChromaDB collection '{collection_name}' retrieved.")
        except Exception:
            self._collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.metric},
                embedding_function=None,
            )
            log.info(f"ChromaDB collection '{collection_name}' created.")
        return self._collection

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the specified collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful
        """
        self.client.delete_collection(collection_name)
        log.info(f"ChromaDB collection '{collection_name}' dropped.")
        return True

    def clear_collection(self) -> int:
        """Delete all documents from the collection.

        Returns:
            Number of documents deleted

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        count = self.collection.count()
        if count == 0:
            return 0
        results = self.collection.get(limit=count, include=[])
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        log.info(f"Cleared {len(ids)} documents from collection.")
        return len(ids)

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        return self.collection.count()

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
            raise ConnectionError("ChromaDB collection is not initialized.")

        # Determine what to include
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
            ConnectionError: If collection is not initialized
            ValueError: If document ID is missing or document not found
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")

        doc_id = pk or extract_id(kwargs)
        if not doc_id:
            raise ValueError("Document ID is required (provide pk or id/_id/pk in kwargs)")

        results = self.collection.get(ids=[doc_id], include=["embeddings", "metadatas", "documents"])
        if not results["ids"]:
            raise ValueError(f"Document with ID '{doc_id}' not found")

        doc_data = {
            "_id": results["ids"][0],
            "vector": results["embeddings"][0],
            "metadata": results["metadatas"][0] if results["metadatas"] else {},
        }
        if results.get("documents"):
            doc_data["text"] = results["documents"][0]

        return VectorDocument.from_kwargs(**doc_data)

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
            ValueError: If vector is missing or document with same ID already exists
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")

        doc = VectorDocument.from_kwargs(**kwargs)
        stored = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        pk = doc.pk
        vector = stored.get("$vector") or stored.get("vector")
        if vector is None:
            raise ValueError("Vector ('$vector' or 'vector') is required for create in ChromaDB.")

        # Conflict check
        existing = self.collection.get(ids=[pk])
        if existing.get("ids"):
            raise ValueError(f"Conflict: document with id '{pk}' already exists.")

        text = stored.get("text") if self.store_text else None
        metadata = {k: v for k, v in stored.items() if k not in ("_id", "$vector", "text")}

        self.collection.add(ids=[pk], embeddings=[vector], metadatas=[metadata], documents=[text] if text else None)
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
            ConnectionError: If collection is not initialized
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")

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
            raise ConnectionError("ChromaDB collection is not initialized.")

        id_val = extract_id(kwargs)
        if not id_val:
            raise ValueError("'id', '_id', or 'pk' is required for update")

        # Get existing document
        existing = self.collection.get(ids=[id_val], include=["embeddings", "metadatas", "documents"])
        if not existing["ids"]:
            raise ValueError(f"Document with ID '{id_val}' not found")

        prepared = prepare_item_for_storage(kwargs, store_text=self.store_text)
        vector = prepared.get("$vector") or prepared.get("vector") or existing["embeddings"][0]
        text = prepared.get("text") if self.store_text else (existing.get("documents", [None])[0])

        # Start from existing metadata, overlay new fields
        metadata = existing["metadatas"][0] if existing["metadatas"] else {}
        for k, v in prepared.items():
            if k not in ("_id", "$vector", "text"):
                metadata[k] = v

        self.collection.update(
            ids=[id_val], embeddings=[vector], metadatas=[metadata], documents=[text] if text else None
        )
        log.info(f"Updated document with id '{id_val}'.")

        # Return refreshed document
        refreshed = self.collection.get(ids=[id_val], include=["embeddings", "metadatas", "documents"])
        doc_data = {
            "_id": refreshed["ids"][0],
            "vector": refreshed["embeddings"][0],
            "metadata": refreshed["metadatas"][0] if refreshed["metadatas"] else {},
        }
        if refreshed.get("documents"):
            doc_data["text"] = refreshed["documents"][0]

        return VectorDocument.from_kwargs(**doc_data)

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
            raise ConnectionError("ChromaDB collection is not initialized.")

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
            raise ConnectionError("ChromaDB collection is not initialized.")

        pks = normalize_pks(ids)
        if not pks:
            return 0

        self.collection.delete(ids=pks)
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
            ConnectionError: If collection is not initialized
            ValueError: If conflict occurs and ignore_conflicts=False
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        if not documents:
            return []

        to_add_ids: List[str] = []
        to_add_vectors: List[List[float]] = []
        to_add_metadatas: List[Dict[str, Any]] = []
        to_add_texts: List[str | None] = []
        created_docs: List[VectorDocument] = []

        for doc in documents:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            pk = doc.pk
            vector = item.get("$vector") or item.get("vector")
            if vector is None:
                raise ValueError("Vector required for bulk_create in ChromaDB.")

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
                raise ValueError(f"Conflict on id '{pk}' during bulk_create.")

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
            raise ConnectionError("ChromaDB collection is not initialized.")
        if not documents:
            return []

        updated_docs: List[VectorDocument] = []
        missing: List[str] = []

        for doc in documents:
            pk = doc.pk
            if not pk:
                if ignore_conflicts:
                    continue
                missing.append("<no_id>")
                continue

            existing = self.collection.get(ids=[pk], include=["embeddings", "metadatas", "documents"])
            if not existing.get("ids"):
                if ignore_conflicts:
                    continue
                missing.append(pk)
                continue

            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(item, update_fields)

            meta_update = {k: v for k, v in update_doc.items() if k not in ("_id", "$vector", "text")}
            vector_update = update_doc.get("$vector") or update_doc.get("vector") or existing["embeddings"][0]
            text_update = update_doc.get("text") if self.store_text else None

            self.collection.update(
                ids=[pk],
                embeddings=[vector_update],
                metadatas=[meta_update] if meta_update else None,
                documents=[text_update] if text_update else None,
            )
            updated_docs.append(doc)

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
            raise ConnectionError("ChromaDB collection is not initialized.")
        if not documents:
            return []

        ids = []
        vectors = []
        metadatas = []
        texts = []

        for doc in documents:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            ids.append(doc.pk)
            vectors.append(item.get("$vector") or item.get("vector"))
            metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "text")}
            metadatas.append(metadata)
            texts.append(item.get("text") if self.store_text else None)

        if batch_size and batch_size > 0:
            for i in range(0, len(ids), batch_size):
                slice_ids = ids[i : i + batch_size]
                slice_vecs = vectors[i : i + batch_size]
                slice_meta = metadatas[i : i + batch_size]
                slice_docs = texts[i : i + batch_size] if self.store_text else None
                self.collection.add(
                    ids=slice_ids,
                    embeddings=slice_vecs,
                    metadatas=slice_meta,
                    documents=slice_docs,
                )
        else:
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=texts if self.store_text else None,
            )

        log.info(f"Upserted {len(documents)} document(s).")
        return documents
