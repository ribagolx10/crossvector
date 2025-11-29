"""
Main engine for orchestrating vector store operations.

This module provides the `VectorEngine`, a high-level class that uses
pluggable adapters for embedding and database operations. It provides
a convenient wrapper around the ABC interface with automatic embedding
generation and flexible input handling.
"""

import logging
from typing import Any, Dict, List, Sequence, Set, Union

from crossvector.settings import settings

from .abc import EmbeddingAdapter, VectorDBAdapter
from .schema import VectorDocument
from .utils import normalize_texts, normalize_metadatas, normalize_pks

log = logging.getLogger(__name__)


class VectorEngine:
    """
    Orchestrates vector database and embedding operations using adapters.
    """

    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        db_adapter: VectorDBAdapter,
        collection_name: str = settings.ASTRA_DB_COLLECTION_NAME,
        store_text: bool = settings.VECTOR_STORE_TEXT,
    ):
        """
        Initializes the engine with specific adapters.

        Args:
            embedding_adapter: An instance of an EmbeddingAdapter subclass.
            db_adapter: An instance of a VectorDBAdapter subclass.
            collection_name: The name of the collection to work with.
            store_text: Whether to store the original text content in the database.
        """
        self.embedding_adapter = embedding_adapter
        self.db_adapter = db_adapter
        self.collection_name = collection_name
        self.store_text = store_text

        log.info(
            f"VectorEngine initialized with "
            f"EmbeddingAdapter: {embedding_adapter.__class__.__name__}, "
            f"DBAdapter: {db_adapter.__class__.__name__}, "
            f"store_text: {store_text}, "
            f"pk_mode: {settings.PRIMARY_KEY_MODE}."
        )

        # Initialize the database collection
        self.db_adapter.initialize(
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_adapter.embedding_dimension,
            store_text=self.store_text,
        )

    def drop_collection(self, collection_name: str) -> bool:
        """
        Drops the collection.
        """
        return self.db_adapter.drop_collection(collection_name)

    def clear_collection(self) -> Dict[str, Any]:
        """
        Deletes all documents from the collection. A dangerous operation.
        """
        log.warning(f"Clearing all documents from collection '{self.collection_name}'.")
        deleted_count = self.db_adapter.clear_collection()
        return {"deleted_count": deleted_count}

    def count(self) -> int:
        """
        Returns the total number of documents in the collection.
        """
        count = self.db_adapter.count()
        log.info(f"Collection '{self.collection_name}' has {count} documents.")
        return count

    def search(
        self,
        query: str,
        limit: int = 5,
        offset: int = 0,
        where: Dict[str, Any] | None = None,
        fields: Set[str] | None = None,
    ) -> List[VectorDocument]:
        """
        Perform vector similarity search with automatic query embedding.

        Args:
            query: Search query text
            limit: Maximum number of results to return (default: 5)
            offset: Number of results to skip for pagination (default: 0)
            where: Optional metadata filter conditions
            fields: Optional set of field names to include in results

        Returns:
            List of VectorDocument instances ordered by similarity

        Examples:
            # Simple search
            docs = engine.search("machine learning", limit=10)
            for doc in docs:
                print(doc.text, doc.metadata)

            # Search with pagination
            docs = engine.search("AI", limit=10, offset=20)

            # Search with metadata filter
            docs = engine.search("python", where={"category": "tutorial", "level": "beginner"})

        TODO: Add rerank feature in next version
              - Support reranking with Cohere, Jina, or custom rerankers
              - Allow hybrid search (vector + keyword)
              - Add score fusion strategies
        """
        log.info(f"Executing search with query: '{query[:50]}...', limit={limit}, offset={offset}")

        # Generate query embedding
        query_embedding = self.embedding_adapter.get_embeddings([query])[0]

        # Perform search with all parameters
        vector_docs = self.db_adapter.search(
            vector=query_embedding,
            limit=limit,
            offset=offset,
            where=where,
            fields=fields,
        )

        log.info(f"Search operation found {len(vector_docs)} results.")

        # TODO: Add rerank step here
        # if rerank_params:
        #     vector_docs = self._rerank(vector_docs, query, rerank_params)

        return vector_docs

    def get(self, pk: str) -> VectorDocument:
        """
        Retrieve a single document by its primary key.

        Args:
            pk: Primary key of the document to retrieve

        Returns:
            VectorDocument instance

        Raises:
            ValueError: If document not found

        Examples:
            doc = engine.get("doc_id_123")
            print(doc.text, doc.metadata)
        """
        log.info(f"Retrieving document with pk: {pk}")
        return self.db_adapter.get(pk=pk)

    def delete(self, ids: Union[str, Sequence[str]]) -> int:
        """
        Delete document(s) by primary key.

        Args:
            ids: Single document pk or list of pks to delete

        Returns:
            Number of documents successfully deleted

        Examples:
            # Single document
            count = engine.delete("doc_id")

            # Multiple documents
            count = engine.delete(["doc1", "doc2", "doc3"])
        """
        log.info(f"Deleting document(s): {ids}")
        return self.db_adapter.delete(ids)

    def upsert(
        self,
        documents: list[VectorDocument],
        batch_size: int | None = None,
    ) -> list[VectorDocument]:
        """
        Insert or update documents (create if not exists, update if exists).

        Args:
            documents: List of VectorDocument instances to upsert
            batch_size: Number of documents per batch (optional)

        Returns:
            List of successfully upserted VectorDocument instances

        Examples:
            # Prepare documents with embeddings
            docs = [
                VectorDocument(text="Hello", vector=embedding1),
                VectorDocument(text="World", vector=embedding2)
            ]
            result = engine.upsert(docs)
        """
        log.info(f"Upserting {len(documents)} document(s)")
        return self.db_adapter.upsert(documents, batch_size=batch_size)

    def bulk_create(
        self,
        documents: list[VectorDocument],
        ignore_conflicts: bool = False,
        update_conflicts: bool = False,
    ) -> list[VectorDocument]:
        """
        Bulk create documents with conflict handling.

        Args:
            documents: List of VectorDocument instances to create
            ignore_conflicts: If True, skip documents with conflicting pk
            update_conflicts: If True, update existing documents on conflict

        Returns:
            List of successfully created VectorDocument instances

        Raises:
            ValueError: If conflict occurs and both flags are False

        Examples:
            docs = [VectorDocument(text="Doc1", vector=v1), ...]
            result = engine.bulk_create(docs, ignore_conflicts=True)
        """
        log.info(f"Bulk creating {len(documents)} document(s)")
        return self.db_adapter.bulk_create(
            documents,
            ignore_conflicts=ignore_conflicts,
            update_conflicts=update_conflicts,
        )

    def bulk_update(
        self,
        documents: list[VectorDocument],
        batch_size: int | None = None,
        ignore_conflicts: bool = False,
    ) -> list[VectorDocument]:
        """
        Bulk update existing documents.

        Args:
            documents: List of VectorDocument instances to update
            batch_size: Number of documents per batch (optional)
            ignore_conflicts: If True, skip non-existent documents

        Returns:
            List of successfully updated VectorDocument instances

        Raises:
            ValueError: If any document doesn't exist and ignore_conflicts=False

        Examples:
            docs = [VectorDocument(pk="id1", text="Updated", vector=v1), ...]
            result = engine.bulk_update(docs, batch_size=100)
        """
        log.info(f"Bulk updating {len(documents)} document(s)")
        return self.db_adapter.bulk_update(
            documents,
            batch_size=batch_size,
            ignore_conflicts=ignore_conflicts,
        )

    def get_collection(self, collection_name: str | None = None) -> Any:
        """
        Get an existing collection object.

        Args:
            collection_name: Name of the collection (default: current collection)

        Returns:
            Collection object (type depends on database adapter)

        Raises:
            ValueError: If collection doesn't exist

        Examples:
            # Get current collection
            collection = engine.get_collection()

            # Get specific collection
            collection = engine.get_collection("my_collection")
        """
        name = collection_name or self.collection_name
        return self.db_adapter.get_collection(name)

    def add_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: str = "cosine",
    ) -> None:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection to create
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dot_product)

        Raises:
            ValueError: If collection already exists

        Examples:
            engine.add_collection("my_collection", dimension=1536)
        """
        log.info(f"Creating collection: {collection_name}")
        self.db_adapter.add_collection(collection_name, dimension, metric)

    def get_or_create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: str = "cosine",
    ) -> Any:
        """
        Get existing collection or create if it doesn't exist.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension (used if creating)
            metric: Distance metric (used if creating)

        Returns:
            Collection object (type depends on database adapter)

        Examples:
            collection = engine.get_or_create_collection("my_collection", 1536)
        """
        return self.db_adapter.get_or_create_collection(
            collection_name,
            dimension,
            metric,
        )

    # Helper methods for flexible input handling with auto-embedding

    def create_from_texts(
        self,
        texts: Union[str, List[str]],
        metadatas: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
        pks: Union[str, List[str], None] = None,
        ignore_conflicts: bool = False,
        update_conflicts: bool = False,
    ) -> list[VectorDocument]:
        """
        Create documents from raw text(s) with automatic embedding generation.

        Args:
            texts: Single text string or list of text strings
            metadatas: Single metadata dict or list of metadata dicts (optional)
            pks: Single pk or list of pks (optional, auto-generated if not provided)
            ignore_conflicts: Skip conflicting documents
            update_conflicts: Update existing documents on conflict

        Returns:
            List of successfully created VectorDocument instances

        Examples:
            # Single text
            docs = engine.create_from_texts("Hello world")
            docs = engine.create_from_texts("Hello", metadatas={"source": "test"})

            # Multiple texts
            docs = engine.create_from_texts(
                ["Text 1", "Text 2"],
                metadatas=[{"id": 1}, {"id": 2}]
            )
        """
        # Normalize inputs using utils
        text_list = normalize_texts(texts)
        metadata_list = normalize_metadatas(metadatas, len(text_list))
        pk_list = normalize_pks(pks, len(text_list))

        # Generate embeddings
        log.info(f"Generating embeddings for {len(text_list)} text(s)")
        embeddings = self.embedding_adapter.get_embeddings(text_list)

        # Create VectorDocuments
        vector_docs = []
        for i, text in enumerate(text_list):
            doc = VectorDocument(
                id=pk_list[i] if i < len(pk_list) else None,
                text=text,
                vector=embeddings[i],
                metadata=metadata_list[i] if i < len(metadata_list) else {},
            )
            vector_docs.append(doc)

        # Bulk create
        return self.bulk_create(
            vector_docs,
            ignore_conflicts=ignore_conflicts,
            update_conflicts=update_conflicts,
        )

    def upsert_from_texts(
        self,
        texts: Union[str, List[str]],
        metadatas: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
        pks: Union[str, List[str], None] = None,
        batch_size: int | None = None,
    ) -> list[VectorDocument]:
        """
        Upsert documents from raw text(s) with automatic embedding generation.

        Args:
            texts: Single text string or list of text strings
            metadatas: Single metadata dict or list of metadata dicts (optional)
            pks: Single pk or list of pks (optional, auto-generated if not provided)
            batch_size: Number of documents per batch (optional)

        Returns:
            List of successfully upserted VectorDocument instances

        Examples:
            # Single text
            docs = engine.upsert_from_texts("Hello world", pks="doc1")

            # Multiple texts
            docs = engine.upsert_from_texts(
                ["Text 1", "Text 2"],
                pks=["doc1", "doc2"],
                metadatas=[{"v": 1}, {"v": 2}]
            )
        """
        # Normalize inputs using utils
        text_list = normalize_texts(texts)
        metadata_list = normalize_metadatas(metadatas, len(text_list))
        pk_list = normalize_pks(pks, len(text_list))

        # Generate embeddings
        log.info(f"Generating embeddings for {len(text_list)} text(s)")
        embeddings = self.embedding_adapter.get_embeddings(text_list)

        # Create VectorDocuments
        vector_docs = []
        for i, text in enumerate(text_list):
            doc = VectorDocument(
                id=pk_list[i] if i < len(pk_list) else None,
                text=text,
                vector=embeddings[i],
                metadata=metadata_list[i] if i < len(metadata_list) else {},
            )
            vector_docs.append(doc)

        # Upsert
        return self.upsert(vector_docs, batch_size=batch_size)

    def update_from_texts(
        self,
        pks: Union[str, List[str]],
        texts: Union[str, List[str]],
        metadatas: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
        batch_size: int | None = None,
        ignore_conflicts: bool = False,
    ) -> list[VectorDocument]:
        """
        Update existing documents from raw text(s) with automatic embedding generation.

        Args:
            pks: Single pk or list of pks (required for updates)
            texts: Single text string or list of text strings
            metadatas: Single metadata dict or list of metadata dicts (optional)
            batch_size: Number of documents per batch (optional)
            ignore_conflicts: Skip non-existent documents

        Returns:
            List of successfully updated VectorDocument instances

        Raises:
            ValueError: If any document doesn't exist and ignore_conflicts=False

        Examples:
            # Single document
            docs = engine.update_from_texts("doc1", "Updated text")

            # Multiple documents
            docs = engine.update_from_texts(
                ["doc1", "doc2"],
                ["Text 1", "Text 2"],
                metadatas=[{"v": 2}, {"v": 2}]
            )
        """
        # Normalize inputs using utils
        pk_list = normalize_pks(pks, 1 if isinstance(pks, (str, int)) else len(pks))  # type: ignore
        text_list = normalize_texts(texts)
        metadata_list = normalize_metadatas(metadatas, len(text_list))

        # Generate embeddings
        log.info(f"Generating embeddings for {len(text_list)} text(s)")
        embeddings = self.embedding_adapter.get_embeddings(text_list)

        # Create VectorDocuments
        vector_docs = []
        for i, text in enumerate(text_list):
            doc = VectorDocument(
                id=pk_list[i],
                text=text,
                vector=embeddings[i],
                metadata=metadata_list[i] if i < len(metadata_list) else {},
            )
            vector_docs.append(doc)

        # Bulk update
        return self.bulk_update(
            vector_docs,
            batch_size=batch_size,
            ignore_conflicts=ignore_conflicts,
        )
