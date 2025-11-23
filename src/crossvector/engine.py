"""
Main engine for orchestrating vector store operations.

This module provides the `VectorEngine`, a high-level class that uses
pluggable adapters for embedding and database operations. It simplifies
interactions with the vector store through a single `execute` method.
"""

import logging
from typing import Any, Dict, List

from crossvector.settings import settings

from .abc import EmbeddingAdapter, VectorDBAdapter
from .schema import SearchRequest, UpsertRequest

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
            f"store_text: {store_text}."
        )

        # Initialize the database collection
        self.db_adapter.initialize(
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_adapter.embedding_dimension,
            store_text=self.store_text,
        )

    def upsert(self, params: UpsertRequest) -> Dict[str, Any]:
        """
        Handles the logic for upserting documents.
        """
        log.info(f"Executing upsert for {len(params.documents)} documents.")
        if not params.documents:
            log.warning("Upsert operation called with no documents.")
            return {"status": "noop", "count": 0}

        texts_to_embed = [doc.text for doc in params.documents]
        embeddings = self.embedding_adapter.get_embeddings(texts_to_embed)

        docs_to_insert = []
        for i, doc in enumerate(params.documents):
            # Add timestamps to metadata
            metadata_with_timestamps = {
                **doc.metadata,
                "created_timestamp": doc.created_timestamp,
                "updated_timestamp": doc.updated_timestamp,
            }

            docs_to_insert.append(
                {
                    "_id": doc.id,
                    "text": doc.text,
                    "vector": embeddings[i],
                    **metadata_with_timestamps,
                }
            )

        self.db_adapter.upsert(docs_to_insert)
        log.info(f"Upsert operation completed for {len(docs_to_insert)} documents.")
        return {"status": "success", "count": len(docs_to_insert)}

    def add(self, params: UpsertRequest) -> Dict[str, Any]:
        """
        Adds new documents to the vector store.
        This is an alias for upsert() - documents will be inserted or updated if they exist.

        Args:
            params: UpsertRequest containing documents to add

        Returns:
            Dict with status and count of documents added
        """
        return self.upsert(params)

    def search(self, params: SearchRequest) -> List[Dict[str, Any]]:
        """
        Handles the logic for searching documents.

        TODO: Add rerank feature in next version
              - Support reranking with Cohere, Jina, or custom rerankers
              - Allow hybrid search (vector + keyword)
              - Add score fusion strategies
        """
        log.info(f"Executing search with query: '{params.query[:50]}...'")
        query_embedding = self.embedding_adapter.get_embeddings([params.query])[0]

        results = self.db_adapter.search(vector=query_embedding, limit=params.limit, fields=params.fields)
        log.info(f"Search operation found {len(results)} results.")

        # TODO: Add rerank step here
        # if rerank_params:
        #     results = self._rerank(results, params.query, rerank_params)

        return results

    def get(self, id: str) -> Dict[str, Any] | None:
        """
        Retrieves a single document by its ID.
        """
        log.info(f"Retrieving document with ID: {id}")
        return self.db_adapter.get(id)

    def count(self) -> int:
        """
        Returns the total number of documents in the collection.
        """
        count = self.db_adapter.count()
        log.info(f"Collection '{self.collection_name}' has {count} documents.")
        return count

    def delete_one(self, id: str) -> int:
        """
        Deletes a single document by its ID.
        """
        log.warning(f"Attempting to delete document with ID: {id}")
        deleted_count = self.db_adapter.delete_one(id)
        log.info(f"Successfully deleted {deleted_count} document.")
        return deleted_count

    def delete_many(self, ids: List[str]) -> int:
        """
        Deletes multiple documents by their IDs.
        """
        if not ids:
            log.warning("delete_many called with an empty list of IDs.")
            return 0
        log.warning(f"Attempting to delete {len(ids)} documents.")
        deleted_count = self.db_adapter.delete_many(ids)
        log.info(f"Successfully deleted {deleted_count} documents.")
        return deleted_count

    def clear_collection(self) -> Dict[str, Any]:
        """
        Deletes all documents from the collection. A dangerous operation.
        """
        log.warning(f"Clearing all documents from collection '{self.collection_name}'.")
        result = self.db_adapter.collection.delete_many({})
        return {"deleted_count": result.deleted_count}
