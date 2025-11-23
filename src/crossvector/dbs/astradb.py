"""Concrete adapter for AstraDB vector database."""

import logging
import os
from typing import Any, Dict, List, Set

from astrapy import DataAPIClient
from astrapy.constants import DOC
from astrapy.constants import VectorMetric as AstraVectorMetric
from astrapy.data.collection import Collection
from astrapy.database import Database
from astrapy.info import CollectionDefinition, CollectionVectorOptions

from crossvector.abc import VectorDBAdapter
from crossvector.constants import VECTOR_METRIC_MAP, VectorMetric
from crossvector.settings import settings

log = logging.getLogger(__name__)


class AstraDBAdapter(VectorDBAdapter):
    """
    Vector database adapter for AstraDB.
    """

    def __init__(self, **kwargs: Any):
        self._client: DataAPIClient | None = None
        self._db: Any | None = None
        self.collection: Collection | None = None
        log.info("AstraDBAdapter initialized.")

    @property
    def client(self) -> DataAPIClient:
        """
        Lazily initializes and returns the AstraDB DataAPIClient.
        """
        if self._client is None:
            if not settings.ASTRA_DB_APPLICATION_TOKEN:
                raise ValueError("ASTRA_DB_APPLICATION_TOKEN is not set. Please configure it in your .env file.")
            self._client = DataAPIClient(token=settings.ASTRA_DB_APPLICATION_TOKEN)
        return self._client

    @property
    def db(self) -> Database:
        """
        Lazily initializes and returns the AstraDB database instance.
        """
        if self._db is None:
            if not settings.ASTRA_DB_API_ENDPOINT:
                raise ValueError("ASTRA_DB_API_ENDPOINT is not set. Please configure it in your .env file.")
            self._db = self.client.get_database(api_endpoint=settings.ASTRA_DB_API_ENDPOINT)
        return self._db

    def initialize(self, collection_name: str, embedding_dimension: int, metric: str = None, store_text: bool = True):
        """
        Creates or retrieves an AstraDB collection with the proper vector configuration.
        """
        self.store_text = store_text
        if metric is None:
            metric = os.getenv("VECTOR_METRIC", VectorMetric.COSINE)
        self.get_collection(collection_name, embedding_dimension, metric)

    def get_collection(
        self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE
    ) -> Collection[DOC]:
        """
        Gets or creates the underlying AstraDB collection object.
        Ensures the collection exists and is ready for use.
        """
        try:
            self.collection_name = collection_name
            self.embedding_dimension = embedding_dimension
            # Default store_text if not set
            if not hasattr(self, "store_text"):
                self.store_text = True

            vector_metric = VECTOR_METRIC_MAP.get(metric.lower(), AstraVectorMetric.COSINE)
            # List existing collections
            existing_collections = self.db.list_collection_names()
            if collection_name in existing_collections:
                # Collection exists, get it
                self.collection = self.db.get_collection(collection_name)
                log.info(f"AstraDB collection '{collection_name}' retrieved successfully.")
            else:
                # Collection doesn't exist, create it
                log.info(f"Creating new collection '{collection_name}'...")
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

    def upsert(self, documents: List[Dict[str, Any]]):
        """
        Inserts or updates multiple documents in the AstraDB collection.
        Each document should follow the standard format:
        {"_id": str, "$vector": List[float], "text": str, ...metadata}
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        items = []
        for doc in documents:
            # AstraDB stores all fields at root level, with _id and $vector as special keys
            item = {}
            # Set _id if present
            if "_id" in doc:
                item["_id"] = doc["_id"]
            elif "id" in doc:
                item["_id"] = doc["id"]
            # Set $vector
            if "$vector" in doc:
                item["$vector"] = doc["$vector"]
            elif "vector" in doc:
                item["$vector"] = doc["vector"]

            # Set text if enabled
            if self.store_text and "text" in doc:
                item["text"] = doc["text"]

            # Add all other fields as metadata
            # In Astra DB, we can store metadata fields at root level or nested.
            # To be consistent with other adapters, let's keep them at root but exclude special keys.
            for k, v in doc.items():
                if k not in ("_id", "$vector", "id", "vector", "text"):
                    item[k] = v
            items.append(item)
        result = self.collection.insert_many(items)
        return result.inserted_ids if hasattr(result, "inserted_ids") else result

    def search(self, vector: List[float], limit: int, fields: Set[str] | None = None) -> List[Dict[str, Any]]:
        """
        Performs a vector similarity search in AstraDB.
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")

        try:
            # Construct projection
            projection = {"$vector": 0}  # Exclude vector by default to save bandwidth
            if fields:
                projection = {field: 1 for field in fields}
            elif self.store_text:
                # If no fields specified, return everything (except vector usually, but let's follow standard)
                # Astra returns everything by default if projection is empty/None
                projection = {"$vector": 0}

            results = list(
                self.collection.find(
                    sort={"$vector": vector},
                    limit=limit,
                    projection=projection,
                )
            )
            log.info(f"Found {len(results)} results in AstraDB.")
            return results
        except Exception as e:
            log.error(f"Failed to search in AstraDB: {e}", exc_info=True)
            raise

    def get(self, id: str) -> Dict[str, Any] | None:
        """
        Retrieves a single document by its ID from AstraDB.
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        return self.collection.find_one({"_id": id})

    def count(self) -> int:
        """
        Counts the total number of documents in the AstraDB collection.
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        return self.collection.count_documents({}, upper_bound=10000)

    def delete_one(self, id: str) -> int:
        """
        Deletes a single document by its ID from AstraDB.
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        result = self.collection.delete_one({"_id": id})
        return result.deleted_count

    def delete_many(self, ids: List[str]) -> int:
        """
        Deletes multiple documents by their IDs from AstraDB.
        """
        if not self.collection:
            raise ConnectionError("AstraDB collection is not initialized.")
        if not ids:
            return 0
        result = self.collection.delete_many({"_id": {"$in": ids}})
        return result.deleted_count
