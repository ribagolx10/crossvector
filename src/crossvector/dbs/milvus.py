"""
Concrete adapter for Milvus vector database.
Design follows AstraDBAdapter: lazy client/collection, adapter interface.
"""

import logging
import os
from typing import Any, Dict, List, Set

from pymilvus import DataType, MilvusClient

from crossvector.constants import VECTOR_METRIC_MAP, VectorMetric

log = logging.getLogger(__name__)


class MilvusDBAdapter:
    """
    Vector database adapter for Milvus (cloud API).
    """

    def __init__(self, **kwargs: Any):
        self._client: MilvusClient | None = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None
        log.info("MilvusDBAdapter initialized.")

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            uri = os.getenv("MILVUS_API_ENDPOINT")
            user = os.getenv("MILVUS_USER")
            password = os.getenv("MILVUS_PASSWORD")
            token = None
            if user and password:
                token = f"{user}:{password}"
            self._client = MilvusClient(uri=uri, token=token)
            log.info(f"MilvusClient initialized with uri={uri}")
        return self._client

    def initialize(self, collection_name: str, embedding_dimension: int, metric: str = None, store_text: bool = True):
        self.store_text = store_text
        if metric is None:
            metric = os.getenv("VECTOR_METRIC", VectorMetric.COSINE)
        self.get_collection(collection_name, embedding_dimension, metric)

    def get_collection(self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE) -> Any:
        """
        Gets or creates the underlying Milvus collection object.
        Ensures the collection exists and is ready for use.
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        # Default store_text if not set
        if not hasattr(self, "store_text"):
            self.store_text = True

        metric_key = VECTOR_METRIC_MAP.get(metric, VectorMetric.COSINE)
        info = self._get_collection_info(collection_name)
        index_info = self._get_index_info(collection_name)

        # Check schema compatibility (simplified)
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
            if "doc_id" not in field_names or "vector" not in field_names:
                self.client.drop_collection(collection_name=collection_name)
                log.info(f"Milvus collection '{collection_name}' dropped due to wrong schema.")
                need_create = True
            elif self.store_text and "text" not in field_names:
                # If we want to store text but the collection doesn't have it, we must recreate
                self.client.drop_collection(collection_name=collection_name)
                log.info(f"Milvus collection '{collection_name}' dropped to add 'text' field.")
                need_create = True
            elif not has_vector_index:
                # Index missing/wrong
                self.client.drop_collection(collection_name=collection_name)
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

    def _get_collection_info(self, collection_name: str):
        try:
            return self.client.describe_collection(collection_name=collection_name)
        except Exception:
            return None

    def _get_index_info(self, collection_name: str):
        try:
            return self.client.describe_index(collection_name=collection_name)
        except Exception:
            return None

    def _build_schema(self, embedding_dimension: int):
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dimension)
        if self.store_text:
            # Max length for VARCHAR in Milvus is 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        return schema

    def _build_index_params(self, embedding_dimension: int, metric: str = VectorMetric.COSINE):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_type="TRIE",  # For VARCHAR primary key
        )
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type=metric.upper(), params={"nlist": 1024}
        )
        return index_params

    def upsert(self, documents: List[Dict[str, Any]]):
        """
        Inserts documents into Milvus collection.
        Each document should follow the standard format:
        {"_id": str, "$vector": List[float], "text": str, ...metadata}
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        if not documents:
            return
        data = []
        for doc in documents:
            doc_id = doc.get("_id") or doc.get("id")
            vector = doc.get("$vector") or doc.get("vector")

            item = {"doc_id": doc_id, "vector": vector}

            if self.store_text:
                text = doc.get("text", "")
                # Truncate if too long (Milvus limit)
                if len(text) > 65535:
                    text = text[:65535]
                item["text"] = text

            # Extract metadata (all fields except _id, $vector, text)
            metadata = {k: v for k, v in doc.items() if k not in ("_id", "$vector", "id", "vector", "text")}
            item["metadata"] = metadata
            data.append(item)

        self.client.insert(collection_name=self.collection_name, data=data)
        log.info(f"Inserted {len(data)} vectors into Milvus.")

    def search(self, vector: List[float], limit: int, fields: Set[str] | None = None) -> List[Dict[str, Any]]:
        self.client.load_collection(collection_name=self.collection_name)
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        output_fields = ["metadata"]
        if self.store_text:
            if fields is None or "text" in fields:
                output_fields.append("text")

        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=limit,
            output_fields=output_fields,
        )
        # MilvusClient returns list of dicts
        return results[0] if results else []

    def get(self, id: str) -> Dict[str, Any] | None:
        """Retrieves a document by its doc_id."""
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        results = self.client.get(collection_name=self.collection_name, ids=[id])
        return results[0] if results else None

    def count(self) -> int:
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        info = self.client.describe_collection(collection_name=self.collection_name)
        return info.get("num_entities", 0)

    def delete_one(self, id: str) -> int:
        """Deletes a document by its doc_id."""
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        self.client.delete(collection_name=self.collection_name, ids=[id])
        return 1

    def delete_many(self, ids: List[str]) -> int:
        """Deletes multiple documents by their doc_ids."""
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        self.client.delete(collection_name=self.collection_name, ids=ids)
        return len(ids)
