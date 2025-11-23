"""
Concrete adapter for Chroma vector database.
Design follows AstraDBAdapter and MilvusDBAdapter: lazy client/collection, adapter interface.
"""

import logging
import os
from typing import Any, Dict, List, Set

import chromadb
from chromadb.config import Settings

from crossvector.constants import VECTOR_METRIC_MAP, VectorMetric
from crossvector.settings import settings as api_settings

log = logging.getLogger(__name__)


class ChromaDBAdapter:
    """
    Vector database adapter for ChromaDB.
    """

    def __init__(self, **kwargs: Any):
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None
        log.info("ChromaDBAdapter initialized.")

    @property
    def client(self) -> chromadb.Client:
        """
        Lazy initializes and returns the ChromaDB client.
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
                    log.info("Chroma CloudClient initialized.")
                    return self._client
                except Exception:
                    try:
                        # Fallback: top-level CloudClient
                        CloudClient = getattr(chromadb, "CloudClient", None)
                        if CloudClient:
                            self._client = CloudClient(
                                tenant=cloud_tenant, database=cloud_database, api_key=cloud_api_key
                            )
                            log.info("Chroma CloudClient (top-level) initialized.")
                            return self._client
                    except Exception:
                        log.exception("Failed to initialize Chroma CloudClient; falling back.")

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
                        log.info(f"Chroma HttpClient initialized (host={http_host}, port={http_port}).")
                        return self._client
                except Exception:
                    log.exception("Failed to initialize Chroma HttpClient; falling back.")

            # 3) Fallback: local persistence client
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", None)
            settings = Settings(persist_directory=persist_dir) if persist_dir else Settings()
            try:
                self._client = chromadb.Client(settings)
                log.info(f"ChromaDB local client initialized. Persist dir: {persist_dir}")
            except Exception:
                log.exception("Failed to initialize local Chroma client.")
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """
        Lazily initializes and returns the ChromaDB collection using get_collection.
        """
        if not self.collection_name or not self.embedding_dimension:
            raise ValueError("Collection name and embedding dimension must be set. Call initialize().")
        return self.get_collection(self.collection_name, self.embedding_dimension)

    def initialize(
        self, collection_name: str, embedding_dimension: int, metric: str = None, store_text: bool = None, **kwargs
    ):
        """
        Creates or retrieves a ChromaDB collection.
        """

        self.store_text = store_text or api_settings.VECTOR_STORE_TEXT
        if metric is None:
            metric = os.getenv("VECTOR_METRIC", VectorMetric.COSINE)
        self.get_collection(collection_name, embedding_dimension, metric)

    def get_collection(self, collection_name: str, embedding_dimension: int, metric: str = "cosine") -> Any:
        """
        Gets or creates the underlying ChromaDB collection object.
        Ensures the collection exists and is ready for use.
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        # Default store_text if not set
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
        self.client.delete_collection(collection_name)
        log.info(f"ChromaDB collection '{collection_name}' dropped.")
        return True

    def upsert(self, documents: List[Dict[str, Any]]):
        """
        Inserts a batch of documents into the ChromaDB collection.
        Each document should follow the standard format:
        {"_id": str, "$vector": List[float], "text": str, ...metadata}
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        if not documents:
            return
        ids = [doc.get("_id") or doc.get("id") for doc in documents]
        vectors = [doc.get("$vector") or doc.get("vector") for doc in documents]

        # Handle text storage
        texts = None
        if self.store_text:
            texts = [doc.get("text") for doc in documents]

        # ChromaDB expects metadata as dict, so extract all fields except _id, $vector, and text
        metadatas = []
        for doc in documents:
            metadata = {k: v for k, v in doc.items() if k not in ("_id", "$vector", "id", "vector", "text")}
            # Flatten metadata if it contains nested dicts (Chroma doesn't support nested metadata)
            flat_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat_metadata[f"{k}.{sub_k}"] = sub_v
                else:
                    flat_metadata[k] = v
            metadatas.append(flat_metadata)

        self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
        log.info(f"Inserted {len(ids)} vectors into ChromaDB.")

    def search(self, vector: List[float], limit: int, fields: Set[str] | None = None) -> List[Dict[str, Any]]:
        """
        Performs a vector similarity search in ChromaDB.
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")

        # Determine what to include
        include = ["metadatas", "distances"]
        if self.store_text:
            if fields is None or "text" in fields:
                include.append("documents")

        results = self.collection.query(query_embeddings=[vector], n_results=limit, include=include)

        # Chroma returns ids, distances, metadatas, documents as lists of lists (one per query)
        # We only query one vector, so take the first element
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [None] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [None] * len(ids)

        out = []
        for id_, dist, meta, doc in zip(ids, distances, metadatas, documents):
            item = {"id": id_, "score": dist, "metadata": meta}
            if doc is not None:
                item["text"] = doc
            out.append(item)
        return out

    def get(self, id: str) -> Dict[str, Any] | None:
        """
        Retrieves a single entity by its ID from ChromaDB.
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        results = self.collection.get(ids=[id], include=["embeddings", "metadatas", "documents"])
        if results["ids"]:
            return {"id": results["ids"][0], "vector": results["embeddings"][0], "metadata": results["metadatas"][0]}
        return None

    def count(self) -> int:
        """
        Counts the total number of entities in the ChromaDB collection.
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        return self.collection.count()

    def delete_one(self, id: str) -> int:
        """
        Deletes a single entity by its ID from ChromaDB.
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        self.collection.delete(ids=[id])
        return 1

    def delete_many(self, ids: List[str]) -> int:
        """
        Deletes multiple entities by their IDs from ChromaDB.
        """
        if not self.collection:
            raise ConnectionError("ChromaDB collection is not initialized.")
        self.collection.delete(ids=ids)
        return len(ids)
