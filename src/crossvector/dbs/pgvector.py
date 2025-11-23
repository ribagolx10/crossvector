"""
Concrete adapter for pgvector (PostgreSQL vector extension).
Design giống các adapter khác: lazy client, collection, interface chuẩn.
"""

import json
import logging
import os
from typing import Any, Dict, List, Set

import psycopg2
import psycopg2.extras

from crossvector.constants import VectorMetric

log = logging.getLogger(__name__)


class PGVectorAdapter:
    """
    Vector database adapter for pgvector (PostgreSQL).
    """

    def __init__(self, **kwargs: Any):
        self._conn = None
        self._cursor = None
        self.table_name: str | None = None
        self.embedding_dimension: int | None = None
        log.info("PGVectorAdapter initialized.")

    @property
    def conn(self):
        if self._conn is None:
            self._conn = psycopg2.connect(
                dbname=os.getenv("PGVECTOR_DBNAME", "postgres"),
                user=os.getenv("PGVECTOR_USER", "postgres"),
                password=os.getenv("PGVECTOR_PASSWORD", "postgres"),
                host=os.getenv("PGVECTOR_HOST", "localhost"),
                port=os.getenv("PGVECTOR_PORT", "5432"),
            )
        return self._conn

    @property
    def cursor(self):
        if self._cursor is None:
            self._cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return self._cursor

    def initialize(self, table_name: str, embedding_dimension: int, metric: str = "cosine"):
        self.get_collection(table_name, embedding_dimension, metric)

    def get_collection(self, table_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE) -> Any:
        """
        Gets or creates the underlying PGVector table object.
        Ensures the table exists and is ready for use.
        """
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) UNIQUE,
            vector vector({embedding_dimension}),
            metadata JSONB
        );
        """
        self.cursor.execute(create_table_sql)
        self.conn.commit()
        log.info(f"PGVector table '{table_name}' initialized.")
        return table_name

    def upsert(self, documents: List[Dict[str, Any]]):
        """
        Inserts a batch of documents into the PGVector table.
        Each document should follow the standard format:
        {"_id": str, "$vector": List[float], "text": str, ...metadata}
        """
        if not self.table_name:
            raise ValueError("Table name must be set. Call initialize().")
        for doc in documents:
            doc_id = doc.get("_id") or doc.get("id")
            vector = doc.get("$vector") or doc.get("vector")
            # Extract metadata (all fields except _id and $vector)
            metadata = {k: v for k, v in doc.items() if k not in ("_id", "$vector", "id", "vector")}
            metadata_json = json.dumps(metadata)
            sql = f"INSERT INTO {self.table_name} (doc_id, vector, metadata) VALUES (%s, %s, %s)"
            self.cursor.execute(sql, (doc_id, vector, metadata_json))
        self.conn.commit()
        log.info(f"Inserted {len(documents)} vectors into PGVector.")

    def search(self, vector: List[float], limit: int, fields: Set[str]) -> List[Dict[str, Any]]:
        if not self.table_name:
            raise ValueError("Table name must be set. Call initialize().")
        # Cast Python list to PostgreSQL vector type
        sql = f"SELECT id, metadata, vector <-> %s::vector AS score FROM {self.table_name} ORDER BY score ASC LIMIT %s"
        self.cursor.execute(sql, (vector, limit))
        results = self.cursor.fetchall()
        return [{"id": r["id"], "score": r["score"], "metadata": r["metadata"]} for r in results]

    def get(self, id: str) -> Dict[str, Any] | None:
        """Retrieves a document by its doc_id."""
        if not self.table_name:
            raise ValueError("Table name must be set. Call initialize().")
        sql = f"SELECT doc_id, vector, metadata FROM {self.table_name} WHERE doc_id = %s"
        self.cursor.execute(sql, (id,))
        result = self.cursor.fetchone()
        if result:
            return {"_id": result["doc_id"], "$vector": result["vector"], **result["metadata"]}
        return None

    def count(self) -> int:
        if not self.table_name:
            raise ValueError("Table name must be set. Call initialize().")
        sql = f"SELECT COUNT(*) FROM {self.table_name}"
        self.cursor.execute(sql)
        return self.cursor.fetchone()["count"]

    def delete_one(self, id: str) -> int:
        """Deletes a document by its doc_id."""
        if not self.table_name:
            raise ValueError("Table name must be set. Call initialize().")
        sql = f"DELETE FROM {self.table_name} WHERE doc_id = %s"
        self.cursor.execute(sql, (id,))
        self.conn.commit()
        return self.cursor.rowcount

    def delete_many(self, ids: List[str]) -> int:
        """Deletes multiple documents by their doc_ids."""
        if not self.table_name:
            raise ValueError("Table name must be set. Call initialize().")
        sql = f"DELETE FROM {self.table_name} WHERE doc_id = ANY(%s)"
        self.cursor.execute(sql, (ids,))
        self.conn.commit()
        return self.cursor.rowcount
