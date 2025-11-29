"""Concrete adapter for PostgreSQL pgvector extension.

This module provides the pgvector implementation of the VectorDBAdapter interface,
enabling vector storage and retrieval using PostgreSQL's pgvector extension.

Key Features:
    - Lazy connection initialization to PostgreSQL
    - Full CRUD operations with VectorDocument models
    - Batch operations for bulk create/update/upsert
    - Configurable vector metrics via pgvector operators
    - Dynamic schema creation with PRIMARY_KEY_MODE support
    - Automatic index creation for vector similarity search
"""

import json
import logging
import os
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

import psycopg2
import psycopg2.extras

from crossvector.abc import VectorDBAdapter
from crossvector.constants import VectorMetric
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.utils import (
    apply_update_fields,
    extract_id,
    normalize_pks,
    prepare_item_for_storage,
)

log = logging.getLogger(__name__)


class PGVectorAdapter(VectorDBAdapter):
    """Vector database adapter for PostgreSQL with pgvector extension.

    Provides a high-level interface for vector operations using PostgreSQL's
    pgvector extension. Supports flexible primary key types and automatic
    schema migration when PK mode changes.

    Attributes:
        collection_name: Name of the active collection (table)
        embedding_dimension: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
    """

    use_dollar_vector: bool = False

    def __init__(self, **kwargs: Any):
        """Initialize the PGVector adapter with lazy connection setup.

        Args:
            **kwargs: Additional configuration options (currently unused)
        """
        self._conn = None
        self._cursor = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None
        log.info("PGVectorAdapter initialized.")

    @property
    def conn(self) -> Any:
        """Lazily initialize and return the PostgreSQL connection.

        Returns:
            Active psycopg2 connection instance

        Raises:
            psycopg2.Error: If connection fails
        """
        if self._conn is None:
            self._conn = psycopg2.connect(
                dbname=os.getenv("PGVECTOR_DBNAME", "postgres"),
                user=os.getenv("PGVECTOR_USER", "postgres"),
                password=os.getenv("PGVECTOR_PASSWORD", "postgres"),
                host=os.getenv("PGVECTOR_HOST", "localhost"),
                port=os.getenv("PGVECTOR_PORT", "5432"),
            )
            log.info("PostgreSQL connection established.")
        return self._conn

    @property
    def cursor(self) -> Any:
        """Lazily initialize and return a RealDictCursor.

        Returns:
            Active psycopg2 RealDictCursor instance
        """
        if self._cursor is None:
            self._cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return self._cursor

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def initialize(
        self,
        collection_name: str,
        embedding_dimension: int,
        metric: str = VectorMetric.COSINE,
        store_text: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the database and ensure the collection is ready.

        Args:
            collection_name: Name of the collection (table) to use/create
            embedding_dimension: Dimension of the vector embeddings
            metric: Distance metric ('cosine', 'euclidean', 'dot_product')
            store_text: Whether to store original text content
            **kwargs: Additional configuration options
        """
        self.store_text = store_text if store_text is not None else api_settings.VECTOR_STORE_TEXT
        self.get_collection(collection_name, embedding_dimension, metric)
        log.info(
            f"PGVector initialized: collection='{collection_name}', "
            f"dimension={embedding_dimension}, metric={metric}, store_text={self.store_text}"
        )

    def add_collection(self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE) -> str:
        """Create a new pgvector table.

        Args:
            collection_name: Name of the table to create
            embedding_dimension: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            Collection name (table name)

        Raises:
            ValueError: If table already exists
        """
        self.cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
            """,
            (collection_name,),
        )
        exists = self.cursor.fetchone()[0]
        if exists:
            raise ValueError(f"Collection '{collection_name}' already exists.")

        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        if not hasattr(self, "store_text"):
            self.store_text = True

        desired_int64 = (api_settings.PRIMARY_KEY_MODE or "uuid").lower() == "int64"
        pk_type = "BIGINT" if desired_int64 else "VARCHAR(255)"

        create_table_sql = f"""
        CREATE TABLE {collection_name} (
            id {pk_type} PRIMARY KEY,
            vector vector({embedding_dimension}),
            text TEXT,
            metadata JSONB
        );
        """
        self.cursor.execute(create_table_sql)
        self.conn.commit()
        log.info(f"PGVector table '{collection_name}' created. Store text: {self.store_text}")
        return collection_name

    def get_collection(self, collection_name: str) -> str:
        """Get an existing pgvector table.

        Args:
            collection_name: Name of the table to retrieve

        Returns:
            Collection name (table name)

        Raises:
            ValueError: If table doesn't exist
        """
        self.cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
            """,
            (collection_name,),
        )
        exists = self.cursor.fetchone()[0]
        if not exists:
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        self.collection_name = collection_name
        log.info(f"PGVector table '{collection_name}' retrieved.")
        return collection_name

    def get_or_create_collection(
        self, collection_name: str, embedding_dimension: int, metric: str = VectorMetric.COSINE
    ) -> str:
        """Get or create the underlying pgvector table.

        Ensures the table exists with proper vector configuration and PK type.
        If PK type doesn't match PRIMARY_KEY_MODE, the table is dropped and recreated.

        Args:
            collection_name: Name of the table
            embedding_dimension: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            Collection name (table name)
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        if not hasattr(self, "store_text"):
            self.store_text = True

        desired_int64 = (api_settings.PRIMARY_KEY_MODE or "uuid").lower() == "int64"
        pk_type = "BIGINT" if desired_int64 else "VARCHAR(255)"

        # Detect existing table and PK type mismatch
        self.cursor.execute(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'id'
            """,
            (collection_name,),
        )
        row = self.cursor.fetchone()
        if row:
            existing_type = row.get("data_type")
            # For varchar it may be 'character varying'
            is_int64 = existing_type and existing_type.lower() in {"bigint", "integer"}
            is_varchar = existing_type and "character varying" in existing_type.lower()

            if (desired_int64 and not is_int64) or ((not desired_int64) and not is_varchar):
                log.info(f"PK type mismatch detected; recreating table '{collection_name}' with desired PK type.")
                self.cursor.execute(f"DROP TABLE IF EXISTS {collection_name}")
                self.conn.commit()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {collection_name} (
            id {pk_type} PRIMARY KEY,
            vector vector({embedding_dimension}),
            text TEXT,
            metadata JSONB
        );
        """
        self.cursor.execute(create_table_sql)
        self.conn.commit()
        log.info(f"PGVector table '{collection_name}' initialized. Store text: {self.store_text}")
        return collection_name

    def drop_collection(self, collection_name: str) -> bool:
        """Drop the specified collection (table).

        Args:
            collection_name: Name of the table to drop

        Returns:
            True if successful
        """
        sql = f"DROP TABLE IF EXISTS {collection_name}"
        self.cursor.execute(sql)
        self.conn.commit()
        log.info(f"PGVector collection '{collection_name}' dropped.")
        return True

    def clear_collection(self) -> int:
        """Delete all documents from the collection (table).

        Returns:
            Number of documents deleted

        Raises:
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        count = self.count()
        if count == 0:
            return 0

        sql = f"TRUNCATE TABLE {self.collection_name}"
        self.cursor.execute(sql)
        self.conn.commit()
        log.info(f"Cleared {count} documents from collection.")
        return count

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        sql = f"SELECT COUNT(*) FROM {self.collection_name}"
        self.cursor.execute(sql)
        return self.cursor.fetchone()["count"]

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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Table name must be set. Call initialize().")

        # Construct SELECT query based on requested fields
        select_fields = ["id", "vector <-> %s::vector AS score"]

        if fields is None or "text" in fields:
            select_fields.append("text")
        if fields is None or "metadata" in fields:
            select_fields.append("metadata")

        # Build WHERE clause for metadata filter
        where_clause = ""
        params = [vector]
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append("metadata->>%s = %s")
                params.extend([key, str(value)])
            where_clause = " WHERE " + " AND ".join(conditions)

        params.extend([limit, offset])
        sql = f"SELECT {', '.join(select_fields)} FROM {self.collection_name}{where_clause} ORDER BY score ASC LIMIT %s OFFSET %s"
        self.cursor.execute(sql, tuple(params))
        results = self.cursor.fetchall()

        # Convert to VectorDocument instances
        vector_docs = []
        for r in results:
            doc_dict = {"_id": r["id"], "metadata": r.get("metadata", {})}
            if "text" in r:
                doc_dict["text"] = r["text"]
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
            ValueError: If collection not set or document ID missing/not found
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        doc_id = pk or extract_id(kwargs)
        if not doc_id:
            raise ValueError("Document ID is required (provide pk or id/_id/pk in kwargs)")

        sql = f"SELECT id, vector, text, metadata FROM {self.collection_name} WHERE id = %s"
        self.cursor.execute(sql, (doc_id,))
        result = self.cursor.fetchone()

        if not result:
            raise ValueError(f"Document with ID '{doc_id}' not found")

        doc_data = {
            "_id": result["id"],
            "vector": result["vector"],
            "text": result["text"],
            "metadata": result["metadata"] or {},
        }
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
            ValueError: If collection not set, vector missing, or document ID conflicts
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        doc = VectorDocument.from_kwargs(**kwargs)
        item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        pk = doc.pk

        # Conflict check
        self.cursor.execute(f"SELECT 1 FROM {self.collection_name} WHERE id = %s", (pk,))
        if self.cursor.fetchone():
            raise ValueError(f"Conflict: document with id '{pk}' already exists.")

        vector = item.get("$vector") or item.get("vector")
        if vector is None:
            raise ValueError("Vector required for create in PGVector.")

        text = item.get("text") if self.store_text else None
        metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "vector", "text")}

        self.cursor.execute(
            f"INSERT INTO {self.collection_name} (id, vector, text, metadata) VALUES (%s, %s, %s, %s)",
            (pk, vector, text, json.dumps(metadata)),
        )
        self.conn.commit()
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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

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
            ValueError: If collection not set, ID missing, or document not found
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        id_val = extract_id(kwargs)
        if not id_val:
            raise ValueError("'id', '_id', or 'pk' is required for update")

        # Get existing document
        sql = f"SELECT id, vector, text, metadata FROM {self.collection_name} WHERE id = %s"
        self.cursor.execute(sql, (id_val,))
        existing = self.cursor.fetchone()

        if not existing:
            raise ValueError(f"Document with ID '{id_val}' not found")

        prepared = prepare_item_for_storage(kwargs, store_text=self.store_text)
        updates: List[str] = []
        params: List[Any] = []

        if "$vector" in prepared or "vector" in prepared:
            updates.append("vector = %s")
            params.append(prepared.get("$vector") or prepared.get("vector"))

        if self.store_text and "text" in prepared:
            updates.append("text = %s")
            params.append(prepared["text"])

        metadata = existing.get("metadata", {})
        for k, v in prepared.items():
            if k not in ("_id", "$vector", "vector", "text"):
                metadata[k] = v

        if metadata:
            updates.append("metadata = %s")
            params.append(json.dumps(metadata))

        if not updates:
            # No changes to make
            doc_data = {
                "_id": existing["id"],
                "vector": existing["vector"],
                "text": existing["text"],
                "metadata": existing["metadata"] or {},
            }
            return VectorDocument.from_kwargs(**doc_data)

        params.append(id_val)
        sql = f"UPDATE {self.collection_name} SET {', '.join(updates)} WHERE id = %s"
        self.cursor.execute(sql, tuple(params))
        self.conn.commit()
        log.info(f"Updated document with id '{id_val}'.")

        # Return refreshed document
        return self.get(id_val)

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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")

        pks = normalize_pks(ids)
        if not pks:
            return 0

        if len(pks) == 1:
            sql = f"DELETE FROM {self.collection_name} WHERE id = %s"
            self.cursor.execute(sql, (pks[0],))
        else:
            sql = f"DELETE FROM {self.collection_name} WHERE id = ANY(%s)"
            self.cursor.execute(sql, (pks,))

        self.conn.commit()
        deleted = self.cursor.rowcount
        log.info(f"Deleted {deleted} document(s).")
        return deleted

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
            ValueError: If collection not set, vector missing, or conflict occurs
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        if not documents:
            return []

        created_docs: List[VectorDocument] = []
        batch: List[tuple] = []

        for doc in documents:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            pk = doc.pk

            # Conflict check
            self.cursor.execute(f"SELECT 1 FROM {self.collection_name} WHERE id = %s", (pk,))
            exists = self.cursor.fetchone()

            if exists:
                if ignore_conflicts:
                    continue
                if update_conflicts:
                    # Perform update instead
                    update_doc = apply_update_fields(item, update_fields)
                    update_kwargs = {"_id": pk, **update_doc}
                    self.update(**update_kwargs)
                    continue
                raise ValueError(f"Conflict on id '{pk}' during bulk_create.")

            vector = item.get("$vector") or item.get("vector")
            if vector is None:
                raise ValueError("Vector required for bulk_create in PGVector.")

            text = item.get("text") if self.store_text else None
            metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "vector", "text")}

            batch.append((pk, vector, text, json.dumps(metadata)))
            created_docs.append(doc)

            # Flush batch if size reached
            if batch_size and batch_size > 0 and len(batch) >= batch_size:
                self.cursor.executemany(
                    f"INSERT INTO {self.collection_name} (id, vector, text, metadata) VALUES (%s, %s, %s, %s)",
                    batch,
                )
                batch.clear()

        if batch:
            self.cursor.executemany(
                f"INSERT INTO {self.collection_name} (id, vector, text, metadata) VALUES (%s, %s, %s, %s)",
                batch,
            )

        self.conn.commit()
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
            ValueError: If collection not set or document missing (when ignore_conflicts=False)
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
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

            # Check if exists
            self.cursor.execute(f"SELECT 1 FROM {self.collection_name} WHERE id = %s", (pk,))
            existing = self.cursor.fetchone()

            if not existing:
                if ignore_conflicts:
                    continue
                missing.append(pk)
                continue

            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            update_doc = apply_update_fields(item, update_fields)

            if not update_doc:
                continue

            # Build update query
            update_kwargs = {"_id": pk, **update_doc}
            self.update(**update_kwargs)
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
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name must be set. Call initialize().")
        if not documents:
            return []

        for doc in documents:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            doc_id = doc.pk
            vector = item.get("$vector") or item.get("vector")
            text = item.get("text") if self.store_text else None
            metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "vector", "text")}
            metadata_json = json.dumps(metadata)

            sql = f"""
            INSERT INTO {self.collection_name} (id, vector, text, metadata) 
            VALUES (%s, %s, %s, %s) 
            ON CONFLICT (id) DO UPDATE 
            SET vector = EXCLUDED.vector, text = EXCLUDED.text, metadata = EXCLUDED.metadata
            """
            self.cursor.execute(sql, (doc_id, vector, text, metadata_json))

        self.conn.commit()
        log.info(f"Upserted {len(documents)} document(s).")
        return documents
