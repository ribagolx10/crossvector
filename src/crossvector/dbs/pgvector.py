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
from typing import Any, Dict, List, Set, Tuple

import psycopg2
import psycopg2.extras

from crossvector.abc import VectorDBAdapter
from crossvector.constants import VectorMetric
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
        super(PGVectorAdapter, self).__init__(**kwargs)
        self._conn = None
        self._cursor = None
        self.collection_name: str | None = None
        self.embedding_dimension: int | None = None

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
                dbname=api_settings.PGVECTOR_DBNAME or "postgres",
                user=api_settings.PGVECTOR_USER or "postgres",
                password=api_settings.PGVECTOR_PASSWORD or "postgres",
                host=api_settings.PGVECTOR_HOST or "localhost",
                port=api_settings.PGVECTOR_PORT or "5432",
            )
            self.logger.message("PostgreSQL connection established.")
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
        self.logger.message(
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
            CollectionExistsError: If table already exists
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
            raise CollectionExistsError("Collection already exists", collection_name=collection_name)

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
        self.logger.message(f"PGVector table '{collection_name}' created. Store text: {self.store_text}")
        return collection_name

    def get_collection(self, collection_name: str) -> str:
        """Get an existing pgvector table.

        Args:
            collection_name: Name of the table to retrieve

        Returns:
            Collection name (table name)

        Raises:
            CollectionNotFoundError: If table doesn't exist
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
            raise CollectionNotFoundError("Collection does not exist", collection_name=collection_name)

        self.collection_name = collection_name
        self.logger.message(f"PGVector table '{collection_name}' retrieved.")
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
                self.logger.message(
                    f"PK type mismatch detected; recreating table '{collection_name}' with desired PK type."
                )
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
        self.logger.message(f"PGVector table '{collection_name}' initialized. Store text: {self.store_text}")
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
        self.logger.message(f"PGVector collection '{collection_name}' dropped.")
        return True

    def clear_collection(self) -> int:
        """Delete all documents from the collection (table).

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection_name:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="clear_collection", adapter="PGVector"
            )

        count = self.count()
        if count == 0:
            return 0

        sql = f"TRUNCATE TABLE {self.collection_name}"
        self.cursor.execute(sql)
        self.conn.commit()
        self.logger.message(f"Cleared {count} documents from collection.")
        return count

    def count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total document count

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="count", adapter="PGVector")
        sql = f"SELECT COUNT(*) FROM {self.collection_name}"
        self.cursor.execute(sql)
        return self.cursor.fetchone()["count"]

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
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="search", adapter="PGVector")

        # Construct SELECT query based on requested fields
        select_fields = ["id"]
        if vector is not None:
            select_fields.append("vector <-> %s::vector AS score")
        if fields is None or "text" in fields:
            select_fields.append("text")
        if fields is None or "metadata" in fields:
            select_fields.append("metadata")

        # Build WHERE clause for metadata filter
        where_clause = ""
        params: List[Any] = []
        if vector is not None:
            params.append(vector)
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append("metadata->>%s = %s")
                params.extend([key, str(value)])
            where_clause = " WHERE " + " AND ".join(conditions)

        if limit is None:
            limit = api_settings.VECTOR_SEARCH_LIMIT
        params.extend([limit, offset])
        order_clause = " ORDER BY score ASC" if vector is not None else ""
        sql = f"SELECT {', '.join(select_fields)} FROM {self.collection_name}{where_clause}{order_clause} LIMIT %s OFFSET %s"
        self.cursor.execute(sql, tuple(params))
        results = self.cursor.fetchall()

        # Convert to VectorDocument instances
        vector_docs = []
        for r in results:
            doc_dict = {"_id": r["id"], "metadata": r.get("metadata", {})}
            if "text" in r:
                doc_dict["text"] = r["text"]
            vector_docs.append(VectorDocument.from_kwargs(**doc_dict))

        self.logger.message(f"Search returned {len(vector_docs)} results.")
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
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="get", adapter="PGVector")

        pk = args[0] if args else None
        doc_id = pk or extract_pk(None, **kwargs)

        # Priority 1: Direct pk lookup
        if doc_id:
            sql = f"SELECT id, vector, text, metadata FROM {self.collection_name} WHERE id = %s LIMIT 2"
            self.cursor.execute(sql, (doc_id,))
            rows = self.cursor.fetchall()
            if not rows:
                raise DoesNotExist(f"Document with ID '{doc_id}' not found")
            if len(rows) > 1:
                raise MultipleObjectsReturned(f"Multiple documents found with ID '{doc_id}'")
            result = rows[0]
            doc_data = {
                "_id": result["id"],
                "vector": result["vector"],
                "text": result["text"],
                "metadata": result["metadata"] or {},
            }
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
            DocumentExistsError: If document ID already exists
            InvalidFieldError: If vector missing
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="create", adapter="PGVector")

        item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)

        pk = doc.pk

        # Conflict check
        self.cursor.execute(f"SELECT 1 FROM {self.collection_name} WHERE id = %s", (pk,))
        if self.cursor.fetchone():
            raise DocumentExistsError("Document already exists", document_id=pk)

        vector = item.get("$vector") or item.get("vector")
        if vector is None:
            raise InvalidFieldError("Vector is required", field="vector", operation="create")

        text = item.get("text") if self.store_text else None
        metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "vector", "text")}

        self.cursor.execute(
            f"INSERT INTO {self.collection_name} (id, vector, text, metadata) VALUES (%s, %s, %s, %s)",
            (pk, vector, text, json.dumps(metadata)),
        )
        self.conn.commit()
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
            MissingFieldError: If ID missing
            DocumentNotFoundError: If document not found
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="update", adapter="PGVector")

        pk = doc.id or extract_pk(None, **kwargs)
        if not pk:
            raise MissingFieldError("'id', '_id', or 'pk' is required for update", field="id", operation="update")

        # Get existing document
        sql = f"SELECT id, vector, text, metadata FROM {self.collection_name} WHERE id = %s"
        self.cursor.execute(sql, (pk,))
        existing = self.cursor.fetchone()

        if not existing:
            raise DocumentNotFoundError("Document not found", document_id=pk, operation="update")

        # Build update payload from doc
        prepared = prepare_item_for_storage(
            doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector),
            store_text=self.store_text,
        )
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

        params.append(pk)
        sql = f"UPDATE {self.collection_name} SET {', '.join(updates)} WHERE id = %s"
        self.cursor.execute(sql, tuple(params))
        self.conn.commit()
        self.logger.message(f"Updated document with id '{pk}'.")

        # Return refreshed document
        return self.get(pk)

    def delete(self, ids: DocIds) -> int:
        """Delete document(s) by ID.

        Args:
            ids: Single document ID or list of IDs to delete

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="delete", adapter="PGVector")

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
        self.logger.message(f"Deleted {deleted} document(s).")
        return deleted

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
            CollectionNotInitializedError: If collection not set
            InvalidFieldError: If vector missing
            DocumentExistsError: If conflict occurs
        """
        if not self.collection_name:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="bulk_create", adapter="PGVector"
            )
        if not docs:
            return []

        created_docs: List[VectorDocument] = []
        batch: List[tuple] = []

        for doc in docs:
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
                raise DocumentExistsError("Document already exists", document_id=pk, operation="bulk_create")

            vector = item.get("$vector") or item.get("vector")
            if vector is None:
                raise InvalidFieldError("Vector is required", field="vector", operation="bulk_create")

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
            CollectionNotInitializedError: If collection not set
            MissingDocumentError: If document missing (when ignore_conflicts=False)
        """
        if not self.collection_name:
            raise CollectionNotInitializedError(
                "Collection is not initialized", operation="bulk_update", adapter="PGVector"
            )
        if not docs:
            return []

        # Collect all PKs and validate
        doc_map: Dict[str, VectorDocument] = {}
        for doc in docs:
            pk = doc.pk
            if not pk:
                if not ignore_conflicts:
                    raise MissingDocumentError("Document missing ID", missing_ids=["<no_id>"], operation="bulk_update")
                continue
            doc_map[pk] = doc

        if not doc_map:
            return []

        # Fetch all existing documents in ONE query
        pks = list(doc_map.keys())
        placeholders = ",".join(["%s"] * len(pks))
        self.cursor.execute(f"SELECT id FROM {self.collection_name} WHERE id IN ({placeholders})", pks)
        existing_pks = {row[0] for row in self.cursor.fetchall()}

        # Check for missing documents
        missing = [pk for pk in pks if pk not in existing_pks]
        if missing:
            if not ignore_conflicts:
                raise MissingDocumentError("Missing documents for update", missing_ids=missing, operation="bulk_update")
            # Remove missing from processing
            for pk in missing:
                doc_map.pop(pk, None)

        # Collect documents that exist for batch upsert
        dataset: List[VectorDocument] = []
        updated_docs: List[VectorDocument] = []

        for pk, doc in doc_map.items():
            if pk in existing_pks:
                dataset.append(doc)
                updated_docs.append(doc)

        # Batch upsert all collected documents
        if dataset:
            self.upsert(dataset, batch_size=batch_size)
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
            InvalidFieldError: If any document is missing a vector
        """
        if not self.collection_name:
            raise CollectionNotInitializedError("Collection is not initialized", operation="upsert", adapter="PGVector")
        if not docs:
            return []

        batch: List[Tuple[Any, Any, Any, str]] = []
        upserted: List[VectorDocument] = []

        for doc in docs:
            item = doc.to_storage_dict(store_text=self.store_text, use_dollar_vector=self.use_dollar_vector)
            doc_id = doc.pk
            vector = item.get("$vector") or item.get("vector")
            if vector is None:
                raise InvalidFieldError("Vector is required", field="vector", operation="upsert")
            text = item.get("text") if self.store_text else None
            metadata = {k: v for k, v in item.items() if k not in ("_id", "$vector", "vector", "text")}
            metadata_json = json.dumps(metadata)
            batch.append((doc_id, vector, text, metadata_json))
            upserted.append(doc)

            if batch_size and batch_size > 0 and len(batch) >= batch_size:
                self._flush_upsert_batch(batch)
                batch.clear()

        if batch:
            self._flush_upsert_batch(batch)

        self.conn.commit()
        self.logger.message(f"Upserted {len(upserted)} document(s).")
        return upserted

    def _flush_upsert_batch(self, batch: List[Tuple[Any, Any, Any, str]]) -> None:
        """Execute a batch of upsert operations using ON CONFLICT.

        Args:
            batch: List of tuples (id, vector, text, metadata_json)
        """
        sql = f"""
        INSERT INTO {self.collection_name} (id, vector, text, metadata)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
        SET vector = EXCLUDED.vector,
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata
        """
        self.cursor.executemany(sql, batch)
