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
    ConnectionError,
    DatabaseNotFoundError,
    DocumentExistsError,
    DocumentNotFoundError,
    DoesNotExist,
    InvalidFieldError,
    MissingConfigError,
    MissingDocumentError,
    MissingFieldError,
    MultipleObjectsReturned,
)
from crossvector.querydsl.compilers.pgvector import PgVectorWhereCompiler, pgvector_where
from crossvector.schema import VectorDocument
from crossvector.settings import settings as api_settings
from crossvector.utils import (
    apply_update_fields,
    extract_pk,
    prepare_item_for_storage,
)


class PgVectorAdapter(VectorDBAdapter):
    """Vector database adapter for PostgreSQL with pgvector extension.

    Provides a high-level interface for vector operations using PostgreSQL's
    pgvector extension. Supports flexible primary key types and automatic
    schema migration when PK mode changes.

    Attributes:
        collection_name: Name of the active collection (table)
        dim: Dimension of vector embeddings
        store_text: Whether to store original text with vectors
    """

    _cursor: Any = None
    use_dollar_vector: bool = False
    where_compiler: PgVectorWhereCompiler = pgvector_where
    supports_metadata_only: bool = True  # PGVector supports JSONB filtering without vector

    @property
    def client(self) -> Any:
        """Lazily initialize and return the PostgreSQL connection.

        Returns:
            Active psycopg2 connection instance

        Raises:
            psycopg2.Error: If connection fails
        """
        if self._client is None:
            # Require explicit VECTOR_COLLECTION_NAME; avoid falling back to system 'postgres'
            target_db = api_settings.VECTOR_COLLECTION_NAME
            if not target_db:
                raise MissingConfigError(
                    "VECTOR_COLLECTION_NAME is not set. Set it via environment variable or .env file (e.g. VECTOR_COLLECTION_NAME=vector_db). Refusing to use system 'postgres' database to avoid accidental writes.",
                    config_key="VECTOR_COLLECTION_NAME",
                    adapter="PGVector",
                    hint="Add VECTOR_COLLECTION_NAME to your .env then reinitialize the engine.",
                )
            user = api_settings.PGVECTOR_USER or "postgres"
            password = api_settings.PGVECTOR_PASSWORD or "postgres"
            host = api_settings.PGVECTOR_HOST or "localhost"
            port = api_settings.PGVECTOR_PORT or "5432"
            try:
                self._client = psycopg2.connect(
                    dbname=target_db,
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                )
                self.logger.message("PostgreSQL connection established (db=%s).", target_db)
            except psycopg2.OperationalError as e:
                msg = str(e)
                if "does not exist" in msg and target_db:
                    # Attempt automatic database creation using maintenance 'postgres'
                    self.logger.message("Database '%s' missing. Attempting creation...", target_db)
                    try:
                        admin_conn = psycopg2.connect(
                            dbname="postgres",
                            user=user,
                            password=password,
                            host=host,
                            port=port,
                        )
                        admin_conn.autocommit = True
                        cur = admin_conn.cursor()
                        try:
                            cur.execute(f"CREATE DATABASE {target_db}")
                            self.logger.message("Database '%s' created successfully.", target_db)
                        finally:
                            cur.close()
                            admin_conn.close()
                        # Re-attempt connection to newly created database
                        self._client = psycopg2.connect(
                            dbname=target_db,
                            user=user,
                            password=password,
                            host=host,
                            port=port,
                        )
                        self.logger.message("PostgreSQL connection established after creation (db=%s).", target_db)
                    except Exception as ce:
                        # Surface a more specific error if creation failed
                        raise DatabaseNotFoundError(
                            "Could not auto-create database. Ensure the role has CREATEDB privilege or create it manually: CREATE DATABASE {target_db};",
                            database=target_db,
                            adapter="PGVector",
                            user=user,
                            host=host,
                            port=port,
                            original_error=str(ce),
                            hint=f"Login with a superuser and run: CREATE DATABASE {target_db}; then retry.",
                        ) from ce
                else:
                    # Re-raise as a structured ConnectionError with context
                    raise ConnectionError(
                        "PostgreSQL connection failed",
                        database=target_db,
                        adapter="PGVector",
                        host=host,
                        port=port,
                        user=user,
                        original_error=msg,
                    ) from e
        return self._client

    @property
    def cursor(self) -> Any:
        """Lazily initialize and return a RealDictCursor.

        Returns:
            Active psycopg2 RealDictCursor instance
        """
        if self._cursor is None:
            self._cursor = self.client.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return self._cursor

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def initialize(
        self,
        collection_name: str,
        dim: int,
        metric: str = VectorMetric.COSINE,
        store_text: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the database and ensure the collection is ready.

        Args:
            collection_name: Name of the collection (table) to use/create
            dim: Dimension of the vector embeddings
            metric: Distance metric ('cosine', 'euclidean', 'dot_product')
            store_text: Whether to store original text content
            **kwargs: Additional configuration options
        """
        self.store_text = store_text if store_text is not None else api_settings.VECTOR_STORE_TEXT
        # Use get_or_create_collection to ensure table exists with proper schema
        self.get_or_create_collection(collection_name, dim, metric)
        self.logger.message(
            f"PGVector initialized: collection='{collection_name}', "
            f"dimension={dim}, metric={metric}, store_text={self.store_text}"
        )

    def add_collection(self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE) -> str:
        """Create a new pgvector table.

        Args:
            collection_name: Name of the table to create
            dim: Vector embedding dimension
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
        self.dim = dim
        if not hasattr(self, "store_text"):
            self.store_text = True

        desired_int64 = (api_settings.PRIMARY_KEY_MODE or "uuid").lower() == "int64"
        pk_type = "BIGINT" if desired_int64 else "VARCHAR(255)"

        # Ensure pgvector extension installed
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.client.commit()
            self.logger.message("pgvector extension ensured (CREATE EXTENSION IF NOT EXISTS vector).")
        except Exception:
            self.client.rollback()
            raise

        create_table_sql = f"""
        CREATE TABLE {collection_name} (
            id {pk_type} PRIMARY KEY,
            vector vector({dim}),
            text TEXT,
            metadata JSONB
        );
        """
        self.cursor.execute(create_table_sql)
        self.client.commit()
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

    def get_or_create_collection(self, collection_name: str, dim: int, metric: str = VectorMetric.COSINE) -> str:
        """Get or create the underlying pgvector table.

        Ensures the table exists with proper vector configuration and PK type.
        If PK type doesn't match PRIMARY_KEY_MODE, the table is dropped and recreated.

        Args:
            collection_name: Name of the table
            dim: Vector embedding dimension
            metric: Distance metric for vector search

        Returns:
            Collection name (table name)
        """
        self.collection_name = collection_name
        self.dim = dim
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
                self.client.commit()

        # Ensure pgvector extension installed before creating table
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.client.commit()
            self.logger.message("pgvector extension ensured (CREATE EXTENSION IF NOT EXISTS vector).")
        except Exception:
            self.client.rollback()
            raise

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {collection_name} (
            id {pk_type} PRIMARY KEY,
            vector vector({dim}),
            text TEXT,
            metadata JSONB
        );
        """
        self.cursor.execute(create_table_sql)
        self.client.commit()
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
        self.client.commit()
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
        self.client.commit()
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
        select_fields = ["id", "vector"]
        if vector is not None:
            select_fields.append("vector <-> %s::vector AS distance")
        if fields is None or "text" in fields:
            select_fields.append("text")
        if fields is None or "metadata" in fields:
            select_fields.append("metadata")

        # Build WHERE clause for metadata filter
        where_clause = ""
        params: List[Any] = []
        if vector is not None:
            params.append(vector)

        if where is not None:
            # Compiler handles both Q objects and dicts
            where_clause = " WHERE " + self.where_compiler.to_where(where)

        if limit is None:
            limit = api_settings.VECTOR_SEARCH_LIMIT
        params.extend([limit, offset])
        # Order by distance (lower distance = higher similarity); we map to score later
        order_clause = " ORDER BY distance ASC" if vector is not None else ""
        sql = f"SELECT {', '.join(select_fields)} FROM {self.collection_name}{where_clause}{order_clause} LIMIT %s OFFSET %s"
        try:
            self.cursor.execute(sql, tuple(params))
        except Exception as exec_err:
            # Ensure aborted transaction does not poison subsequent operations
            try:
                self.client.rollback()
            except Exception:
                pass
            raise exec_err
        results = self.cursor.fetchall()

        # Convert to VectorDocument instances
        vector_docs = []
        for r in results:
            metadata_block = r.get("metadata", {}) or {}
            # Map distance to similarity score if present (assume cosine distance range [0,1])
            if vector is not None and "distance" in r and isinstance(r["distance"], (int, float)):
                dist = r["distance"]
                score = 1.0 - dist if 0.0 <= dist <= 1.0 else float(dist)
                if "score" not in metadata_block and isinstance(metadata_block, dict):
                    metadata_block["score"] = score
            raw_vec = r.get("vector")
            vec: List[float] = []
            if raw_vec is not None:
                if isinstance(raw_vec, list):
                    vec = raw_vec
                elif hasattr(raw_vec, "tolist"):
                    try:
                        vec = list(raw_vec.tolist())
                    except Exception:
                        vec = []
            doc_dict = {"_id": r["id"], "vector": vec, "metadata": metadata_block}
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
            raw_vec = result.get("vector")
            parsed_vec: List[float] = []
            if isinstance(raw_vec, list):
                parsed_vec = raw_vec
            elif isinstance(raw_vec, str):
                # Parse pgvector textual representation: '[v1,v2,...]'
                trimmed = raw_vec.strip().strip("[]")
                if trimmed:
                    try:
                        parsed_vec = [float(x) for x in trimmed.split(",") if x.strip()]
                    except Exception:
                        parsed_vec = []
            elif hasattr(raw_vec, "tolist"):
                try:
                    parsed_vec = list(raw_vec.tolist())
                except Exception:
                    parsed_vec = []
            doc_data = {
                "_id": result["id"],
                "vector": parsed_vec,
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
        self.client.commit()
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
            # No changes to make; parse vector for consistency
            raw_vec = existing.get("vector")
            parsed_vec: List[float] = []
            if isinstance(raw_vec, list):
                parsed_vec = raw_vec
            elif isinstance(raw_vec, str):
                trimmed = raw_vec.strip().strip("[]")
                if trimmed:
                    try:
                        parsed_vec = [float(x) for x in trimmed.split(",") if x.strip()]
                    except Exception:
                        parsed_vec = []
            elif hasattr(raw_vec, "tolist"):
                try:
                    parsed_vec = list(raw_vec.tolist())
                except Exception:
                    parsed_vec = []
            doc_data = {
                "_id": existing["id"],
                "vector": parsed_vec,
                "text": existing["text"],
                "metadata": existing["metadata"] or {},
            }
            return VectorDocument.from_kwargs(**doc_data)

        params.append(pk)
        sql = f"UPDATE {self.collection_name} SET {', '.join(updates)} WHERE id = %s"
        self.cursor.execute(sql, tuple(params))
        self.client.commit()
        self.logger.message(f"Updated document with id '{pk}'.")

        # Return refreshed document
        return self.get(pk)

    def delete(self, *args) -> int:
        """Delete documents by ID.

        Args:
                *args: One or more document IDs (varargs) to delete

        Returns:
            Number of documents deleted

        Raises:
            CollectionNotInitializedError: If collection is not initialized
        """
        if not self._client:
            raise CollectionNotInitializedError("Connection is not initialized", operation="delete", adapter="PgVector")

        if not args:
            return 0

        if len(args) == 1:
            sql = f"DELETE FROM {self.collection_name} WHERE id = %s"
            self.cursor.execute(sql, (args[0],))
        else:
            sql = f"DELETE FROM {self.collection_name} WHERE id = ANY(%s)"
            self.cursor.execute(sql, (list(args),))

        self.client.commit()
        deleted = self.cursor.rowcount
        self.logger.message(f"Deleted {deleted} documents.")
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

        self.client.commit()
        self.logger.message(f"Bulk created {len(created_docs)} documents.")
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
        # RealDictCursor returns dicts, not tuples
        existing_pks = {row["id"] for row in self.cursor.fetchall()}

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

        for pk, doc in doc_map.items():
            if pk in existing_pks:
                dataset.append(doc)

        # Batch upsert all collected documents and return the upserted results
        if dataset:
            return self.upsert(dataset, batch_size=batch_size)
        return []

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

        self.client.commit()
        self.logger.message(f"Upserted {len(upserted)} documents.")
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
