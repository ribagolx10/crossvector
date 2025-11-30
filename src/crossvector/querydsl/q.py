"""Query DSL core utilities.

This module defines the `Q` class used to compose structured filter
expressions in a backend-agnostic way. A `Q` node can be turned into a
universal dict representation and then compiled into backend-specific
"where" clauses or string expressions via the compilers.

Typical usage:

- Build filters: `Q(age__gte=18) & Q(age__lte=30)`
- Negate: `~Q(is_active__eq=True)`
- Compile: `q.to_where("pgvector")` or `q.to_expr("milvus")`
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from .compilers.base import BaseWhere

BackendType = Literal["generic", "milvus", "chromadb", "astradb", "pgvector"]


class Q:
    """Composable boolean query node.

    A `Q` instance holds leaf-level filters (e.g., `field__op=value`) or
    boolean combinations of child `Q` nodes using `$and` / `$or` connectors.

    - Use `&` to combine with logical AND.
    - Use `|` to combine with logical OR.
    - Use `~` to negate a node.

    Filter keys follow the `field__lookup` convention where lookup is one
    of: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`.

    These operators are verified to work across all supported backends:
    - Milvus, PgVector, AstraDB, ChromaDB
    """

    # Common operators - verified to work on ALL backends (Milvus, PgVector, AstraDB, Chroma)
    _COMMON_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"}

    # Supported operator mappings - only universally supported operators
    _OP_MAP = {
        "eq": "$eq",
        "ne": "$ne",
        "gt": "$gt",
        "gte": "$gte",
        "lt": "$lt",
        "lte": "$lte",
        "in": "$in",
        "nin": "$nin",
    }

    def __init__(self, negate: bool = False, **filters: Any):
        """Initialize a `Q` node.

        - negate: whether this node is negated.
        - filters: leaf-level filters using `field__lookup=value` pairs.
        """
        self.filters: Dict[str, Any] = filters
        self.children: List["Q"] = []
        self.connector = "$and"
        self.negate = negate

    def __and__(self, other: "Q") -> "Q":
        """Return a new node representing logical AND of two nodes."""
        node = Q()
        node.connector = "$and"
        node.children = [self, other]
        return node

    def __or__(self, other: "Q") -> "Q":
        """Return a new node representing logical OR of two nodes."""
        node = Q()
        node.connector = "$or"
        node.children = [self, other]
        return node

    def __invert__(self) -> "Q":
        """Return a negated copy of this node (logical NOT)."""
        q = deepcopy(self)
        q.negate = not self.negate
        return q

    def __str__(self) -> str:
        """Human-friendly string form of the universal dict representation."""
        return str(self.to_dict())

    def __repr__(self) -> str:
        return f"<Q: {self.to_dict()}>"

    # -------------------
    # Universal dict representation
    # -------------------
    def _leaf_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert leaf filters to the universal dict form.

        Returns a mapping where keys are field names (dots for nested)
        and values are dicts of universal operators (e.g., `$eq`).
        """
        result: Dict[str, Dict[str, Any]] = {}
        for key, value in self.filters.items():
            if "__" in key:
                # Split from the right to get the lookup operator
                # e.g., "info__lang__eq" -> field="info__lang", lookup="eq"
                parts = key.rsplit("__", 1)
                if len(parts) == 2:
                    field, lookup = parts
                    op = self._OP_MAP.get(lookup)
                    if op is not None:
                        # Valid lookup found - convert __ to . for nested paths
                        field_key = field.replace("__", ".")
                        result.setdefault(field_key, {})
                        result[field_key][op] = value
                        continue
                # No valid lookup operator - treat whole key as field name with implicit $eq
                field_key = key.replace("__", ".")
                result.setdefault(field_key, {})
                result[field_key]["$eq"] = value
            else:
                field_key = key.replace("__", ".")
                result.setdefault(field_key, {})
                result[field_key]["$eq"] = value
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return the universal dict representation of this node.

        - Leaves become `{field: {op: value}}` mappings.
        - Boolean combinations use `{"$and": [...]}`, `{"$or": [...]}`.
        - Negation wraps with `{"$not": node}`.
        """
        if self.children:
            node = {self.connector: [child.to_dict() for child in self.children]}
        else:
            node = self._leaf_to_dict()
        if self.negate:
            return {"$not": node}
        return node

    # -------------------
    # Backend-specific expression/dict
    # -------------------

    def _get_where_compiler(self, backend: BackendType) -> Optional[BaseWhere]:
        """Return the backend-specific where compiler, if any."""
        if backend == "milvus":
            from .compilers.milvus import milvus_where

            return milvus_where
        elif backend == "chromadb":
            from .compilers.chroma import chroma_where

            return chroma_where
        elif backend == "astradb":
            from .compilers.astradb import astradb_where

            return astradb_where
        elif backend == "pgvector":
            from .compilers.pgvector import pgvector_where

            return pgvector_where
        else:
            return None

    def to_where(self, backend: BackendType = "generic") -> Any:
        """Compile to a backend-native "where" representation.

        - For string-evaluated backends (SQL/CQL), returns a string.
        - For dict-evaluated backends (Chroma, Milvus), returns a dict.
        - For `generic`, returns the universal dict.
        """
        node = self.to_dict()
        where_compiler = self._get_where_compiler(backend)
        if where_compiler:
            return where_compiler.to_where(node)
        return node

    def to_expr(self, backend: BackendType = "generic") -> str:
        """Compile to a string expression for debugging/evaluation.

        If a backend compiler is available, uses its string formatter;
        otherwise returns `str(universal_dict)`.
        """
        node = self.to_dict()
        where_compiler = self._get_where_compiler(backend)
        if where_compiler:
            return where_compiler.to_expr(node)
        return str(node)
