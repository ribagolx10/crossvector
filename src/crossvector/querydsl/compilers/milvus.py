"""Milvus-specific where compiler.

Transforms universal Q node dicts into Milvus filter strings (boolean expressions).

Milvus supports:
- Comparison: ==, !=, >, <, >=, <=
- Range: IN, LIKE
- Logical: AND, OR, NOT
- Null checks: IS NULL, IS NOT NULL
- Arithmetic: +, -, *, /, %, **

Limitations:
- Requires vector for all searches (no metadata-only)
- Array $contains may not work as expected (JSON array support varies)
- Nested fields stored flattened with dot notation
"""

from typing import Any, Dict, Union

from crossvector.exceptions import InvalidFieldError

from .base import BaseWhere
from .utils import normalize_where_input

__all__ = (
    "MilvusWhereCompiler",
    "milvus_where",
)


class MilvusWhereCompiler(BaseWhere):
    """Compile universal query nodes into Milvus boolean filter expressions.

    Capabilities:
    - SUPPORTS_NESTED: False (flattened metadata with dot notation, may not work correctly)
    - REQUIRES_VECTOR: True (metadata-only search not supported)
    - REQUIRES_AND_WRAPPER: False (implicit AND with multiple fields)
    """

    # Capability flags
    SUPPORTS_NESTED = False  # Flattened metadata, nested may not work as expected
    REQUIRES_VECTOR = True  # Milvus requires vector for all searches
    REQUIRES_AND_WRAPPER = False  # Multiple fields use implicit AND

    # Operator mapping from universal to Milvus syntax
    _OP_MAP = {
        "$eq": "==",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
        "$in": "IN",
        "$nin": "NOT IN",
    }

    def to_where(self, where: Union[Dict[str, Any], Any]) -> str:
        """Convert Q object or universal dict to Milvus filter string.

        Args:
            where: Q object or universal dict format

        Returns:
            Milvus boolean expression string
        """
        node = normalize_where_input(where)
        return self._node_to_expr(node)

    def to_expr(self, node: Dict[str, Any]) -> str:
        """Convert universal node to Milvus filter string (same as to_where)."""
        return self._node_to_expr(node)

    def _node_to_expr(self, node: Dict[str, Any]) -> str:
        """Recursively transform node into Milvus filter expression."""
        if "$and" in node:
            return " && ".join(self._node_to_expr(x) for x in node["$and"])
        if "$or" in node:
            return " || ".join(self._node_to_expr(x) for x in node["$or"])
        if "$not" in node:
            return "!(" + self._node_to_expr(node["$not"]) + ")"

        parts = []
        for field, expr in node.items():
            for op, val in expr.items():
                if op not in self._OP_MAP:
                    raise InvalidFieldError(
                        field=field,
                        operation="search",
                        message=f"Operator {op} is not supported. Supported: {', '.join(sorted(self._OP_MAP.keys()))}",
                    )

                # Map non-root fields to JSON path inside 'metadata'
                target_field = field if field in ("id", "text") else f"metadata['{field}']"
                parts.append(f"{target_field} {self._OP_MAP[op]} {repr(val)}")

        return " && ".join(parts)


milvus_where = MilvusWhereCompiler()
