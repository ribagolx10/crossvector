"""Chroma-specific where compiler.

Transforms universal Q node dicts into Chroma's native filter structure and
string expression format.
"""

from typing import Any, Dict, Union

from crossvector.exceptions import InvalidFieldError

from .base import BaseWhere
from .utils import normalize_where_input

__all__ = (
    "ChromaWhereCompiler",
    "chroma_where",
)


class ChromaWhereCompiler(BaseWhere):
    """Compile universal query nodes into Chroma filter dicts and expressions.

    Chroma Capabilities:
    - Supports nested fields via dot-notation (e.g., 'info.lang') when metadata is flattened
    - Requires $and wrapper for multiple top-level field filters
    - Does NOT support: $not, $contains, $regex, $iregex, $icontains in get() method
    """

    # Capability flags
    SUPPORTS_NESTED = True  # Via dot-notation on flattened metadata
    REQUIRES_VECTOR = False  # Can search metadata-only
    REQUIRES_AND_WRAPPER = True  # Multiple fields must be wrapped in $and

    # Operator mapping (Chroma uses same syntax as universal format)
    _OP_MAP = {
        "$eq": "$eq",
        "$ne": "$ne",
        "$gt": "$gt",
        "$gte": "$gte",
        "$lt": "$lt",
        "$lte": "$lte",
        "$in": "$in",
        "$nin": "$nin",
    }

    def to_where(self, where: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Convert Q object or universal dict to Chroma filter dict.

        Args:
            where: Q object or universal dict format

        Returns:
            Chroma-compatible filter dict
        """
        node = normalize_where_input(where)
        return self._node_to_dict(node)

    def to_expr(self, node: Dict[str, Any]) -> str:
        """Convert universal node to Chroma string expression."""
        return self._node_to_expr(node)

    def _node_to_dict(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively transform node into Chroma's dict format.

        Raises:
            InvalidFieldError: If unsupported operators are used
        """
        if "$and" in node:
            return {"$and": [self._node_to_dict(n) for n in node["$and"]]}
        if "$or" in node:
            return {"$or": [self._node_to_dict(n) for n in node["$or"]]}
        if "$not" in node:
            raise InvalidFieldError(
                field="$not", operation="search", message="Operator $not is not supported by Chroma"
            )

        # Build per-field dicts
        field_dicts = []
        for field, expr in node.items():
            if not isinstance(expr, dict):
                field_dicts.append({field: {"$eq": expr}})
                continue
            compiled: Dict[str, Any] = {}
            for op, val in expr.items():
                if op not in self._OP_MAP:
                    raise InvalidFieldError(
                        field=field,
                        operation="search",
                        message=f"Operator {op} is not supported. Supported: {', '.join(sorted(self._OP_MAP.keys()))}",
                    )
                compiled[op] = val
            if compiled:
                field_dicts.append({field: compiled})
        # If multiple fields, wrap in $and; if one, unwrap; if none, return {}
        if not field_dicts:
            return {}
        if len(field_dicts) == 1:
            return field_dicts[0]
        return {"$and": field_dicts}

    def _node_to_expr(self, node: Dict[str, Any]) -> str:
        """Recursively transform node into Chroma's string expression."""
        return str(self._node_to_dict(node))


chroma_where = ChromaWhereCompiler()
