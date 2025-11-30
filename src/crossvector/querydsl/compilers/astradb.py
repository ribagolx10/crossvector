"""AstraDB Data API where compiler.

Transforms universal Q node dicts into AstraDB Data API filter format.
AstraDB uses MongoDB-like filter syntax with operators like $eq, $gt, $in, etc.

AstraDB Data API supports:
- Comparison: $eq, $ne, $gt, $gte, $lt, $lte
- Range: $in, $nin
- Logical: $and, $or (but NOT $not)
- Nested fields via dot notation

Limitations:
- $not operator not supported by AstraDB Data API
- $contains for arrays may need special handling
- $in requires proper array type
"""

from typing import Any, Dict, Union

from ...exceptions import InvalidFieldError
from .base import BaseWhere
from .utils import normalize_where_input

__all__ = (
    "AstraDBWhereCompiler",
    "astradb_where",
)


class AstraDBWhereCompiler(BaseWhere):
    """Compile universal query nodes into AstraDB Data API filter dicts.

    AstraDB Data API uses MongoDB-like filter syntax, so we can directly
    return the universal dict format with minimal transformation.

    Capabilities:
    - SUPPORTS_NESTED: True (via dot notation)
    - REQUIRES_VECTOR: False (metadata-only search supported)
    - REQUIRES_AND_WRAPPER: False (multiple fields use implicit AND)
    """

    # Capability flags
    SUPPORTS_NESTED = True  # Supports nested fields via dot notation
    REQUIRES_VECTOR = False  # Can search metadata-only
    REQUIRES_AND_WRAPPER = False  # Multiple fields use implicit AND

    # Operator mapping (AstraDB uses same syntax as universal format)
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
        """Convert Q object or universal dict to AstraDB Data API filter dict.

        AstraDB uses MongoDB-like syntax, so the universal dict format
        is already compatible. We return it directly or with minor adjustments.

        Args:
            where: Q object or universal dict format

        Returns:
            AstraDB Data API filter dict
        """
        node = normalize_where_input(where)
        return self._node_to_dict(node)

    def to_expr(self, node: Dict[str, Any]) -> str:
        """Convert universal node to string representation for debugging."""
        return str(self._node_to_dict(node))

    def _node_to_dict(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Transform node into AstraDB Data API filter format.

        Since AstraDB uses MongoDB-like filter syntax, the universal dict
        format is already compatible. We just need to handle any special cases.
        """
        # Handle logical operators
        if "$and" in node:
            return {"$and": [self._node_to_dict(n) for n in node["$and"]]}
        if "$or" in node:
            return {"$or": [self._node_to_dict(n) for n in node["$or"]]}
        if "$not" in node:
            raise InvalidFieldError(
                field="$not", operation="search", message="Operator $not is not supported by AstraDB"
            )

        # Handle field-level filters
        result = {}
        for field, expr in node.items():
            result[field] = {}
            for op, val in expr.items():
                if op not in self._OP_MAP:
                    raise InvalidFieldError(
                        field=field,
                        operation="search",
                        message=f"Operator {op} is not supported. Supported: {', '.join(sorted(self._OP_MAP.keys()))}",
                    )
                result[field][op] = val

        return result


astradb_where = AstraDBWhereCompiler()
