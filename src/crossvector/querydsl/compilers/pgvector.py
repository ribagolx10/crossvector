"""PostgreSQL / pgvector where compiler.

Transforms universal Q node dicts into SQL WHERE clauses suitable for
PostgreSQL queries.

PgVector (PostgreSQL with JSONB) supports:
- Comparison: =, !=, >, <, >=, <= (with type casting for JSONB)
- Range: IN, NOT IN
- String: LIKE, ILIKE
- JSONB: @>, ?, ?&, ?| for JSON operations
- Logical: AND, OR, NOT

Limitations:
- Numeric comparisons require explicit type casting (text > numeric fails)
- Array $contains needs JSONB array contains operator
- Nested fields use -> and ->> operators
"""

from typing import Any, Dict, List, Union

from crossvector.exceptions import InvalidFieldError

from .base import BaseWhere
from .utils import format_value_sql, normalize_where_input, quote_identifier

__all__ = (
    "PgVectorWhereCompiler",
    "pgvector_where",
)


class PgVectorWhereCompiler(BaseWhere):
    """Compile universal query nodes into PostgreSQL WHERE clauses.

    Capabilities:
    - SUPPORTS_NESTED: True (via JSONB -> and ->> operators)
    - REQUIRES_VECTOR: False (metadata-only search supported)
    - REQUIRES_AND_WRAPPER: False (multiple fields use implicit AND)
    """

    # Capability flags
    SUPPORTS_NESTED = True  # JSONB supports nested fields
    REQUIRES_VECTOR = False  # Can search metadata-only
    REQUIRES_AND_WRAPPER = False  # Multiple fields use implicit AND

    # Operator mapping from universal to PostgreSQL SQL syntax
    _OP_MAP = {
        "$eq": "=",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
        "$in": "IN",
        "$nin": "NOT IN",
    }

    def to_where(self, where: Union[Dict[str, Any], Any]) -> str:
        """Convert Q object or universal dict to SQL WHERE clause.

        Args:
            where: Q object or universal dict format

        Returns:
            SQL WHERE clause string
        """
        node = normalize_where_input(where)
        return self._node_to_expr(node)

    def to_expr(self, node: Dict[str, Any]) -> str:
        """Convert universal node to SQL WHERE clause (same as to_where)."""
        return self._node_to_expr(node)

    def _node_to_expr(self, node: Dict[str, Any]) -> str:
        """Recursively transform node into SQL WHERE clause."""
        if "$and" in node:
            return " AND ".join(self._node_to_expr(x) for x in node["$and"])
        if "$or" in node:
            return " OR ".join(self._node_to_expr(x) for x in node["$or"])
        if "$not" in node:
            return "NOT (" + self._node_to_expr(node["$not"]) + ")"
        parts: List[str] = []
        base_columns = {"id", "vector", "text", "metadata"}
        for field, expr in node.items():
            # Route non-base fields through JSONB metadata extraction
            if field in base_columns:
                ident = quote_identifier(field)
            else:
                # Support nested paths using JSONB path operators.
                # For dot-paths like "info.lang", use metadata #>> '{info,lang}' to get text
                if "." in field:
                    path_elems = ",".join(p.replace("'", "''") for p in field.split("."))
                    ident_text = f"metadata #>> '{{{path_elems}}}'"
                else:
                    # Single-level key
                    ident_text = f"metadata->>'{field}'"
                ident = ident_text
            for op, val in expr.items():
                if op not in self._OP_MAP:
                    raise InvalidFieldError(
                        field=field,
                        operation="search",
                        message=f"Operator {op} is not supported. Supported: {', '.join(sorted(self._OP_MAP.keys()))}",
                    )
                sql_op = self._OP_MAP[op]
                # Decide whether to cast the left-hand side to numeric for numeric comparisons
                cast_numeric = False
                if op in {"$gt", "$gte", "$lt", "$lte"}:
                    cast_numeric = True
                elif op in {"$eq", "$ne"} and isinstance(val, (int, float)):
                    cast_numeric = True
                elif (
                    op in {"$in", "$nin"}
                    and isinstance(val, (list, tuple))
                    and all(isinstance(x, (int, float)) for x in val)
                ):
                    cast_numeric = True
                lhs = f"({ident})::numeric" if cast_numeric else ident
                parts.append(f"{lhs} {sql_op} {format_value_sql(val)}")
        return " AND ".join(parts)


pgvector_where = PgVectorWhereCompiler()
