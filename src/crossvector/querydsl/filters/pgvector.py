from typing import Dict, Tuple, List, Any
from .base import BaseWhere
from .utils import quote_identifier, format_value_sql

__all__ = ("PgVectorWhere", "pgvector_where",)

class PgVectorWhere(BaseWhere):

    _OP_MAP = {
        "$eq": "=",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
        "$in": "IN",
        "$nin": "NOT IN",
        "$contains": "LIKE",
    }
    
    def where(self, node: Dict[str, Any]) -> str:
        return self._node_to_expr(node)

    def _node_to_expr(self, node):
        if "$and" in node:
            return " AND ".join(self._node_to_expr(x) for x in node["$and"])
        if "$or" in node:
            return " OR ".join(self._node_to_expr(x) for x in node["$or"])
        if "$not" in node:
            return "NOT (" + self._node_to_expr(node["$not"]) + ")"

        parts = []
        for field, expr in node.items():
            ident = quote_identifier(field)
            for op, val in expr.items():
                sql_op = self._OP_MAP[op]
                if op == "$contains":
                    parts.append(f"{ident} {sql_op} '%{val}%'")
                elif op in ("$in", "$nin"):
                    parts.append(f"{ident} {sql_op} {format_value_sql(val)}")
                else:
                    parts.append(f"{ident} {sql_op} {format_value_sql(val)}")
        return " AND ".join(parts)

pgvector_where = PgVectorWhere()
