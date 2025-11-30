from typing import Dict, Any
from .base import BaseWhere
from .utils import quote_identifier, format_value_sql


__all__ = ("AstraDBCompiler", "astradb_compiler",)

class AstraDBCompiler(BaseWhere):

    _OP_MAP = {
        "$eq": "=",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
        "$in": "IN",
        "$nin": "NOT IN",
        "$contains": "LIKE",  # CQL supports LIKE only in 3.x with indexing
    }
    
    def where(self, node: Dict[str, Any]) -> str:
        return self._node_to_expr(node)

    def _node_to_expr(self, node):
        if "$and" in node:
            return " AND ".join(self._node_to_expr(x) for x in node["$and"])
        if "$or" in node:
            # CQL does not support OR in WHERE; we can emulate by returning multiple WHERE clauses
            # For safety, we raise to force developer to handle client-side OR or secondary queries.
            raise NotImplementedError("CQL WHERE does not support OR. Run multiple queries or use ALLOW FILTERING (not recommended).")
        if "$not" in node:
            # CQL supports NOT only in certain contexts; we do best-effort
            return "NOT (" + self._node_to_expr(node["$not"]) + ")"

        parts = []
        for field, expr in node.items():
            ident = quote_identifier(field)
            for op, val in expr.items():
                cql_op = self._OP_MAP[op]
                if op == "$contains":
                    parts.append(f"{ident} {cql_op} '%{val}%'")
                elif op in ("$in", "$nin"):
                    parts.append(f"{ident} {cql_op} {format_value_sql(val)}")
                else:
                    parts.append(f"{ident} {cql_op} {format_value_sql(val)}")
        return " AND ".join(parts)


astradb_compiler = AstraDBCompiler()
