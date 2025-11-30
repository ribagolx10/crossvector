from typing import Any, Dict
from .base import BaseWhere

__all__ = ("ChromaWhere", "chroma_where",)

class ChromaWhere(BaseWhere):

    _OP_MAP = {
        "$eq": "==",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
        "$in": "in",
        "$nin": "not in",
        "$contains": "in",
    }
    def where(self, node: Dict[str, Any]) -> str:
        return self._node_to_expr(node)

    def _node_to_expr(self, node):
        if "$and" in node:
            parts = [self._node_to_expr(n) for n in node["$and"]]
            return "(" + " and ".join(parts) + ")"
        if "$or" in node:
            parts = [self._node_to_expr(n) for n in node["$or"]]
            return "(" + " or ".join(parts) + ")"
        if "$not" in node:
            return "not (" + self._node_to_expr(node["$not"]) + ")"

        parts = []
        for field, expr in node.items():
            for op, val in expr.items():
                if op == "$contains":
                    # "'sub' in metadata_field"
                    parts.append(f"'{val}' in {field}")
                elif op in ("$in", "$nin"):
                    parts.append(f"{field} {self._OP_MAP[op]} {repr(val)}")
                else:
                    parts.append(f"{field} {self._OP_MAP[op]} {repr(val)}")
        return " and ".join(parts)


chroma_where = ChromaWhere()
