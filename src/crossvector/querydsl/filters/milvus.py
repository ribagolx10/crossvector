from typing import Dict, Any
from .base import BaseWhere

__all__ = ("MilvusWhere", "milvus_where",)



class MilvusWhere(BaseWhere):
    _OP_MAP = {
        "$eq": "==",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
        "$in": "in",
        "$nin": "not in",
        "$contains": "like",  # no native substring; may vary depending on milvus metadata support
    }

    def where(self, node: Dict[str, Any]) -> str:
        return self._node_to_expr(node)

    def _node_to_expr(self, node):
        if "$and" in node:
            return " && ".join(self._node_to_expr(x) for x in node["$and"])
        if "$or" in node:
            return " || ".join(self._node_to_expr(x) for x in node["$or"])
        if "$not" in node:
            return "!(" + self._node_to_expr(node["$not"]) + ")"

        parts = []
        for field, expr in node.items():
            for op, val in expr.items():
                if op in ("$in", "$nin"):
                    parts.append(f"{field} {self._OP_MAP[op]} {repr(val)}")
                elif op == "$contains":
                    # best-effort; Milvus metadata may not support substring, depends on version
                    parts.append(f"{field} LIKE '%{val}%'")
                else:
                    parts.append(f"{field} {self._OP_MAP[op]} {repr(val)}")
        return " && ".join(parts)

milvus_where = MilvusWhere()
