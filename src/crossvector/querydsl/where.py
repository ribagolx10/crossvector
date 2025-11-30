from copy import deepcopy
from typing import Any, Dict, List


__all__ = ("Where",)

LOOKUP_MAP = {
    "eq": "$eq",
    "ne": "$ne",
    "gt": "$gt",
    "gte": "$gte",
    "lt": "$lt",
    "lte": "$lte",
    "in": "$in",
    "nin": "$nin",
    "contains": "$contains",
    "icontains": "$contains",
    "regex": "$regex",
    "iregex": "$regex",
    "startswith": "$regex",
    "endswith": "$regex",
}

def _field_to_key(field: str, nested_as_dotted: bool = True) -> str:
    # default: convert a__b__c -> "a.b.c" so engines that accept dotted path work
    if nested_as_dotted:
        return field.replace("__", ".")
    return field.split("__")  # optional: return list for nested dict building

class Where:
    def __init__(self, negate: bool = False, **filters):
        self.filters = filters  # raw kwargs like pk__in=[...], age__gte=18
        self.children: List["Q"] = []
        self.connector = "$and"
        self.negate = negate

    def __and__(self, other: "Where") -> "Where":
        node = Where()
        node.connector = "$and"
        node.children = [self, other]
        return node

    def __or__(self, other: "Where") -> "Where":
        node = Where()
        node.connector = "$or"
        node.children = [self, other]
        return node

    def __invert__(self) -> "Where":
        # Return a shallow negated Q node: keep structure, flip negate flag
        q = deepcopy(self)
        q.negate = not self.negate
        return q

    def _leaf_to_dict(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for key, value in self.filters.items():
            if "__" in key:
                field, lookup = key.split("__", 1)
                op = LOOKUP_MAP.get(lookup)
                if op is None:
                    raise ValueError(f"Unsupported lookup: {lookup}")
                field_key = _field_to_key(field)
                result.setdefault(field_key, {})
                # special handling for startswith/endswith expressed as regex
                if lookup == "startswith":
                    result[field_key][op] = f"^{value}"
                elif lookup == "endswith":
                    result[field_key][op] = f"{value}$"
                else:
                    result[field_key][op] = value
            else:
                # default equality
                field_key = _field_to_key(key)
                result.setdefault(field_key, {})
                result[field_key]["$eq"] = value
        return result

    def to_dict(self) -> Dict[str, Any]:
        if self.children:
            node = {self.connector: [child.to_dict() for child in self.children]}
        else:
            node = self._leaf_to_dict()

        if self.negate:
            return {"$not": node}
        return node
