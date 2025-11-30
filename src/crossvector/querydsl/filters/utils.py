from typing import Dict, Any, Tuple, List

def merge_field_conditions(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a dict like {"$and": [ {"age":{"$gte":18}}, {"age":{"$lte":30}} , ... ]}
    into the original structure (we keep as-is), but this helper can be used
    by compilers to coalesce same-field ops when beneficial.
    """
    # For simplicity, return node unchanged here. Compilers can merge when generating SQL/CQL.
    return node

def quote_identifier(name: str) -> str:
    # safe quoting for SQL/CQL identifiers (very basic)
    if "." in name:
        # dotted path: leave as is, or quote each part
        return ".".join(f'"{p}"' for p in name.split("."))
    return f'"{name}"'

def format_value_sql(v):
    # very simple formatter: param binding recommended instead
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(format_value_sql(x) for x in v)
        return f"({inner})"
    return str(v)
