"""Compiler utility functions.

Provides helpers for quoting identifiers, formatting SQL values, and
merging/normalizing filter node structures.
"""

from typing import Any, Dict, List, Tuple, Union


def normalize_where_input(where: Any) -> Dict[str, Any]:
    """Normalize Q object or dict to universal dict format.

    Args:
        where: Q object (with .to_dict() method) or dict

    Returns:
        Universal dict format ready for compilation

    Raises:
        TypeError: If input is neither Q object nor dict
    """
    if hasattr(where, "to_dict") and callable(where.to_dict):
        # Q object - convert to universal dict
        return where.to_dict()
    elif isinstance(where, dict):
        # Already a dict - return as-is
        return where
    else:
        raise TypeError(f"where parameter must be a Q object or dict, got {type(where).__name__}")


def merge_field_conditions(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a dict like {"$and": [ {"age":{"$gte":18}}, {"age":{"$lte":30}} , ... ]}
    into the original structure (we keep as-is), but this helper can be used
    by compilers to coalesce same-field ops when beneficial.
    """
    # For simplicity, return node unchanged here. Compilers can merge when generating SQL/CQL.
    return node


def quote_identifier(name: str) -> str:
    """Quote SQL/CQL identifier with double quotes.

    Handles dotted field paths by quoting each segment separately.
    """
    if "." in name:
        # dotted path: leave as is, or quote each part
        return ".".join(f'"{p}"' for p in name.split("."))
    return f'"{name}"'


def format_value_sql(v: Union[None, str, int, float, List[Any], Tuple[Any, ...]]) -> str:
    """Format Python value for SQL literal embedding (basic approach).

    Use parameterized queries in production for safety.
    """
    # very simple formatter: param binding recommended instead
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(format_value_sql(x) for x in v)
        return f"({inner})"
    return str(v)
