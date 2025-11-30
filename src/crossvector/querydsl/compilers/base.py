"""Base compiler interface.

Defines the abstract contract all backend-specific where compilers must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

__all__ = ("BaseWhere",)


class BaseWhere(ABC):
    """Abstract base class for where clause compilers.

    Subclasses implement `to_where` and `to_expr` to produce backend-specific
    filter structures.
    """

    @abstractmethod
    def to_where(self, node: Dict[str, Any]) -> Any:
        """
        Convert Q/Where node into backend-native filter representation.
        - dict for Chroma, AstraDB (MongoDB-like syntax)
        - string for Milvus, PostgreSQL (expression/SQL evaluation)
        """
        raise NotImplementedError

    @abstractmethod
    def to_expr(self, node: Dict[str, Any]) -> str:
        """Convert a Where/Q node into a string expression for evaluation backends."""
        raise NotImplementedError
