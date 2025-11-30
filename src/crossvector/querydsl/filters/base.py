from typing import Any, Dict, Union
from abc import ABC, abstractmethod

__all__ = ("BaseWhere",)

class BaseWhere(ABC):
    
    @abstractmethod
    def where(self, node: Dict[str, Any]) -> Any:
        """Convert a Where/Q node into backend-specific WHERE representation."""
        raise NotImplementedError
