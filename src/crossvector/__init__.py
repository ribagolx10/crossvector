"""
This __init__.py file makes the vector_store directory a Python package
and exposes the main `VectorEngine` and schema classes for easy access.
"""

from .abc import EmbeddingAdapter, VectorDBAdapter
from .engine import VectorEngine
from .schema import VectorDocument

__version__ = "0.1.1"

__all__ = [
    "VectorEngine",
    "EmbeddingAdapter",
    "VectorDBAdapter",
    "VectorDocument",
]
