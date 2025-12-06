"""
This __init__.py file makes the vector_store directory a Python package
and exposes the main `VectorEngine` and schema classes for easy access.
"""

from .abc import EmbeddingAdapter, VectorDBAdapter
from .engine import VectorEngine
from .schema import VectorDocument
from .types import Doc, DocId, DocIds

__version__ = "0.2.0"

__all__ = [
    "VectorEngine",
    "EmbeddingAdapter",
    "VectorDBAdapter",
    "VectorDocument",
    "Doc",
    "DocId",
    "DocIds",
]
