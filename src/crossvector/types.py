"""Type aliases for crossvector package.

This module provides reusable type definitions to ensure consistency
across the codebase and improve code readability.
"""

from typing import Any, Dict, Sequence, Union

from .schema import VectorDocument

# Document input types - flexible input for create/update operations
Doc = Union[VectorDocument, Dict[str, Any], str]

# Primary key types - single or multiple document identifiers
DocId = str
DocIds = Union[str, Sequence[str]]
