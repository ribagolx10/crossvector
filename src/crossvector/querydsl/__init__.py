"""Query DSL module.

Exports the `Q` class for building composable, backend-agnostic filter
expressions. Compiled representations are handled by the `compilers` subpackage.
"""

from .q import Q

__all__ = ("Q",)
