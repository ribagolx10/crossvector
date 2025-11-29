"""Custom exceptions for CrossVector library.

This module defines all custom exceptions used throughout the library for
consistent error handling and clear error messaging.
"""

from typing import Any, Dict


# Base exception
class CrossVectorError(Exception):
    """Base exception for all CrossVector errors.

    Attributes:
        message: Error message
        details: Additional error context as key-value pairs
    """

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        """Initialize exception with message and additional details.

        Args:
            message: Human-readable error message
            **kwargs: Additional context (e.g., document_id, collection_name, field_name)
        """
        self.message = message
        self.details: Dict[str, Any] = kwargs
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message with details."""
        if not self.details:
            return self.message

        details_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
        if self.message:
            return f"{self.message} ({details_str})"
        return details_str

    def __repr__(self) -> str:
        """Return detailed representation of the exception."""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


# Document operation exceptions
class DoesNotExist(CrossVectorError):
    """Raised when a document does not exist for the given query.

    Example:
        >>> raise DoesNotExist("Document not found", document_id="doc123")
    """


class MultipleObjectsReturned(CrossVectorError):
    """Raised when multiple objects are returned for a query that expects exactly one.

    Example:
        >>> raise MultipleObjectsReturned("Expected 1 document, found 3", count=3, filter={"status": "active"})
    """


class DocumentNotFoundError(CrossVectorError):
    """Raised when a document is not found by ID or filter.

    Example:
        >>> raise DocumentNotFoundError("Document not found", document_id="doc123")
    """


class DocumentExistsError(CrossVectorError):
    """Raised when attempting to create a document with an ID that already exists.

    Example:
        >>> raise DocumentExistsError("Document already exists", document_id="doc123")
    """


class MissingDocumentError(CrossVectorError):
    """Raised when required documents are missing in batch operations.

    Example:
        >>> raise MissingDocumentError("Documents not found", missing_ids=["doc1", "doc2"])
    """


# Validation exceptions
class ValidationError(CrossVectorError):
    """Raised when document validation fails.

    Example:
        >>> raise ValidationError("Invalid document format", field="vector", expected_type="list")
    """


class MissingFieldError(ValidationError):
    """Raised when a required field is missing.

    Example:
        >>> raise MissingFieldError("Required field missing", field="id", operation="update")
    """


class InvalidFieldError(ValidationError):
    """Raised when a field has an invalid value or type.

    Example:
        >>> raise InvalidFieldError("Invalid field value", field="vector", value=None, expected="list[float]")
    """


class MismatchError(ValidationError):
    """Raised when data consistency check fails (e.g., text mismatch in get_or_create).

    Example:
        >>> raise MismatchError("Text content mismatch", provided="Hello", existing="Hi", document_id="doc123")
    """


# Configuration exceptions
class ConfigurationError(CrossVectorError):
    """Raised when configuration is invalid or missing.

    Example:
        >>> raise ConfigurationError("Invalid configuration", setting="metric", value="invalid")
    """


class MissingConfigError(ConfigurationError):
    """Raised when required configuration values are not set.

    Example:
        >>> raise MissingConfigError("Configuration not set", config_key="OPENAI_API_KEY")
    """


class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid.

    Example:
        >>> raise InvalidConfigError("Invalid config value", config_key="dimension", value=-1, expected=">0")
    """


# Collection exceptions
class CollectionError(CrossVectorError):
    """Base exception for collection-related errors.

    Example:
        >>> raise CollectionError("Collection operation failed", collection_name="my_collection")
    """


class CollectionExistsError(CollectionError):
    """Raised when attempting to create a collection that already exists.

    Example:
        >>> raise CollectionExistsError("Collection already exists", collection_name="my_collection")
    """


class CollectionNotFoundError(CollectionError):
    """Raised when a collection does not exist.

    Example:
        >>> raise CollectionNotFoundError("Collection not found", collection_name="my_collection")
    """


class CollectionNotInitializedError(CollectionError):
    """Raised when attempting operations on an uninitialized collection.

    Example:
        >>> raise CollectionNotInitializedError("Collection not initialized", operation="search")
    """


# Connection exceptions
class ConnectionError(CrossVectorError):
    """Raised when database connection fails or is not initialized.

    Example:
        >>> raise ConnectionError("Database not connected", adapter="AstraDB", endpoint="https://api.example.com")
    """


# Search exceptions
class SearchError(CrossVectorError):
    """Raised when search operation fails or has invalid parameters.

    Example:
        >>> raise SearchError("Invalid search parameters", reason="vector and where both missing")
    """


# Import exceptions
class DependencyError(CrossVectorError):
    """Raised when a required dependency is not installed.

    Example:
        >>> raise DependencyError("Required package not installed", package="google-genai", install_cmd="pip install google-genai")
    """
