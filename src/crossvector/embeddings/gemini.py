"""Concrete adapter for Google Gemini embedding models."""

from typing import Any, List, Optional

from crossvector.abc import EmbeddingAdapter
from crossvector.exceptions import InvalidFieldError, MissingConfigError, SearchError
from crossvector.settings import settings as api_settings


class GeminiEmbeddingAdapter(EmbeddingAdapter):
    """
    Embedding adapter for Google Gemini models.
    Supports text-embedding-004 and gemini-embedding-001 with dynamic dimensionality.
    """

    # Default dimensions for Gemini models (when dim is not specified)
    _DEFAULT_DIMENSIONS = {
        "text-embedding-004": 768,
        "text-embedding-005": 768,
        "text-multilingual-embedding-002": 768,
        "embedding-001": 768,
        "gemini-embedding-001": 1536,  # Default optimized to 1536, supports up to 3072
        # Full model names
        "models/text-embedding-004": 768,
        "models/text-embedding-005": 768,
        "models/text-multilingual-embedding-002": 768,
        "models/embedding-001": 768,
    }

    # Valid output dimensions for gemini-embedding-001
    _VALID_DIMENSIONS_GEMINI_001 = [768, 1536, 3072]

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        task_type: str = "retrieval_document",
        dim: Optional[int] = None,
    ):
        """
        Initialize Gemini embedding adapter.

        Args:
            model_name: The Gemini model to use for embeddings.
                - "gemini-embedding-001": State-of-the-art performance across English, multilingual and code tasks.
                  It unifies the previously specialized models like text-embedding-005 and text-multilingual-embedding-002
                  and achieves better performance in their respective domains.
                  Max 3072 dims (default 768). 2048 tokens context. Supported text languages.
                - "text-embedding-005": Specialized in English and code tasks.
                  Max 768 dims. 2048 tokens context. English only.
                - "text-multilingual-embedding-002": Specialized in multilingual tasks.
                  Max 768 dims. 2048 tokens context. Supported text languages.
                - "text-embedding-004": Legacy English model. 768 dims.
            api_key: Google API key (defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var)
            task_type: Task type for embedding:
                - retrieval_document: For storing documents
                - retrieval_query: For search queries
                - semantic_similarity: For similarity comparison
                - classification: For classification tasks
            dim: Output dimension (primarily for gemini-embedding-001)
                - None: Use default (768 for most models)
                - 768, 1536, or 3072: Supported by gemini-embedding-001
        """
        # Determine model: explicit > VECTOR_EMBEDDING_MODEL > default
        model_name = model_name or api_settings.VECTOR_EMBEDDING_MODEL or "gemini-embedding-001"

        # Normalize model name
        normalized_model = model_name
        if not normalized_model.startswith("models/"):
            normalized_model = f"models/{normalized_model}"

        # Determine embedding dimension
        resolved_dim = dim
        if dim is not None:
            # User specified dimension
            if "gemini-embedding-001" in normalized_model:
                if dim not in self._VALID_DIMENSIONS_GEMINI_001:
                    raise InvalidFieldError(
                        "Invalid dim for gemini-embedding-001",
                        field="dim",
                        value=dim,
                        expected=self._VALID_DIMENSIONS_GEMINI_001,
                    )
            else:
                # Other models don't support dynamic dimensionality
                import logging

                logging.warning(f"dim is only supported for gemini-embedding-001. Using default for {normalized_model}")
                resolved_dim = self._DEFAULT_DIMENSIONS.get(
                    normalized_model, self._DEFAULT_DIMENSIONS.get(model_name, 1536)
                )
        else:
            # Use default dimension
            resolved_dim = self._DEFAULT_DIMENSIONS.get(
                normalized_model, self._DEFAULT_DIMENSIONS.get(model_name, 1536)
            )

        # Initialize parent with resolved dim
        super().__init__(model_name=model_name, dim=resolved_dim)

        self._client = None
        self._api_key = api_key or api_settings.GEMINI_API_KEY
        self.task_type = task_type
        self.model_name = normalized_model

        self.logger.message(
            f"GeminiEmbeddingAdapter initialized: model={self.model_name}, dim={self._dim}, task_type={self.task_type}"
        )

    @property
    def client(self) -> Any:
        """
        Lazily initializes and returns the Gemini client.
        """
        if self._client is None:
            if not self._api_key:
                raise MissingConfigError(
                    "API key not configured",
                    config_key="GEMINI_API_KEY",
                )
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
                self.logger.message("Google Generative AI client initialized successfully.")
            except ImportError:
                raise MissingConfigError(
                    "Required package not installed",
                    config_key="google-genai",
                    suggestion="pip install google-genai",
                )
        return self._client

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the Gemini API.

        Args:
            texts: A list of strings to embed

        Returns:
            A list of embedding vectors
        """
        if not texts:
            return []

        try:
            from google.genai import types

            results = []
            # Process texts individually
            for text in texts:
                # Build config object to ensure dimensionality is passed explicitly
                config = types.EmbedContentConfig(task_type=self.task_type, output_dimensionality=self.dim)
                # Call API
                result = self.client.models.embed_content(model=self.model_name, contents=text, config=config)

                # Extract embedding
                embedding = result.embeddings[0].values
                results.append(embedding)

            self.logger.message(
                f"Generated {len(results)} embeddings using {self.model_name} "
                f"(dimension={len(results[0]) if results else 'N/A'})"
            )
            return results

        except Exception as e:
            self.logger.error(f"Failed to get embeddings from Gemini: {e}", exc_info=True)
            raise SearchError(
                "Embedding generation failed",
                model=self.model_name,
                task_type=self.task_type,
            ) from e
