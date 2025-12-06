"""Concrete adapter for OpenAI embedding models."""

from typing import List, Optional

from openai import OpenAI

from crossvector.abc import EmbeddingAdapter
from crossvector.exceptions import InvalidFieldError, MissingConfigError, SearchError
from crossvector.settings import settings


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """
    Embedding adapter for OpenAI models.
    """

    # Known dimensions for OpenAI models
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        # Determine model: explicit > VECTOR_EMBEDDING_MODEL > default
        model_name = model_name or settings.VECTOR_EMBEDDING_MODEL or "text-embedding-3-small"
        # Validate model and get its default dimension
        if model_name not in self._DIMENSIONS:
            raise InvalidFieldError(
                "Unknown embedding dimension",
                field="model_name",
                value=model_name,
                expected=list(self._DIMENSIONS.keys()),
            )
        # Use model's default dimension if dim not provided
        model_dim = dim or self._DIMENSIONS[model_name]
        super().__init__(model_name=model_name, dim=model_dim)
        self._client: OpenAI | None = None
        self.logger.message(f"OpenAIEmbeddingAdapter initialized with model '{model_name}', dim={self._dim}.")

    @property
    def client(self) -> OpenAI:
        """
        Lazily initializes and returns the OpenAI client.
        Raises ValueError if the API key is not configured.
        """
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise MissingConfigError(
                    "API key not configured",
                    config_key="OPENAI_API_KEY",
                )
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the OpenAI API.
        """
        if not texts:
            return []
        try:
            # OpenAI API recommends replacing newlines
            texts = [text.replace("\n", " ") for text in texts]
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            self.logger.error(f"Failed to get embeddings from OpenAI: {e}", exc_info=True)
            raise SearchError(
                "Embedding generation failed",
                model=self.model_name,
            ) from e
