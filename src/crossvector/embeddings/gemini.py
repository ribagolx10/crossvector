"""Concrete adapter for Google Gemini embedding models."""

import logging
import os
from typing import Any, Dict, List, Optional

from crossvector.abc import EmbeddingAdapter

log = logging.getLogger(__name__)


class GeminiEmbeddingAdapter(EmbeddingAdapter):
    """
    Embedding adapter for Google Gemini models.
    Supports text-embedding-004 and gemini-embedding-001 with dynamic dimensionality.
    """

    # Default dimensions for Gemini models (when output_dimensionality is not specified)
    _DEFAULT_DIMENSIONS = {
        "text-embedding-004": 768,
        "text-embedding-005": 768,
        "text-multilingual-embedding-002": 768,
        "embedding-001": 768,
        "gemini-embedding-001": 768,  # Default optimized to 768, supports up to 3072
        # Full model names
        "models/text-embedding-004": 768,
        "models/text-embedding-005": 768,
        "models/text-multilingual-embedding-002": 768,
        "models/embedding-001": 768,
        "models/gemini-embedding-001": 768,
    }

    # Valid output dimensions for gemini-embedding-001
    _VALID_DIMENSIONS_GEMINI_001 = [768, 1536, 3072]

    def __init__(
        self,
        model_name: str = "models/gemini-embedding-001",
        api_key: Optional[str] = None,
        task_type: str = "retrieval_document",
        output_dimensionality: Optional[int] = None,
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
            output_dimensionality: Output dimension (primarily for gemini-embedding-001)
                - None: Use default (768 for most models)
                - 768, 1536, or 3072: Supported by gemini-embedding-001
        """
        super().__init__(model_name)
        self._client = None
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality

        # Normalize model name
        if not model_name.startswith("models/"):
            self.model_name = f"models/{model_name}"

        # Determine embedding dimension
        if output_dimensionality is not None:
            # User specified dimension
            if "gemini-embedding-001" in self.model_name:
                if output_dimensionality not in self._VALID_DIMENSIONS_GEMINI_001:
                    raise ValueError(
                        f"Invalid output_dimensionality {output_dimensionality} for gemini-embedding-001. "
                        f"Valid options: {self._VALID_DIMENSIONS_GEMINI_001}"
                    )
                self._embedding_dimension = output_dimensionality
            else:
                # Other models don't support dynamic dimensionality
                log.warning(
                    f"output_dimensionality is only supported for gemini-embedding-001. Ignoring for {self.model_name}"
                )
                self._embedding_dimension = self._DEFAULT_DIMENSIONS.get(
                    self.model_name, self._DEFAULT_DIMENSIONS.get(model_name, 768)
                )
        else:
            # Use default dimension
            self._embedding_dimension = self._DEFAULT_DIMENSIONS.get(
                self.model_name, self._DEFAULT_DIMENSIONS.get(model_name, 768)
            )

        log.info(
            f"GeminiEmbeddingAdapter initialized: model={self.model_name}, "
            f"dimension={self._embedding_dimension}, task_type={self.task_type}"
        )

    @property
    def client(self) -> Any:
        """
        Lazily initializes and returns the Gemini client.
        """
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY is not set. "
                    "Please configure it in your .env file or pass it to the constructor."
                )
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
                log.info("Google Generative AI client initialized successfully.")
            except ImportError:
                raise ImportError("google-genai package is not installed. Install it with: pip install google-genai")
        return self._client

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

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
                # Build config
                config_params: Dict[str, Any] = {"task_type": self.task_type}

                # Add output_dimensionality if specified (only for gemini-embedding-001)
                if self.output_dimensionality is not None and "gemini-embedding-001" in self.model_name:
                    config_params["output_dimensionality"] = self.output_dimensionality

                config = types.EmbedContentConfig(**config_params)

                # Call API
                result = self.client.models.embed_content(model=self.model_name, contents=text, config=config)

                # Extract embedding
                embedding = result.embeddings[0].values
                results.append(embedding)

            log.info(
                f"Generated {len(results)} embeddings using {self.model_name} "
                f"(dimension={len(results[0]) if results else 'N/A'})"
            )
            return results

        except Exception as e:
            log.error(f"Failed to get embeddings from Gemini: {e}", exc_info=True)
            raise
