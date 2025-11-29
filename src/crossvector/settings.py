"""Settings for CrossVector engine."""

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class CrossVectorSettings(BaseSettings):
    """CrossVector configuration settings."""

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Gemini
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # AstraDB
    ASTRA_DB_APPLICATION_TOKEN: Optional[str] = None
    ASTRA_DB_API_ENDPOINT: Optional[str] = None
    ASTRA_DB_COLLECTION_NAME: str = "vector_documents"

    # Milvus
    MILVUS_API_ENDPOINT: Optional[str] = None
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None

    # PGVector
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: str = "5432"
    PGVECTOR_DBNAME: str = "postgres"
    PGVECTOR_USER: str = "postgres"
    PGVECTOR_PASSWORD: str = "postgres"

    # ChromaDB
    CHROMA_API_KEY: Optional[str] = None
    CHROMA_CLOUD_TENANT: Optional[str] = None
    CHROMA_CLOUD_DATABASE: Optional[str] = None
    CHROMA_HTTP_HOST: Optional[str] = None
    CHROMA_HTTP_PORT: Optional[str] = None
    CHROMA_PERSIST_DIR: Optional[str] = None

    # Vector settings
    VECTOR_METRIC: str = "cosine"
    VECTOR_STORE_TEXT: bool = False
    LOG_LEVEL: str = "DEBUG"
    VECTOR_SEARCH_LIMIT: int = 10
    PRIMARY_KEY_MODE: Literal["uuid", "hash_text", "hash_vector", "int64", "auto"] = (
        "uuid"  # choices: uuid, hash_text, hash_vector, int64, auto
    )

    # Optional dotted path to custom PK factory callable: fn(text: str|None, vector: List[float]|None, metadata: dict) -> str
    PRIMARY_KEY_FACTORY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = CrossVectorSettings()
