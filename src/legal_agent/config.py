from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # OpenAI
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_regulatory_collection: str = "regulatory_updates"
    qdrant_policies_collection: str = "internal_policies"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Arize Phoenix
    phoenix_endpoint: str = "http://localhost:6006"

    # LangWatch
    langwatch_api_key: str = ""

    # Scraping
    sources_file: str = "data/sources/targets.json"

    # Chunking
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64


def get_settings() -> Settings:
    return Settings()
