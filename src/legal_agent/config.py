from __future__ import annotations

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL")
    openai_llm_model_enrichment: str = os.getenv("OPENAI_LLM_MODEL_ENRICHMENT")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL")
    openai_embedding_dim: int = 768

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    qdrant_regulatory_collection: str = os.getenv("REGULATORY_COLLECTION", "regulatory_updates")
    qdrant_policies_collection: str = os.getenv("POLICIES_COLLECTION", "internal_policies")
    # Named vector field names (must match Qdrant collection definition)
    qdrant_regulatory_dense_name: str = os.getenv("COMPLIANCE_DENSE_NAME", "compliance")
    qdrant_policies_dense_name: str = os.getenv("POLICY_DENSE_NAME", "internal_policy")
    qdrant_sparse_name: str = os.getenv("POLICY_SPARSE_NAME", "legal_clause")
    # FastEmbed
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"


    # Local legal models
    legal_embedding_model: str = "nlpaueb/legal-bert-base-uncased" #"Equall/saul-embeddings"
    legal_embedding_dim: int = 768 # 4096
    legal_slm_model: str = "Equall/Saul-7B-Instruct-v1"
    legal_slm_device: str = "cuda"  # or "cpu"
    legal_slm_load_in_4bit: bool = True

    # Switch: use legal models vs OpenAI
    use_legal_embeddings: bool = False
    use_legal_slm: bool = False

    # Langfuse
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", '')
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", '')
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") 

    # Arize Phoenix
    phoenix_endpoint: str = os.getenv("PHOENIX_ENDPOINT")

    # LangWatch
    langwatch_api_key: str = os.getenv("LANGWATCH_API_KEY")

    # Scraping
    sources_file: str = "data/sources/targets.json"

    # Chunking
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunking_tokenizer_model: str = "nlpaueb/legal-bert-base-uncased"


def get_settings() -> Settings:
    return Settings()
