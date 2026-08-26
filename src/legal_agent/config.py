from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # OpenAI
    openai_api_key: str | None = None
    openai_llm_model: str = "gpt-5.6-luna"
    openai_llm_model_enrichment: str = "gpt-4o-mini"
    openai_llm_model_relevance_check: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 768

    # Local OpenAI-compatible llama.cpp server
    use_local_llm: bool = True
    local_llm_base_url: str = "http://localhost:8001/v1"
    local_llm_api_key: str = "local"
    local_llm_model: str = "qwen3.8-27b"
    local_llm_context_window: int = 65536
    local_llm_max_tokens: int = 32768
    local_llm_timeout_seconds: float = 300.0

    # Gemini
    use_gemini: bool = False
    use_gemini_enrichment: bool = False
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    analyst_thinking: str = "medium"
    redliner_thinking: str = "low"
    auditor_thinking: str = "high"
    enrichment_thinking: str = "minimal"
    relevance_check_thinking: str = "minimal"

    # Qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_regulatory_collection: str = "regulatory_updates"
    qdrant_policies_collection: str = "internal_policies"
    # Named vector field names (must match Qdrant collection definition)
    qdrant_regulatory_dense_name: str = "compliance"
    qdrant_policies_dense_name: str = "internal_policy"
    qdrant_sparse_name: str = "legal_clause"
    # FastEmbed
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # Optional local legal embeddings
    legal_embedding_model: str = "nlpaueb/legal-bert-base-uncased"
    legal_embedding_dim: int = 768
    legal_embedding_device: str = "cuda"
    use_legal_embeddings: bool = False

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Arize Phoenix
    phoenix_endpoint: str | None = None
    phoenix_external: bool = True

    # LangWatch
    langwatch_api_key: str | None = None

    # Scraping
    sources_file: str = "data/sources/targets.json"
    content_validation_min_chars: int = 500
    content_validation_llm_enabled: bool = True
    content_validation_llm_model: str = "gpt-4o-mini"
    content_validation_confidence_threshold: float = 0.8
    quarantine_file: str = "data/quarantine/rejected_documents.jsonl"
    manifest_file: str = "data/state/document_manifest.sqlite3"

    # Chunking
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunking_tokenizer_model: str = "nlpaueb/legal-bert-base-uncased"

    # Reranker layer
    voyage_api_key: str | None = None
    voyage_model: str = "rerank-2.5"


def get_settings() -> Settings:
    return Settings()
