from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legal_agent.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_legal_embedder(model_name: str, device: str = "cuda"):
    """Load the legal embedding model once and cache it."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading legal embedding model: %s", model_name)
    model = SentenceTransformer(model_name, device=device)
    return model


@lru_cache(maxsize=1)
def get_legal_slm(model_name: str, device: str = "cuda", load_in_4bit: bool = True):
    """Load the legal SLM for metadata extraction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading legal SLM: %s (4-bit=%s)", model_name, load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    return tokenizer, model


def embed_texts_legal(texts: list[str], settings: "Settings") -> list[list[float]]:
    """Embed a batch of texts using the legal embedding model."""
    model = get_legal_embedder(settings.legal_embedding_model, settings.legal_slm_device)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_texts_openai(texts: list[str], settings: "Settings") -> list[list[float]]:
    """Embed texts using OpenAI (fallback)."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(input=texts, model=settings.openai_embedding_model)
    return [d.embedding for d in resp.data]


def embed_texts(texts: list[str], settings: "Settings") -> list[list[float]]:
    """Route to legal or OpenAI embeddings based on config."""
    if settings.use_legal_embeddings:
        return embed_texts_legal(texts, settings)
    return embed_texts_openai(texts, settings)