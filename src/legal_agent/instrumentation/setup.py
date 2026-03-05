"""Initialise Langfuse, Arize Phoenix, and LangWatch before any LLM call."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legal_agent.config import Settings

logger = logging.getLogger(__name__)

_INITIALISED = False


def init_observability(settings: "Settings") -> None:
    """Wire up all observability backends. Safe to call multiple times."""
    global _INITIALISED  # noqa: PLW0603
    if _INITIALISED:
        return

    _init_langfuse(settings)
    _init_phoenix(settings)
    _init_langwatch(settings)

    _INITIALISED = True
    logger.info("Observability stack initialised (Langfuse, Phoenix, LangWatch).")


def _init_langfuse(settings: "Settings") -> None:
    if not settings.langfuse_secret_key:
        logger.warning("Langfuse secret key not set – skipping Langfuse init.")
        return

    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    langfuse.auth_check()
    logger.info("Langfuse connected at %s", settings.langfuse_host)


def _init_phoenix(settings: "Settings") -> None:
    try:
        import phoenix as px
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        px.launch_app() if settings.phoenix_endpoint == "http://localhost:6006" else None
        LlamaIndexInstrumentor().instrument()
        logger.info("Arize Phoenix instrumentation active.")
    except Exception:
        logger.warning("Phoenix init failed – RAG metrics unavailable.", exc_info=True)


def _init_langwatch(settings: "Settings") -> None:
    if not settings.langwatch_api_key:
        logger.warning("LangWatch API key not set – skipping LangWatch init.")
        return
    try:
        import langwatch

        langwatch.login(api_key=settings.langwatch_api_key)
        logger.info("LangWatch guardrails active.")
    except Exception:
        logger.warning("LangWatch init failed.", exc_info=True)
