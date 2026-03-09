"""Initialise Langfuse, Arize Phoenix, and LangWatch before any LLM call."""

from __future__ import annotations

import logging
import os
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

    # The Python SDK v3 reads these env vars automatically and sets up the
    # OTel TracerProvider + exporter to send spans to Langfuse.
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host

    from langfuse import get_client

    langfuse = get_client()
    if langfuse.auth_check():
        logger.info("Langfuse connected at %s", settings.langfuse_host)
    else:
        logger.warning("Langfuse auth_check failed – check your keys and host.")


def _init_phoenix(settings: "Settings") -> None:
    try:
        import phoenix as px
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Launch local Phoenix server if using localhost endpoint
        if "localhost" in settings.phoenix_endpoint:
            px.launch_app()

        # Add Phoenix as a second OTel exporter so spans go to BOTH
        # Langfuse (already configured above) and Phoenix simultaneously.
        phoenix_exporter = OTLPSpanExporter(
            endpoint=f"{settings.phoenix_endpoint}/v1/traces"
        )
        provider = otel_trace.get_tracer_provider()
        provider.add_span_processor(SimpleSpanProcessor(phoenix_exporter))

        # Instrument LlamaIndex – this hooks into the OTel provider that
        # Langfuse already configured, so all @step spans flow to Langfuse.
        LlamaIndexInstrumentor().instrument()
        logger.info("Arize Phoenix instrumentation active at %s", settings.phoenix_endpoint)
    except Exception:
        logger.warning("Phoenix init failed – RAG metrics unavailable.", exc_info=True)


def _init_langwatch(settings: "Settings") -> None:
    if not settings.langwatch_api_key:
        logger.warning("LangWatch API key not set – skipping LangWatch init.")
        return
    try:
        import langwatch

        # langwatch>=0.2.0 uses setup(), not login()
        langwatch.setup(api_key=settings.langwatch_api_key)
        logger.info("LangWatch guardrails active.")
    except Exception:
        logger.warning("LangWatch init failed.", exc_info=True)