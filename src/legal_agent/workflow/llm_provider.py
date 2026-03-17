"""LLM factory: OpenAI or Gemini with per-step thinking_level."""
from __future__ import annotations

from typing import TYPE_CHECKING

from legal_agent.config import Settings

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import LLM

def get_llm_for_step(settings: Settings, step_name: str) -> LLM:
    """Return an LLM configured for the given workflow step."""
    if settings.use_gemini:
        from llama_index.llms.google_genai import GoogleGenAI
        from google.genai import types

        level_map = {
            "analyst": settings.analyst_thinking,
            "redliner": settings.redliner_thinking,
            "auditor": settings.auditor_thinking,
            "enrichment": settings.enrichment_thinking,
        }
        thinking_level = level_map.get(step_name, "medium")

        return GoogleGenAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            generation_config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
            ),
        )
    else:
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        return LlamaOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
        )