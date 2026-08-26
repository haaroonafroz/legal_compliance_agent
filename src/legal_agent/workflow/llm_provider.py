"""LLM factory for local llama.cpp, OpenAI, or Gemini providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from legal_agent.config import Settings

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import LLM


_LOCAL_REASONING: dict[str, tuple[bool, str | None]] = {
    "enrichment": (False, None),
    "relevance_check": (False, None),
    "analyst": (True, "medium"),
    "redliner": (True, "low"),
    "auditor": (True, "xHigh"),
}


def local_chat_template_kwargs(step_name: str) -> dict[str, bool | str]:
    """Return llama.cpp chat-template controls for a workflow step."""
    enabled, effort = _LOCAL_REASONING.get(step_name, (True, "medium"))
    kwargs: dict[str, bool | str] = {"enable_thinking": enabled}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    return kwargs


def _get_local_llm(settings: Settings, step_name: str) -> LLM:
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=settings.local_llm_model,
        api_base=settings.local_llm_base_url,
        api_key=settings.local_llm_api_key,
        context_window=settings.local_llm_context_window,
        max_tokens=settings.local_llm_max_tokens,
        timeout=settings.local_llm_timeout_seconds,
        max_retries=2,
        is_chat_model=True,
        is_function_calling_model=False,
        additional_kwargs={
            "extra_body": {
                "chat_template_kwargs": local_chat_template_kwargs(step_name),
            }
        },
    )


def get_llm_for_step(settings: Settings, step_name: str) -> LLM:
    """Return an LLM configured for the given workflow step."""
    if settings.use_local_llm:
        return _get_local_llm(settings, step_name)

    if settings.use_gemini:
        from google.genai import types
        from llama_index.llms.google_genai import GoogleGenAI

        level_map = {
            "analyst": settings.analyst_thinking,
            "redliner": settings.redliner_thinking,
            "auditor": settings.auditor_thinking,
            "enrichment": settings.enrichment_thinking,
            "relevance_check": settings.relevance_check_thinking,
        }
        thinking_level = level_map.get(step_name, "medium")

        return GoogleGenAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            generation_config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
            ),
        )

    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    if step_name == "relevance_check":
        model = settings.openai_llm_model_relevance_check
    elif step_name == "enrichment":
        model = settings.openai_llm_model_enrichment
    else:
        model = settings.openai_llm_model

    return LlamaOpenAI(model=model, api_key=settings.openai_api_key)
