"""Tests for local llama.cpp reasoning controls."""

from types import SimpleNamespace

from legal_agent.workflow.llm_provider import get_llm_for_step, local_chat_template_kwargs


def test_local_reasoning_profiles():
    assert local_chat_template_kwargs("enrichment") == {"enable_thinking": False}
    assert local_chat_template_kwargs("relevance_check") == {"enable_thinking": False}
    assert local_chat_template_kwargs("analyst") == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert local_chat_template_kwargs("redliner") == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }
    assert local_chat_template_kwargs("auditor") == {
        "enable_thinking": True,
        "reasoning_effort": "xHigh",
    }


def test_local_provider_forwards_auditor_reasoning(monkeypatch):
    captured = {}

    def fake_openai_like(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "llama_index.llms.openai_like.OpenAILike",
        fake_openai_like,
    )
    settings = SimpleNamespace(
        use_local_llm=True,
        local_llm_model="qwen",
        local_llm_base_url="http://localhost:8001/v1",
        local_llm_api_key="local",
        local_llm_context_window=32768,
        local_llm_max_tokens=8192,
        local_llm_timeout_seconds=300,
    )

    get_llm_for_step(settings, "auditor")

    assert captured["additional_kwargs"]["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "xHigh",
        }
    }
