"""Tests for local OpenAI-compatible metadata enrichment."""

import json
from types import SimpleNamespace

from legal_agent.scraping import pipelines


def test_local_enrichment_disables_thinking(monkeypatch):
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(
                content=json.dumps(
                    {
                        "topic_tags": ["privacy"],
                        "compliance_domain": "data_protection",
                        "applies_to_departments": ["Legal"],
                        "obligation_type": "requirement",
                    }
                )
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(pipelines, "OpenAI", _Client)
    pipeline = pipelines.MetadataEnrichmentPipeline()
    pipeline._settings = SimpleNamespace(
        use_local_llm=True,
        use_gemini_enrichment=False,
        local_llm_base_url="http://localhost:8001/v1",
        local_llm_api_key="local",
        local_llm_model="qwen",
        local_llm_timeout_seconds=300,
    )

    result = pipeline.extract_metadata("Article 1 requires privacy controls.")

    assert result["compliance_domain"] == "data_protection"
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
