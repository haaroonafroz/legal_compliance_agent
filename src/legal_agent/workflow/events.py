"""Custom events that flow between workflow steps."""

from __future__ import annotations

from llama_index.core.workflow import Event
from pydantic import Field


class NewLawEvent(Event):
    """Emitted by the Horizon Scanner after grouping unprocessed regulation chunks."""

    regulation_text: str
    header_path: str
    jurisdiction: str
    source_url: str
    effective_date: str
    chunk_ids: list[int] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    compliance_domain: str = ""
    applies_to_departments: list[str] = Field(default_factory=list)
    obligation_type: str = ""


class RetrievedContextEvent(Event):
    """Emitted by the Librarian after retrieving matching internal policies."""

    regulation: NewLawEvent
    matched_policies: list[dict]
    retrieval_scores: list[float] = Field(default_factory=list)


class AnalysisCompleteEvent(Event):
    """Emitted by the Analyst with the gap analysis between regulation and policies."""

    regulation: NewLawEvent
    matched_policies: list[dict]
    gap_analysis: str
    audit_notes: str | None = None
    previous_draft: str | None = None


class DraftCompleteEvent(Event):
    """Emitted by the Redliner with proposed policy updates."""

    regulation: NewLawEvent
    gap_analysis: str
    proposed_updates: str


class AuditResultEvent(Event):
    """Emitted by the Auditor after reviewing for hallucinations."""

    regulation: NewLawEvent
    gap_analysis: str
    proposed_updates: str
    audit_notes: str
    passed: bool


class FinalReportEvent(Event):
    """Terminal event containing the complete compliance report for one regulation."""

    jurisdiction: str
    source_url: str
    gap_analysis: str
    proposed_updates: str
    audit_notes: str
    passed: bool
