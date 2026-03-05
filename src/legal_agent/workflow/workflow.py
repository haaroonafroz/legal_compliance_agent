"""ComplianceWorkflow – the core LlamaIndex event-driven workflow."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from legal_agent.config import Settings
from legal_agent.db.client import client_from_settings
from legal_agent.workflow.events import (
    AnalysisCompleteEvent,
    AuditResultEvent,
    DraftCompleteEvent,
    FinalReportEvent,
    NewLawEvent,
    RetrievedContextEvent,
)
from legal_agent.workflow.prompts import (
    ANALYST_SYSTEM,
    ANALYST_USER,
    AUDITOR_SYSTEM,
    AUDITOR_USER,
    REDLINER_SYSTEM,
    REDLINER_USER,
)

logger = logging.getLogger(__name__)

MAX_AUDIT_RETRIES = 2


class ComplianceWorkflow(Workflow):
    """End-to-end compliance gap analysis: scan → retrieve → analyse → draft → audit."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self.llm = LlamaOpenAI(model=settings.openai_llm_model, api_key=settings.openai_api_key)
        self.qdrant = client_from_settings(settings)

        from openai import OpenAI

        self._openai = OpenAI(api_key=settings.openai_api_key)
        self._embed_model = settings.openai_embedding_model

    # ------------------------------------------------------------------
    # Step 1 – Horizon Scanner
    # ------------------------------------------------------------------
    @step
    async def horizon_scanner(self, ctx: Context, ev: StartEvent) -> NewLawEvent | StopEvent:
        """Pull unprocessed regulatory chunks from Qdrant and group them."""
        logger.info("Horizon Scanner: fetching unprocessed regulations…")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results = self.qdrant.scroll(
            collection_name=self.settings.qdrant_regulatory_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="is_processed", match=MatchValue(value=False))]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        points = results[0]
        if not points:
            logger.info("No unprocessed regulations found.")
            return StopEvent(result={"reports": []})

        grouped: dict[str, list] = {}
        for pt in points:
            key = pt.payload["source_url"]
            grouped.setdefault(key, []).append(pt)

        reports = []
        await ctx.set("reports", reports)
        await ctx.set("grouped_keys", list(grouped.keys()))
        await ctx.set("pending_count", len(grouped))

        for source_url, chunks in grouped.items():
            chunks.sort(key=lambda p: p.payload.get("chunk_index", 0))
            combined_text = "\n\n".join(c.payload["text"] for c in chunks)
            first = chunks[0].payload

            return NewLawEvent(
                regulation_text=combined_text,
                header_path=first.get("header_path", ""),
                jurisdiction=first.get("jurisdiction", ""),
                source_url=source_url,
                effective_date=first.get("effective_date", ""),
                chunk_ids=[c.id for c in chunks],
                topic_tags=first.get("topic_tags", []),
                compliance_domain=first.get("compliance_domain", ""),
                applies_to_departments=first.get("applies_to_departments", []),
                obligation_type=first.get("obligation_type", ""),
            )

        return StopEvent(result={"reports": []})

    # ------------------------------------------------------------------
    # Step 2 – Librarian (RAG retrieval against internal_policies)
    # ------------------------------------------------------------------
    @step
    async def librarian(self, ctx: Context, ev: NewLawEvent) -> RetrievedContextEvent:
        """Retrieve internal policies relevant to the new regulation."""
        logger.info("Librarian: searching internal policies for '%s'…", ev.source_url)
        from legal_agent.utils.models import embed_texts
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        domain = ev.compliance_domain
        tags = ev.topic_tags
        embed_input = f"[{domain}] [{', '.join(tags)}] {ev.regulation_text[:8000]}"
        vectors = embed_texts([embed_input], self.settings)
        query_vector = vectors[0]
        # Build optional metadata filter
        filter_conditions = []
        if tags:
            filter_conditions.append(
                FieldCondition(key="topic_tags", match=MatchAny(any=tags))
            )
        if domain:
            filter_conditions.append(
                FieldCondition(key="compliance_domain", match=MatchValue(value=domain))
            )
        query_filter = Filter(should=filter_conditions) if filter_conditions else None
        hits = self.qdrant.query_points(
            collection_name=self.settings.qdrant_policies_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=10,
            with_payload=True,
        )

        matched_policies = []
        scores = []
        for hit in hits.points:
            matched_policies.append(hit.payload)
            scores.append(hit.score)

        logger.info("Librarian: found %d matching policies.", len(matched_policies))
        return RetrievedContextEvent(
            regulation=ev,
            matched_policies=matched_policies,
            retrieval_scores=scores,
        )

    # ------------------------------------------------------------------
    # Step 3 – Analyst (gap analysis)
    # ------------------------------------------------------------------
    @step
    async def analyst(self, ctx: Context, ev: RetrievedContextEvent) -> AnalysisCompleteEvent:
        """Compare regulation against internal policies and produce gap analysis."""
        logger.info("Analyst: generating gap analysis for '%s'…", ev.regulation.source_url)

        policies_text = "\n---\n".join(
            f"**Policy:** {p.get('policy_id', 'N/A')} (Dept: {p.get('department', 'N/A')})\n{p.get('text', '')}"
            for p in ev.matched_policies
        ) or "_No matching policies found._"

        prompt = ANALYST_USER.format(
            jurisdiction=ev.regulation.jurisdiction,
            source_url=ev.regulation.source_url,
            effective_date=ev.regulation.effective_date,
            regulation_text=ev.regulation.regulation_text[:6000],
            policies_text=policies_text[:6000],
        )

        response = await self.llm.acomplete(
            prompt,
            system_prompt=ANALYST_SYSTEM,
        )

        return AnalysisCompleteEvent(
            regulation=ev.regulation,
            matched_policies=ev.matched_policies,
            gap_analysis=response.text,
        )

    # ------------------------------------------------------------------
    # Step 4 – Redliner (draft amendments)
    # ------------------------------------------------------------------
    @step
    async def redliner(self, ctx: Context, ev: AnalysisCompleteEvent) -> DraftCompleteEvent:
        """Draft policy amendments to close identified gaps."""
        logger.info("Redliner: drafting amendments for '%s'…", ev.regulation.source_url)

        prompt = REDLINER_USER.format(gap_analysis=ev.gap_analysis[:8000])
        response = await self.llm.acomplete(prompt, system_prompt=REDLINER_SYSTEM)

        return DraftCompleteEvent(
            regulation=ev.regulation,
            gap_analysis=ev.gap_analysis,
            proposed_updates=response.text,
        )

    # ------------------------------------------------------------------
    # Step 5 – Auditor (hallucination / consistency check)
    # ------------------------------------------------------------------
    @step
    async def auditor(
        self, ctx: Context, ev: DraftCompleteEvent | AuditResultEvent
    ) -> FinalReportEvent | AnalysisCompleteEvent:
        """Review outputs for hallucinations. Retry via Analyst if audit fails."""
        if isinstance(ev, AuditResultEvent):
            retries = await ctx.get("audit_retries", default=0)
            if ev.passed or retries >= MAX_AUDIT_RETRIES:
                return FinalReportEvent(
                    jurisdiction=ev.regulation.jurisdiction,
                    source_url=ev.regulation.source_url,
                    gap_analysis=ev.gap_analysis,
                    proposed_updates=ev.proposed_updates,
                    audit_notes=ev.audit_notes,
                    passed=ev.passed,
                )
            logger.warning("Audit FAILED (retry %d) – sending back to Analyst.", retries + 1)
            await ctx.set("audit_retries", retries + 1)
            return AnalysisCompleteEvent(
                regulation=ev.regulation,
                matched_policies=[],
                gap_analysis=ev.gap_analysis,
            )

        logger.info("Auditor: reviewing outputs for '%s'…", ev.regulation.source_url)
        prompt = AUDITOR_USER.format(
            gap_analysis=ev.gap_analysis[:6000],
            proposed_updates=ev.proposed_updates[:6000],
        )
        response = await self.llm.acomplete(prompt, system_prompt=AUDITOR_SYSTEM)

        passed = response.text.strip().upper().startswith("PASS")

        audit_ev = AuditResultEvent(
            regulation=ev.regulation,
            gap_analysis=ev.gap_analysis,
            proposed_updates=ev.proposed_updates,
            audit_notes=response.text,
            passed=passed,
        )

        retries = await ctx.get("audit_retries", default=0)
        if passed or retries >= MAX_AUDIT_RETRIES:
            return FinalReportEvent(
                jurisdiction=ev.regulation.jurisdiction,
                source_url=ev.regulation.source_url,
                gap_analysis=ev.gap_analysis,
                proposed_updates=ev.proposed_updates,
                audit_notes=response.text,
                passed=passed,
            )

        logger.warning("Audit FAILED – sending back to Analyst for re-analysis.")
        await ctx.set("audit_retries", retries + 1)
        return AnalysisCompleteEvent(
            regulation=ev.regulation,
            matched_policies=[],
            gap_analysis=ev.gap_analysis,
        )

    # ------------------------------------------------------------------
    # Collect final reports
    # ------------------------------------------------------------------
    @step
    async def collect_report(self, ctx: Context, ev: FinalReportEvent) -> StopEvent:
        """Mark regulation as processed in Qdrant and emit the final report."""
        logger.info("Report ready for '%s' (passed=%s).", ev.source_url, ev.passed)

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.qdrant.set_payload(
            collection_name=self.settings.qdrant_regulatory_collection,
            payload={"is_processed": True},
            points=Filter(
                must=[FieldCondition(key="source_url", match=MatchValue(value=ev.source_url))]
            ),
        )

        return StopEvent(
            result={
                "jurisdiction": ev.jurisdiction,
                "source_url": ev.source_url,
                "gap_analysis": ev.gap_analysis,
                "proposed_updates": ev.proposed_updates,
                "audit_notes": ev.audit_notes,
                "passed": ev.passed,
            }
        )
