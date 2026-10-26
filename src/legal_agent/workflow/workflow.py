"""ComplianceWorkflow – the core LlamaIndex event-driven workflow."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from legal_agent.config import Settings
from legal_agent.db.client import client_from_settings
from legal_agent.utils.obligations import (
    LegalObligation,
    ObligationEvidence,
    obligation_query,
    parse_obligations,
)
from legal_agent.workflow.events import (
    AnalysisCompleteEvent,
    AuditResultEvent,
    DraftCompleteEvent,
    FinalReportEvent,
    NewLawEvent,
    ObligationsExtractedEvent,
    RelevantContextEvent,
    RetrievedContextEvent,
)
from legal_agent.workflow.llm_provider import get_llm_for_step
from legal_agent.workflow.prompts import (
    ANALYST_SYSTEM,
    ANALYST_USER,
    AUDITOR_SYSTEM,
    AUDITOR_USER,
    OBLIGATION_EXTRACTOR_SYSTEM,
    OBLIGATION_EXTRACTOR_USER,
    REDLINER_SYSTEM,
    REDLINER_USER,
    RELEVANCE_CHECK_SYSTEM,
    RELEVANCE_CHECK_USER,
)

logger = logging.getLogger(__name__)

MAX_AUDIT_RETRIES = 2


def _chat_messages(system: str, user: str) -> list[ChatMessage]:
    """Build a [system, user] ChatMessage list for ``llm.achat()``."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=system),
        ChatMessage(role=MessageRole.USER, content=user),
    ]


def _format_policy_evidence(policies: list[dict], *, max_chars: int = 12000) -> str:
    text = "\n---\n".join(
        (
            f"Policy: {p.get('policy_id', 'N/A')} | "
            f"Chunk: {p.get('chunk_id', 'N/A')} | "
            f"Section: {p.get('header_path', 'N/A')} | "
            f"Department: {p.get('department', 'N/A')}\n"
            f"{p.get('text', '')}"
        )
        for p in policies
    )
    return (text or "_No matching policies found._")[:max_chars]


def _policy_provenance(policies: list[dict]) -> list[dict]:
    fields = ("policy_id", "document_id", "chunk_id", "header_path", "department")
    return [{field: policy.get(field) for field in fields} for policy in policies]


def _format_obligation_evidence(
    evidence_bundles: list[dict], *, max_chars: int = 12000
) -> str:
    """Format evidence grouped by obligation for analyst and auditor prompts."""
    sections = []
    for bundle in evidence_bundles:
        obligation = bundle.get("obligation", {})
        evidence = bundle.get("matched_policies", [])
        status = "SUFFICIENT" if bundle.get("sufficient") else "INSUFFICIENT"
        lines = [
            f"Obligation {bundle.get('obligation_id', 'unknown')} [{status}]",
            f"Regulation quote: {obligation.get('quote', '')}",
            f"Required action: {obligation.get('action', '')}",
        ]
        if bundle.get("insufficiency_reason"):
            lines.append(f"Evidence note: {bundle['insufficiency_reason']}")
        lines.append(_format_policy_evidence(evidence, max_chars=4000))
        sections.append("\n".join(lines))
    return "\n\n---\n\n".join(sections)[:max_chars] or "_No obligation evidence found._"


class ComplianceWorkflow(Workflow):
    """End-to-end compliance gap analysis: scan → retrieve → analyse → draft → audit."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self.llm_analyst = get_llm_for_step(settings, "analyst")
        self.llm_redliner = get_llm_for_step(settings, "redliner")
        self.llm_auditor = get_llm_for_step(settings, "auditor")
        self.llm_relevance_check = get_llm_for_step(settings, "relevance_check")
        self.qdrant = client_from_settings(settings)

    # ------------------------------------------------------------------
    # Step 1 – Horizon Scanner
    # ------------------------------------------------------------------
    @step
    async def horizon_scanner(self, ctx: Context, ev: StartEvent) -> NewLawEvent | StopEvent:
        """Pull unprocessed regulatory chunks from Qdrant, validate sequence integrity, and rebuild full documents."""
        logger.info("Horizon Scanner: fetching unprocessed regulations…")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        pending_filter = Filter(
            must=[FieldCondition(key="is_processed", match=MatchValue(value=False))],
            must_not=[FieldCondition(key="is_current", match=MatchValue(value=False))],
        )
        points = []
        next_offset = None
        while True:
            page, next_offset = self.qdrant.scroll(
                collection_name=self.settings.qdrant_regulatory_collection,
                scroll_filter=pending_filter,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(page)
            if next_offset is None:
                break

        if not points:
            logger.info("No unprocessed regulations found.")
            return StopEvent(result={"reports": []})

        grouped: dict[str, list] = {}
        for pt in points:
            payload = pt.payload or {}
            document_id = payload.get("document_id")
            if not document_id:
                logger.warning("Skipping point without document_id: %s", pt.id)
                continue
            grouped.setdefault(document_id, []).append(pt)

        if not grouped:
            logger.info("No valid unprocessed regulatory documents found.")
            return StopEvent(result={"reports": []})

        for document_id, chunks in grouped.items():
            chunks.sort(key=lambda p: p.payload.get("chunk_index", 0))
            first = chunks[0].payload
            expected_chunk_count = first.get("chunk_count", len(chunks))
            document_hash = first.get("document_hash", "")

            indices = [c.payload.get("chunk_index") for c in chunks]
            expected_indices = list(range(expected_chunk_count))

            if len(chunks) != expected_chunk_count or indices != expected_indices:
                logger.warning(
                    "Skipping incomplete document %s: have %d/%d chunks with indices %s",
                    document_id,
                    len(chunks),
                    expected_chunk_count,
                    indices,
                )
                continue

            if any(c.payload.get("document_hash") != document_hash for c in chunks):
                logger.warning("Skipping inconsistent document_hash set for %s", document_id)
                continue

            combined_text = "\n\n".join(c.payload["text"] for c in chunks)

            return NewLawEvent(
                document_id=document_id,
                document_hash=document_hash,
                regulation_text=combined_text,
                regulation_chunks=[c.payload["text"] for c in chunks],
                header_path=first.get("header_path", ""),
                jurisdiction=first.get("jurisdiction", ""),
                source_url=first.get("source_url", ""),
                effective_date=first.get("effective_date", ""),
                chunk_ids=[c.id for c in chunks],
                chunk_count=expected_chunk_count,
                topic_tags=first.get("topic_tags", []),
                compliance_domain=first.get("compliance_domain", ""),
                applies_to_departments=first.get("applies_to_departments", []),
                obligation_type=first.get("obligation_type", ""),
                source_id=first.get("source_id", ""),
                source_key=first.get("source_key", ""),
                revision_id=first.get("revision_id", document_id),
                document_type=first.get("document_type", ""),
                published_at=first.get("published_at", ""),
                legal_identifier=first.get("legal_identifier", ""),
                language=first.get("language", ""),
                is_current=first.get("is_current", True),
            )

        return StopEvent(result={"reports": []})

    # ------------------------------------------------------------------
    # Step 2 – Obligation extraction
    # ------------------------------------------------------------------
    @step
    async def obligation_extractor(
        self, ctx: Context, ev: NewLawEvent
    ) -> ObligationsExtractedEvent:
        """Extract atomic obligations once before running focused retrieval."""
        logger.info("Obligation Extractor: decomposing '%s'…", ev.source_url)
        prompt = OBLIGATION_EXTRACTOR_USER.format(
            regulation_text=ev.regulation_text[:16000],
        )
        response = await self.llm_analyst.achat(
            messages=_chat_messages(OBLIGATION_EXTRACTOR_SYSTEM, prompt)
        )
        obligations = parse_obligations(
            response.message.content,
            ev.regulation_text,
            maximum=self.settings.obligation_extraction_max,
        )
        await ctx.store.set("obligations", [obligation.model_dump() for obligation in obligations])
        logger.info("Obligation Extractor: found %d obligations.", len(obligations))
        return ObligationsExtractedEvent(regulation=ev, obligations=obligations)

    # ------------------------------------------------------------------
    # Step 3 – Librarian (obligation-centric RAG retrieval)
    # ------------------------------------------------------------------
    @step
    async def librarian(
        self, ctx: Context, ev: ObligationsExtractedEvent
    ) -> RetrievedContextEvent:
        """Retrieve and validate policy evidence independently for each obligation."""
        logger.info("Librarian: searching policy evidence for '%s'…", ev.regulation.source_url)
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FusionQuery,
            MatchAny,
            MatchValue,
            Prefetch,
            SparseVector,
        )

        from legal_agent.utils.models import embed_texts, sparse_encode_texts

        regulation = ev.regulation
        obligations = ev.obligations or [
            LegalObligation(
                obligation_id="obligation-1",
                quote=regulation.regulation_text[:1000],
                action="Comply with the stated requirement",
                confidence=0.0,
            )
        ]
        query_texts = [obligation_query(obligation) for obligation in obligations]
        dense_vectors = embed_texts(query_texts, self.settings)
        sparse_pairs = sparse_encode_texts(query_texts, self.settings)

        evidence_bundles: list[dict] = []
        all_policies: list[dict] = []
        all_scores: list[float] = []
        reranker_used = False
        import voyageai

        for obligation, dense_vector, (sparse_indices, sparse_values) in zip(
            obligations, dense_vectors, sparse_pairs, strict=True
        ):
            required_filters = []
            optional_filters = []
            if regulation.compliance_domain:
                required_filters.append(
                    FieldCondition(
                        key="compliance_domain",
                        match=MatchValue(value=regulation.compliance_domain),
                    )
                )
            if obligation.responsible_departments:
                optional_filters.append(
                    FieldCondition(
                        key="department",
                        match=MatchAny(any=obligation.responsible_departments),
                    )
                )
            query_filter = (
                Filter(must=required_filters, should=optional_filters)
                if required_filters or optional_filters
                else None
            )
            def run_policy_query(
                active_filter,
                dense_vector=dense_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
            ):
                return self.qdrant.query_points(
                    collection_name=self.settings.qdrant_policies_collection,
                    prefetch=[
                        Prefetch(
                            query=dense_vector,
                            using=self.settings.qdrant_policies_dense_name,
                            limit=self.settings.obligation_retrieval_top_k,
                            filter=active_filter,
                        ),
                        Prefetch(
                            query=SparseVector(indices=sparse_indices, values=sparse_values),
                            using=self.settings.qdrant_sparse_name,
                            limit=self.settings.obligation_retrieval_top_k,
                            filter=active_filter,
                        ),
                    ],
                    query=FusionQuery(fusion="rrf"),
                    limit=self.settings.obligation_retrieval_top_k,
                    with_payload=True,
                )

            hits = run_policy_query(query_filter)
            if not hits.points and query_filter is not None:
                logger.info(
                    "Librarian: relaxing metadata filters for obligation '%s'.",
                    obligation.obligation_id,
                )
                hits = run_policy_query(None)
            direct_policies = [hit.payload for hit in hits.points if hit.payload]
            direct_scores = [hit.score for hit in hits.points if hit.payload]

            documents = [
                (
                    f"[Section: {policy.get('header_path', '')}]\n\n{policy.get('text', '')}"
                    if policy.get("header_path")
                    else policy.get("text", "")
                )
                for policy in direct_policies
            ]
            obligation_reranker_used = False
            if self.settings.voyage_api_key and len(direct_policies) > 5:
                reranking = voyageai.Client(api_key=self.settings.voyage_api_key).rerank(
                    query=obligation_query(obligation),
                    documents=documents,
                    model=self.settings.voyage_model,
                    top_k=self.settings.obligation_retrieval_final_k,
                    truncation=True,
                )
                ranked_indices = [result.index for result in reranking.results]
                direct_policies = [direct_policies[index] for index in ranked_indices]
                direct_scores = [result.relevance_score for result in reranking.results]
                obligation_reranker_used = True
                reranker_used = True
            else:
                direct_policies = direct_policies[: self.settings.obligation_retrieval_final_k]
                direct_scores = direct_scores[: self.settings.obligation_retrieval_final_k]

            expanded_policies = self._expand_policy_neighbors(
                direct_policies,
                hops=self.settings.obligation_neighbor_hops,
            )
            direct_score_by_id = {
                policy.get("chunk_id") or policy.get("document_id"): score
                for policy, score in zip(direct_policies, direct_scores, strict=True)
            }
            expanded_scores = [
                direct_score_by_id.get(policy.get("chunk_id") or policy.get("document_id"), 0.0)
                for policy in expanded_policies
            ]
            sufficient = (
                len(direct_policies) >= self.settings.obligation_min_evidence_hits
                and bool(direct_scores)
                and max(direct_scores) >= self.settings.obligation_min_retrieval_score
            )
            evidence = ObligationEvidence(
                obligation_id=obligation.obligation_id,
                obligation=obligation,
                matched_policies=expanded_policies,
                retrieval_scores=expanded_scores,
                reranker_used=obligation_reranker_used,
                sufficient=sufficient,
                insufficiency_reason=""
                if sufficient
                else "No policy evidence met the configured retrieval gate.",
            ).model_dump()
            evidence_bundles.append(evidence)
            for policy, score in zip(expanded_policies, expanded_scores, strict=True):
                policy_id = policy.get("chunk_id") or policy.get("document_id")
                if policy_id not in {
                    existing.get("chunk_id") or existing.get("document_id")
                    for existing in all_policies
                }:
                    all_policies.append(policy)
                    all_scores.append(score)

        await ctx.store.set("matched_policies", all_policies)
        await ctx.store.set("retrieval_scores", all_scores)
        await ctx.store.set("reranker_used", reranker_used)
        await ctx.store.set("obligation_evidence", evidence_bundles)

        logger.info("Librarian: built evidence for %d obligations.", len(evidence_bundles))
        return RetrievedContextEvent(
            regulation=regulation,
            matched_policies=all_policies,
            retrieval_scores=all_scores,
            reranker_used=reranker_used,
            obligation_evidence=evidence_bundles,
        )

    def _expand_policy_neighbors(
        self, policies: list[dict], *, hops: int
    ) -> list[dict]:
        """Add adjacent policy chunks to preserve the surrounding control context."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        expanded = list(policies)
        known_ids = {
            policy.get("chunk_id") or policy.get("document_id")
            for policy in expanded
            if policy.get("chunk_id") or policy.get("document_id")
        }
        frontier = list(known_ids)
        for _ in range(max(0, hops)):
            neighbor_ids = set()
            for policy in expanded:
                if (policy.get("chunk_id") or policy.get("document_id")) not in frontier:
                    continue
                for field in ("prev_chunk_id", "next_chunk_id"):
                    neighbor_id = policy.get(field)
                    if neighbor_id and neighbor_id not in known_ids:
                        neighbor_ids.add(neighbor_id)
            if not neighbor_ids:
                break

            points, _ = self.qdrant.scroll(
                collection_name=self.settings.qdrant_policies_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchAny(any=list(neighbor_ids)),
                        )
                    ]
                ),
                limit=len(neighbor_ids),
                with_payload=True,
                with_vectors=False,
            )
            frontier = []
            for point in points:
                payload = point.payload or {}
                policy_id = payload.get("chunk_id") or payload.get("document_id")
                if policy_id and policy_id not in known_ids:
                    expanded.append(payload)
                    known_ids.add(policy_id)
                    frontier.append(policy_id)
            if not frontier:
                break
        return expanded

    # ------------------------------------------------------------------
    # Step Intermediate – Relevance check
    # ------------------------------------------------------------------
    @step
    async def relevance_check(
        self, ctx: Context, ev: RetrievedContextEvent
    ) -> RelevantContextEvent | StopEvent:
        """Determine if the regulation is relevant or noise."""

        logger.info(
            "Relevance check: checking '%s'…",
            ev.regulation.source_url,
        )
        prompt = RELEVANCE_CHECK_USER.format(
            regulation_text=ev.regulation.regulation_text,
        )
        response = await self.llm_relevance_check.achat(
            messages=_chat_messages(RELEVANCE_CHECK_SYSTEM, prompt)
        )
        decision = response.message.content.strip().upper()

        if decision not in {"RELEVANT", "IRRELEVANT"}:
            raise ValueError(f"Unexpected relevance decision: {decision!r}")

        if decision == "IRRELEVANT":
            logger.info("Document marked as IRRELEVANT. Stopping workflow and starting again.")

            from qdrant_client.models import FieldCondition, Filter, MatchValue

            self.qdrant.set_payload(
                collection_name=self.settings.qdrant_regulatory_collection,
                payload={"is_processed": True, "workflow_status": "irrelevant"},
                points=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=ev.regulation.document_id),
                        )
                    ]
                ),
            )
            return StopEvent(
                result={
                    "document_id": ev.regulation.document_id,
                    "document_hash": ev.regulation.document_hash,
                    "jurisdiction": ev.regulation.jurisdiction,
                    "source_url": ev.regulation.source_url,
                    "source_id": ev.regulation.source_id,
                    "source_key": ev.regulation.source_key,
                    "revision_id": ev.regulation.revision_id,
                    "document_type": ev.regulation.document_type,
                    "published_at": ev.regulation.published_at,
                    "legal_identifier": ev.regulation.legal_identifier,
                    "language": ev.regulation.language,
                    "is_current": ev.regulation.is_current,
                    "status": "irrelevant",
                    "skipped": True,
                    "reason": "irrelevant",
                }
            )

        logger.info("Document is RELEVANT. Continuing workflow.")
        if any(not bundle.get("sufficient", False) for bundle in ev.obligation_evidence):
            logger.warning("Evidence gate failed for '%s'.", ev.regulation.source_url)
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            self.qdrant.set_payload(
                collection_name=self.settings.qdrant_regulatory_collection,
                payload={"is_processed": True, "workflow_status": "review_required"},
                points=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=ev.regulation.document_id),
                        )
                    ]
                ),
            )
            return StopEvent(
                result={
                    "document_id": ev.regulation.document_id,
                    "document_hash": ev.regulation.document_hash,
                    "jurisdiction": ev.regulation.jurisdiction,
                    "source_url": ev.regulation.source_url,
                    "source_id": ev.regulation.source_id,
                    "source_key": ev.regulation.source_key,
                    "revision_id": ev.regulation.revision_id,
                    "document_type": ev.regulation.document_type,
                    "published_at": ev.regulation.published_at,
                    "legal_identifier": ev.regulation.legal_identifier,
                    "language": ev.regulation.language,
                    "is_current": ev.regulation.is_current,
                    "status": "review_required",
                    "skipped": True,
                    "reason": "insufficient_evidence",
                    "obligation_evidence": ev.obligation_evidence,
                }
            )
        return RelevantContextEvent(
            regulation=ev.regulation,
            matched_policies=ev.matched_policies,
            retrieval_scores=ev.retrieval_scores,
            reranker_used=ev.reranker_used,
            obligation_evidence=ev.obligation_evidence,
        )

    # ------------------------------------------------------------------
    # Step 3 – Analyst (gap analysis)
    # ------------------------------------------------------------------
    @step
    async def analyst(self, ctx: Context, ev: RelevantContextEvent) -> AnalysisCompleteEvent:
        """Compare regulation against internal policies and produce gap analysis."""
        logger.info("Analyst: generating gap analysis for '%s'…", ev.regulation.source_url)

        policies_text = _format_policy_evidence(ev.matched_policies, max_chars=6000)
        obligation_evidence = _format_obligation_evidence(
            ev.obligation_evidence,
            max_chars=10000,
        )

        prompt = ANALYST_USER.format(
            jurisdiction=ev.regulation.jurisdiction,
            source_url=ev.regulation.source_url,
            effective_date=ev.regulation.effective_date,
            regulation_text=ev.regulation.regulation_text[:6000],
            policies_text=policies_text[:6000],
            obligation_evidence=obligation_evidence,
        )

        response = await self.llm_analyst.achat(messages=_chat_messages(ANALYST_SYSTEM, prompt))

        return AnalysisCompleteEvent(
            regulation=ev.regulation,
            matched_policies=ev.matched_policies,
            gap_analysis=response.message.content,
            obligation_evidence=ev.obligation_evidence,
        )

    # ------------------------------------------------------------------
    # Step 4 – Redliner (draft amendments)
    # ------------------------------------------------------------------
    @step
    async def redliner(self, ctx: Context, ev: AnalysisCompleteEvent) -> DraftCompleteEvent:
        """Draft policy amendments to close identified gaps."""
        logger.info("Redliner: drafting amendments for '%s'…", ev.regulation.source_url)
        audit_feedback = ""
        if ev.audit_notes and ev.previous_draft:
            audit_feedback = (
                "\n## Previous Draft (Rejected by Auditor)\n"
                f"{ev.previous_draft[:3000]}\n\n"
                "## Auditor Rejection Notes\n"
                f"{ev.audit_notes}\n\n"
                "Revise the draft above to address the auditor's specific objections.\n"
            )
        prompt = REDLINER_USER.format(
            gap_analysis=ev.gap_analysis[:8000],
            audit_feedback=audit_feedback,
        )
        response = await self.llm_redliner.achat(messages=_chat_messages(REDLINER_SYSTEM, prompt))

        return DraftCompleteEvent(
            regulation=ev.regulation,
            gap_analysis=ev.gap_analysis,
            proposed_updates=response.message.content,
        )

    # ------------------------------------------------------------------
    # Step 5 – Auditor (hallucination / consistency check)
    # ------------------------------------------------------------------
    @step
    async def auditor(
        self, ctx: Context, ev: DraftCompleteEvent | AuditResultEvent
    ) -> FinalReportEvent | AnalysisCompleteEvent | AuditResultEvent:
        """Review outputs for hallucinations and return failed drafts to the Redliner."""
        if isinstance(ev, AuditResultEvent):
            retries = await ctx.store.get("audit_retries", default=0)
            if ev.passed or retries >= MAX_AUDIT_RETRIES:
                matched_policies = await ctx.store.get("matched_policies", default=[])
                retrieval_scores = await ctx.store.get("retrieval_scores", default=[])
                reranker_used = await ctx.store.get("reranker_used", default=False)
                return FinalReportEvent(
                    document_id=ev.regulation.document_id,
                    document_hash=ev.regulation.document_hash,
                    jurisdiction=ev.regulation.jurisdiction,
                    source_url=ev.regulation.source_url,
                    gap_analysis=ev.gap_analysis,
                    proposed_updates=ev.proposed_updates,
                    audit_notes=ev.audit_notes,
                    passed=ev.passed,
                    status="approved" if ev.passed else "review_required",
                    matched_policies=_policy_provenance(matched_policies),
                    retrieval_scores=retrieval_scores,
                    reranker_used=reranker_used,
                    audit_retries=retries,
                    obligation_evidence=await ctx.store.get("obligation_evidence", default=[]),
                    source_id=ev.regulation.source_id,
                    source_key=ev.regulation.source_key,
                    revision_id=ev.regulation.revision_id,
                    document_type=ev.regulation.document_type,
                    published_at=ev.regulation.published_at,
                    legal_identifier=ev.regulation.legal_identifier,
                    language=ev.regulation.language,
                    is_current=ev.regulation.is_current,
                )
            logger.warning("Audit FAILED (retry %d) – sending back to the Redliner.", retries + 1)
            await ctx.store.set("audit_retries", retries + 1)
            return AnalysisCompleteEvent(
                regulation=ev.regulation,
                matched_policies=await ctx.store.get("matched_policies", default=[]),
                gap_analysis=ev.gap_analysis,
                audit_notes=ev.audit_notes,
                previous_draft=ev.proposed_updates,
                obligation_evidence=await ctx.store.get("obligation_evidence", default=[]),
            )

        logger.info("Auditor: reviewing outputs for '%s'…", ev.regulation.source_url)
        matched_policies = await ctx.store.get("matched_policies", default=[])
        obligation_evidence = await ctx.store.get("obligation_evidence", default=[])
        prompt = AUDITOR_USER.format(
            regulation_text=ev.regulation.regulation_text[:12000],
            policies_text=_format_policy_evidence(matched_policies),
            obligation_evidence=_format_obligation_evidence(obligation_evidence),
            gap_analysis=ev.gap_analysis[:6000],
            proposed_updates=ev.proposed_updates[:6000],
        )

        response = await self.llm_auditor.achat(messages=_chat_messages(AUDITOR_SYSTEM, prompt))

        passed = response.message.content.strip().upper() == "PASS"

        audit_ev = AuditResultEvent(
            regulation=ev.regulation,
            gap_analysis=ev.gap_analysis,
            proposed_updates=ev.proposed_updates,
            audit_notes=response.message.content,
            passed=passed,
        )
        return audit_ev

    # ------------------------------------------------------------------
    # Collect final reports
    # ------------------------------------------------------------------
    @step
    async def collect_report(self, ctx: Context, ev: FinalReportEvent) -> StopEvent:
        """Finalize workflow state and emit a provenance-bearing report."""
        logger.info("Report ready for '%s' (passed=%s).", ev.source_url, ev.passed)

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.qdrant.set_payload(
            collection_name=self.settings.qdrant_regulatory_collection,
            payload={"is_processed": True, "workflow_status": ev.status},
            points=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=ev.document_id))]
            ),
        )

        return StopEvent(
            result={
                "document_id": ev.document_id,
                "document_hash": ev.document_hash,
                "jurisdiction": ev.jurisdiction,
                "source_url": ev.source_url,
                "source_id": ev.source_id,
                "source_key": ev.source_key,
                "revision_id": ev.revision_id,
                "document_type": ev.document_type,
                "published_at": ev.published_at,
                "legal_identifier": ev.legal_identifier,
                "language": ev.language,
                "is_current": ev.is_current,
                "status": ev.status,
                "gap_analysis": ev.gap_analysis,
                "proposed_updates": ev.proposed_updates,
                "audit_notes": ev.audit_notes,
                "passed": ev.passed,
                "matched_policies": ev.matched_policies,
                "retrieval_scores": ev.retrieval_scores,
                "reranker_used": ev.reranker_used,
                "audit_retries": ev.audit_retries,
                    "obligation_evidence": ev.obligation_evidence,
            }
        )
