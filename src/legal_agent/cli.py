"""CLI entrypoint for the Legal Agent system."""

from __future__ import annotations

import asyncio
import logging

import click

from legal_agent.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Legal Agent – Regulatory compliance automation."""


@cli.command()
def init_db() -> None:
    """Create PostgreSQL schema and Qdrant collections."""
    settings = get_settings()
    from legal_agent.db import ensure_collections, get_qdrant_client
    from legal_agent.db.manifest import DocumentManifest

    client = get_qdrant_client(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ensure_collections(client, settings)
    DocumentManifest(settings.database_url).migrate()
    click.echo("PostgreSQL schema and Qdrant collections initialised.")


@cli.command("migrate-db")
def migrate_db() -> None:
    """Apply the idempotent PostgreSQL schema migration."""
    from legal_agent.db.manifest import DocumentManifest

    DocumentManifest(get_settings().database_url).migrate()
    click.echo("PostgreSQL migration complete.")


@cli.command("import-legacy-state")
@click.option("--sqlite", "sqlite_file", type=click.Path(exists=True, dir_okay=False),
              default="data/state/document_manifest.sqlite3", show_default=True)
@click.option("--reports-dir", type=click.Path(file_okay=False),
              default="data/reports", show_default=True)
@click.option("--apply", is_flag=True, help="Import into an empty migrated PostgreSQL database.")
def import_legacy_state(sqlite_file: str, reports_dir: str, apply: bool) -> None:
    """Preview or import existing SQLite state without changing the source files."""
    from pathlib import Path

    from legal_agent.db.legacy import import_legacy, preview_legacy
    from legal_agent.db.manifest import DocumentManifest

    source = Path(sqlite_file)
    reports = Path(reports_dir)
    rows, matched, warnings = preview_legacy(source, reports)
    click.echo(f"Found {len(rows)} revisions and {len(matched)} matching reports.")
    for warning in warnings:
        click.echo(f"  {warning}")
    if not apply:
        click.echo("Preview only. Run with --apply after a backup and legal-agent migrate-db.")
        return
    imported_rows, imported_reports, _ = import_legacy(
        DocumentManifest(get_settings().database_url), source, reports
    )
    click.echo(f"Imported {imported_rows} revisions and {imported_reports} reports.")


@cli.command()
@click.option(
    "--sources",
    default=None,
    help="Path to targets.json (overrides the value from config/env).",
    show_default=True,
)
@click.option(
    "--log-level",
    default="ERROR",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Scrapy log level.",
    show_default=True,
)
def scrape(sources: str | None, log_level: str) -> None:
    """Run the regulatory harvesting spider.

    Scrapes all start_urls defined in the targets.json sources file and stores
    extracted regulatory documents into the pipeline (Qdrant by default).
    """
    from scrapy.crawler import CrawlerProcess

    from legal_agent.scraping.settings import (
        BOT_NAME,
        CONCURRENT_REQUESTS,
        COOKIES_ENABLED,
        DEFAULT_REQUEST_HEADERS,
        DOWNLOAD_DELAY,
        ITEM_PIPELINES,
    )
    from legal_agent.scraping.spiders.regulatory_spider import RegulatorySpider

    settings = get_settings()
    sources_file = sources or settings.sources_file

    process = CrawlerProcess(
        settings={
            "BOT_NAME": BOT_NAME,
            "CONCURRENT_REQUESTS": CONCURRENT_REQUESTS,
            "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
            "COOKIES_ENABLED": COOKIES_ENABLED,
            "DEFAULT_REQUEST_HEADERS": DEFAULT_REQUEST_HEADERS,
            "ITEM_PIPELINES": ITEM_PIPELINES,
            "LOG_LEVEL": log_level.upper(),
        }
    )

    process.crawl(RegulatorySpider, sources_file=sources_file)
    process.start()
    click.echo("Scraping complete.")


@cli.command("retry-dead-letters")
@click.option("--limit", default=100, show_default=True, type=click.IntRange(min=1))
def retry_dead_letters(limit: int) -> None:
    """Requeue failed acquisitions for the next scrape run."""
    settings = get_settings()
    from legal_agent.db.manifest import DocumentManifest

    manifest = DocumentManifest(settings.database_url)
    records = manifest.retryable(limit=limit)
    requeued = sum(manifest.reset_for_retry(record["revision_id"]) for record in records)
    click.echo(f"Requeued {requeued} dead-lettered document(s).")
    if records:
        click.echo("Run `legal-agent scrape` to retry acquisition.")


@cli.command()
def run_workflow() -> None:
    """Process unprocessed regulations through the compliance workflow."""
    settings = get_settings()

    import datetime

    from langfuse import get_client, propagate_attributes

    from legal_agent.db.manifest import DocumentManifest
    from legal_agent.instrumentation import init_observability
    from legal_agent.utils.state import persist_workflow_result
    from legal_agent.workflow import ComplianceWorkflow

    init_observability(settings)

    async def _run() -> None:
        manifest = DocumentManifest(settings.database_url)
        if settings.workflow_lease_seconds <= 1200:
            raise click.UsageError("WORKFLOW_LEASE_SECONDS must exceed the 1200-second workflow timeout")
        model_name = (
            settings.local_llm_model if settings.use_local_llm else settings.openai_llm_model
        )
        processed = 0
        with propagate_attributes(
            session_id=f"batch-{datetime.datetime.now(datetime.UTC).date().isoformat()}",
            tags=["compliance-workflow", model_name],
            version="0.1.0",
        ):
            while True:
                claim = manifest.claim_next(settings.workflow_lease_seconds)
                if claim is None:
                    break
                try:
                    wf = ComplianceWorkflow(settings=settings, timeout=1200)
                    result = await wf.run(
                        document_id=claim["revision_id"], document_hash=claim["document_hash"]
                    )
                    if not isinstance(result, dict) or result.get("reports") == []:
                        raise RuntimeError("Claimed document is missing or incomplete in Qdrant")
                    path = persist_workflow_result(settings, result, claim)
                except Exception:
                    manifest.release_claim(claim["revision_id"], claim["claim_token"])
                    raise
                processed += 1
                click.echo(f"Report saved to {path} (status={result.get('status', 'unknown')})")

        if processed == 0:
            click.echo("No unprocessed regulations found.")
        else:
            click.echo(f"Processed {processed} regulation document(s).")

    asyncio.run(_run())

    try:
        get_client().flush()
        logger.info("Langfuse spans flushed.")
    except Exception:
        logger.debug("Langfuse flush skipped.", exc_info=True)


@cli.command("review-report")
@click.argument("report_reference")
@click.option("--decision", type=click.Choice(["approve", "reject"]), required=True)
@click.option("--reviewer", required=True, help="Human reviewer name or ID.")
@click.option("--reason", required=True, help="Rationale for the decision.")
def review_report(report_reference: str, decision: str, reviewer: str, reason: str) -> None:
    """Review a document by ID or by a local report export path."""
    from pathlib import Path

    from legal_agent.db.manifest import DocumentManifest
    from legal_agent.utils.report import load_report, write_report
    from legal_agent.utils.state import mirror_document_state

    if not reviewer.strip() or not reason.strip():
        raise click.UsageError("Reviewer and reason must be non-empty.")
    path = Path(report_reference)
    document_id = load_report(path)["document_id"] if path.exists() else report_reference
    settings = get_settings()
    manifest = DocumentManifest(settings.database_url)
    result = manifest.get_report(document_id)
    current_status = result.get("status")
    if current_status not in {"pending_human_review", "review_required"}:
        raise click.UsageError(f"Report cannot be reviewed from status {current_status!r}.")
    if decision == "approve" and current_status != "pending_human_review":
        raise click.UsageError("Only a report that passed automated checks can be approved.")
    try:
        reviewed = manifest.review_report(
            document_id, result["document_hash"], current_status,
            decision, reviewer, reason,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if path.exists():
        try:
            write_report(path, reviewed)
        except OSError:
            logger.warning("Review committed; local report export could not be updated", exc_info=True)
    try:
        mirror_document_state(settings, document_id, reviewed["status"])
    except Exception:
        logger.warning("Review committed; Qdrant status mirror failed", exc_info=True)
    click.echo(f"Recorded {decision} decision for {document_id}.")


@cli.command("load-policies-pdf")
@click.argument(
    "pdf_file",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--department",
    default="",
    help=(
        "Hard-override department for every chunk (e.g. 'Legal', 'HR'). "
        "When omitted the enrichment model infers it per chunk."
    ),
)
@click.option(
    "--domain",
    default="",
    help=(
        "Hard-override compliance_domain (e.g. 'GDPR', 'ISO27001'). "
        "When omitted the enrichment model infers it per chunk."
    ),
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="Chunks per embedding + Qdrant upsert batch.",
)
def load_policies_pdf(
    pdf_file: str,
    department: str,
    domain: str,
    batch_size: int,
) -> None:
    from pathlib import Path

    from legal_agent.utils.loader import ingest_policy_pdf

    settings = get_settings()
    pdf_path = Path(pdf_file)

    click.echo(f"Ingesting '{pdf_path.name}' …")
    click.echo(f"  Target collection : {settings.qdrant_policies_collection}")
    if settings.use_local_llm:
        enrichment_model = f"local {settings.local_llm_model}"
    elif settings.use_gemini_enrichment:
        enrichment_model = settings.gemini_model
    else:
        enrichment_model = settings.openai_llm_model_enrichment
    click.echo(f"  Enrichment model  : {enrichment_model}")
    if department:
        click.echo(f"  Department override : {department}")
    if domain:
        click.echo(f"  Domain override     : {domain}")

    n = ingest_policy_pdf(
        pdf_path,
        settings,
        department=department,
        compliance_domain=domain,
        batch_size=batch_size,
    )

    click.echo(f"\nDone – {n} chunks upserted into '{settings.qdrant_policies_collection}'.")


@cli.command()
def status() -> None:
    """Show authoritative PostgreSQL lifecycle counts and Qdrant collection health."""
    from legal_agent.db.client import client_from_settings
    from legal_agent.db.manifest import DocumentManifest

    settings = get_settings()
    manifest = DocumentManifest(settings.database_url)
    click.echo(f"  Acquisition: {manifest.counts()}")
    click.echo(f"  Workflow: {manifest.workflow_counts()}")
    client = client_from_settings(settings)
    for name in (settings.qdrant_regulatory_collection, settings.qdrant_policies_collection):
        try:
            info = client.get_collection(name)
            click.echo(f"  {name}: {info.points_count} search points")
        except Exception:  # noqa: BLE001 - search health is advisory
            click.echo(f"  {name}: unavailable")


if __name__ == "__main__":
    cli()
