"""CLI entrypoint for the Legal Agent system."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

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
    """Create Qdrant collections and payload indexes."""
    settings = get_settings()
    from legal_agent.db import ensure_collections, get_qdrant_client

    client = get_qdrant_client(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ensure_collections(client, settings)
    click.echo("Qdrant collections initialised.")


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
    from scrapy.utils.project import get_project_settings as _scrapy_project_settings

    from legal_agent.scraping.spiders.regulatory_spider import RegulatorySpider
    from legal_agent.scraping.settings import (
        BOT_NAME,
        CONCURRENT_REQUESTS,
        COOKIES_ENABLED,
        DEFAULT_REQUEST_HEADERS,
        DOWNLOAD_DELAY,
        ITEM_PIPELINES,
    )

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


@cli.command()
def run_workflow() -> None:
    """Process unprocessed regulations through the compliance workflow."""
    settings = get_settings()

    from legal_agent.instrumentation import init_observability
    from langfuse import get_client, propagate_attributes
    from legal_agent.workflow import ComplianceWorkflow
    from legal_agent.utils.report import save_report
    import datetime

    init_observability(settings)

    async def _run() -> None:
        wf = ComplianceWorkflow(settings=settings, timeout=300)
        with propagate_attributes(
            session_id=f"batch-{datetime.date.today().isoformat()}",
            tags=["compliance-workflow", settings.openai_llm_model],
            version="0.1.0",
        ):
            result = await wf.run()

        if isinstance(result, dict) and result.get("reports") == []:
            click.echo("No unprocessed regulations found.")
            return
        path = save_report(result)
        click.echo(f"Report saved to {path}")

    asyncio.run(_run())

    try:
        get_client().flush()
        logger.info("Langfuse spans flushed.")
    except Exception:
        pass


@cli.command("load-policies-pdf")
@click.argument("pdf_file", default="data\policies\dl-binding-corporate-rules-privacy.pdf", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--department", default="",
    help=(
        "Hard-override department for every chunk (e.g. 'Legal', 'HR'). "
        "When omitted the enrichment model infers it per chunk."
    ),
)
@click.option(
    "--domain", default="",
    help=(
        "Hard-override compliance_domain (e.g. 'GDPR', 'ISO27001'). "
        "When omitted the enrichment model infers it per chunk."
    ),
)
@click.option(
    "--batch-size", default=32, show_default=True,
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
    click.echo(
        f"  Enrichment model  : "
        f"{'SLM (' + settings.legal_slm_model + ')' if settings.use_legal_slm else settings.openai_llm_model_enrichment}"
    )
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
    """Show system status: Qdrant collections and unprocessed document count."""
    settings = get_settings()
    from legal_agent.db.client import client_from_settings

    client = client_from_settings(settings)

    for name in [settings.qdrant_regulatory_collection, settings.qdrant_policies_collection]:
        try:
            info = client.get_collection(name)
            click.echo(f"  {name}: {info.points_count} points")
        except Exception:
            click.echo(f"  {name}: NOT FOUND")

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        unprocessed = client.count(
            collection_name=settings.qdrant_regulatory_collection,
            count_filter=Filter(
                must=[FieldCondition(key="is_processed", match=MatchValue(value=False))]
            ),
        )
        click.echo(f"  Unprocessed regulations: {unprocessed.count}")
    except Exception:
        click.echo("  Could not count unprocessed regulations.")


if __name__ == "__main__":
    cli()