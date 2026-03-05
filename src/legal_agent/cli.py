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
@click.option("--sources", default=None, help="Path to targets.json (overrides config).")
def scrape(sources: str | None) -> None:
    """Run the regulatory harvesting spider."""
    from scrapy.crawler import CrawlerProcess

    from legal_agent.scraping.settings import (
        BOT_NAME,
        CONCURRENT_REQUESTS,
        COOKIES_ENABLED,
        DEFAULT_REQUEST_HEADERS,
        DOWNLOAD_DELAY,
        ITEM_PIPELINES,
        LOG_LEVEL,
        ROBOTSTXT_OBEY,
    )

    settings = get_settings()
    sources_file = sources or settings.sources_file

    process = CrawlerProcess(
        settings={
            "BOT_NAME": BOT_NAME,
            "ROBOTSTXT_OBEY": ROBOTSTXT_OBEY,
            "CONCURRENT_REQUESTS": CONCURRENT_REQUESTS,
            "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
            "COOKIES_ENABLED": COOKIES_ENABLED,
            "DEFAULT_REQUEST_HEADERS": DEFAULT_REQUEST_HEADERS,
            "ITEM_PIPELINES": ITEM_PIPELINES,
            "LOG_LEVEL": LOG_LEVEL,
        }
    )
    process.crawl(
        "legal_agent.scraping.spiders.regulatory_spider.RegulatorySpider",
        sources_file=sources_file,
    )
    process.start()
    click.echo("Scraping complete.")


@cli.command()
def run_workflow() -> None:
    """Process unprocessed regulations through the compliance workflow."""
    settings = get_settings()

    from legal_agent.instrumentation import init_observability

    init_observability(settings)

    from legal_agent.workflow import ComplianceWorkflow

    async def _run() -> None:
        wf = ComplianceWorkflow(settings=settings, timeout=300)
        result = await wf.run()

        if isinstance(result, dict) and result.get("reports") == []:
            click.echo("No unprocessed regulations found.")
            return

        from legal_agent.utils.report import save_report

        path = save_report(result)
        click.echo(f"Report saved to {path}")

    asyncio.run(_run())


@cli.command()
@click.argument("json_file", type=click.Path(exists=True))
def load_policies(json_file: str) -> None:
    settings = get_settings()
    from legal_agent.db.client import client_from_settings
    from legal_agent.utils.models import compute_vectors

    client = client_from_settings(settings)
    policies = json.loads(open(json_file, encoding="utf-8").read())
    from qdrant_client.models import PointStruct

    points = []
    for idx, policy in enumerate(policies):
        text = policy["text"][:8000]
        domain = policy.get("compliance_domain", "")
        tags = policy.get("topic_tags", [])
        embed_input = f"[{domain}] [{', '.join(tags)}] {text}"

        named_vectors = compute_vectors(
            [embed_input],
            settings,
            dense_name=settings.qdrant_policies_dense_name,  # "internal_policy"
            sparse_name=settings.qdrant_sparse_name,          # "legal_clause"
        )[0]

        points.append(
            PointStruct(
                id=idx,
                vector=named_vectors,
                payload={
                    "text": policy["text"],
                    "policy_id": policy.get("policy_id", f"POL-{idx:04d}"),
                    "department": policy.get("department", "General"),
                    "last_updated": policy.get("last_updated", ""),
                    "topic_tags": tags,
                    "compliance_domain": domain,
                    "obligation_type": policy.get("obligation_type", ""),
                },
            )
        )

    client.upsert(collection_name=settings.qdrant_policies_collection, points=points)
    click.echo(f"Loaded {len(points)} policies into '{settings.qdrant_policies_collection}'.")


@cli.command()
def status() -> None:
    """Show system status: Qdrant collections, unprocessed count."""
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
