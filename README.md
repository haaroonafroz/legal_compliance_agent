# Legal Agent

Automated regulatory change detection, compliance gap analysis, and policy update drafting.

The system scrapes regulatory portals, stores updates in Qdrant, and runs a multi-agent LlamaIndex Workflow to compare new regulations against internal company policies — producing structured compliance gap reports with redlined amendments.

## Architecture

```
Scrapy Spider ──► Qdrant (regulatory_updates)
                          │
          ComplianceWorkflow (LlamaIndex)
          ┌─────────┼─────────────────────┐
    Horizon     Librarian        Analyst
    Scanner   (RAG retrieval)  (gap analysis)
                                     │
                              Redliner ──► Auditor ──► Report
```

**Observability:** Langfuse (tracing/cost), Arize Phoenix (RAG metrics), LangWatch (guardrails).

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Fill in your API keys

# 3. Initialise Qdrant collections
legal-agent init-db

# 4. Load internal policies
legal-agent load-policies data/policies.json

# 5. Scrape regulatory updates
legal-agent scrape

# 6. Run compliance workflow
legal-agent run-workflow

# 7. Check status
legal-agent status
```

## CLI Commands

| Command | Description |
|---|---|
| `legal-agent init-db` | Create Qdrant collections and indexes |
| `legal-agent scrape` | Run the Scrapy regulatory harvesting spider |
| `legal-agent run-workflow` | Process unprocessed regulations through the compliance workflow |
| `legal-agent load-policies FILE` | Bulk-load internal policies from a JSON file |
| `legal-agent status` | Show Qdrant collection stats and unprocessed count |

## Project Structure

```
src/legal_agent/
├── cli.py                     # Click CLI entrypoint
├── config.py                  # Pydantic settings (reads .env)
├── db/
│   ├── client.py              # Qdrant client singleton
│   └── schemas.py             # Collection definitions & indexes
├── instrumentation/
│   └── setup.py               # Langfuse, Phoenix, LangWatch init
├── scraping/
│   ├── items.py               # Scrapy item definitions
│   ├── pipelines.py           # PDF→MD, chunking, Qdrant upsert
│   ├── settings.py            # Scrapy settings
│   └── spiders/
│       └── regulatory_spider.py
├── utils/
│   └── report.py              # Markdown/JSON report writer
└── workflow/
    ├── events.py              # LlamaIndex Workflow events
    ├── prompts.py             # LLM prompt templates
    └── workflow.py            # ComplianceWorkflow (5 @step agents)
```

## Workflow Steps

1. **Horizon Scanner** — Fetches unprocessed chunks from `regulatory_updates`, groups by source
2. **Librarian** — Performs vector similarity search against `internal_policies`
3. **Analyst** — Generates compliance gap analysis comparing regulation vs. policy
4. **Redliner** — Drafts concrete policy amendments in redline format
5. **Auditor** — Reviews for hallucinations/inconsistencies; retries via Analyst if failed

## Testing

```bash
pytest
```

## License

See [LICENSE](LICENSE).
