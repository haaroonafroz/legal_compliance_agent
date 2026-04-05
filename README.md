# Legal Agent

`legal-agent` is a regulatory intelligence pipeline for compliance teams. It scrapes regulatory sources, stages the resulting documents in Qdrant, retrieves the most relevant internal policy chunks, and runs an event-driven workflow that produces:

- a gap analysis between external regulation and internal policy
- proposed policy amendments in redline format
- an auditor verdict and a saved report artifact

The project was built around one core requirement from the original project plan: make the workflow inspectable and easy to evaluate. The architecture therefore favors explicit workflow steps, stored intermediate artifacts, and trace-friendly integrations over opaque agent loops.

## Overview

The pipeline has two main phases:

1. **Harvest and stage regulations**
   Scrapy crawls sites defined in `data/sources/targets.json`, extracts HTML or PDF content, converts PDFs to Markdown with Docling, chunks the text, enriches it with metadata, and stores it in the `regulatory_updates` Qdrant collection.
2. **Run compliance analysis**
   A `LlamaIndex` workflow picks up unprocessed regulations, retrieves relevant policy chunks from `internal_policies`, performs gap analysis, drafts policy updates, audits the draft, and writes a report to `data/reports/`.

Current workflow steps:

1. `Horizon Scanner`: groups unprocessed regulation chunks back into complete documents.
2. `Librarian`: runs hybrid retrieval against internal policies.
3. `Relevance Check`: filters out site noise and non-regulatory content.
4. `Analyst`: compares regulation text against internal policy text.
5. `Redliner`: drafts policy amendments in redline form.
6. `Auditor`: checks for hallucinations, weak reasoning, and draft mismatch before finalizing.

## Quick Start

### Requirements

- Python 3.11+
- Access to a Qdrant instance
- OpenAI API key for the default setup
- Optional: Gemini, VoyageAI, Langfuse, Phoenix, and LangWatch

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in the values you actually want to use.

Minimum practical setup:

- `OPENAI_API_KEY`
- `OPENAI_LLM_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `QDRANT_URL`
- `QDRANT_API_KEY`

Optional but useful:

- `GEMINI_API_KEY`, `GEMINI_MODEL` if you want Gemini for workflow steps
- `VOYAGE_API_KEY` for reranking retrieved policy chunks
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `PHOENIX_ENDPOINT`
- `LANGWATCH_API_KEY`

### 3. Initialize Qdrant collections

```bash
legal-agent init-db
```

This creates the two collections used by the system:

- `regulatory_updates`
- `internal_policies`

### 4. Load internal policies

The current ingestion path is PDF-based:

```bash
legal-agent load-policies-pdf "data/policies/your-policy.pdf"
```

This pipeline parses the PDF with Docling, chunks it, enriches each chunk with metadata, computes embeddings, and upserts the result into `internal_policies`.

### 5. Configure crawl targets

Edit `data/sources/targets.json`. Each target defines:

- one or more `start_urls`
- the `jurisdiction`
- crawl limits such as `max_depth`, `max_pages`, and `max_pdfs`
- whether HTML links and PDF links should be followed

### 6. Scrape regulations

```bash
legal-agent scrape
```

You can also point to a different source file:

```bash
legal-agent scrape --sources path/to/targets.json
```

### 7. Run the workflow

```bash
legal-agent run-workflow
```

The workflow processes one unprocessed regulatory document per run, marks it as processed in Qdrant, and writes Markdown and JSON reports into `data/reports/`.

### 8. Check system status

```bash
legal-agent status
```

## Structural Choices

### Why LlamaIndex Workflows

For an event-driven agentic system that stays evaluable. `LlamaIndex Workflows` fits that well because it gives:

- explicit step boundaries instead of a free-form agent loop
- typed events between steps
- easier tracing in observability tools
- a simpler control flow than a large graph orchestration layer

This repo uses one workflow class, `ComplianceWorkflow`, with explicit `@step` methods and custom event models in `workflow/events.py`.

### Why Qdrant

Qdrant is used as both the staging layer and the retrieval layer:

- `regulatory_updates` stores scraped regulation chunks plus processing state
- `internal_policies` stores chunked internal policy text for retrieval

The collections are configured with dense and sparse vectors so the system can support hybrid retrieval rather than pure cosine similarity alone.

### LLM and embedding choices

The code supports a few operating modes:

- **Default mode**: OpenAI for workflow LLM calls and OpenAI embeddings
- **Gemini workflow mode**: Gemini for step execution, while embeddings can still remain OpenAI unless local legal embeddings are enabled
- **Local legal model mode**: optional legal-domain embedding and small language models for offline or domain-specific experimentation

The current model routing is intentional:

- `analyst`, `redliner`, and `auditor` can use a stronger model
- `relevance_check` can use a cheaper model
- enrichment can be split onto a lower-cost model

That separation keeps cost-sensitive steps cheap while preserving quality where reasoning quality matters most.

### Retrieval design

Policy retrieval is not just plain vector search. The `Librarian` step uses:

- dense embeddings for semantic similarity
- sparse embeddings for lexical/legal-term recall
- reciprocal rank fusion in Qdrant
- optional VoyageAI reranking for the final shortlist

This is a sensible fit for legal/compliance text, where exact clause language often matters as much as semantic similarity.

### Why an explicit auditor step

The project plan emphasized hallucination control and evaluability. Instead of trusting a single generation pass, the workflow adds an `Auditor` step that:

- checks for unsupported references and logic gaps
- returns `PASS` or `FAIL`
- can trigger a retry loop back through drafting

That makes failure modes visible and gives you an auditable artifact instead of a silent best-effort answer.

## Evaluation And Observability

The repository already includes instrumentation hooks for the evaluation stack described in the project plan:

- **Langfuse**: trace, latency, and cost visibility
- **Arize Phoenix**: OpenInference/LlamaIndex tracing for RAG and workflow inspection
- **LangWatch**: guardrail and monitoring integration

### How to run eval-oriented workflow sessions

1. Configure the relevant observability keys in `.env`.
2. Run `legal-agent run-workflow`.
3. Inspect the saved report in `data/reports/`.
4. Review traces in Langfuse and Phoenix for the same run.

### What you can evaluate today

- **Retrieval quality**: inspect which policy chunks the `Librarian` retrieved for a regulation.
- **Reasoning quality**: compare the `Analyst` gap table against the retrieved policy evidence.
- **Draft quality**: review whether the `Redliner` output actually closes the listed gaps.
- **Hallucination resistance**: use the `Auditor` result and notes as a first-pass factuality check.
- **Operational behavior**: use Langfuse and Phoenix to inspect cost, latency, and step execution.

### Important note on eval scope

This repo currently provides instrumentation and inspectable artifacts, not a full standalone benchmark harness with labeled datasets and automated scoring scripts. In practice, evaluation is performed by:

- running representative scrape and workflow jobs
- inspecting traces and retrieved context
- reviewing generated reports
- extending the current test suite with task-specific regression cases when needed

### Tests

Run the current automated tests with:

```bash
pytest
```

The existing tests are lightweight smoke tests around chunking and workflow event construction. They are useful for sanity checks, but they are not a substitute for workflow-level evaluation on real regulatory samples.

## CLI Commands

| Command | Purpose |
|---|---|
| `legal-agent init-db` | Create the Qdrant collections and payload indexes |
| `legal-agent scrape` | Crawl configured regulatory sources and stage results |
| `legal-agent run-workflow` | Process unprocessed regulations through the compliance workflow |
| `legal-agent load-policies-pdf FILE` | Ingest a policy PDF into the `internal_policies` collection |
| `legal-agent status` | Show collection sizes and unprocessed regulation count |

## Repository Structure

```text
.
├── data/
│   ├── policies/        # Source policy PDFs used for ingestion
│   ├── reports/         # Generated Markdown and JSON workflow outputs
│   └── sources/         # Crawl target definitions
├── src/legal_agent/
│   ├── cli.py           # Click CLI entrypoint
│   ├── config.py        # Environment-backed application settings
│   ├── db/              # Qdrant client and collection schema setup
│   ├── instrumentation/ # Langfuse, Phoenix, and LangWatch setup
│   ├── scraping/        # Spider, item models, and ingestion pipelines
│   ├── utils/           # Embeddings, report writing, policy loaders
│   └── workflow/        # Event models, prompts, LLM routing, workflow logic
├── tests/               # Smoke tests for chunking and workflow events
├── Project_map.md       # Original implementation plan and evaluation goals
└── README.md
```

## End-To-End Flow

```text
targets.json
   -> Scrapy spider
   -> Docling PDF conversion / HTML extraction
   -> chunking + metadata enrichment
   -> Qdrant.regulatory_updates
   -> ComplianceWorkflow
   -> hybrid retrieval from Qdrant.internal_policies
   -> analysis + drafting + audit
   -> report written to data/reports
```

## License

See `LICENSE`.
