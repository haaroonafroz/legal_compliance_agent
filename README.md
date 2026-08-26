# Legal Agent

A tool to help legal teams track new laws and update company rules.

The agent finds new rules online. It checks them against your company rules. Then, it tells you what needs to change.

## How It Works

The agent runs in five clear steps:

1. **Scrape:** It finds new laws on the web.
2. **Retrieve:** It finds your company rules that match the new law.
3. **Analyze:** It spots the gaps between the law and your rules.
4. **Draft:** It writes text to fix your rules.
5. **Audit:** It checks its own work for mistakes.

Workflow steps (Agents):

1. `Horizon Scanner`: groups unprocessed regulation chunks back into complete documents.
2. `Librarian`: runs hybrid retrieval against internal policies.
3. `Relevance Check`: filters out site noise and non-regulatory content.
4. `Analyst`: compares regulation text against internal policy text.
5. `Redliner`: drafts policy amendments in redline form.
6. `Auditor`: checks for hallucinations, weak reasoning, and draft mismatch before finalizing.

## Tools in Use

We use a few key tools to build this agent:

- **Scrapy:** Crawls websites to find new legal texts.
- **Docling:** Turns PDF files into plain text.
- **Qdrant:** A smart database. It stores and finds the right text chunks fast.
- **LlamaIndex:** Runs the step-by-step agent logic.
- **Langfuse & Phoenix:** Tracks costs and AI choices so you can test the agent.

# Architecture

The project was built around one core requirement from the original project plan: make the workflow inspectable and easy to evaluate. The architecture therefore favors explicit workflow steps, stored intermediate artifacts, and trace-friendly integrations over opaque agent loops.

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

## Quick Start

Here is how to run the agent on your machine.

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Set Up Keys

Copy the example file to make your own config.

```bash
cp .env.example .env
```

Open the `.env` file. Add your keys.

#### Minimum requirements:

- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `LOCAL_LLM_BASE_URL` when `USE_LOCAL_LLM=true`

#### Optional but useful:

- `GEMINI_API_KEY`, `GEMINI_MODEL` if you want Gemini for workflow steps
- `VOYAGE_API_KEY`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `PHOENIX_ENDPOINT`
- `LANGWATCH_API_KEY`

### 3. Set Up the Database

Create the tables in Qdrant.

```bash
legal-agent init-db
```

### 4. Add Your Policy

Load your company rules into the database.

```bash
legal-agent load-policies-pdf "data/policies/your-policy.pdf"
```

### 5. Scrape New Laws

Find and save new rules from the web.

```bash
legal-agent scrape
```

You can change the sites to scrape in `data/sources/targets.json`.

### 6. Run the Agent

Start the main job.

```bash
legal-agent run-workflow
```

The agent processes all complete, pending regulatory documents and saves one report per
document in the `data/reports/` folder.

## CLI Commands


| Command                         | What It Does                           |
| ------------------------------- | -------------------------------------- |
| `legal-agent init-db`           | Sets up the database.                  |
| `legal-agent scrape`            | Finds new laws on the web.             |
| `legal-agent retry-dead-letters` | Requeues failed acquisitions for retry. |
| `legal-agent run-workflow`      | Runs the main agent task.              |
| `legal-agent load-policies-pdf` | Loads a PDF policy into the database.  |
| `legal-agent status`            | Shows how many laws are left to check. |




## Tracing and evals (quick start)

Observability runs when you call `legal-agent run-workflow`: `init_observability()` wires **Langfuse**, **Arize Phoenix**, and optionally **LangWatch**. Set the keys below in `.env`.

### Langfuse (cloud)

1. Create a project in [Langfuse Cloud](https://cloud.langfuse.com) and create API keys.
2. In `.env`:
  - `LANGFUSE_PUBLIC_KEY=...`
  - `LANGFUSE_SECRET_KEY=...`
  - `LANGFUSE_HOST=https://cloud.langfuse.com` (this is also the default in `config.py` if you omit it)
3. Run a workflow, then open the Langfuse UI and find your traces for that session.

### Langfuse (local with Docker)

Self‑hosted Langfuse needs its full stack (see the official [self‑hosting guide](https://langfuse.com/docs/deployment/self-host)). Typical pattern:

1. Use the upstream project’s `docker-compose` (Postgres, ClickHouse, MinIO, workers, web) and start it.
2. Finish first‑time setup in the web UI and create project keys.
3. Point the app at your instance, for example:
  - `LANGFUSE_HOST=http://localhost:3000`
  - `LANGFUSE_PUBLIC_KEY=...`
  - `LANGFUSE_SECRET_KEY=...`
4. **Dashboard:** open `http://localhost:3000` in your browser (or the host/port you published in compose).

### Arize Phoenix (local UI + traces)

This repo sends OTEL traces to Phoenix at `PHOENIX_ENDPOINT` (HTTP OTLP path `/v1/traces` is appended in code).

**Option A – Phoenix already running (Docker or manual server)**

1. Start Phoenix so the UI is on port **6006** (see [Arize Phoenix docs](https://docs.arize.com/phoenix) for the current Docker image / command).
2. In `.env`:
  - `PHOENIX_ENDPOINT=http://localhost:6006`
  - `PHOENIX_EXTERNAL=true`
   So the app does **not** try to spawn another server; it only exports spans to the running instance.
3. **Dashboard:** open `http://localhost:6006` and inspect traces / LlamaIndex spans.

**Option B – Let the workflow launch Phoenix**

1. In `.env`:
  - `PHOENIX_ENDPOINT=http://localhost:6006`
  - `PHOENIX_EXTERNAL=false`
   When `localhost` is in the URL and `PHOENIX_EXTERNAL` is false, `phoenix.launch_app()` can start the server for you during `run-workflow`.
2. Open `http://localhost:6006` for the Phoenix UI.

### Run the workflow with tracing on

```bash
legal-agent run-workflow
```

If Langfuse keys are missing, Langfuse init is skipped (warning in logs). If PHOENIX_ENDPOINT is wrong or Phoenix is not reachable, Phoenix init may log a warning and skip RAG-style observability for that run.



# Configurations

**Note:** The `src/legal_agent/config.py` file is the central configuration hub for the entire workflow. You should use it to configure all LLMs, embedding models, tokenizers, and other system parameters before running the agent.

## Vector Collection

Qdrant is used as both the staging layer and the retrieval layer:

- `regulatory_updates` stores scraped regulation chunks plus processing state
- `internal_policies` stores chunked internal policy text for retrieval
The collections are configured with dense and sparse vectors so the system can support hybrid retrieval rather than pure cosine similarity alone.

## Regulatory acquisition

Configured sources use source-specific adapters. The EDPB adapter polls the official
publications feed first and uses the configured publication pages as a targeted
backfill; it does not follow arbitrary links by default. EUR-Lex/CELLAR documents are
configured explicitly by URL or CELEX identifier.

Before a document reaches enrichment or Qdrant, deterministic checks validate its
status, extracted length, navigation content, and legal-content signals. Only
borderline candidates use the cheaper content-validation model. Rejected documents
are recorded in `QUARANTINE_FILE`.

The document manifest at `MANIFEST_FILE` tracks canonical source identities and
content revisions. Unchanged revisions are skipped, while changed revisions
supersede older ones. Conversion, model, and persistence failures become
dead-lettered records that can be requeued with `legal-agent retry-dead-letters`.

Regulatory documents move from `pending` to `approved`, `review_required`, or
`irrelevant`. A failed final audit is retained as `review_required`; it is not silently
treated as an approved compliance result.

## LLM and Embedding Model choices

The code supports several operating modes to balance cost, performance, and domain specificity:

- **Local llama.cpp mode (default):** workflow and enrichment calls use the OpenAI-compatible server configured by `LOCAL_LLM_BASE_URL` (default `http://localhost:8001/v1`). Qwen thinking is disabled for enrichment and relevance, medium for the Analyst, low for the Redliner, and `xHigh` for the Auditor.
- **OpenAI mode:** set `USE_LOCAL_LLM=false`. Enrichment uses `gpt-4o-mini`; relevance, analysis, redlining, and audit use `gpt-5.6-luna` by default.
- **Gemini workflow mode:** set `USE_LOCAL_LLM=false` and `USE_GEMINI=true`. Per-step thinking levels remain configurable in `src/legal_agent/config.py`.
- **Local Legal Embeddings:** you can use `nlpaueb/legal-bert-base-uncased` instead of OpenAI embeddings by setting `USE_LEGAL_EMBEDDINGS=true`.

### Agent Routing

The current model routing is intentionally designed to optimize both cost and quality:

- `analyst`, `redliner`, and `auditor` steps require deep legal reasoning, so they are routed to stronger models (or higher thinking modes if using Gemini).
- `relevance_check` acts as a fast filter and can safely use a cheaper, faster model.
- `enrichment` runs with thinking disabled on the local Qwen model, or uses `gpt-4o-mini` in OpenAI mode.

This separation keeps high-volume, cost-sensitive steps cheap while preserving top-tier quality where complex reasoning matters most.

## License

See `LICENSE`.
