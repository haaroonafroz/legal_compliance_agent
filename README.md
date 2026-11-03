# Legal Agent

A tool to help legal teams track new laws and update company rules.

The agent finds new regulatory material online and compares extracted obligations with
written company policies. Its reports are draft policy-coverage assessments for human
review; they do not establish that operational controls work or that the organisation
is legally compliant.

## How It Works

The process has eight steps:

1. **Scrape:** It finds new laws on the web.
2. **Screen:** It filters irrelevant material with a bounded, low-cost relevance check.
3. **Extract:** It identifies quoted obligations in every source chunk.
4. **Retrieve:** It finds candidate policy text for each obligation.
5. **Analyze:** It assesses written policy coverage for each obligation.
6. **Draft:** It proposes text for identified policy gaps.
7. **Model audit:** It checks the draft for inconsistencies.
8. **Human review:** A person decides whether to approve or reject the report.

Workflow steps (Agents):

1. `Horizon Scanner`: groups unprocessed regulation chunks back into complete documents.
2. `Relevance Check`: filters out site noise before extraction and retrieval.
3. `Obligation Extractor`: returns source-quoted duties for each document chunk.
4. `Librarian`: runs hybrid retrieval against internal policies per obligation.
5. `Analyst`: assesses policy coverage with validated source and policy citations.
6. `Redliner`: drafts policy amendments in redline form.
7. `Auditor`: checks for hallucinations, weak reasoning, and draft mismatch before review.

## Tools in Use

We use a few key tools to build this agent:

- **Scrapy:** Crawls websites to find new legal texts.
- **Docling:** Turns PDF files into plain text.
- **PostgreSQL:** Durable document, job-claim, report, and review state.
- **Qdrant:** Stores and retrieves text chunks and vectors; it is not the source of truth for workflow state.
- **LlamaIndex:** Runs the step-by-step agent logic.
- **Langfuse & Phoenix:** Tracks costs and AI choices so you can test the agent.

# Architecture

The project was built around one core requirement from the original project plan: make the workflow inspectable and easy to evaluate. The architecture therefore favors explicit workflow steps, stored intermediate artifacts, and trace-friendly integrations over opaque agent loops.

```
Scrapy Spider ──► Qdrant (regulatory_updates)
                          │
          ComplianceWorkflow (LlamaIndex)
          ┌─────────┼─────────────────────┐
    Horizon ──► Obligation ──► Librarian ──► Analyst
    Scanner      Extractor     (RAG retrieval)  (coverage)
                                                │
                                     Redliner ──► Model audit
                                                    │
                                             Report ──► Human review
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

Start local PostgreSQL and Qdrant, then create the PostgreSQL schema and Qdrant collections.

```bash
docker compose up -d postgres qdrant
legal-agent init-db
```

For a managed database, set `DATABASE_URL` first and run `legal-agent migrate-db` as a separate, one-time release step. `init-db` also initializes Qdrant.

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

The agent processes all complete, pending regulatory documents and saves one JSON and
one durable JSON report per document in PostgreSQL, then tries to export JSON and
Markdown files to `data/reports/`. A model audit pass produces `pending_human_review`.

### 7. Record a human decision

Inspect the source, policy citations, proposed changes, and model audit notes in the
report. Then record a decision with a reviewer identity and rationale:

```bash
legal-agent review-report data/reports/<report-name>.md \
  --decision approve --reviewer "your-name" --reason "Reviewed source and policy evidence"
```

`approve` is allowed only for `pending_human_review`. A reviewer can `reject` a report
that is pending review or marked `review_required`. The decision and timestamp are
committed to PostgreSQL, then mirrored to local exports and Qdrant when available. The command does not edit an
internal policy or implement a control.

## CLI Commands


| Command                         | What It Does                           |
| ------------------------------- | -------------------------------------- |
| `legal-agent init-db`           | Sets up the database.                  |
| `legal-agent scrape`            | Finds new laws on the web.             |
| `legal-agent retry-dead-letters` | Requeues failed acquisitions for retry. |
| `legal-agent run-workflow`      | Runs the main agent task.              |
| `legal-agent review-report`     | Records a human approval or rejection. |
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
   Install the optional local server first with `pip install -e '.[phoenix]'`.
2. Open `http://localhost:6006` for the Phoenix UI.

### Run the workflow with tracing on

```bash
legal-agent run-workflow
```

If Langfuse keys are missing, Langfuse init is skipped (warning in logs). If PHOENIX_ENDPOINT is wrong or Phoenix is not reachable, Phoenix init may log a warning and skip RAG-style observability for that run.



# Configurations

**Note:** The `src/legal_agent/config.py` file is the central configuration hub for the entire workflow. You should use it to configure all LLMs, embedding models, tokenizers, and other system parameters before running the agent.

## Vector Collection

PostgreSQL governs acquisition and workflow state; Qdrant stages and retrieves text:

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

The PostgreSQL document manifest configured by `DATABASE_URL` tracks canonical source identities and
content revisions. Unchanged revisions are skipped, while changed revisions
supersede older ones. Conversion, model, and persistence failures become
dead-lettered records that can be requeued with `legal-agent retry-dead-letters`.

Regulatory documents move from `pending` to `pending_human_review`, `review_required`,
or `irrelevant`. Only an explicit `review-report --decision approve` transition marks
a document `approved`. Extraction failures, unsupported assessments, uncertain policy
coverage, weak retrieval, and failed model audits require review. No model output alone
can approve a report.

## Obligation-centric workflow

`legal-agent run-workflow` extracts duties, prohibitions, deadlines, and conditional
requirements from every source chunk. Malformed extraction or an empty result requires
review. The Librarian then runs a
separate dense+sparse policy search for every obligation, expands adjacent policy
chunks, and optionally reranks the evidence. The Analyst and Auditor receive this
obligation-to-evidence mapping, rather than one undifferentiated policy bundle.

The retrieval gate checks whether candidate evidence can be trusted. Weak matches
become `review_required`; no matches in a populated policy corpus produce a potential
gap for human review. An empty or unavailable policy corpus becomes `review_required`. The Analyst then returns one structured
policy-coverage assessment for each obligation. The code checks obligation IDs and
verbatim source and policy citations; malformed, missing, or uncertain assessments
also require review. Tune the bounded retrieval loop with
`OBLIGATION_RETRIEVAL_TOP_K`, `OBLIGATION_RETRIEVAL_FINAL_K`,
`OBLIGATION_MIN_EVIDENCE_HITS`, `OBLIGATION_MIN_RETRIEVAL_SCORE`, and
`OBLIGATION_NEIGHBOR_HOPS`.

## Container architecture and run

The Dockerfile builds one Python image with the CLI, workflow, scraping code, and
dependencies. The default command starts a batch workflow; the same image can run
`init-db`, `load-policies-pdf`, `scrape`, or `review-report` by overriding the command.

```mermaid
flowchart LR
    Dockerfile --> Image[Image: Python 3.12 slim, OS libraries, CPU PyTorch, Python package, source targets]
    Env[.env / runtime secrets] --> CLI
    Image --> CLI
    subgraph Container[Single batch-job container]
        CLI[legal-agent CLI]
        CLI --> Scrape[scrape]
        CLI --> Load[load-policies-pdf]
        CLI --> Workflow[run-workflow]
        CLI --> Review[review-report]
        Workflow --> Extract[Obligation extraction]
        Extract --> Retrieve[Policy retrieval]
        Retrieve --> Assess[Assessment, redline, model audit]
        Assess --> Report[JSON + Markdown report]
    end
    Web[Regulatory websites] --> Scrape
    PDFs[Policy PDFs] --> Load
    Scrape --> Regs[(Qdrant: regulatory_updates)]
    Load --> Policies[(Qdrant: internal_policies)]
    Regs --> Workflow
    Policies --> Retrieve
    Report --> Data[(PostgreSQL: manifest, claims, reports, reviews)]
    Data --> Review
    Review --> Regs
    Scrape --> Data
    Scrape --> Models
    Load --> Models
    Extract --> Models[External LLM and embedding APIs]
    Retrieve --> Models
    Assess --> Models
    Container --> Traces[Optional tracing services]
    Human[Human reviewer] --> Review
```

No queue, scheduler, API server, or multi-container deployment is included yet.
The container exits when its command finishes. Keep `data/` on durable storage so
optional local report exports and input policies survive container replacement. The
authoritative manifest and reports live in PostgreSQL. Qdrant, model endpoints, and
optional tracing services must be reachable from the container.
PostgreSQL claims use row locks and expiring leases so concurrent workers do not process
the same current document. The worker must finish within the lease; expired claims can
be retried. Reports and review decisions remain in PostgreSQL across container replacement.

Build and run the workflow manually:

```bash
docker build -t legal-agent .
docker run --rm --env-file .env \
  -e QDRANT_URL=https://your-qdrant-host \
  -v "$PWD/data:/app/data" legal-agent legal-agent run-workflow
```

The image defaults to hosted LLM mode, but `.env` can override it. A Docker container
needs database and Qdrant addresses reachable from inside the container; the sample
local `localhost` URLs in `.env.example` work only for commands run on the host.
The Docker image uses the CPU PyTorch wheels described in the
[PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).
Deployment to a cloud account still requires environment-specific infrastructure
and credentials.

The three synthetic cases in `evals/cases.json` check the Analyst's policy-coverage
outcomes against labeled expectations. With a configured model, run
`python evals/run_analyst_eval.py`. This is a small prompt check, not a legal-quality
benchmark; expand it with reviewed examples and retrieval/end-to-end cases before
relying on model performance claims.

## LLM and Embedding Model choices

The code supports several operating modes to balance cost, performance, and domain specificity:

- **Local llama.cpp mode (default):** workflow and enrichment calls use the OpenAI-compatible server configured by `LOCAL_LLM_BASE_URL` (default `http://localhost:8001/v1`). Qwen thinking is disabled for enrichment and relevance, medium for the Analyst, low for the Redliner, and `xHigh` for the Auditor.
- **OpenAI mode:** set `USE_LOCAL_LLM=false`. Enrichment uses `gpt-4o-mini`; relevance, analysis, redlining, and audit use `gpt-5.6-luna` by default.
- **Gemini workflow mode:** set `USE_LOCAL_LLM=false` and `USE_GEMINI=true`. Per-step thinking levels remain configurable in `src/legal_agent/config.py`.
- **Local Legal Embeddings:** you can use `nlpaueb/legal-bert-base-uncased` instead of OpenAI embeddings by setting `USE_LEGAL_EMBEDDINGS=true`.

### Agent Routing

The current model routing is intentionally designed to optimize both cost and quality:

- `analyst`, `redliner`, and `auditor` steps require deep legal reasoning, so they are routed to stronger models (or higher thinking modes if using Gemini).
- `obligation_extractor` uses the analyst route once per document to produce structured duties for focused retrieval.
- `relevance_check` acts as a fast filter and can safely use a cheaper, faster model.
- `enrichment` runs with thinking disabled on the local Qwen model, or uses `gpt-4o-mini` in OpenAI mode.

This separation keeps high-volume, cost-sensitive steps cheap while preserving top-tier quality where complex reasoning matters most.

## License

See `LICENSE`.

## PostgreSQL persistence and migration

`DATABASE_URL` is required for scraping, workflow processing, report review, and
status. PostgreSQL holds document revisions, deduplicated indexed chunk IDs, current
revision pointers, workflow claims and leases, JSON reports, and human decisions.
Qdrant holds text and vectors for retrieval. A Qdrant status payload is only a
best-effort mirror; a failed mirror cannot undo a committed review. Local JSON and
Markdown reports are exports and may be regenerated from PostgreSQL. The application
fails fast without `DATABASE_URL` instead of silently creating a new SQLite file.

For local development:

```bash
docker compose up -d postgres qdrant
cp .env.example .env
# Add your model and embedding credentials to .env.
legal-agent migrate-db
legal-agent init-db
```

The Compose PostgreSQL URL is
`postgresql://legal_agent:local_dev_only@localhost:5433/legal_agent` and is for
local development only. `init-db` applies the migration again safely before setting
up Qdrant. Use `docker compose down` to stop services; the named volumes retain
local data. Do not use `down -v` unless you intend to erase them.

For [Supabase](https://supabase.com/docs/guides/database/connecting-to-postgres),
create a project in your own account, then copy the PostgreSQL
connection string from **Connect** in its dashboard into a runtime secret named
`DATABASE_URL`. Use a direct connection for migrations when your network supports
its endpoint. If the runtime is IPv4 only, choose the session pooler URL from the
dashboard. Percent-encode reserved password characters. Require TLS with
`sslmode=require`; for certificate verification, download the project CA and use
`sslmode=verify-full&sslrootcert=/path/to/ca.crt`. Run `legal-agent migrate-db`
once with a role that can create tables and indexes, and use a separate role with
only table read/write permissions for routine workers. Restrict network access,
rotate credentials, and keep both the URL and CA path out of Git and image layers.
Back up the database before schema changes. This application does not require a
Supabase API key or service-role key. The SQL uses transaction-scoped locks and row
claims, so PostgreSQL is needed for the lifecycle guarantees; Redis is not a
replacement for durable reports or accountable review decisions.

[Neon](https://neon.com/docs/connect/connect-from-any-app) is another managed PostgreSQL option if you want its branching workflow.
[Cloud SQL for PostgreSQL](https://docs.cloud.google.com/sql/docs/postgres/connect-run) is a natural later choice when deploying Cloud Run Jobs on
Google Cloud. Neither is needed to validate the code locally; the standard
`DATABASE_URL` works across providers. No hosted service is provisioned by this
repository.

Existing local state is retained. Before switching a populated installation,
back up `data/state/document_manifest.sqlite3` and `data/reports/`, then preview:

```bash
legal-agent import-legacy-state
legal-agent import-legacy-state --apply
```

`--apply` requires an empty, already migrated PostgreSQL manifest and leaves the
SQLite database and report files untouched. It matches reports by document ID and
source hash, flags unmatched or uncredited decisions, and imports in one
transaction. Verify Qdrant still has the corresponding source chunks before
running workers. Old reports with no reliable identity remain local for manual
review. Run the preview first and inspect any warnings.

### Worker and review behavior

A workflow worker claims one current document in PostgreSQL for at most
`WORKFLOW_LEASE_SECONDS` (default 1800). The workflow timeout is 1200 seconds, so
the lease must be longer. A worker commits its result only if the same claim and
source hash are still current. Failed runs release their claim and exit with an
error; they are not marked complete. Expired claims can be taken by another
worker. Review decisions are conditional updates of the durable report and source
revision. Use either `legal-agent review-report <document-id> ...` or pass a local
report export path. Local exports and Qdrant status fields can lag the database.

### Fictional evaluation set

`evals/synthetic_cases.json` contains explicitly fictional internal policies and
GDPR/EDPB-inspired notices. They are **not legal advice or verbatim legal text**.
The cases cover full, partial, missing, ambiguous, outdated, conflicting, multiple,
and changed requirements, plus fabricated citations and prompt injection. Offline
tests verify the fixture and reject unsupported quotations. To seed one scenario
through the application's ingestion paths, run:

```bash
python -m evals.seed_synthetic --case full-coverage
python -m evals.seed_synthetic --case multiple-obligations
```

Seeding requires configured PostgreSQL, Qdrant, and embedding models and makes
network/model calls. Run one case at a time in a disposable collection because
unrelated fictional policies can affect retrieval. To evaluate the analyst LLM
without seeding or scraping, run:

```bash
python -m evals.run_analyst_eval --output evals/results.json
```

The live harness reports outcome accuracy, invalid output rate, mean latency, a
confusion table, and case details. It uses the configured LLM and may incur model
charges. `evals/results.json` is a local artifact and should not be committed.
