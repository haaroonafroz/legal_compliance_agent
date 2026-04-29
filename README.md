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

## Tools in Use

We use a few key tools to build this agent:

* **Scrapy:** Crawls websites to find new legal texts.
* **Docling:** Turns PDF files into plain text.
* **Qdrant:** A smart database. It stores and finds the right text chunks fast.
* **LlamaIndex:** Runs the step-by-step agent logic.
* **Langfuse & Phoenix:** Tracks costs and AI choices so you can test the agent.

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
- `OPENAI_LLM_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `QDRANT_URL`
- `QDRANT_API_KEY`

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

The agent will process one new law. It will save a report in the `data/reports/` folder.

## CLI Commands

| Command | What It Does |
|---|---|
| `legal-agent init-db` | Sets up the database. |
| `legal-agent scrape` | Finds new laws on the web. |
| `legal-agent run-workflow` | Runs the main agent task. |
| `legal-agent load-policies-pdf` | Loads a PDF policy into the database. |
| `legal-agent status` | Shows how many laws are left to check. |

## License

See `LICENSE`.
