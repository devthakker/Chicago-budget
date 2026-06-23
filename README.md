# Chicago Budget RAG

This project is a public-facing RAG application for the Chicago FY2026 budget documents. It lets users ask plain-English questions about the Annual Appropriation Ordinance and the Grant Details Ordinance, then returns answers with page-level citations and direct links back to the source PDFs.

The system is designed around three goals:

- retrieval quality for long civic PDFs
- transparent source grounding with page citations
- practical deployment for public use

## What It Does

- extracts text from the two source PDFs
- chunks and indexes the content with page metadata
- blends BM25 and optional vector retrieval
- optionally reranks results with a cross-encoder
- generates cited answers using OpenAI, Bedrock, or Ollama
- lets users open exact source pages in a built-in viewer
- normalizes appropriation records into a structured explorer dataset
- exposes SEO-friendly department pages, fund pages, and a budget simulator
- supports exporting answers to Markdown, JSON, and CSV
- includes evaluation tooling and tuning scripts for retrieval quality

## Source Documents

- `chicago_Annual_Appropriation_Ordinance_2026.pdf`
- `chicago_Grant_Details_Ordinance_2026.pdf`

## Project Structure

- [app.py](</Users/devin/Documents/GitHub/Chicago budget/app.py>): FastAPI app, routes, SEO pages, export endpoints, analytics hooks, runtime controls
- [src/chicago_budget_rag/engine.py](</Users/devin/Documents/GitHub/Chicago budget/src/chicago_budget_rag/engine.py>): indexing, retrieval, reranking, answer generation, provider abstraction
- [templates/](</Users/devin/Documents/GitHub/Chicago budget/templates>): HTML templates for search, guides, disabled page, and rate-limit page
- [static/styles.css](</Users/devin/Documents/GitHub/Chicago budget/static/styles.css>): shared styling for the web UI
- [build_index.py](</Users/devin/Documents/GitHub/Chicago budget/build_index.py>): offline index build entrypoint
- [build_structured_budget.py](</Users/devin/Documents/GitHub/Chicago budget/build_structured_budget.py>): offline structured dataset build entrypoint
- [query_rag.py](</Users/devin/Documents/GitHub/Chicago budget/query_rag.py>): CLI query tool
- [eval_rag.py](</Users/devin/Documents/GitHub/Chicago budget/eval_rag.py>): retrieval evaluation and tuning harness
- [src/chicago_budget_rag/structured_budget.py](</Users/devin/Documents/GitHub/Chicago budget/src/chicago_budget_rag/structured_budget.py>): parser and aggregator for explorer/simulator data
- [eval/questions.sample.json](</Users/devin/Documents/GitHub/Chicago budget/eval/questions.sample.json>): starter benchmark set
- [docker-compose.yml](</Users/devin/Documents/GitHub/Chicago budget/docker-compose.yml>): containerized runtime configuration
- [Dockerfile](</Users/devin/Documents/GitHub/Chicago budget/Dockerfile>): image build definition
- [.env.openai.example](</Users/devin/Documents/GitHub/Chicago budget/.env.openai.example>): OpenAI-based runtime template
- [.env.bedrock.example](</Users/devin/Documents/GitHub/Chicago budget/.env.bedrock.example>): Bedrock-based runtime template for Lightsail or other AWS hosts

## Documentation

- Local launch: [docs/local/README.md](</Users/devin/Documents/GitHub/Chicago budget/docs/local/README.md>)
- AWS launch: [docs/aws/README.md](</Users/devin/Documents/GitHub/Chicago budget/docs/aws/README.md>)

## License

This project is released under the MIT License. See [LICENSE](</Users/devin/Documents/GitHub/Chicago budget/LICENSE>).

## Runtime Notes

- `POST /` redirects to canonical `GET /search?q=...` URLs for crawlable search pages.
- Curated search pages and guide pages are indexable; arbitrary search pages are `noindex,follow`.
- `/explorer`, `/simulator`, `/departments/*`, and `/funds/*` add structured, crawlable surfaces beyond the RAG query UI.
- `robots.txt` and `sitemap.xml` are served by the app.
- The public site can be disabled with `SITE_ENABLED=false` while keeping health checks alive.
- Query export is available through the UI and `GET /export`.

## Model Providers

The app supports:

- OpenAI
- AWS Bedrock
- Ollama

Provider selection and model configuration are environment-driven.

## SEO Surface

The app includes:

- canonical search result pages
- guide landing pages
- `robots.txt`
- `sitemap.xml`
- Open Graph and Twitter metadata
- JSON-LD for home, search, and guide pages
- internal linking through guides and curated searches

## Evaluation

Use [eval_rag.py](</Users/devin/Documents/GitHub/Chicago budget/eval_rag.py>) to measure hit rate and MRR across a benchmark set and tune BM25/vector blend settings before shipping retrieval changes.
