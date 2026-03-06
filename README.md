# Chicago Budget RAG

RAG pipeline over:
- `chicago_Annual_Appropriation_Ordinance_2026.pdf`
- `chicago_Grant_Details_Ordinance_2026.pdf`

The system does:
- PDF text extraction (`pdftotext -layout`)
- Chunking with page metadata
- Hybrid retrieval (BM25 + optional embeddings)
- Reranking heuristic
- Answers with page-level citations

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional for semantic retrieval + LLM answer synthesis:

```bash
export OPENAI_API_KEY=your_key
export LLM_PROVIDER=openai
export EMBEDDING_PROVIDER=openai
# optional model overrides
export OPENAI_CHAT_MODEL=gpt-4.1-mini
export OPENAI_EMBED_MODEL=text-embedding-3-small
```

Alternative providers:

Ollama (local):
```bash
export LLM_PROVIDER=ollama
export EMBEDDING_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_CHAT_MODEL=llama3.1
export OLLAMA_EMBED_MODEL=nomic-embed-text
```

AWS Bedrock:
```bash
export LLM_PROVIDER=bedrock
export EMBEDDING_PROVIDER=bedrock
export AWS_REGION=us-east-1
export BEDROCK_CHAT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
export BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0
# and standard AWS credentials env vars or IAM role
```

## 2) Build index

```bash
python3 build_index.py --pdf-dir . --index-dir data/index
```

## 3) Query from CLI

```bash
python3 query_rag.py "What is budgeted for the Office of the Mayor?"
```

JSON output:

```bash
python3 query_rag.py "What grants mention ARPA?" --json
```

## 4) Run web app

```bash
uvicorn app:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## Docker (one command)

```bash
docker compose up --build
```

Then open [http://localhost:8000](http://localhost:8000).

Notes:
- On first start, the container auto-builds `data/index/index.json` from the two PDFs.
- The index is stored in a persistent Docker volume (`rag_index`), so subsequent starts are faster.
- To force a fresh rebuild:

```bash
FORCE_REINDEX=1 docker compose up --build
```

- To enable OpenAI embeddings and answer synthesis:
 - To enable OpenAI embeddings and answer synthesis:

```bash
LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai OPENAI_API_KEY=your_key docker compose up --build
```

- To use local Ollama from Docker (default points to `host.docker.internal:11434`):

```bash
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama OLLAMA_CHAT_MODEL=llama3.2:latest OLLAMA_EMBED_MODEL=qwen3-embedding:4b docker compose up --build
```

- To use AWS Bedrock:

```bash
LLM_PROVIDER=bedrock EMBEDDING_PROVIDER=bedrock AWS_REGION=us-east-1 docker compose up --build
```

Important:
- If you change `EMBEDDING_PROVIDER` or embedding model, rebuild the index so vectors match the provider/model:

```bash
FORCE_REINDEX=1 docker compose up --build
```

## Index format

Index is stored in `data/index/index.json` and includes:
- chunk text + metadata (`source_file`, `page_start`, `page_end`, `section`)
- BM25 stats and per-chunk term frequencies
- optional embedding vectors

## Can this be shared as a public web app?

Yes. This project is ready to deploy after indexing.

Recommended deployment pattern:
1. Build index locally and commit/publish `data/index/index.json` if size is acceptable.
2. Deploy FastAPI app to Render, Fly.io, Railway, or a Docker host.
3. Set `OPENAI_API_KEY` in host secrets for best answer quality.
4. Add simple auth/rate limits if opening to the public.

For larger corpora later, move index storage to a vector DB and keep this same app interface.
