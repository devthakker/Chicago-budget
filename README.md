# Chicago Budget RAG

RAG pipeline over:
- `chicago_Annual_Appropriation_Ordinance_2026.pdf`
- `chicago_Grant_Details_Ordinance_2026.pdf`

The system does:
- PDF text extraction (`pdftotext -layout`)
- Improved chunking (smaller chunks + section-aware boundaries)
- TOC suppression (TOC-like chunks are penalized and optionally filtered)
- Hybrid retrieval (BM25 + optional embeddings)
- Optional cross-encoder reranking path
- Answers with page-level citations

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional cross-encoder reranker dependencies:

```bash
pip install -r requirements-reranker.txt
```

Optional provider setup examples:

OpenAI:
```bash
export LLM_PROVIDER=openai
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your_key
export OPENAI_CHAT_MODEL=gpt-4.1-mini
export OPENAI_EMBED_MODEL=text-embedding-3-small
```

Ollama:
```bash
export LLM_PROVIDER=ollama
export EMBEDDING_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_CHAT_MODEL=llama3.2:latest
export OLLAMA_EMBED_MODEL=qwen3-embedding:4b
```

AWS Bedrock:
```bash
export LLM_PROVIDER=bedrock
export EMBEDDING_PROVIDER=bedrock
export AWS_REGION=us-east-1
export BEDROCK_CHAT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
export BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0
```

## 2) Build index

Default chunking now uses `max_tokens=450` and `overlap_tokens=70`:

```bash
python3 build_index.py --pdf-dir . --index-dir data/index
```

Override if needed:

```bash
python3 build_index.py --max-tokens 400 --overlap-tokens 60
```

## 3) Query from CLI

```bash
python3 query_rag.py "What is budgeted for the Office of the Mayor?"
```

Override retrieval blend from CLI:

```bash
python3 query_rag.py "What grants mention ARPA?" --bm25-weight 0.9 --vector-weight 0.1
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

## Docker

```bash
docker compose up --build
```

Then open [http://localhost:8000](http://localhost:8000).

Notes:
- On first start, the container auto-builds `data/index/index.json`.
- Index is stored in a persistent Docker volume (`rag_index`).
- Force rebuild when embedding provider/model changes:

```bash
FORCE_REINDEX=1 docker compose up --build
```

Provider examples in Docker:

```bash
LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai OPENAI_API_KEY=your_key docker compose up --build
```

```bash
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama OLLAMA_CHAT_MODEL=llama3.2:latest OLLAMA_EMBED_MODEL=qwen3-embedding:4b docker compose up --build
```

## Ollama install + models

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the two models used in this project:

```bash
ollama pull llama3.2:latest
ollama pull qwen3-embedding:4b
```

Run Docker with those two models:

```bash
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama OLLAMA_BASE_URL=http://host.docker.internal:11434 OLLAMA_CHAT_MODEL=llama3.2:latest OLLAMA_EMBED_MODEL=qwen3-embedding:4b docker compose up --build
```

```bash
LLM_PROVIDER=bedrock EMBEDDING_PROVIDER=bedrock AWS_REGION=us-east-1 docker compose up --build
```

## Retrieval tuning flags

Set these as env vars (local or Docker):
- `RAG_BM25_WEIGHT` (default `0.85`)
- `RAG_VECTOR_WEIGHT` (default `0.15`)
- `RAG_TOC_PENALTY` (default `0.35`)
- `RAG_SUPPRESS_TOC` (default `true`)
- `RAG_CANDIDATE_MULTIPLIER` (default `8`)

Reranking controls:
- `RAG_RERANKER=auto|cross-encoder|none` (default `auto`)
- `RAG_RERANK_CANDIDATES` (default `30`)
- `RAG_RERANKER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`)

If cross-encoder dependencies are not installed, reranking falls back to heuristic ranking.
