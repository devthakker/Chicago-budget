# Local Launch

This runbook is for fully launching the project on your machine for development or local demos.

## What You Need

- Python 3
- `pdftotext` available on your machine
- Optional: Docker
- Optional model provider:
  - OpenAI
  - Ollama
  - AWS Bedrock

## Python Launch

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional reranker support:

```bash
pip install -r requirements-reranker.txt
```

## Provider Setup

### OpenAI

```bash
export LLM_PROVIDER=openai
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your_key
export OPENAI_CHAT_MODEL=gpt-4.1-mini
export OPENAI_EMBED_MODEL=text-embedding-3-small
```

### Ollama

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the models used by this project:

```bash
ollama pull llama3.2:latest
ollama pull qwen3-embedding:4b
```

Set env vars:

```bash
export LLM_PROVIDER=ollama
export EMBEDDING_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_CHAT_MODEL=llama3.2:latest
export OLLAMA_EMBED_MODEL=qwen3-embedding:4b
```

### AWS Bedrock

For this app, the cost-efficient Bedrock starting point is:

- chat: `Amazon Nova Micro`
- embeddings: `Amazon Titan Text Embeddings V2`

For `Amazon Nova Micro`, Bedrock on-demand inference uses an inference profile. Use an inference profile ID or ARN for `BEDROCK_CHAT_MODEL`, not the raw model ID.

Verify the available Nova Micro inference profiles in your account and region:

```bash
aws bedrock list-inference-profiles \
  --region us-east-2 \
  --type-equals SYSTEM_DEFINED \
  --query 'inferenceProfileSummaries[?contains(inferenceProfileName, `Nova Micro`) || contains(inferenceProfileId, `nova-micro`)].[inferenceProfileId,inferenceProfileArn]' \
  --output table
```

```bash
export LLM_PROVIDER=bedrock
export EMBEDDING_PROVIDER=bedrock
export AWS_REGION=us-east-2
export BEDROCK_CHAT_MODEL=replace_with_nova_micro_inference_profile_id_or_arn
export BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0
```

## Build The Index

Default chunking is already tuned:

```bash
python3 build_index.py --pdf-dir . --index-dir data/index
python3 build_structured_budget.py
```

If you change embedding provider or embedding model, rebuild the index.

## Run The App

```bash
uvicorn app:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

## Query From CLI

```bash
python3 query_rag.py "What grants mention ARPA?"
```

Override retrieval blend:

```bash
python3 query_rag.py "What grants mention ARPA?" --bm25-weight 0.9 --vector-weight 0.1
```

## Evaluate And Tune Retrieval

Run evaluation:

```bash
python3 eval_rag.py --questions-file eval/questions.sample.json --top-k 8 --show-queries
```

Run tuning grid:

```bash
python3 eval_rag.py --questions-file eval/questions.sample.json --top-k 8 --tune --bm25-grid "0.7,0.8,0.85,0.9,0.95"
```

## Docker Launch

Basic launch:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

First start builds the index automatically.

Force rebuild if embedding config changed:

```bash
FORCE_REINDEX=1 docker compose up --build
```

### OpenAI With Docker

```bash
LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai OPENAI_API_KEY=your_key docker compose up --build
```

### Ollama With Docker

```bash
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama OLLAMA_BASE_URL=http://host.docker.internal:11434 OLLAMA_CHAT_MODEL=llama3.2:latest OLLAMA_EMBED_MODEL=qwen3-embedding:4b docker compose up --build
```

### Bedrock With Docker

```bash
LLM_PROVIDER=bedrock EMBEDDING_PROVIDER=bedrock AWS_REGION=us-east-2 docker compose up --build
```

## Export

The UI supports exporting query results to Markdown, JSON, and CSV.

Direct export endpoint:

```bash
curl -L "http://localhost:8000/export?query=What%20grants%20mention%20ARPA%3F&fmt=markdown" -o export.md
```

Supported `fmt` values:

- `markdown`
- `json`
- `csv`

## Runtime Controls

Rate limiting defaults:

- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_MAX_REQUESTS=20`
- `RATE_LIMIT_WINDOW_SECONDS=60`

Disable the public site without taking the service down:

```bash
SITE_ENABLED=false
SITE_DISABLED_REPO_URL=https://github.com/your-org/your-repo
```

## Ubuntu Docker Fix

If `docker-compose-plugin` is missing:

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git
```
