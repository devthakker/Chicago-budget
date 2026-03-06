# Full AWS Deployment (With Vercel DNS)

This guide deploys the app publicly on AWS EC2 using Docker, with DNS managed in Vercel.

## Architecture
- Compute: 1 EC2 instance (Ubuntu 22.04)
- App: Docker Compose (`web` service)
- TLS + reverse proxy: Caddy on host
- DNS: Vercel DNS (A records to EC2 Elastic IP)

## 1) AWS prerequisites

1. Launch an EC2 instance:
- AMI: Ubuntu 22.04 LTS
- Type: `t3.large` minimum for basic use
- Storage: at least 30 GB

2. Security Group inbound rules:
- `22` (SSH) from your IP only
- `80` (HTTP) from `0.0.0.0/0`
- `443` (HTTPS) from `0.0.0.0/0`

3. Allocate and attach an Elastic IP to this instance.

4. SSH into instance:

```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

## 2) Install Docker + Git

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git curl
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

Confirm:

```bash
docker --version
docker compose version
```

## 3) Deploy the app

```bash
git clone <YOUR_REPO_URL>
cd "Chicago budget"
```

Create `.env` for runtime config.

Option A: OpenAI (`.env` example):

```bash
cat > .env <<'ENV'
PORT=8000
FORCE_REINDEX=1

LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=replace_with_your_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

RAG_BM25_WEIGHT=0.85
RAG_VECTOR_WEIGHT=0.15
RAG_RERANKER=auto
RAG_SUPPRESS_TOC=true

RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX_REQUESTS=20
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_METHOD=POST
RATE_LIMIT_PATH=/
RATE_LIMIT_TRUST_PROXY=true
ENV
```

Option B: Bedrock (`.env` example):

```bash
cat > .env <<'ENV'
PORT=8000
FORCE_REINDEX=1

LLM_PROVIDER=bedrock
EMBEDDING_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_CHAT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0

RAG_BM25_WEIGHT=0.85
RAG_VECTOR_WEIGHT=0.15
RAG_RERANKER=auto
RAG_SUPPRESS_TOC=true

RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX_REQUESTS=20
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_METHOD=POST
RATE_LIMIT_PATH=/
RATE_LIMIT_TRUST_PROXY=true
ENV
```

You can also copy the OpenAI template from this repo:

```bash
cp .env.openai.example .env
# then edit OPENAI_API_KEY
```

Start:

```bash
docker compose --env-file .env up --build -d
```

Check logs:

```bash
docker compose logs -f --tail=200
```

After first successful index build, set `FORCE_REINDEX=0` in `.env` and restart:

```bash
docker compose --env-file .env up -d
```

## 4) Put Caddy in front for HTTPS

Install Caddy:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

Create Caddy config (replace domain names):

```bash
sudo tee /etc/caddy/Caddyfile > /dev/null <<'CADDY'
example.com, www.example.com {
    reverse_proxy 127.0.0.1:8000
}
CADDY
```

Reload:

```bash
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

## 5) Configure Vercel DNS

Your DNS is already on Vercel, so add records in your Vercel domain settings:

1. Apex/root record:
- Type: `A`
- Name: `@`
- Value: `<YOUR_ELASTIC_IP>`
- TTL: default

2. `www` record (pick one):
- Preferred: `CNAME` name `www` -> `@`
- Or `A` name `www` -> `<YOUR_ELASTIC_IP>`

3. Wait for propagation and verify:

```bash
dig +short example.com
dig +short www.example.com
```

Expected: both resolve to your Elastic IP.

## 6) Final verification

From your local machine:

```bash
curl -I http://example.com
curl -I https://example.com
curl -I https://www.example.com
```

Then open in browser:
- `https://example.com`
- `https://www.example.com`

## 7) Operations

Update app:

```bash
cd "Chicago budget"
git pull
docker compose --env-file .env up --build -d
```

View logs:

```bash
docker compose logs -f --tail=200
```

Restart:

```bash
docker compose restart
```

## 8) Optional: Ollama on AWS

If you want Ollama in AWS instead of Bedrock:
- Use a GPU instance for acceptable latency/quality.
- Install Ollama on host, pull models.
- Set in `.env`:

```bash
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434
OLLAMA_CHAT_MODEL=llama3.2:latest
OLLAMA_EMBED_MODEL=qwen3-embedding:4b
```

If Ollama runs on the same EC2 host and app runs in Docker, point `OLLAMA_BASE_URL` to a host-reachable address (not `localhost` inside the container).

## 9) Common issues

- HTTPS not issuing:
  - Confirm ports `80/443` are open in EC2 security group.
  - Confirm Vercel DNS records point to Elastic IP.
  - Check `sudo journalctl -u caddy -n 200 --no-pager`.

- App not reachable:
  - `docker compose ps`
  - `docker compose logs --tail=200`
  - Confirm Caddy points to `127.0.0.1:8000`.

- Bad/old retrieval results after config change:
  - Set `FORCE_REINDEX=1` once and redeploy.
