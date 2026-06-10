# AWS Launch

This runbook is for fully launching the app publicly on AWS EC2, with DNS managed in Vercel.

## Architecture

- Compute: one EC2 instance
- App: Docker Compose
- TLS and reverse proxy: Caddy
- DNS: Vercel DNS pointing to an Elastic IP

## 1. Provision EC2

Create an Ubuntu 22.04 instance with at least:

- `t3.large`
- 30 GB storage

Open inbound rules:

- `22` from your IP
- `80` from the public internet
- `443` from the public internet

Allocate and attach an Elastic IP.

SSH in:

```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

## 2. Install Docker

Use Docker's official repo:

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
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git curl
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker --version
docker compose version
```

## 3. Deploy The App

Clone the repo:

```bash
git clone <YOUR_REPO_URL>
cd "Chicago budget"
```

## 4. Create Runtime Config

### Option A: OpenAI

```bash
cp .env.openai.example .env
nano .env
```

Set at minimum:

```env
OPENAI_API_KEY=replace_with_your_key
FORCE_REINDEX=1
SITE_DISABLED_REPO_URL=https://github.com/your-org/your-repo
```

### Option B: Bedrock

Create `.env`:

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

SITE_ENABLED=true
SITE_DISABLED_REPO_URL=https://github.com/your-org/your-repo
ENV
```

## 5. Start The App

## 5. Verify Bedrock Before Launch

Run this only if you are using Bedrock.

First confirm the instance or shell can authenticate to AWS:

```bash
aws sts get-caller-identity
```

Then verify Bedrock model access with a direct invocation:

```bash
python3 - <<'PY'
import json
import os
import boto3

region = os.environ.get("AWS_REGION", "us-east-1")
model_id = os.environ.get("BEDROCK_CHAT_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")

client = boto3.client("bedrock-runtime", region_name=region)

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 32,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Reply with the single word: ok"}]}
    ],
}

response = client.invoke_model(
    modelId=model_id,
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json",
)

payload = json.loads(response["body"].read())
print(json.dumps(payload, indent=2))
PY
```

If this succeeds, your AWS credentials, region, and Bedrock chat model access are working.

If it fails:

- check that the EC2 instance role or AWS keys are present
- confirm `AWS_REGION` matches the region where the model is enabled
- confirm your AWS account has access to the configured `BEDROCK_CHAT_MODEL`

## 6. Start The App

```bash
docker compose --env-file .env up --build -d
```

Check logs:

```bash
docker compose logs -f --tail=200
```

After the first successful index build, set:

```env
FORCE_REINDEX=0
```

Then restart:

```bash
docker compose --env-file .env up -d
```

## 7. Put Caddy In Front

Install Caddy:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

Example Caddyfile:

```bash
sudo tee /etc/caddy/Caddyfile > /dev/null <<'CADDY'
example.com, www.example.com, budget.example.com {
    reverse_proxy 127.0.0.1:8000
}
CADDY
```

Reload:

```bash
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

## 8. Configure Vercel DNS

If you want a root domain:

- Type: `A`
- Name: `@`
- Value: `<YOUR_ELASTIC_IP>`

If you want a subdomain like `budget.example.com`:

- Type: `A`
- Name: `budget`
- Value: `<YOUR_ELASTIC_IP>`

Optional `www`:

- Type: `CNAME`
- Name: `www`
- Value: `@`

Verify:

```bash
dig +short example.com
dig +short www.example.com
dig +short budget.example.com
```

## 9. Verify

From your local machine:

```bash
curl -I http://example.com
curl -I https://example.com
curl -I https://budget.example.com
```

Also check:

```bash
curl -sS https://example.com/health
curl -sS https://example.com/robots.txt
curl -sS https://example.com/sitemap.xml
```

## 10. Operations

Update deployment:

```bash
cd "Chicago budget"
git pull
docker compose --env-file .env up --build -d
```

Force one-time reindex after retrieval/index changes:

```bash
nano .env
```

Set:

```env
FORCE_REINDEX=1
```

Restart:

```bash
docker compose --env-file .env up -d
```

After reindex completes, set it back to `0`.

View logs:

```bash
docker compose logs -f --tail=200
```

## 11. Optional Ollama On AWS

If you want Ollama instead of OpenAI or Bedrock:

- use a GPU instance
- install Ollama on the host
- pull the required models
- point the container at a host-reachable Ollama URL

Example `.env` values:

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434
OLLAMA_CHAT_MODEL=llama3.2:latest
OLLAMA_EMBED_MODEL=qwen3-embedding:4b
```

## 12. Common Issues

- HTTPS not issuing:
  - confirm ports `80` and `443` are open
  - confirm Vercel DNS points to the Elastic IP
  - check `sudo journalctl -u caddy -n 200 --no-pager`

- App not reachable:
  - `docker compose ps`
  - `docker compose logs --tail=200`
  - confirm Caddy points to `127.0.0.1:8000`

- Stale retrieval after config changes:
  - run one deploy with `FORCE_REINDEX=1`
