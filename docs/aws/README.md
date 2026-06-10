# AWS Launch

This runbook is for fully launching the app publicly on AWS Lightsail, with DNS managed in Vercel.

## Architecture

- Compute: one Ubuntu Lightsail instance
- App: Docker Compose
- TLS and reverse proxy: Caddy
- DNS: Vercel DNS pointing to a Lightsail static IP
- LLM provider: OpenAI or Bedrock

## Recommended Size

Start with:

- Lightsail Linux/Unix
- `2 GB RAM / 2 vCPUs / 60 GB SSD`

That is the practical low-cost starting point for this app. It leaves enough room for Docker, the local search index, and a public web process without pushing memory too hard.

## 1. Provision Lightsail

In AWS Lightsail:

1. Create an instance.
2. Choose:
   - platform: Linux/Unix
   - blueprint: Ubuntu 22.04 LTS
   - plan: `2 GB RAM / 2 vCPUs / 60 GB SSD`
3. Name the instance.
4. Create it.

Then attach a static IP to that instance from the Lightsail console.

## 2. Open The Required Ports

Lightsail uses instance firewalls. Open:

- `22` for SSH
- `80` for HTTP
- `443` for HTTPS

Prefer restricting `22` to your own IP range if possible.

Reference: [Control instance traffic with firewalls in Lightsail](https://docs.aws.amazon.com/lightsail/latest/userguide/understanding-firewall-and-port-mappings-in-amazon-lightsail.html)

## 3. SSH Into The Box

```bash
ssh -i /path/to/key.pem ubuntu@<LIGHTSAIL_STATIC_IP>
```

## 4. Install Docker

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

## 5. Deploy The App

Clone the repo:

```bash
git clone https://github.com/devthakker/Chicago-budget.git
cd "Chicago budget"
```

## 6. Create Runtime Config

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

Lightsail does not give you the same straightforward EC2-instance-profile path most people use on EC2. For the simplest setup, configure AWS credentials explicitly on the server with `aws configure`, or export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`.

For this app, the cost-efficient Bedrock recommendation is:

- chat model: `Amazon Nova Micro`
- embedding model: `Amazon Titan Text Embeddings V2`

Why:

- AWS documents Titan Text Embeddings V2 as their retrieval-oriented embedding model for RAG and document search, with model ID `amazon.titan-embed-text-v2:0`. [Titan embedding docs](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
- AWS pricing shows Anthropic Claude 3.5 Sonnet extended access in `US East (Ohio)` at `$6.00` per 1M input tokens and `$30.00` per 1M output tokens, which is substantially more expensive than the low-cost Bedrock-native path you should start with for a public search app. [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)

Because Bedrock's public model catalog is now surfaced through an interactive listing, verify the exact chat model ID available in your account and region before launch:

```bash
aws bedrock list-foundation-models \
  --region us-east-2 \
  --query 'modelSummaries[?providerName==`Amazon`].[modelId]' \
  --output text
```

Use the cheapest text-generation-capable Amazon Nova model returned there. The template defaults to `amazon.nova-micro-v1:0`, which is the intended low-cost starting point.

Copy the template:

```bash
cp .env.bedrock.example .env
nano .env
```

Set at minimum:

```env
AWS_ACCESS_KEY_ID=replace_with_your_access_key
AWS_SECRET_ACCESS_KEY=replace_with_your_secret_key
SITE_DISABLED_REPO_URL=https://github.com/your-org/your-repo
```

## 7. Verify Bedrock Before Launch

Run this only if you are using Bedrock.

First confirm the server can authenticate to AWS:

```bash
aws sts get-caller-identity
```

If the `aws` CLI is not installed:

```bash
sudo apt update
sudo apt install -y awscli
```

Then verify Bedrock model access with a direct invocation:

```bash
python3 - <<'PY'
import os
import boto3

region = os.environ.get("AWS_REGION", "us-east-2")
model_id = os.environ.get("BEDROCK_CHAT_MODEL", "amazon.nova-micro-v1:0")

client = boto3.client("bedrock-runtime", region_name=region)

response = client.converse(
    modelId=model_id,
    messages=[{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}],
    inferenceConfig={"maxTokens": 32, "temperature": 0.1},
)
print(response["output"]["message"]["content"])
PY
```

If this succeeds, your AWS credentials, region, and Bedrock chat model access are working.

If it fails:

- check that `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are valid
- confirm `AWS_REGION` matches the region where the model is enabled
- confirm your AWS account has access to the configured `BEDROCK_CHAT_MODEL`

## 8. Start The App

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

## 9. Put Caddy In Front

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

## 10. Configure Vercel DNS

If you want a root domain:

- Type: `A`
- Name: `@`
- Value: `<YOUR_LIGHTSAIL_STATIC_IP>`

If you want a subdomain like `budget.example.com`:

- Type: `A`
- Name: `budget`
- Value: `<YOUR_LIGHTSAIL_STATIC_IP>`

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

## 11. Verify

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

## 12. Operations

Update deployment:

```bash
cd "Chicago budget"
git pull
docker compose --env-file .env up --build -d
```

Force one-time reindex after retrieval or embedding changes:

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

## 13. Optional Ollama On AWS

If you want Ollama instead of OpenAI or Bedrock:

- do not use the small Lightsail plan
- use a larger box, ideally outside Lightsail if you need serious local inference
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

If cost is the priority, use Bedrock or OpenAI instead of Ollama on AWS.

## 14. Common Issues

- HTTPS not issuing:
  - confirm Lightsail ports `80` and `443` are open
  - confirm Vercel DNS points to the Lightsail static IP
  - check `sudo journalctl -u caddy -n 200 --no-pager`

- App not reachable:
  - `docker compose ps`
  - `docker compose logs --tail=200`
  - confirm Caddy points to `127.0.0.1:8000`

- Bedrock authentication failing:
  - verify `aws sts get-caller-identity`
  - verify `AWS_REGION`
  - verify the account has model access in Bedrock

- Stale retrieval after config changes:
  - run one deploy with `FORCE_REINDEX=1`
