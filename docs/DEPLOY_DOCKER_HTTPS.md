# Deploy Docker + HTTPS

This guide runs the EXACT FastAPI agent in Docker and keeps Ollama on the host
machine. Public traffic should hit only the FastAPI container. The API exposes:

- `POST /predict`
- `GET /v1/models`

The `/v1/models` route proxies to Ollama and filters the response to the
configured model, `qwen2.5:7b`.

## 1. Prerequisites

Install and run:

- Docker Desktop
- Ollama
- `qwen2.5:7b` pulled in Ollama

Check Ollama:

```bash
ollama list
ollama ps
curl http://127.0.0.1:11434/v1/models
```

Keep only `qwen2.5:7b` active and keep it loaded:

```bash
cd /Users/appleidvpi/exact-2026
scripts/ollama_keep_qwen_loaded.sh
```

Expected:

```text
qwen2.5:7b    ...    100% GPU    ...    2 hours from now
```

## 2. Build And Run The API Container

From the project root:

```bash
cd /Users/appleidvpi/exact-2026
docker compose up --build -d
```

Check container status:

```bash
docker compose ps
docker compose logs -f exact-api
```

Local tests:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

Expected `/v1/models` should only show:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5:7b",
      "object": "model"
    }
  ]
}
```

Test `/predict`:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "T2_0001",
    "type": "type2",
    "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
    "premises": [],
    "options": []
  }'
```

## 3. HTTPS Option A: Cloudflare Tunnel

Use this if you cannot open ports 80/443 on your network.

Quick temporary tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8080
```

Cloudflare prints an HTTPS URL similar to:

```text
https://example-random.trycloudflare.com
```

Submit these URLs:

```text
Prediction URL: https://example-random.trycloudflare.com/predict
Model URL: https://example-random.trycloudflare.com/v1/models
```

For a stable domain, create a named tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create exact-2026
cloudflared tunnel route dns exact-2026 exact-api.yourdomain.com
```

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: exact-2026
credentials-file: /Users/appleidvpi/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: exact-api.yourdomain.com
    service: http://127.0.0.1:8080
  - service: http_status:404
```

Run:

```bash
cloudflared tunnel run exact-2026
```

Submit:

```text
Prediction URL: https://exact-api.yourdomain.com/predict
Model URL: https://exact-api.yourdomain.com/v1/models
```

## 4. HTTPS Option B: Caddy With Your Own Domain

Use this if your machine/server has a public IP and inbound ports 80/443 are
open.

Point DNS:

```text
exact-api.yourdomain.com -> your public IP
```

Create a `Caddyfile` outside the repo or in your deploy directory:

```caddyfile
exact-api.yourdomain.com {
    reverse_proxy 127.0.0.1:8080
}
```

Run Caddy:

```bash
caddy run --config Caddyfile
```

Submit:

```text
Prediction URL: https://exact-api.yourdomain.com/predict
Model URL: https://exact-api.yourdomain.com/v1/models
```

## 5. Before The Grading Slot

Run these checks:

```bash
scripts/ollama_keep_qwen_loaded.sh
docker compose up -d
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

Then check the public HTTPS URL:

```bash
curl https://exact-api.yourdomain.com/health
curl https://exact-api.yourdomain.com/v1/models
```

For submission speed, keep LangSmith disabled:

```text
EXACT_ENABLE_LANGSMITH=false
```

On macOS, prevent sleep during the slot:

```bash
caffeinate -dimsu
```

## 6. Stop Services

Stop the API container:

```bash
docker compose down
```

Unload Ollama model if needed:

```bash
ollama stop qwen2.5:7b
```
