#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-2h}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

echo "Active Ollama models before cleanup:"
ollama ps

while read -r active_model; do
  if [[ -n "$active_model" && "$active_model" != "$MODEL" ]]; then
    echo "Stopping non-target model: $active_model"
    ollama stop "$active_model"
  fi
done < <(ollama ps | awk 'NR > 1 {print $1}')

echo "Warming $MODEL with keep_alive=$KEEP_ALIVE"
curl -s "$OLLAMA_BASE_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":\"$KEEP_ALIVE\"}" \
  >/dev/null

echo "Active Ollama models after warmup:"
ollama ps
