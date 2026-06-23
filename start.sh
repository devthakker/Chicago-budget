#!/usr/bin/env sh
set -eu

INDEX_FILE="/app/data/index/index.json"
STRUCTURED_FILE="/app/data/structured/budget_dataset.json"

if [ "${FORCE_REINDEX:-0}" = "1" ] || [ ! -f "$INDEX_FILE" ]; then
  echo "[startup] Building index..."
  python3 /app/build_index.py --pdf-dir /app --index-dir /app/data/index
else
  echo "[startup] Using existing index at $INDEX_FILE"
fi

if [ "${FORCE_REINDEX:-0}" = "1" ] || [ ! -f "$STRUCTURED_FILE" ]; then
  echo "[startup] Building structured budget dataset..."
  python3 /app/build_structured_budget.py
else
  echo "[startup] Using existing structured dataset at $STRUCTURED_FILE"
fi

echo "[startup] Starting web app on 0.0.0.0:${PORT:-8000}"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
