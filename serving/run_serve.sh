#!/usr/bin/env bash
# Start (or restart) the UTMOSv2 serving process, detached, with sane env.
# Usage:
#   MAX_BATCH=8 PP_WORKERS=8 DTYPE=bf16 bash run_serve.sh
set -uo pipefail
cd "$(dirname "$0")"
pkill -9 -f "uvicorn server:app" 2>/dev/null || true
sleep 2

export UTMOS_CONFIG="${UTMOS_CONFIG:-fusion_stage3}"
export FOLD="${FOLD:-0}"
export MAX_BATCH="${MAX_BATCH:-8}"
export MAX_WAIT_MS="${MAX_WAIT_MS:-10}"
export DTYPE="${DTYPE:-bf16}"
export NUM_FRAMES="${NUM_FRAMES:-}"          # empty = config default (2); "1" = ~4.7x fewer spec images
export PIPELINE="${PIPELINE:-1}"
export WARMUP="${WARMUP:-1}"
export PP_WORKERS="${PP_WORKERS:-8}"
export CPU_WORKERS="${CPU_WORKERS:-8}"
export PREDICT_DATASET="${PREDICT_DATASET:-sarulab}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# One uvicorn worker: the process pool (not uvicorn workers) provides preprocessing
# parallelism, and a single GPU wants a single owner. Scale via PP_WORKERS/MAX_BATCH.
exec python -m uvicorn server:app --host "$HOST" --port "$PORT"
