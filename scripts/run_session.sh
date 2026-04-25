#!/usr/bin/env bash
# Scheduled agent session runner (cron-friendly).
#
# Usage in cron (run every day at 03:00):
#   0 3 * * * cd /path/to/ai-research && bash scripts/run_session.sh >> reports/agent/cron.log 2>&1

set -euo pipefail

ROUNDS="${AUTOSIGNALX_ROUNDS:-5}"
MODE="${AUTOSIGNALX_MODE:-debate}"

cd "$(dirname "$0")/.."

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting agent session (mode=$MODE, rounds=$ROUNDS)"

uv run autosignalx agent run --mode "$MODE" --max-rounds "$ROUNDS" --record-replay
uv run autosignalx agent score-traces || true
uv run autosignalx agent consolidate || true

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Session complete."
