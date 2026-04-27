#!/usr/bin/env bash
# Scheduled agent session runner (cron-friendly).
#
# Pipeline (every step is idempotent and resumable):
#   1. agent run --mode lab|debate|single        -- discover hypotheses
#   2. agent score-traces                        -- LLM-as-judge per round
#   3. agent consolidate                         -- append session to lessons.md
#   4. agent harden                              -- FDR + adversarial + Phase 8 + Phase 12 gates
#   5. agent eval-suite                          -- Phase 15 calibration + RedTeam + coherence + prompt scoring
#   6. snapshot build                            -- regenerate the static cockpit snapshot
#
# Usage in cron (run every day at 03:00):
#   0 3 * * * cd /path/to/ai-research && bash scripts/run_session.sh >> reports/agent/cron.log 2>&1

set -euo pipefail

ROUNDS="${AUTOSIGNALX_ROUNDS:-5}"
MODE="${AUTOSIGNALX_MODE:-debate}"

cd "$(dirname "$0")/.."

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] Starting agent session (mode=$MODE, rounds=$ROUNDS)"

uv run autosignalx agent run --mode "$MODE" --max-rounds "$ROUNDS" --record-replay
echo "[$(ts)] agent run complete"

uv run autosignalx agent score-traces        || true
uv run autosignalx agent consolidate         || true
uv run autosignalx agent harden              || true
uv run autosignalx agent eval-suite          || true
uv run autosignalx snapshot build            || true

echo "[$(ts)] Session complete (discovery + hardening + eval-suite + snapshot)."
