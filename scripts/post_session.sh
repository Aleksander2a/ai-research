#!/usr/bin/env bash
# Post-live-session pipeline: score traces, consolidate lessons, harden the
# growing finding set, run the eval suite, and rebuild the snapshot.
#
# Idempotent: every step writes to its canonical artifact path and re-runs
# safely. Designed to be invoked between successive lab sessions so each
# session starts with up-to-date hardening + lessons in scope.

set -euo pipefail

cd "$(dirname "$0")/.."

PY=D:/ai-research/.venv/Scripts/python.exe
export PYTHONPATH=src

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[post-session $(ts)] $*"; }

log "score-traces"
$PY -m autosignalx.cli agent score-traces 2>&1 | tail -3 || log "score-traces failed (continuing)"

log "consolidate"
$PY -m autosignalx.cli agent consolidate 2>&1 | tail -3 || log "consolidate failed (continuing)"

log "self-critique"
$PY -m autosignalx.cli agent self-critique 2>&1 | tail -3 || log "self-critique failed (continuing)"

log "harden"
$PY -m autosignalx.cli agent harden 2>&1 | tail -3 || log "harden failed (continuing)"

log "eval-suite"
$PY -m autosignalx.cli agent eval-suite 2>&1 | tail -5 || log "eval-suite failed (continuing)"

log "ablate-capability"
$PY -m autosignalx.cli eval ablate-capability 2>&1 | tail -3 || log "ablate-capability failed (continuing)"

log "snapshot build"
$PY -m autosignalx.cli snapshot build 2>&1 | tail -3 || log "snapshot build failed (continuing)"

log "complete"
