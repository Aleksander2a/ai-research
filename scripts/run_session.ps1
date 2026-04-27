# Scheduled agent session runner for Windows Task Scheduler.
#
# Pipeline (idempotent, resumable):
#   1. agent run --mode lab|debate|single   -- discover hypotheses
#   2. agent score-traces                   -- LLM-as-judge per round
#   3. agent consolidate                    -- append session to lessons.md
#   4. agent harden                         -- FDR + adversarial + Phase 8 + Phase 12 gates
#   5. agent eval-suite                     -- Phase 15 calibration + RedTeam + coherence
#   6. snapshot build                       -- regenerate static cockpit snapshot
#
# Usage:
#   $action = New-ScheduledTaskAction -Execute "pwsh.exe" -Argument "-File C:\path\to\run_session.ps1"
#   $trigger = New-ScheduledTaskTrigger -Daily -At 3:00am
#   Register-ScheduledTask -TaskName "AutoSignalX-Agent" -Action $action -Trigger $trigger

$ErrorActionPreference = "Stop"

$Rounds = if ($env:AUTOSIGNALX_ROUNDS) { $env:AUTOSIGNALX_ROUNDS } else { "5" }
$Mode   = if ($env:AUTOSIGNALX_MODE)   { $env:AUTOSIGNALX_MODE }   else { "debate" }

Set-Location (Join-Path (Split-Path $PSScriptRoot -Parent) ".")

function Log($msg) { Write-Host "[$(Get-Date -Format 'o')] $msg" }
function TryStep($cmd) { try { Invoke-Expression $cmd } catch { Log "step failed (continuing): $_" } }

Log "Starting agent session (mode=$Mode, rounds=$Rounds)"

uv run autosignalx agent run --mode $Mode --max-rounds $Rounds --record-replay
Log "agent run complete"

TryStep "uv run autosignalx agent score-traces"
TryStep "uv run autosignalx agent consolidate"
TryStep "uv run autosignalx agent harden"
TryStep "uv run autosignalx agent eval-suite"
TryStep "uv run autosignalx snapshot build"

Log "Session complete (discovery + hardening + eval-suite + snapshot)."
