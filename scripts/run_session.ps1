# Scheduled agent session runner for Windows Task Scheduler.
#
# Usage:
#   $action = New-ScheduledTaskAction -Execute "pwsh.exe" -Argument "-File C:\path\to\run_session.ps1"
#   $trigger = New-ScheduledTaskTrigger -Daily -At 3:00am
#   Register-ScheduledTask -TaskName "AutoSignalX-Agent" -Action $action -Trigger $trigger

$ErrorActionPreference = "Stop"

$Rounds = if ($env:AUTOSIGNALX_ROUNDS) { $env:AUTOSIGNALX_ROUNDS } else { "5" }
$Mode   = if ($env:AUTOSIGNALX_MODE)   { $env:AUTOSIGNALX_MODE }   else { "debate" }

Set-Location (Join-Path (Split-Path $PSScriptRoot -Parent) ".")

Write-Host "[$(Get-Date -Format 'o')] Starting agent session (mode=$Mode, rounds=$Rounds)"

uv run autosignalx agent run --mode $Mode --max-rounds $Rounds --record-replay
uv run autosignalx agent score-traces
uv run autosignalx agent consolidate

Write-Host "[$(Get-Date -Format 'o')] Session complete."
