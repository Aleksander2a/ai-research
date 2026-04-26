"""Phase 4: static HTML snapshot of the cockpit.

Renders a self-contained, navigable HTML report from the on-disk
artifacts so reviewers can browse forecasts, regimes, findings,
backtests, agent traces, and chat-corpus citations without running
Streamlit. Output is published via GitHub Pages on every push to main.
"""

from autosignalx.snapshot.builder import build_snapshot

__all__ = ["build_snapshot"]
