"""Streamlit Community Cloud entrypoint shim.

Cloud's default entrypoint is ``streamlit_app.py`` at the repo root.
The real app lives under ``app/streamlit_app.py``; we exec it here so a
local ``streamlit run app/streamlit_app.py`` and the cloud deployment
both render the same code path.

When the deployed instance has no DEEPINFRA_API_KEY (the default), the
agent and chat panels run in deterministic replay mode against the
artifacts committed under ``replay/`` and ``reports/``.
"""

from __future__ import annotations

import os
from pathlib import Path

# On Streamlit Cloud the working dir is the repo root; force replay mode
# unless a key has been set in Streamlit secrets.
os.environ.setdefault("AUTOSIGNALX_REPLAY", "true" if not os.environ.get("DEEPINFRA_API_KEY") else "false")

_app_path = Path(__file__).parent / "app" / "streamlit_app.py"
exec(compile(_app_path.read_text(encoding="utf-8"), str(_app_path), "exec"))
