"""Build a static, multi-page HTML snapshot of the cockpit.

Each page is a self-contained HTML file under ``reports/cockpit_snapshot/``
that pulls Plotly from CDN (``include_plotlyjs="cdn"``) to keep file
sizes manageable. The pages link to each other through a shared header
nav so reviewers can walk the same journey as the live cockpit.

Designed to be robust against partial artifacts: every section gracefully
degrades to a "not built yet" notice when its inputs are absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.config import settings

PAGES = [
    ("index", "Overview"),
    ("forecasts", "Forecasts"),
    ("regimes", "Regimes"),
    ("findings", "Findings"),
    ("backtest", "Backtest"),
    ("agent", "Agent"),
    ("chat", "Chat corpus"),
]


@dataclass
class SnapshotResult:
    out_dir: Path
    pages_written: list[str]
    figures: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _nav_html(active: str) -> str:
    items = []
    for slug, label in PAGES:
        href = "index.html" if slug == "index" else f"{slug}.html"
        cls = "active" if slug == active else ""
        items.append(f'<a href="{href}" class="{cls}">{label}</a>')
    return f'<nav class="topnav">{"".join(items)}</nav>'


_BASE_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 0; background: #fafafa; color: #1a1a1a; }
.topnav { display: flex; gap: 0; background: #1a1a1a; padding: 0; flex-wrap: wrap; }
.topnav a { color: #ddd; text-decoration: none; padding: 12px 18px; font-weight: 500; transition: background 0.15s; }
.topnav a:hover { background: #333; color: #fff; }
.topnav a.active { background: #ff4b4b; color: #fff; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { font-weight: 700; margin: 0 0 8px 0; }
h2 { margin-top: 32px; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; }
.caption { color: #666; font-size: 0.95em; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0 24px 0; background: #fff; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.92em; }
th { background: #f5f5f5; font-weight: 600; }
tr:hover td { background: #fcfcfc; }
.card { background: #fff; padding: 16px 20px; border-radius: 6px;
        margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); border: 1px solid #eee; }
.metric { display: inline-block; margin-right: 28px; margin-bottom: 8px; }
.metric .label { color: #666; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }
.metric .value { font-size: 1.5em; font-weight: 700; }
.empty { color: #999; font-style: italic; padding: 16px; background: #f5f5f5; border-radius: 4px; }
.footer { text-align: center; color: #999; padding: 24px; font-size: 0.85em; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.92em; }
.cite { background: #fff4e6; padding: 1px 6px; border-radius: 3px; font-family: monospace; font-size: 0.85em; }
"""


def _wrap_page(title: str, active_slug: str, body: str, n_chunks: int | None = None) -> str:
    info = (
        f"<span style='color:#999;font-size:0.85em;margin-left:auto;align-self:center;padding:0 16px'>"
        f"{n_chunks} corpus chunks indexed</span>"
        if n_chunks is not None
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoSignal-X — {title}</title>
<style>{_BASE_CSS}</style>
</head>
<body>
{_nav_html(active_slug)}
{info}
<div class="container">
{body}
</div>
<div class="footer">
Static snapshot of AutoSignal-X cockpit · regenerate with <code>autosignalx snapshot build</code>
</div>
</body>
</html>
"""


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "<p class='empty'>No rows.</p>"
    sliced = df.head(max_rows)
    return sliced.to_html(index=False, classes="data-table", border=0, float_format=lambda x: f"{x:.4f}")


def _empty_section(reason: str) -> str:
    return f"<p class='empty'>{reason}</p>"


def _figure_to_html(fig) -> str:  # noqa: ANN001
    return fig.to_html(include_plotlyjs="cdn", full_html=False, default_height="420px")


# ---------- page builders ---------- #


def _page_index(reports_dir: Path) -> tuple[str, int]:
    findings = _read_jsonl(reports_dir / "agent" / "findings.jsonl")
    ledger = _read_jsonl(reports_dir / "agent" / "ledger.jsonl")
    telemetry = _read_jsonl(reports_dir / "agent" / "telemetry.jsonl")
    total_cost = sum(float(t.get("cost_usd", 0) or 0) for t in telemetry)

    metrics_html = (
        f"<div class='card'>"
        f"<div class='metric'><div class='label'>Promoted findings</div><div class='value'>{len(findings)}</div></div>"
        f"<div class='metric'><div class='label'>Ledger entries</div><div class='value'>{len(ledger)}</div></div>"
        f"<div class='metric'><div class='label'>LLM calls</div><div class='value'>{len(telemetry)}</div></div>"
        f"<div class='metric'><div class='label'>Total spend</div><div class='value'>${total_cost:.4f}</div></div>"
        f"</div>"
    )

    layer_rows = [
        ("L1 Forecasting", "Chronos-2 multivariate + ARIMA / naive baselines"),
        ("L2 Representation", "Contrastive 1D-CNN encoder + KMeans + HMM"),
        ("L3 Reasoning", "Per-regime HistGradientBoosting + permutation importance"),
        ("L4 Relational", "GLASSO + Granger + centrality"),
        ("L5 Agentic", "LangGraph debate (Theorist / Skeptic / Adjudicator) + DSL + sandboxed codegen"),
    ]
    layers_html = (
        "<table><tr><th>Layer</th><th>Implementation</th></tr>"
        + "".join(f"<tr><td><strong>{n}</strong></td><td>{d}</td></tr>" for n, d in layer_rows)
        + "</table>"
    )

    body = f"""
<h1>AutoSignal-X — research cockpit snapshot</h1>
<p class='caption'>Static, navigable view of every cockpit panel rendered from on-disk artifacts.
For the interactive version, run <code>make demo</code>.</p>

{metrics_html}

<h2>Architecture</h2>
{layers_html}

<h2>What this snapshot contains</h2>
<ul>
<li><strong>Forecasts</strong> — per-method MAE / MAPE / dir-acc, per-(method, regime) stratification.</li>
<li><strong>Regimes</strong> — KMeans + HMM regime timelines and embedding scatter.</li>
<li><strong>Findings</strong> — promoted hypotheses with DM + bootstrap evidence.</li>
<li><strong>Backtest</strong> — simulated trading metrics on the test window.</li>
<li><strong>Agent</strong> — ledger timeline, trace-quality, self-critique, cost breakdown.</li>
<li><strong>Chat corpus</strong> — citation index that powers Ask the Memory.</li>
</ul>

<h2>Live demo & code</h2>
<p>Live cockpit (Streamlit Community Cloud): <a href="https://autosignal-x.streamlit.app" target="_blank">autosignal-x.streamlit.app</a>.</p>
<p>Source: <a href="https://github.com/Aleksander2a/ai-research" target="_blank">github.com/Aleksander2a/ai-research</a>.</p>
"""
    return body, 0


def _page_forecasts(reports_dir: Path) -> tuple[str, int]:
    abl_dir = reports_dir / "ablations"
    if not abl_dir.exists() or not list(abl_dir.glob("*.parquet")):
        return _empty_section("No ablations on disk. Run <code>make baseline</code> + <code>make forecast</code>."), 0

    rows = []
    figs_html = []
    for fp in sorted(abl_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "target" not in df.columns or "prediction" not in df.columns:
            continue
        err = (df["prediction"] - df["target"]).abs()
        mape = (err / df["target"].abs().clip(lower=1e-9)).mean()
        rows.append({
            "method": fp.stem,
            "n": int(len(df)),
            "mae": float(err.mean()),
            "rmse": float(((df["prediction"] - df["target"]) ** 2).mean() ** 0.5),
            "mape": float(mape),
        })

    metrics_df = pd.DataFrame(rows)
    n_figs = 0

    # Equity-of-error chart per method, per asset average
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for fp in sorted(abl_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(fp)
            except Exception:  # noqa: BLE001
                continue
            if df.empty or "target" not in df.columns or "timestamp" not in df.columns:
                continue
            df = df.copy()
            df["abs_err"] = (df["prediction"] - df["target"]).abs()
            agg = df.groupby("timestamp")["abs_err"].mean().sort_index()
            fig.add_trace(go.Scatter(x=agg.index, y=agg.values, mode="lines", name=fp.stem))
        fig.update_layout(
            title="Mean absolute error over time (averaged across assets)",
            xaxis_title="Timestamp", yaxis_title="MAE", template="plotly_white", height=420,
        )
        figs_html.append(_figure_to_html(fig))
        n_figs += 1
    except Exception:  # noqa: BLE001
        pass

    body = (
        "<h1>Forecast Arena</h1>"
        "<p class='caption'>Per-method overall metrics over the walk-forward test window.</p>"
        f"{_df_to_html_table(metrics_df.sort_values('mae'))}"
        + "".join(f"<div class='card'>{h}</div>" for h in figs_html)
    )
    return body, n_figs


def _page_regimes(reports_dir: Path) -> tuple[str, int]:
    km = reports_dir / "regimes" / "kmeans.parquet"
    hmm = reports_dir / "regimes" / "hmm.parquet"
    if not km.exists() and not hmm.exists():
        return _empty_section("No regime artifacts. Run <code>make regime</code>."), 0

    n_figs = 0
    figs = []
    try:
        import plotly.express as px

        for label, path in [("KMeans", km), ("HMM", hmm)]:
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
            reg_col = "regime_id" if "regime_id" in df.columns else df.columns[-1]
            fig = px.scatter(
                df, x=ts_col, y=reg_col, color=reg_col,
                title=f"{label} regime labels over time",
                template="plotly_white", height=320,
            )
            fig.update_traces(marker=dict(size=4))
            figs.append(_figure_to_html(fig))
            n_figs += 1
    except Exception:  # noqa: BLE001
        pass

    body = (
        "<h1>Regime Explorer</h1>"
        "<p class='caption'>Per-timestep regime labels from KMeans on contrastive embeddings (left) "
        "and a parallel Gaussian HMM on raw features (right).</p>"
        + "".join(f"<div class='card'>{h}</div>" for h in figs)
    )
    return body, n_figs


def _page_findings(reports_dir: Path) -> tuple[str, int]:
    findings = _read_jsonl(reports_dir / "agent" / "findings.jsonl")
    if not findings:
        return _empty_section("No promoted findings. Run <code>make agent</code>."), 0

    cards = []
    for f in findings:
        ev = f.get("evidence", {}) or {}
        filt = f.get("filters", {}) or {}
        cards.append(f"""
<div class='card'>
  <h3 style='margin:0 0 6px 0'>{f.get('id', '?')} <span class='cite'>{f.get('method', '?')}</span></h3>
  <p style='margin:6px 0'>{f.get('hypothesis', '')}</p>
  <div class='metric'><div class='label'>Skill vs naive</div><div class='value'>{ev.get('skill_vs_baseline', 0):.4f}</div></div>
  <div class='metric'><div class='label'>p-value (DM)</div><div class='value'>{ev.get('p_value', float('nan')):.4f}</div></div>
  <div class='metric'><div class='label'>Bootstrap CI</div><div class='value'>[{ev.get('bootstrap_ci_low', 0):.4f}, {ev.get('bootstrap_ci_high', 0):.4f}]</div></div>
  <div class='metric'><div class='label'>n</div><div class='value'>{ev.get('n', 0)}</div></div>
  <div class='metric'><div class='label'>Filters</div><div class='value' style='font-size:1em'>{filt}</div></div>
  <div class='metric'><div class='label'>Replications</div><div class='value'>{f.get('replication_count', 1)}</div></div>
</div>
""")
    body = (
        "<h1>Findings</h1>"
        "<p class='caption'>Hypotheses that survived the Diebold–Mariano + bootstrap promotion gate "
        "(p &lt; 0.05, skill &gt; 0, bootstrap CI strictly above zero).</p>"
        + "".join(cards)
    )
    return body, 0


def _page_backtest(reports_dir: Path) -> tuple[str, int]:
    runs_dir = reports_dir / "backtest" / "runs"
    if not runs_dir.exists():
        return _empty_section("No backtest runs on disk. Run <code>autosignalx backtest run</code>."), 0
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        return _empty_section("No backtest runs on disk. Run <code>autosignalx backtest run</code>."), 0

    latest = run_dirs[-1]
    metrics_path = latest / "metrics.json"
    portfolio_path = latest / "portfolio_daily.parquet"

    rows = []
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_strategy = m.get("per_strategy") or {
            k: v for k, v in m.items() if isinstance(v, dict) and "sharpe" in v
        }
        for sname, sm in per_strategy.items():
            rows.append({
                "strategy": sname,
                "cagr": sm.get("cagr"),
                "sharpe": sm.get("sharpe"),
                "max_drawdown": sm.get("max_drawdown"),
                "calmar": sm.get("calmar"),
                "turnover": sm.get("avg_turnover", sm.get("turnover")),
            })
    table_html = _df_to_html_table(pd.DataFrame(rows).sort_values("sharpe", ascending=False)) if rows else _empty_section("metrics.json missing.")

    n_figs = 0
    fig_html = ""
    if portfolio_path.exists():
        try:
            import plotly.graph_objects as go

            df = pd.read_parquet(portfolio_path)
            if {"timestamp", "strategy", "equity"} <= set(df.columns):
                fig = go.Figure()
                for s, sdf in df.groupby("strategy"):
                    sdf = sdf.sort_values("timestamp")
                    fig.add_trace(go.Scatter(x=sdf["timestamp"], y=sdf["equity"], mode="lines", name=s))
                fig.update_layout(
                    title=f"Equity curves — run {latest.name}",
                    xaxis_title="Date", yaxis_title="Equity (start=1.0)",
                    template="plotly_white", height=460, hovermode="x unified",
                )
                fig_html = _figure_to_html(fig)
                n_figs += 1
        except Exception:  # noqa: BLE001
            pass

    body = (
        f"<h1>Backtest — latest run <code>{latest.name}</code></h1>"
        "<p class='caption'>Simulated trading on the test window (strict no-look-ahead). "
        "Strategies compete against passive benchmarks under realistic per-trade costs.</p>"
        f"<div class='card'>{table_html}</div>"
        f"<div class='card'>{fig_html or _empty_section('No portfolio_daily.parquet to plot.')}</div>"
        f"<p style='color:#666'>Total runs on disk: {len(run_dirs)}.</p>"
    )
    return body, n_figs


def _page_agent(reports_dir: Path) -> tuple[str, int]:
    ledger = _read_jsonl(reports_dir / "agent" / "ledger.jsonl")
    telemetry = _read_jsonl(reports_dir / "agent" / "telemetry.jsonl")
    trace = _read_jsonl(reports_dir / "agent" / "trace_quality.jsonl")

    if not ledger:
        return _empty_section("No agent ledger. Run <code>make agent</code>."), 0

    timeline_rows = []
    for e in ledger[-25:]:
        c = e.get("content")
        text = json.dumps(c, default=str) if not isinstance(c, str) else c
        if len(text) > 250:
            text = text[:247] + "..."
        timeline_rows.append({"round": e.get("round"), "step": e.get("step"), "content": text})
    timeline_html = _df_to_html_table(pd.DataFrame(timeline_rows), max_rows=25)

    n_figs = 0
    figs_html = []
    try:
        import plotly.graph_objects as go

        if trace:
            tdf = pd.DataFrame(trace)
            if {"round", "clarity", "novelty", "falsifiability", "evidence_citing"} <= set(tdf.columns):
                fig = go.Figure()
                for col in ("clarity", "novelty", "falsifiability", "evidence_citing"):
                    fig.add_trace(go.Scatter(x=tdf["round"], y=tdf[col], mode="lines+markers", name=col))
                fig.update_layout(
                    title="Trace quality per round (LLM-as-judge, scale 1-5)",
                    xaxis_title="Round", yaxis_title="Score", yaxis_range=[0, 5.5],
                    template="plotly_white", height=380,
                )
                figs_html.append(_figure_to_html(fig))
                n_figs += 1

        if telemetry:
            tdf = pd.DataFrame(telemetry)
            if "cost_usd" in tdf.columns:
                tdf = tdf.sort_values("ts") if "ts" in tdf.columns else tdf
                tdf["cum_cost"] = tdf["cost_usd"].cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=tdf["cum_cost"], mode="lines", name="cumulative cost"))
                fig.update_layout(
                    title="Cumulative LLM spend (USD)", xaxis_title="Call #",
                    yaxis_title="USD", template="plotly_white", height=320,
                )
                figs_html.append(_figure_to_html(fig))
                n_figs += 1
    except Exception:  # noqa: BLE001
        pass

    body = (
        "<h1>Agent activity</h1>"
        "<p class='caption'>Append-only ledger of every agent step, plus per-round trace quality "
        "and cumulative LLM spend.</p>"
        f"<div class='card'><h2 style='margin-top:0'>Recent ledger entries</h2>{timeline_html}</div>"
        + "".join(f"<div class='card'>{h}</div>" for h in figs_html)
    )
    return body, n_figs


def _page_chat(reports_dir: Path) -> tuple[str, int]:
    chat_dir = reports_dir / "chat"
    chunks_path = chat_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return _empty_section("No chat index. Run <code>autosignalx chat index</code>."), 0

    chunks = _read_jsonl(chunks_path)
    if not chunks:
        return _empty_section("Chat index empty."), 0

    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.get("kind", "?")] = by_kind.get(c.get("kind", "?"), 0) + 1

    rows = [
        {"citation_id": c.get("citation_id"), "kind": c.get("kind"),
         "text": (c.get("text") or "")[:160] + ("..." if len(c.get("text") or "") > 160 else "")}
        for c in chunks[:60]
    ]

    breakdown = "<table><tr><th>Kind</th><th>Count</th></tr>" + "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(by_kind.items())
    ) + "</table>"

    body = (
        "<h1>Chat corpus</h1>"
        "<p class='caption'>The retrieval index that powers <em>Ask the Memory</em>. Every "
        "chunk has a stable <code>citation_id</code> the LLM must cite verbatim or refuse.</p>"
        f"<div class='card'><h2 style='margin-top:0'>Index breakdown</h2>{breakdown}</div>"
        f"<div class='card'><h2 style='margin-top:0'>Sample chunks (first 60)</h2>"
        f"{_df_to_html_table(pd.DataFrame(rows), max_rows=60)}</div>"
    )
    return body, 0


# ---------- orchestration ---------- #


def build_snapshot(
    reports_dir: Path | None = None,
    out_dir: Path | None = None,
) -> SnapshotResult:
    """Render every page to ``out_dir`` (default: reports/cockpit_snapshot/)."""
    rd = reports_dir or settings.reports_dir
    od = out_dir or (rd / "cockpit_snapshot")
    od.mkdir(parents=True, exist_ok=True)

    chunks_path = rd / "chat" / "chunks.jsonl"
    n_chunks = sum(1 for _ in chunks_path.open("r", encoding="utf-8")) if chunks_path.exists() else 0

    builders = [
        ("index", _page_index),
        ("forecasts", _page_forecasts),
        ("regimes", _page_regimes),
        ("findings", _page_findings),
        ("backtest", _page_backtest),
        ("agent", _page_agent),
        ("chat", _page_chat),
    ]
    pages_written = []
    total_figs = 0
    for slug, fn in builders:
        body, nfigs = fn(rd)
        total_figs += nfigs
        title = dict(PAGES)[slug]
        html = _wrap_page(title, slug, body, n_chunks=n_chunks if slug == "index" else None)
        target = od / ("index.html" if slug == "index" else f"{slug}.html")
        target.write_text(html, encoding="utf-8")
        pages_written.append(target.name)

    # Tiny manifest so reviewers can see freshness at a glance.
    (od / "manifest.json").write_text(
        json.dumps({"pages": pages_written, "figures": total_figs, "n_corpus_chunks": n_chunks}, default=str),
        encoding="utf-8",
    )
    return SnapshotResult(out_dir=od, pages_written=pages_written, figures=total_figs)
