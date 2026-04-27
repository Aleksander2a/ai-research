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
    ("regime_graph", "Regime graph"),
    ("signal_stability", "Signal stability"),
    ("findings", "Findings"),
    ("survival", "Survival"),
    ("bayesian", "Bayesian"),
    ("calibration", "Calibration"),
    ("red_team", "RedTeam"),
    ("coherence", "Coherence"),
    ("coverage", "Coverage"),
    ("preregistration", "Pre-Registration"),
    ("holdout", "Holdout vault"),
    ("synthetic", "Synthetic benchmark"),
    ("ablation", "Capability ablation"),
    ("backtest", "Backtest"),
    ("agent", "Agent"),
    ("chat", "Chat corpus"),
    ("reproducibility", "Reproducibility"),
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
        ("L1 — Forecasting", "Probabilistic point + interval forecasts. Frozen Chronos-2 (multivariate, with macro covariates) + classical baselines (naive, seasonal-naive, ARIMA(1,1,1) on log-prices). Returns / excess-return / vol / cross-sectional-rank target types are supported via the Phase-7 contract extension."),
        ("L2 — Representation", "Latent regime labels. Contrastive 1D-CNN encoder (16-dim embeddings, 60-day windows, triplet loss) + KMeans on the embeddings; Gaussian HMM on raw features as a parallel detector."),
        ("L3 — Reasoning", "Per-regime feature importance. HistGradientBoostingClassifier per regime + custom permutation importance + walk-forward stability (mean rank, rank std, top-K share)."),
        ("L4 — Relational", "Cross-asset structure. GLASSO partial correlations + Granger causality + NetworkX centrality; per-regime variant rebuilds the graph inside each regime."),
        ("L5 — Agentic", "Hypothesis generation, experimentation, statistical promotion. LangGraph state machine in three modes: <code>single</code>, <code>debate</code> (Theorist / Skeptic / Adjudicator), <code>lab</code> (full specialist team with persistent KG memory and pre-registration verifier)."),
    ]
    layers_html = (
        "<table><tr><th>Layer</th><th>What it does</th></tr>"
        + "".join(f"<tr><td><strong>{n}</strong></td><td>{d}</td></tr>" for n, d in layer_rows)
        + "</table>"
    )

    body = f"""
<h1>AutoSignal-X — system overview</h1>
<p class='caption'>AI research system for discovering predictive structure in liquid daily ETF prices,
with an autonomous research loop that grades its own discoveries through a multi-stage statistical
methodology before any finding ships. Static, navigable view of the cockpit rendered from on-disk
artifacts; for the interactive version, run <code>make demo</code>.</p>

{metrics_html}

<h2>What the system does</h2>
<p>Five model layers feed a hardening pipeline that runs every promoted finding through
Diebold-Mariano + block-bootstrap → BH-FDR → adversarial replication (full-test, placebo
regime-shuffle, 50/50 block-holdout) → Combinatorial Purged Cross-Validation → Probability of
Backtest Overfitting → Deflated Sharpe Ratio → Romano-Wolf joint stepdown → hierarchical
Normal-Normal Bayesian shrinkage with a Bayes-factor floor. The strict bar
<code>survives_all_strict</code> is the conjunction of every gate. The contribution is the
methodology and the agent that operates it; any single trade is incidental.</p>

{layers_html}

<h2>What this snapshot contains</h2>
<ul>
<li><strong>Forecasts</strong> — per-method MAE / MAPE / directional accuracy, per-(method, regime) stratification, mean absolute error over time.</li>
<li><strong>Regimes</strong> — KMeans + HMM regime timelines.</li>
<li><strong>Regime graph</strong> — per-regime GLASSO + Granger + centrality; surfaces hubs and bridges that only matter in certain regimes plus the per-asset cross-regime sensitivity.</li>
<li><strong>Signal stability</strong> — walk-forward feature-importance rankings; distinguishes stable signals from averaging artefacts.</li>
<li><strong>Findings</strong> — promoted hypotheses with DM + bootstrap evidence.</li>
<li><strong>Survival</strong> — every promoted finding under BH-FDR + adversarial replication.</li>
<li><strong>Bayesian</strong> — per-finding posterior mean / sd, P(θ&gt;0), Bayes factor BF₁₀ from the hierarchical Normal-Normal model.</li>
<li><strong>Calibration</strong> — Brier score and Expected Calibration Error of the Theorist's predicted confidence vs finding-survival outcomes.</li>
<li><strong>RedTeam</strong> — asset-shuffle and time-shift adversarial attacks beyond the original gates.</li>
<li><strong>Coherence</strong> — per-session lessons-uptake, lineage branching factor, theme-persistence entropy.</li>
<li><strong>Coverage</strong> — hypothesis search-space heatmap of (method × asset × regime) coloured by Expected Information Gain.</li>
<li><strong>Pre-registration</strong> — hash-committed hypotheses + resolutions (open vs promoted vs refuted).</li>
<li><strong>Holdout vault</strong> — never-touched final test slice (lock state + one-time evaluation results).</li>
<li><strong>Synthetic benchmark</strong> — per-gate recall + FDR on a controlled synthetic universe with deliberately planted causal structure; the apparatus' own audited discriminative power.</li>
<li><strong>Capability ablation</strong> — layer-by-layer marginal MAE-skill vs cost-proxy frontier.</li>
<li><strong>Backtest</strong> — simulated trading on the test window with paired moving-block bootstrap.</li>
<li><strong>Agent</strong> — ledger timeline, trace-quality scoring, cumulative LLM spend.</li>
<li><strong>Chat corpus</strong> — citation index that powers <em>Ask the Memory</em>.</li>
<li><strong>Reproducibility</strong> — git + environment + per-artifact SHA-256 + a single bundle hash for the snapshot state.</li>
</ul>

<h2>Live cockpit &amp; source</h2>
<p>Live interactive cockpit on Streamlit Community Cloud: <a href="https://ai-research-aleksander2a.streamlit.app" target="_blank">ai-research-aleksander2a.streamlit.app</a>.</p>
<p>Source repository: <a href="https://github.com/Aleksander2a/ai-research" target="_blank">github.com/Aleksander2a/ai-research</a>.</p>
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


def _page_survival(reports_dir: Path) -> tuple[str, int]:
    p = reports_dir / "agent" / "survival.jsonl"
    if not p.exists():
        return _empty_section(
            "No survival records. Run <code>autosignalx agent harden</code>."
        ), 0
    records = _read_jsonl(p)
    if not records:
        return _empty_section("Survival file empty."), 0

    def _mark(v) -> str:  # noqa: ANN001
        if v is True:
            return "✅"
        if v is False:
            return "❌"
        return "—"

    rows = []
    for r in records:
        rows.append({
            "finding_id": r.get("finding_id"),
            "method": r.get("method"),
            "filters": str(r.get("filters")),
            "p_orig": f"{r.get('original_p', float('nan')):.4f}" if r.get("original_p") is not None else "—",
            "q_FDR": f"{r.get('fdr_q', float('nan')):.4f}" if r.get("fdr_q") is not None else "—",
            "FDR": _mark(r.get("survives_fdr")),
            "full-test": _mark(r.get("survives_full_test")),
            "placebo": _mark(r.get("survives_placebo")),
            "block-holdout": _mark(r.get("survives_block_holdout")),
            "all": _mark(r.get("survives_all")),
        })

    n = len(records)
    n_all = sum(1 for r in records if r.get("survives_all"))
    n_fdr = sum(1 for r in records if r.get("survives_fdr"))
    n_block = sum(1 for r in records if r.get("survives_block_holdout"))

    headline = (
        f"<div class='card'>"
        f"<div class='metric'><div class='label'>Promoted</div><div class='value'>{n}</div></div>"
        f"<div class='metric'><div class='label'>Survive FDR</div><div class='value'>{n_fdr}/{n}</div></div>"
        f"<div class='metric'><div class='label'>Survive block-holdout</div><div class='value'>{n_block}/{n}</div></div>"
        f"<div class='metric'><div class='label'>Survive all attacks</div><div class='value'>{n_all}/{n}</div></div>"
        f"</div>"
    )

    note = ""
    if n > 0 and n_all == 0:
        note = (
            "<div class='card'><strong>Zero of the promoted findings survive every attack.</strong> "
            "The hardening surfaced exactly which fragility each finding has -- the "
            "methodology is the artifact, not the count.</div>"
        )

    body = (
        "<h1>Survival Analysis</h1>"
        "<p class='caption'>Every promoted finding re-evaluated under BH-FDR + adversarial "
        "replication (full-test, placebo regime-shuffle, 50/50 block-holdout). A red X is a "
        "research insight, not a failure.</p>"
        + headline
        + note
        + f"<div class='card'>{_df_to_html_table(pd.DataFrame(rows), max_rows=50)}</div>"
    )
    return body, 0


def _page_per_regime_graph(reports_dir: Path) -> tuple[str, int]:
    root = reports_dir / "graph" / "per_regime"
    sens_p = root / "regime_sensitivity.parquet"
    if not root.exists() or not sens_p.exists():
        return _empty_section(
            "No per-regime graph artifacts. Run <code>autosignalx graph build-per-regime</code>."
        ), 0

    sens = pd.read_parquet(sens_p)
    cards: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("regime_"):
            continue
        cent_p = d / "centrality.parquet"
        if not cent_p.exists():
            continue
        cent = pd.read_parquet(cent_p).sort_values("eigenvector_centrality", ascending=False)
        rid = d.name.split("_", 1)[1]
        rows = cent[["node", "degree_centrality", "eigenvector_centrality", "betweenness_centrality"]]
        cards.append(
            f"<div class='card'><h3 style='margin-top:0'>Regime {rid}</h3>"
            f"{_df_to_html_table(rows.head(10))}</div>"
        )

    body = (
        "<h1>Regime-Conditioned Graph</h1>"
        "<p class='caption'>Cross-asset structure recomputed within each regime's data subset. "
        "Surfaces hubs and bridges that flip role across regimes -- structural information the "
        "global graph averages away.</p>"
        f"<h2>Regime sensitivity (assets ranked by betweenness range)</h2>"
        f"<div class='card'>{_df_to_html_table(sens, max_rows=12)}</div>"
        + "".join(cards)
    )
    return body, 0


def _page_signal_stability(reports_dir: Path) -> tuple[str, int]:
    p = reports_dir / "signals" / "signal_stability.parquet"
    if not p.exists():
        return _empty_section(
            "No signal-stability summary. Run <code>autosignalx signal stability</code>."
        ), 0
    df = pd.read_parquet(p)
    if df.empty:
        return _empty_section("Stability summary empty."), 0

    cards: list[str] = []
    for rid, group in df.groupby("regime_id", observed=True):
        cols = [c for c in [
            "feature", "mean_importance", "mean_rank", "rank_std", "stability",
        ] + [c for c in group.columns if c.startswith("top")] if c in group.columns]
        cards.append(
            f"<div class='card'><h3 style='margin-top:0'>Regime {rid}</h3>"
            f"{_df_to_html_table(group[cols].head(10))}</div>"
        )

    body = (
        "<h1>Signal Stability</h1>"
        "<p class='caption'>Walk-forward feature-importance stability. A feature with high "
        "mean importance AND high stability AND high top-K share is research-grade; high "
        "importance with low stability is an averaging artefact.</p>"
        + "".join(cards)
    )
    return body, 0


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


# ---------- Phase 8/12/14/15/16 page builders ---------- #


def _page_bayesian(reports_dir: Path) -> tuple[str, int]:
    """Phase 12: render the per-finding posterior + Bayes factor table."""
    findings = _read_jsonl(reports_dir / "agent" / "findings.jsonl")
    survival = _read_jsonl(reports_dir / "agent" / "survival.jsonl")
    if not findings:
        return _empty_section("No promoted findings yet."), 0
    rows = []
    for r in survival:
        b = r.get("bayesian") or {}
        if not b:
            continue
        rows.append({
            "finding_id": r.get("finding_id"),
            "method": r.get("method"),
            "n": b.get("n"),
            "posterior_mean": b.get("posterior_mean"),
            "posterior_sd": b.get("posterior_sd"),
            "P(theta>0)": b.get("prob_positive"),
            "BF_10": b.get("bayes_factor"),
            "survives_bayes": r.get("survives_bayes"),
        })
    if not rows:
        return _empty_section(
            "No Bayesian fields in survival.jsonl. Run "
            "<code>autosignalx agent harden</code> against current findings."
        ), 0
    body = (
        "<h1>Bayesian evidence</h1>"
        "<p class='caption'>Hierarchical Normal-Normal posterior with empirical-Bayes "
        "shrinkage; Bayes factor BF_10 vs the null. Strict bar: BF_10 &ge; 10 and "
        "P(&theta; &gt; 0) &ge; 0.95.</p>"
        f"<div class='card'>{_df_to_html_table(pd.DataFrame(rows), max_rows=50)}</div>"
    )
    return body, 0


def _page_calibration(reports_dir: Path) -> tuple[str, int]:
    """Phase 15: confidence-vs-survival reliability bins."""
    rows = _read_jsonl(reports_dir / "agent" / "calibration.jsonl")
    if not rows:
        return _empty_section(
            "No calibration record. Run <code>autosignalx agent eval-suite</code>."
        ), 0
    rec = rows[-1]
    bins = pd.DataFrame(rec.get("bins") or [])
    summary = (
        f"<div class='card'>"
        f"<div class='metric'><div class='label'>n</div><div class='value'>{rec.get('n', 0)}</div></div>"
        f"<div class='metric'><div class='label'>Brier</div><div class='value'>{rec.get('brier', 0):.3f}</div></div>"
        f"<div class='metric'><div class='label'>ECE</div><div class='value'>{rec.get('ece', 0):.3f}</div></div>"
        f"<div class='metric'><div class='label'>Role</div><div class='value' style='font-size:1em'>{rec.get('role', '?')}</div></div>"
        f"</div>"
    )
    body = (
        "<h1>Agent calibration</h1>"
        "<p class='caption'>Predicted confidence vs observed survival rate of the agent's findings. "
        "Brier closer to 0 is better; ECE measures the average bin-level deviation from the identity line.</p>"
        + summary
        + f"<div class='card'>{_df_to_html_table(bins, max_rows=20)}</div>"
    )
    return body, 0


def _page_red_team(reports_dir: Path) -> tuple[str, int]:
    """Phase 15: asset-shuffle + time-shift attacks per finding."""
    rows = _read_jsonl(reports_dir / "agent" / "red_team.jsonl")
    if not rows:
        return _empty_section(
            "No RedTeam records. Run <code>autosignalx agent eval-suite</code>."
        ), 0
    flat = []
    for r in rows:
        a = r.get("asset_shuffle", {}) or {}
        t = r.get("time_shift", {}) or {}
        flat.append({
            "finding_id": r.get("finding_id"),
            "asset_shuffle_survives": a.get("survives"),
            "promotable_elsewhere": a.get("promotable_elsewhere"),
            "time_shift_promotable": t.get("promotable_after_shift"),
            "time_shift_skill": t.get("skill"),
            "survives_red_team": r.get("survives_red_team"),
        })
    body = (
        "<h1>RedTeam attacks</h1>"
        "<p class='caption'>Beyond Phase-5 (FDR + full-test + placebo + block-holdout): "
        "asset-shuffle (re-test on every other asset in the same regime) and time-shift "
        "(shift forecast_origin by 5 days). Both must survive.</p>"
        f"<div class='card'>{_df_to_html_table(pd.DataFrame(flat), max_rows=50)}</div>"
    )
    return body, 0


def _page_coherence(reports_dir: Path) -> tuple[str, int]:
    """Phase 15: per-session lessons-uptake / branching / theme entropy."""
    rows = _read_jsonl(reports_dir / "agent" / "coherence.jsonl")
    if not rows:
        return _empty_section(
            "No coherence records. Run <code>autosignalx agent eval-suite</code>."
        ), 0
    df = pd.DataFrame(rows)
    body = (
        "<h1>Agent coherence</h1>"
        "<p class='caption'>Per-session multi-horizon coherence: lessons uptake from earlier "
        "sessions, lineage-DAG branching factor, theme-persistence entropy, composite score.</p>"
        f"<div class='card'>{_df_to_html_table(df, max_rows=20)}</div>"
    )
    return body, 0


def _page_coverage(reports_dir: Path) -> tuple[str, int]:
    """Phase 14/16: hypothesis search-space coverage map."""
    abl_dir = reports_dir / "ablations"
    if not abl_dir.exists() or not list(abl_dir.glob("*.parquet")):
        return _empty_section("No ablations to compute coverage."), 0
    try:
        from autosignalx.agent import eig as eig_mod
        from autosignalx.agent import findings as findings_mod
    except Exception as e:  # noqa: BLE001
        return _empty_section(f"Coverage layer unavailable: {e}"), 0
    frames = []
    for fp in sorted(abl_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception:  # noqa: BLE001
            continue
        if "method" not in df.columns:
            df = df.copy()
            df["method"] = fp.stem
        frames.append(df)
    if not frames:
        return _empty_section("No readable ablations."), 0
    forecasts = pd.concat(frames, ignore_index=True)
    rl_path = reports_dir / "regimes" / "kmeans.parquet"
    if rl_path.exists() and "forecast_origin" in forecasts.columns:
        try:
            rl = pd.read_parquet(rl_path)
            rl_join = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
            rl_join["forecast_origin"] = pd.to_datetime(rl_join["forecast_origin"])
            forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
            forecasts = forecasts.merge(rl_join, on="forecast_origin", how="left")
        except Exception:  # noqa: BLE001
            pass
    if "asset" not in forecasts.columns or "method" not in forecasts.columns:
        return _empty_section(
            "Forecasts present but missing columns (`asset` / `method`); "
            "coverage map needs the standard contract."
        ), 0
    methods = sorted(forecasts["method"].unique())
    assets = sorted(forecasts["asset"].unique())
    regimes = (
        sorted(forecasts["regime_id"].dropna().astype(int).unique().tolist())
        if "regime_id" in forecasts.columns else []
    )
    df = eig_mod.coverage_map(
        forecasts=forecasts, methods=methods, assets=assets, regimes=regimes,
        findings=findings_mod.load(),
    )
    if df.empty or "eig_score" not in df.columns:
        return _empty_section(
            "Coverage map produced no rows. Need at least one regime label "
            "and a populated ablations cache."
        ), 0
    body = (
        "<h1>Coverage map</h1>"
        "<p class='caption'>Where has the agent looked? Each (method &times; asset &times; regime) "
        "cell scored by Expected Information Gain (untested + sample-rich + not yet promoted = high).</p>"
        f"<div class='card'>{_df_to_html_table(df.sort_values('eig_score', ascending=False), max_rows=80)}</div>"
    )
    return body, 0


def _page_preregistration(reports_dir: Path) -> tuple[str, int]:
    """Phase 8: hash-committed hypothesis ledger + resolutions."""
    regs = _read_jsonl(reports_dir / "agent" / "preregistrations.jsonl")
    resols = _read_jsonl(reports_dir / "agent" / "preregistration_resolutions.jsonl")
    if not regs:
        return _empty_section(
            "No pre-registrations. Run <code>autosignalx agent run --mode lab</code> "
            "(the lab-mode verifier auto-registers each hypothesis)."
        ), 0
    by_id = {r.get("preregistration_id"): r for r in resols}
    flat = [{
        "id": r.get("id"),
        "method": r.get("method"),
        "filters": r.get("filters"),
        "registered_at": r.get("registered_at"),
        "status": "open" if r.get("id") not in by_id else (
            "promoted" if (by_id.get(r.get("id")) or {}).get("promoted") else "refuted"
        ),
    } for r in regs]
    cols = (
        f"<div class='card'>"
        f"<div class='metric'><div class='label'>Registered</div><div class='value'>{len(regs)}</div></div>"
        f"<div class='metric'><div class='label'>Resolved</div><div class='value'>{len(by_id)}</div></div>"
        f"<div class='metric'><div class='label'>Open</div><div class='value'>{len(regs) - len(by_id)}</div></div>"
        f"</div>"
    )
    body = (
        "<h1>Pre-registration ledger</h1>"
        "<p class='caption'>Every hypothesis hash-committed before its experiment runs. "
        "Resolutions are appended to a separate file so registration history is never rewritten.</p>"
        + cols
        + f"<div class='card'>{_df_to_html_table(pd.DataFrame(flat), max_rows=50)}</div>"
    )
    return body, 0


def _page_holdout(reports_dir: Path) -> tuple[str, int]:
    """Phase 8: holdout vault status + one-time results."""
    vault_meta = reports_dir / "agent" / "holdout_vault" / "vault.json"
    vault_results = reports_dir / "agent" / "holdout_vault" / "results.json"
    if not vault_meta.exists():
        return _empty_section(
            "No holdout vault initialized. Run <code>autosignalx eval vault-init &lt;start&gt; &lt;end&gt;</code>."
        ), 0
    try:
        meta = json.loads(vault_meta.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        meta = {}
    state = "OPENED" if meta.get("opened") else "LOCKED"
    summary = (
        f"<div class='card'>"
        f"<div class='metric'><div class='label'>Start</div><div class='value' style='font-size:1.1em'>{meta.get('start', '?')}</div></div>"
        f"<div class='metric'><div class='label'>End</div><div class='value' style='font-size:1.1em'>{meta.get('end', '?')}</div></div>"
        f"<div class='metric'><div class='label'>Status</div><div class='value'>{state}</div></div>"
        f"<div class='metric'><div class='label'>Lock hash</div><div class='value' style='font-size:1em;font-family:monospace'>{meta.get('lock_hash', '?')}</div></div>"
        f"</div>"
    )
    results_html = ""
    if vault_results.exists():
        try:
            res = json.loads(vault_results.read_text(encoding="utf-8"))
            per = res.get("per_method_mae") or {}
            skill = res.get("skill_vs_baseline") or {}
            rows = [{
                "method": m, "mae": per.get(m), "skill_vs_baseline": skill.get(m),
            } for m in sorted(per)]
            results_html = (
                f"<h2>One-time evaluation</h2>"
                f"<div class='card'>{_df_to_html_table(pd.DataFrame(rows), max_rows=20)}</div>"
            )
        except Exception:  # noqa: BLE001
            pass
    body = (
        "<h1>Holdout vault</h1>"
        "<p class='caption'>Never-touched final test slice. Discovery code asserts no "
        "leakage of forecast_origin into the locked range; "
        "<code>autosignalx eval vault-open</code> is the one-time evaluation method.</p>"
        + summary + results_html
    )
    return body, 0


def _page_synthetic(reports_dir: Path) -> tuple[str, int]:
    """Phase 7-style item: synthetic-known-answer benchmark results."""
    p = reports_dir / "agent" / "synthetic_benchmark.json"
    if not p.exists():
        return _empty_section(
            "No synthetic benchmark results. Run "
            "<code>autosignalx eval synthetic --n-trials 1</code>."
        ), 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return _empty_section("Could not parse synthetic_benchmark.json."), 0
    rows = data.get("ablations") or []
    head = (
        f"<div class='card'>"
        f"<div class='metric'><div class='label'>Planted truths</div><div class='value'>{data.get('planted_truths', 0)}</div></div>"
        f"<div class='metric'><div class='label'>Distractors</div><div class='value'>{data.get('distractors', 0)}</div></div>"
        f"<div class='metric'><div class='label'>n_trials</div><div class='value'>{data.get('n_trials', 0)}</div></div>"
        f"<div class='metric'><div class='label'>Generated at</div><div class='value' style='font-size:0.95em'>{data.get('generated_at', '?')}</div></div>"
        f"</div>"
    )
    body = (
        "<h1>Synthetic-known-answer benchmark</h1>"
        "<p class='caption'>The apparatus is given a synthetic regime-switching market with "
        "deliberately planted causal structure plus distractor null cells. Each row is one "
        "ablation (single vs debate vs lab; memory off vs on); the metrics are recall (planted "
        "truths recovered), false-discovery rate (distractors promoted), and net signal-to-noise.</p>"
        + head
        + f"<div class='card'>{_df_to_html_table(pd.DataFrame(rows), max_rows=24)}</div>"
    )
    return body, 0


def _page_ablation(reports_dir: Path) -> tuple[str, int]:
    """Phase 16: smallest-capability-preserving ablation -- drop each layer, measure delta."""
    p = reports_dir / "agent" / "capability_ablation.json"
    if not p.exists():
        return _empty_section(
            "No capability-ablation results. Run "
            "<code>autosignalx eval ablate-capability</code>."
        ), 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return _empty_section("Could not parse capability_ablation.json."), 0
    rows = data.get("rows") or []
    body = (
        "<h1>Capability-preserving ablation</h1>"
        "<p class='caption'>For each layer (L1 forecasting / L2 regime / L3 signal / "
        "L4 graph / L5 agent), drop that layer's contribution to the survival pipeline "
        "and re-grade. Marginal-skill column is the lift the layer provides; cells with "
        "high cost / low marginal-skill are compression candidates.</p>"
        f"<div class='card'>{_df_to_html_table(pd.DataFrame(rows), max_rows=12)}</div>"
    )
    return body, 0


def _page_reproducibility(reports_dir: Path) -> tuple[str, int]:
    """Phase 16: reproducibility badge."""
    p = reports_dir / "reproducibility_badge.json"
    if not p.exists():
        return _empty_section(
            "No badge committed. Run <code>autosignalx snapshot build</code> "
            "or open the cockpit's Reproducibility panel."
        ), 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return _empty_section("Could not parse reproducibility_badge.json."), 0
    git = data.get("git", {}) or {}
    env = data.get("env", {}) or {}
    libs = env.get("libraries", {}) or {}
    head = (
        f"<div class='card'>"
        f"<div class='metric'><div class='label'>Bundle hash</div><div class='value' style='font-family:monospace'>{data.get('artifacts_bundle_hash', '?')}</div></div>"
        f"<div class='metric'><div class='label'>Artifacts</div><div class='value'>{data.get('n_artifacts', 0)}</div></div>"
        f"<div class='metric'><div class='label'>Replay mode</div><div class='value'>{data.get('replay_mode', False)}</div></div>"
        f"<div class='metric'><div class='label'>Generated</div><div class='value' style='font-size:0.95em'>{data.get('generated_at', '?')}</div></div>"
        f"</div>"
    )
    git_html = (
        f"<div class='card'><h2 style='margin-top:0'>Git</h2>"
        f"<p>Commit <code>{git.get('commit', '?')}</code> on <code>{git.get('branch', '?')}</code> · dirty={git.get('dirty', '?')}</p></div>"
    )
    libs_html = (
        "<div class='card'><h2 style='margin-top:0'>Library versions</h2>"
        + _df_to_html_table(pd.DataFrame(
            [{"library": k, "version": v} for k, v in sorted(libs.items())]
        ), max_rows=20)
        + "</div>"
    )
    body = (
        "<h1>Reproducibility badge</h1>"
        "<p class='caption'>Cryptographic fingerprint of the cockpit state: git + env + "
        "per-artifact SHA-256 + a single bundle hash. Two badges with the same bundle hash "
        "produce the same panels.</p>"
        + head + git_html + libs_html
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
        ("regime_graph", _page_per_regime_graph),
        ("signal_stability", _page_signal_stability),
        ("findings", _page_findings),
        ("survival", _page_survival),
        ("bayesian", _page_bayesian),
        ("calibration", _page_calibration),
        ("red_team", _page_red_team),
        ("coherence", _page_coherence),
        ("coverage", _page_coverage),
        ("preregistration", _page_preregistration),
        ("holdout", _page_holdout),
        ("synthetic", _page_synthetic),
        ("ablation", _page_ablation),
        ("backtest", _page_backtest),
        ("agent", _page_agent),
        ("chat", _page_chat),
        ("reproducibility", _page_reproducibility),
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
