"""Build a citable corpus from on-disk run artifacts.

Each artifact source is rendered to a list of ``Chunk`` records with a
stable ``citation_id`` (e.g. ``ledger:r3/critique``,
``finding:f_9395cd1bd1be``). Chunks are kept short and self-describing
so the LLM can cite them verbatim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autosignalx.config import settings


@dataclass
class Chunk:
    citation_id: str
    kind: str
    text: str
    meta: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
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


def _trim(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _ledger_chunks(reports_dir: Path) -> list[Chunk]:
    entries = _read_jsonl(reports_dir / "agent" / "ledger.jsonl")
    chunks: list[Chunk] = []
    for e in entries:
        rnd = e.get("round", 0)
        step = str(e.get("step", "step"))
        content = e.get("content", "")
        text = (
            json.dumps(content, default=str, indent=2)
            if not isinstance(content, str)
            else content
        )
        chunks.append(
            Chunk(
                citation_id=f"ledger:r{rnd}/{step}",
                kind="ledger",
                text=_trim(f"[round {rnd} {step}] {text}"),
                meta={"round": rnd, "step": step, "session_id": e.get("session_id")},
            )
        )
    return chunks


def _finding_chunks(reports_dir: Path) -> list[Chunk]:
    entries = _read_jsonl(reports_dir / "agent" / "findings.jsonl")
    chunks: list[Chunk] = []
    for e in entries:
        fid = e.get("id", "f_unknown")
        ev = e.get("evidence", {}) or {}
        filt = e.get("filters", {}) or {}
        text = (
            f"Promoted finding {fid} (passed DM + bootstrap gate). "
            f"Hypothesis: {e.get('hypothesis', '')}\n"
            f"method={e.get('method')} filters={filt}\n"
            f"skill_vs_baseline={ev.get('skill_vs_baseline'):.4f} "
            f"p={ev.get('p_value'):.4f} "
            f"bootstrap_ci=[{ev.get('bootstrap_ci_low'):.4f}, "
            f"{ev.get('bootstrap_ci_high'):.4f}] n={ev.get('n')}"
            if isinstance(ev.get("skill_vs_baseline"), (int, float))
            else f"Finding {fid}: {e.get('hypothesis', '')}"
        )
        chunks.append(
            Chunk(
                citation_id=f"finding:{fid}",
                kind="finding",
                text=_trim(text),
                meta={
                    "method": e.get("method"),
                    "filters": filt,
                    "session_id": e.get("session_id"),
                    "round": e.get("round"),
                    "replication_count": e.get("replication_count", 1),
                },
            )
        )
    return chunks


def _lessons_chunks(reports_dir: Path) -> list[Chunk]:
    path = reports_dir / "agent" / "lessons.md"
    if not path.exists():
        return []
    blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    return [
        Chunk(
            citation_id=f"lesson:{i}",
            kind="lesson",
            text=_trim(b),
            meta={"index": i},
        )
        for i, b in enumerate(blocks)
    ]


def _self_critique_chunks(reports_dir: Path) -> list[Chunk]:
    entries = _read_jsonl(reports_dir / "agent" / "self_critique.jsonl")
    return [
        Chunk(
            citation_id=f"self_critique:{e.get('finding_id', f'idx{i}')}",
            kind="self_critique",
            text=_trim(
                f"Self-critique on {e.get('finding_id')}: state={e.get('current_state')}. "
                f"{e.get('rationale', '')}"
            ),
            meta={"finding_id": e.get("finding_id"), "state": e.get("current_state")},
        )
        for i, e in enumerate(entries)
    ]


def _trace_quality_chunks(reports_dir: Path) -> list[Chunk]:
    entries = _read_jsonl(reports_dir / "agent" / "trace_quality.jsonl")
    return [
        Chunk(
            citation_id=f"trace_quality:r{e.get('round')}",
            kind="trace_quality",
            text=_trim(
                f"Trace quality round {e.get('round')}: clarity={e.get('clarity')} "
                f"novelty={e.get('novelty')} falsifiability={e.get('falsifiability')} "
                f"evidence_citing={e.get('evidence_citing')}. "
                f"{e.get('rationale', '')}"
            ),
            meta={"round": e.get("round")},
        )
        for e in entries
    ]


def _telemetry_summary_chunks(reports_dir: Path) -> list[Chunk]:
    entries = _read_jsonl(reports_dir / "agent" / "telemetry.jsonl")
    if not entries:
        return []
    by_model: dict[str, dict[str, float]] = {}
    for e in entries:
        m = str(e.get("model", "?"))
        agg = by_model.setdefault(m, {"calls": 0, "cost_usd": 0.0, "tokens": 0})
        agg["calls"] += 1
        agg["cost_usd"] += float(e.get("cost_usd", 0) or 0)
        agg["tokens"] += int(e.get("total_tokens", 0) or 0)
    chunks: list[Chunk] = []
    for m, agg in by_model.items():
        chunks.append(
            Chunk(
                citation_id=f"telemetry:{m}",
                kind="telemetry",
                text=(
                    f"Telemetry for {m}: {int(agg['calls'])} calls, "
                    f"${agg['cost_usd']:.4f} total, {int(agg['tokens'])} tokens."
                ),
                meta={"model": m, **agg},
            )
        )
    total_cost = sum(a["cost_usd"] for a in by_model.values())
    chunks.append(
        Chunk(
            citation_id="telemetry:total",
            kind="telemetry",
            text=f"Total LLM spend across {len(entries)} calls: ${total_cost:.4f}.",
            meta={"calls": len(entries), "cost_usd": total_cost},
        )
    )
    return chunks


def _backtest_chunks(reports_dir: Path) -> list[Chunk]:
    runs_dir = reports_dir / "backtest" / "runs"
    if not runs_dir.exists():
        return []
    chunks: list[Chunk] = []
    for run_dir in sorted(runs_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        # Real artifacts use a flat {strategy: {...metrics}} layout; the
        # legacy/test fixture wraps under "per_strategy". Accept either.
        per_strategy = metrics.get("per_strategy")
        if not per_strategy:
            per_strategy = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, dict) and "sharpe" in v
            }
        for sname, m in per_strategy.items():
            chunks.append(
                Chunk(
                    citation_id=f"backtest:{run_dir.name}/{sname}",
                    kind="backtest",
                    text=_trim(
                        f"Backtest run {run_dir.name} -- strategy {sname} "
                        f"performance: Sharpe ratio={m.get('sharpe', float('nan')):.4f}, "
                        f"CAGR={m.get('cagr', float('nan')):.4f}, "
                        f"max drawdown={m.get('max_drawdown', float('nan')):.4f}, "
                        f"turnover={m.get('avg_turnover', m.get('turnover', float('nan'))):.4f}."
                    ),
                    meta={"run_id": run_dir.name, "strategy": sname, **m},
                )
            )
    return chunks


def build_corpus(
    reports_dir: Path | None = None,
) -> list[Chunk]:
    """Walk every artifact source and produce a flat list of citable chunks."""
    rd = reports_dir or settings.reports_dir
    chunks: list[Chunk] = []
    chunks.extend(_finding_chunks(rd))
    chunks.extend(_lessons_chunks(rd))
    chunks.extend(_self_critique_chunks(rd))
    chunks.extend(_trace_quality_chunks(rd))
    chunks.extend(_telemetry_summary_chunks(rd))
    chunks.extend(_backtest_chunks(rd))
    chunks.extend(_ledger_chunks(rd))
    return chunks


def chunks_to_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), default=str) + "\n")


def chunks_from_jsonl(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                Chunk(
                    citation_id=d["citation_id"],
                    kind=d["kind"],
                    text=d["text"],
                    meta=d.get("meta", {}),
                )
            )
    return out
