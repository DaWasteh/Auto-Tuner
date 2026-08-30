"""Self-contained HTML performance report for AutoTuner benchmark evidence."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import app_settings

_TEST_ORDER = ("fast", "quick", "custom")
_TEST_TITLES = {
    "fast": "Quick pass · ≤3.125% context",
    "quick": "Standard benchmark · 12.5% context",
    "custom": "Custom context benchmark",
}
_TARGET_ORDER = {"safe": 0, "balanced": 1, "throughput": 2, "low_vram": 3}


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0.0 else 0.0


def _integer(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _text(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _model_name(record: dict) -> str:
    name = _text(record.get("model_name"))
    if name:
        return name
    path = _text(record.get("model_path"))
    return Path(path).stem if path else "Unknown model"


def _winner(record: dict) -> Optional[dict]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        return None
    winner_id = _text(record.get("winner_id"))
    valid = [candidate for candidate in candidates if isinstance(candidate, dict)]
    if winner_id:
        for candidate in valid:
            if _text(candidate.get("id")) == winner_id:
                return candidate
    return valid[0] if valid else None


def _acceptance(candidate: Optional[dict]) -> Tuple[int, int, float]:
    if not isinstance(candidate, dict):
        return 0, 0, 0.0
    drafted = _integer(candidate.get("draft_tokens"))
    accepted = min(drafted, _integer(candidate.get("draft_tokens_accepted")))
    ratio = _float(candidate.get("draft_acceptance"))
    if ratio <= 0.0 and drafted > 0:
        ratio = accepted / drafted
    return drafted, accepted, min(1.0, ratio)


def _candidate_settings(candidate: dict) -> dict:
    value = candidate.get("settings")
    return value if isinstance(value, dict) else {}


def _candidate_rows(record: dict) -> List[dict]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        return []
    winner_id = _text(record.get("winner_id"))
    rows = [candidate for candidate in candidates if isinstance(candidate, dict)]
    return sorted(
        rows,
        key=lambda item: (
            0 if _text(item.get("id")) == winner_id else 1,
            -_float(item.get("overall_tps")),
            _text(item.get("id")),
        ),
    )


def _format_tps(value: object) -> str:
    parsed = _float(value)
    return f"{parsed:.2f}" if parsed > 0.0 else "—"


def _format_seconds(value: object) -> str:
    parsed = _float(value)
    return f"{parsed:.2f}" if parsed > 0.0 else "—"


def _svg_text(x: float, y: float, value: object, *, css: str = "") -> str:
    class_attr = f' class="{css}"' if css else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{class_attr}>{escape(str(value))}</text>'


def _throughput_chart(record: dict, chart_id: str) -> str:
    rows = [row for row in _candidate_rows(record) if not row.get("error")][:24]
    if not rows:
        return '<p class="muted">No valid candidate timings to graph.</p>'
    metrics = (
        ("PP", "prompt_tps", "bar-pp"),
        ("Decode", "generation_tps", "bar-decode"),
        ("End-to-end", "overall_tps", "bar-overall"),
    )
    maxima = {
        field: max((_float(row.get(field)) for row in rows), default=0.0)
        for _label, field, _css in metrics
    }
    winner_id = _text(record.get("winner_id"))
    groups: List[str] = []
    for row in rows:
        identifier = _text(row.get("label"), _text(row.get("id"), "candidate"))
        winner = _text(row.get("id")) == winner_id
        bars: List[str] = []
        for label, field, css in metrics:
            value = _float(row.get(field))
            maximum = maxima[field]
            height = value / maximum * 100.0 if maximum > 0.0 else 0.0
            bars.append(
                '<div class="mini-metric">'
                f'<span class="bar-number">{value:.1f}</span>'
                '<div class="bar-stage">'
                f'<div class="vertical-bar {css}" style="height:{height:.3f}%" '
                f'title="{escape(label)}: {value:.2f} tok/s"></div>'
                "</div>"
                f'<span class="bar-caption">{escape(label)}</span>'
                "</div>"
            )
        groups.append(
            '<div class="candidate-column">'
            f'<div class="candidate-bars">{"".join(bars)}</div>'
            f'<div class="column-label"><b>{"★ " if winner else ""}{escape(identifier)}</b></div>'
            "</div>"
        )
    title_id = f"throughput-title-{chart_id}"
    desc_id = f"throughput-desc-{chart_id}"
    return (
        f'<div class="vertical-scroll chart" role="img" aria-labelledby="{title_id} {desc_id}">'
        f'<span class="sr-only" id="{title_id}">Candidate throughput comparison</span>'
        f'<span class="sr-only" id="{desc_id}">Vertical prompt processing, decode, and end-to-end bars. Candidates are side by side; each metric is normalized only against the same metric.</span>'
        f'<div class="candidate-chart">{"".join(groups)}</div></div>'
    )


def _acceptance_chart(record: dict, chart_id: str) -> str:
    rows: List[Tuple[dict, int, int, float]] = []
    for candidate in _candidate_rows(record):
        drafted, accepted, ratio = _acceptance(candidate)
        if drafted > 0:
            rows.append((candidate, drafted, accepted, ratio))
    if not rows:
        return (
            '<p class="muted">This run reported no drafted-token counters. '
            "Throughput remains available above.</p>"
        )
    winner_id = _text(record.get("winner_id"))
    columns: List[str] = []
    for candidate, drafted, accepted, ratio in rows[:24]:
        label = _text(candidate.get("label"), _text(candidate.get("id"), "candidate"))
        winner = _text(candidate.get("id")) == winner_id
        columns.append(
            '<div class="acceptance-column">'
            f'<span class="bar-number">{ratio * 100:.1f}%</span>'
            '<div class="bar-stage acceptance-stage">'
            f'<div class="vertical-bar bar-accept" style="height:{ratio * 100:.3f}%" '
            f'title="{accepted}/{drafted} accepted ({ratio * 100:.1f}%)"></div>'
            "</div>"
            f'<div class="column-label"><b>{"★ " if winner else ""}{escape(label)}</b>'
            f"<br><span>{accepted}/{drafted}</span></div></div>"
        )
    title_id = f"acceptance-title-{chart_id}"
    desc_id = f"acceptance-desc-{chart_id}"
    return (
        f'<div class="vertical-scroll chart" role="img" aria-labelledby="{title_id} {desc_id}">'
        f'<span class="sr-only" id="{title_id}">Drafted-token acceptance</span>'
        f'<span class="sr-only" id="{desc_id}">Vertical acceptance percentage bars for side-by-side candidates.</span>'
        f'<div class="acceptance-chart">{"".join(columns)}</div></div>'
    )


def _samples_summary(candidate: dict) -> str:
    samples = candidate.get("samples")
    if not isinstance(samples, list):
        return "—"
    valid = [sample for sample in samples if isinstance(sample, dict)]
    if not valid:
        return "—"
    pp = [_float(sample.get("prompt_tps")) for sample in valid]
    decode = [_float(sample.get("generation_tps")) for sample in valid]
    elapsed = [_float(sample.get("elapsed_s")) for sample in valid]
    return (
        f"{len(valid)} sample(s); median PP {median(pp):.2f}, "
        f"decode {median(decode):.2f} tok/s, wall {median(elapsed):.2f}s"
    )


def _candidate_table(record: dict) -> str:
    winner_id = _text(record.get("winner_id"))
    rows: List[str] = []
    for candidate in _candidate_rows(record):
        settings = _candidate_settings(candidate)
        drafted, accepted, acceptance = _acceptance(candidate)
        identifier = _text(
            candidate.get("label"), _text(candidate.get("id"), "candidate")
        )
        winner = _text(candidate.get("id")) == winner_id
        error = _text(candidate.get("error"))
        cls = ' class="winner"' if winner else (' class="failed"' if error else "")
        settings_text = (
            f"threads {settings.get('threads', '—')} / batch threads {settings.get('batch_threads', '—')}; "
            f"batch {settings.get('batch', '—')} / ubatch {settings.get('ubatch', '—')}; "
            f"draft n-max {settings.get('draft_n_max', '—')}"
        )
        acceptance_text = (
            f"{acceptance * 100:.1f}% ({accepted}/{drafted})" if drafted else "—"
        )
        rows.append(
            f"<tr{cls}>"
            f"<td>{'★ ' if winner else ''}{escape(identifier)}</td>"
            f"<td>{escape(settings_text)}</td>"
            f"<td>{_format_tps(candidate.get('prompt_tps'))}</td>"
            f"<td>{_format_tps(candidate.get('generation_tps'))}</td>"
            f"<td>{_format_tps(candidate.get('overall_tps'))}</td>"
            f"<td>{_format_seconds(candidate.get('inference_s'))}</td>"
            f"<td>{acceptance_text}</td>"
            f"<td>{escape(_samples_summary(candidate))}</td>"
            f"<td>{escape(error) if error else '—'}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="9">No candidate details were stored.</td></tr>')
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Candidate</th><th>Runtime settings</th><th>PP tok/s</th>"
        "<th>Decode tok/s</th><th>End-to-end tok/s</th><th>Inference s</th>"
        "<th>Draft acceptance</th><th>Samples</th><th>Error</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _backend_text(record: dict) -> str:
    backend = app_settings.normalise_performance_backend(
        _text(record.get("benchmark_backend"))
    )
    label = app_settings.performance_backend_label(backend)
    runtime_label = _text(record.get("runtime_label"))
    return runtime_label or label


def _quality_text(record: dict) -> str:
    quality = record.get("quality_frozen")
    if not isinstance(quality, dict):
        return "—"
    fields = (
        "cache_k",
        "cache_v",
        "flash_attn",
        "ngl",
        "n_cpu_moe",
        "tensor_split",
        "main_gpu",
        "rope_scaling",
        "rope_factor",
    )
    return ", ".join(f"{key}={quality.get(key)}" for key in fields if key in quality)


def _record_detail(record: dict, index: int) -> str:
    model = _model_name(record)
    target = _text(record.get("performance_target"), "unknown")
    backend = _backend_text(record)
    drafter = _text(record.get("drafter_label"), "No drafter")
    context = _integer(record.get("desired_context"))
    fraction = _float(record.get("prompt_context_fraction"))
    elapsed = _float(record.get("elapsed_s"))
    winner = _winner(record)
    winner_label = (
        _text(winner.get("label"), _text(winner.get("id"), "—"))
        if winner is not None
        else "—"
    )
    summary = (
        f"{model} · {target} · {backend} · {drafter} · ctx {context:,} · "
        f"winner {winner_label}"
    )
    path = _text(record.get("model_path"))
    runtime = _text(record.get("runtime_binary"), "—")
    build = record.get("runtime_build", "—")
    return (
        f'<details class="run" id="run-{index}"><summary>{escape(summary)}</summary>'
        '<div class="run-body">'
        '<div class="facts">'
        f"<div><b>Model path</b><span>{escape(path or '—')}</span></div>"
        f"<div><b>Workload</b><span>{fraction * 100:.4g}% context · "
        f"{_integer(record.get('generated_token_target'))} decode target</span></div>"
        f"<div><b>Elapsed</b><span>{elapsed:.1f}s</span></div>"
        f"<div><b>Strategy</b><span>{escape(_text(record.get('search_strategy'), 'legacy/full'))} "
        f"· {escape(_text(record.get('profile_confidence'), 'unknown confidence'))}</span></div>"
        f"<div><b>Runtime</b><span>{escape(backend)} · {escape(runtime)} · build {escape(str(build))}</span></div>"
        f"<div><b>Decision</b><span>{escape(_text(record.get('reason'), '—'))}</span></div>"
        f'<div class="wide"><b>Frozen quality/placement</b><span>{escape(_quality_text(record))}</span></div>'
        "</div>"
        "<h4>All candidate runs</h4>"
        + _candidate_table(record)
        + "<h4>Throughput comparison</h4>"
        + _throughput_chart(record, str(index))
        + "<h4>MTP / speculative drafted-token acceptance</h4>"
        + _acceptance_chart(record, str(index))
        + "</div></details>"
    )


def _summary_table(records: Sequence[dict]) -> str:
    rows: List[str] = []
    for index, record in enumerate(records, start=1):
        winner = _winner(record)
        drafted, accepted, acceptance = _acceptance(winner)
        acceptance_text = (
            f"{acceptance * 100:.1f}% ({accepted}/{drafted})" if drafted else "—"
        )
        rows.append(
            "<tr>"
            f'<td><a href="#run-{index}">{escape(_model_name(record))}</a></td>'
            f"<td>{escape(_text(record.get('performance_target'), 'unknown'))}</td>"
            f"<td>{escape(_backend_text(record))}</td>"
            f"<td>{escape(_text(record.get('drafter_label'), 'No drafter'))}</td>"
            f"<td>{_integer(record.get('desired_context')):,}</td>"
            f"<td>{escape(_text(winner.get('label'), _text(winner.get('id'), '—')) if winner else '—')}</td>"
            f"<td>{_format_tps(winner.get('prompt_tps') if winner else None)}</td>"
            f"<td>{_format_tps(winner.get('generation_tps') if winner else None)}</td>"
            f"<td>{_format_tps(winner.get('overall_tps') if winner else None)}</td>"
            f"<td>{acceptance_text}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="10">No saved benchmark runs.</td></tr>')
    return (
        '<div class="table-wrap"><table><thead><tr><th>Model</th><th>Mode</th>'
        "<th>Backend / build</th><th>Drafter</th><th>Context</th><th>Winner</th><th>PP tok/s</th>"
        "<th>Decode tok/s</th><th>End-to-end tok/s</th><th>Draft acceptance</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _winner_overview(records: Sequence[dict], chart_id: str) -> str:
    """Render winner bars vertically so model/backend runs sit side by side."""
    rows: List[Tuple[dict, dict]] = []
    for record in records:
        winner = _winner(record)
        if isinstance(winner, dict) and not winner.get("error"):
            rows.append((record, winner))
    if not rows:
        return '<p class="muted">No valid winning timings to compare.</p>'
    metrics = (
        ("Prompt processing", "prompt_tps", "bar-pp"),
        ("Decode", "generation_tps", "bar-decode"),
        ("End-to-end", "overall_tps", "bar-overall"),
    )
    panels: List[str] = []
    for metric_index, (metric_label, field, css) in enumerate(metrics):
        maximum = max(
            (_float(winner.get(field)) for _record, winner in rows), default=0.0
        )
        columns: List[str] = []
        for record, winner in rows:
            value = _float(winner.get(field))
            height = value / maximum * 100.0 if maximum > 0.0 else 0.0
            model = _model_name(record)
            target = _text(record.get("performance_target"), "unknown")
            backend = _backend_text(record)
            columns.append(
                '<div class="model-column">'
                f'<span class="bar-number">{value:.1f}</span>'
                '<div class="bar-stage overview-stage">'
                f'<div class="vertical-bar {css}" style="height:{height:.3f}%" '
                f'title="{escape(model)} · {escape(metric_label)}: {value:.2f} tok/s"></div>'
                "</div>"
                f'<div class="column-label"><b>{escape(model)}</b><br>'
                f"<span>{escape(target)} · {escape(backend)}</span></div></div>"
            )
        title_id = f"overview-{chart_id}-{metric_index}"
        panels.append(
            '<section class="metric-panel">'
            f'<h3 id="{title_id}">{escape(metric_label)} <small>tok/s</small></h3>'
            f'<div class="overview-chart" role="img" aria-labelledby="{title_id}">'
            f'<div class="model-chart">{"".join(columns)}</div></div></section>'
        )
    # One outer overflow surface moves every metric panel together.  Keeping the
    # panels in a vertical stack and giving each the same intrinsic width makes
    # a model/backend lane stay in exactly the same column for PP, decode, and
    # end-to-end even for collections with dozens of models.
    chart_width = max(1, len(rows)) * 138 + 8
    return (
        '<div class="overview-note">Vertical bars; model/backend runs are side by '
        "side. Scroll the stacked panels together; each model stays in the same "
        "column. Each panel has its own scale, so compare only bars of the same metric."
        '</div><div class="overview-scroll" role="region" '
        'aria-label="Synchronized model metric charts" tabindex="0">'
        f'<div class="overview-grid" style="width:max(100%,{chart_width}px)">'
        + "".join(panels)
        + "</div></div>"
    )


def _flatten_records(records_by_test: Dict[str, List[dict]]) -> List[dict]:
    records: List[dict] = []
    for test_type in _TEST_ORDER:
        for raw in records_by_test.get(test_type, []):
            if isinstance(raw, dict):
                item = dict(raw)
                item["benchmark_type"] = test_type
                records.append(item)
    return sorted(
        records,
        key=lambda record: (
            _TEST_ORDER.index(_text(record.get("benchmark_type"), "custom")),
            _model_name(record).casefold(),
            _TARGET_ORDER.get(_text(record.get("performance_target")), 99),
            _text(record.get("drafter_label")).casefold(),
        ),
    )


def build_performance_report_html(
    records_by_test: Dict[str, List[dict]],
    *,
    generated_at: Optional[datetime] = None,
) -> str:
    """Render benchmark evidence as dependency-free, escaped HTML."""
    generated = generated_at or datetime.now(timezone.utc)
    records = _flatten_records(records_by_test)
    sections: List[str] = []
    offset = 0
    for test_type in _TEST_ORDER:
        typed = [
            record
            for record in records
            if _text(record.get("benchmark_type")) == test_type
        ]
        details = "".join(
            _record_detail(record, offset + index)
            for index, record in enumerate(typed, start=1)
        )
        offset += len(typed)
        sections.append(
            f'<section id="workload-{test_type}"><h2>{escape(_TEST_TITLES[test_type])}</h2>'
            f'<p class="muted">{len(typed)} model/mode/backend/drafter run(s).</p>'
            + (_winner_overview(typed, test_type) if typed else "")
            + f"{details or '<p>No saved runs for this workload.</p>'}</section>"
        )

    css = """
:root{color-scheme:dark;--bg:#0b1018;--panel:#151d29;--panel2:#1d2837;--panel3:#253448;--text:#f2f6fb;--muted:#a8b7c9;--line:#33445a;--accent:#66bbff;--good:#58d68d;--violet:#b794f6;--warn:#ffd166;--bad:#ff7f8a;--shadow:0 16px 45px #0006}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 15% -10%,#1d3958 0,transparent 32rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1,h2,h3,h4{line-height:1.2}h1{font-size:clamp(2rem,5vw,3.6rem);margin:.15em 0}h2{margin-top:46px;border-bottom:1px solid var(--line);padding-bottom:10px}h3 small{color:var(--muted);font-weight:500}a{color:var(--accent)}.muted{color:var(--muted)}.hero{padding:28px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,#1a2738dd,#111923dd);box-shadow:var(--shadow)}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.13em;text-transform:uppercase}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:22px}.kpi{padding:14px;border:1px solid var(--line);border-radius:12px;background:#0b111acc}.kpi b{display:block;font-size:1.6rem}.kpi span{color:var(--muted)}nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 24px}nav a{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:var(--panel);text-decoration:none}.callout,.overview-note{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);padding:12px 16px;border-radius:10px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px;box-shadow:0 8px 24px #0003}table{border-collapse:collapse;width:100%;min-width:980px;background:var(--panel)}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--panel3);z-index:1}tbody tr:hover{background:#ffffff08}tr.winner{background:color-mix(in srgb,var(--good) 13%,transparent)}tr.failed{background:color-mix(in srgb,var(--bad) 11%,transparent)}details.run{margin:14px 0;border:1px solid var(--line);border-radius:12px;background:var(--panel);box-shadow:0 7px 22px #0002}details.run>summary{cursor:pointer;padding:14px 16px;font-weight:700}details.run[open]>summary{border-bottom:1px solid var(--line);background:var(--panel2)}.run-body{padding:4px 16px 20px}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:8px;margin:12px 0 20px}.facts div{background:var(--panel2);padding:10px;border-radius:8px;display:flex;flex-direction:column}.facts .wide{grid-column:1/-1}.facts span{color:var(--muted);overflow-wrap:anywhere}.overview-scroll{overflow-x:auto;overflow-y:hidden;margin:12px 0 20px;padding:0 0 6px;scrollbar-gutter:stable}.overview-scroll:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.overview-grid{display:grid;grid-template-columns:1fr;gap:12px}.metric-panel{width:100%;min-width:0;margin:0;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}.metric-panel h3{position:sticky;left:12px;width:max-content;max-width:calc(100vw - 80px);margin:0 0 8px;padding:2px 8px;border-radius:6px;background:var(--panel);z-index:1}.overview-chart{padding:8px 4px 4px}.vertical-scroll{overflow-x:auto;overflow-y:hidden;padding:8px 4px 4px}.model-chart,.candidate-chart,.acceptance-chart{display:flex;align-items:flex-end;gap:12px;min-width:max-content}.model-column{width:126px;text-align:center}.candidate-column{width:220px;text-align:center}.acceptance-column{width:130px;text-align:center}.candidate-bars{display:flex;align-items:flex-end;justify-content:center;gap:10px}.mini-metric{display:flex;flex-direction:column;align-items:center;width:58px}.bar-stage{height:190px;width:32px;display:flex;align-items:flex-end;border-radius:7px 7px 3px 3px;background:linear-gradient(#ffffff0b,#ffffff03);border-bottom:2px solid var(--line);overflow:hidden}.overview-stage{height:220px;width:42px;margin:auto}.acceptance-stage{height:180px;width:46px;margin:auto}.vertical-bar{width:100%;min-height:2px;border-radius:6px 6px 2px 2px;box-shadow:inset 0 1px #fff5;transition:filter .15s}.vertical-bar:hover{filter:brightness(1.2)}.bar-number{display:block;color:var(--text);font-variant-numeric:tabular-nums;margin-bottom:4px}.bar-caption{color:var(--muted);font-size:.76rem;margin-top:5px}.column-label{margin-top:8px;overflow-wrap:anywhere}.column-label span{color:var(--muted);font-size:.82rem}.bar-pp{background:linear-gradient(#86ccff,#298ee3)}.bar-decode{background:linear-gradient(#ccb3ff,#805ad5)}.bar-overall{background:linear-gradient(#83e8aa,#2cab68)}.bar-accept{background:linear-gradient(#ffe49b,#e5a923)}.chart{width:100%;background:var(--panel2);border:1px solid var(--line);border-radius:10px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:1050px){.bar-stage{height:160px}.overview-stage{height:190px}}@media(max-width:640px){main{padding:14px}.hero{padding:18px}.candidate-column{width:190px}.mini-metric{width:50px}.bar-stage{height:140px}}@media print{body{background:#fff;color:#111}.hero,.run,.table-wrap,.metric-panel,.chart{break-inside:avoid;box-shadow:none}.muted,.facts span,.column-label span{color:#444}.vertical-scroll,.overview-scroll{overflow:visible}.bar-pp,.bar-decode,.bar-overall,.bar-accept{print-color-adjust:exact}}
"""
    generated_text = generated.astimezone(timezone.utc).isoformat(timespec="seconds")
    model_count = len({_model_name(record).casefold() for record in records})
    backend_labels = {
        _backend_text(record)
        for record in records
        if _backend_text(record) != "Unknown"
    }
    candidates = [
        candidate
        for record in records
        for candidate in (
            record.get("candidates")
            if isinstance(record.get("candidates"), list)
            else []
        )
        if isinstance(candidate, dict)
    ]
    failed_candidates = sum(
        bool(_text(candidate.get("error"))) for candidate in candidates
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'\">"
        "<title>AutoTuner performance report</title>"
        f"<style>{css}</style></head><body><main>"
        '<header class="hero"><div class="eyebrow">Measured llama.cpp evidence</div>'
        "<h1>AutoTuner performance report</h1>"
        f'<p class="muted">Generated {escape(generated_text)} · backend-specific, dependency-free report</p>'
        '<div class="kpis">'
        f'<div class="kpi"><b>{model_count}</b><span>models</span></div>'
        f'<div class="kpi"><b>{len(records)}</b><span>saved runs</span></div>'
        f'<div class="kpi"><b>{len(candidates)}</b><span>candidate runs</span></div>'
        f'<div class="kpi"><b>{failed_candidates}</b><span>failed candidates retained</span></div>'
        f'<div class="kpi"><b>{len(backend_labels)}</b><span>backend/build lanes</span></div>'
        "</div></header>"
        '<nav aria-label="Report sections"><a href="#winner-overview">Winner table</a>'
        '<a href="#workload-fast">Quick pass</a><a href="#workload-quick">Standard</a>'
        '<a href="#workload-custom">Custom</a></nav>'
        '<p class="callout"><b>How to read this report:</b> vertical bars place models and candidates side by side. Compare heights only within one metric panel. Each saved model/performance-mode/backend/drafter run expands into every measured candidate, including failed settings. ★ marks the applied winner. Prompt processing, decode, end-to-end throughput, and drafted-token acceptance use native llama.cpp response timings/counters. Legacy 25% records are classified as Custom.</p>'
        '<h2 id="winner-overview">Winner overview</h2>'
        + _summary_table(records)
        + "".join(sections)
        + "</main></body></html>"
    )


def write_performance_report(
    records_by_test: Dict[str, List[dict]], destination: Optional[Path] = None
) -> Path:
    """Atomically write a timestamped report under ~/.autotuner/reports."""
    now = datetime.now(timezone.utc)
    if destination is None:
        report_dir = app_settings.app_data_dir() / "reports"
        destination = report_dir / now.strftime("performance-report-%Y%m%d-%H%M%S.html")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = build_performance_report_html(records_by_test, generated_at=now)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8", newline="\n")
    os.replace(tmp, destination)
    latest = destination.parent / "performance-report-latest.html"
    latest_tmp = latest.with_suffix(latest.suffix + ".tmp")
    latest_tmp.write_text(html, encoding="utf-8", newline="\n")
    os.replace(latest_tmp, latest)
    return destination
