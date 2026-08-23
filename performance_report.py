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


def _throughput_chart(record: dict) -> str:
    rows = [row for row in _candidate_rows(record) if not row.get("error")]
    if not rows:
        return '<p class="muted">No valid candidate timings to graph.</p>'
    rows = rows[:24]
    metrics = (
        ("PP", "prompt_tps", "bar-pp"),
        ("Decode", "generation_tps", "bar-decode"),
        ("End-to-end", "overall_tps", "bar-overall"),
    )
    maximum = max(
        (_float(row.get(field)) for row in rows for _label, field, _css in metrics),
        default=0.0,
    )
    width = 1120
    label_width = 245
    plot_width = 780
    row_height = 58
    height = 48 + row_height * len(rows)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="throughput-title throughput-desc">',
        '<title id="throughput-title">Candidate throughput comparison</title>',
        '<desc id="throughput-desc">Prompt processing, decode and end-to-end tokens per second for every valid candidate.</desc>',
    ]
    winner_id = _text(record.get("winner_id"))
    for index, row in enumerate(rows):
        top = 35 + index * row_height
        identifier = _text(row.get("label"), _text(row.get("id"), "candidate"))
        if _text(row.get("id")) == winner_id:
            identifier = f"★ {identifier}"
        parts.append(_svg_text(8, top + 20, identifier[:40], css="chart-label"))
        for metric_index, (label, field, css) in enumerate(metrics):
            value = _float(row.get(field))
            y = top + metric_index * 12
            bar_width = (value / maximum * plot_width) if maximum > 0 else 0
            parts.append(
                f'<rect x="{label_width}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="9" rx="3" class="{css}"><title>{escape(label)}: {value:.2f} tok/s</title></rect>'
            )
            parts.append(
                _svg_text(
                    label_width + bar_width + 6,
                    y + 8,
                    f"{label} {value:.1f}",
                    css="chart-value",
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def _acceptance_chart(record: dict) -> str:
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
    rows = rows[:24]
    width = 1000
    label_width = 270
    plot_width = 600
    row_height = 34
    height = 42 + row_height * len(rows)
    parts = [
        f'<svg class="chart acceptance-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="acceptance-title acceptance-desc">',
        '<title id="acceptance-title">Drafted-token acceptance</title>',
        '<desc id="acceptance-desc">Accepted drafted tokens divided by all drafted tokens for each candidate.</desc>',
    ]
    winner_id = _text(record.get("winner_id"))
    for index, (candidate, drafted, accepted, ratio) in enumerate(rows):
        y = 24 + index * row_height
        label = _text(candidate.get("label"), _text(candidate.get("id"), "candidate"))
        if _text(candidate.get("id")) == winner_id:
            label = f"★ {label}"
        parts.append(_svg_text(8, y + 14, label[:44], css="chart-label"))
        parts.append(
            f'<rect x="{label_width}" y="{y:.1f}" width="{plot_width}" height="18" rx="5" class="bar-track" />'
        )
        parts.append(
            f'<rect x="{label_width}" y="{y:.1f}" width="{ratio * plot_width:.1f}" '
            f'height="18" rx="5" class="bar-accept"><title>{accepted}/{drafted} accepted ({ratio * 100:.1f}%)</title></rect>'
        )
        parts.append(
            _svg_text(
                label_width + plot_width + 8,
                y + 14,
                f"{ratio * 100:.1f}% · {accepted}/{drafted}",
                css="chart-value",
            )
        )
    parts.append("</svg>")
    return "".join(parts)


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
        identifier = _text(candidate.get("label"), _text(candidate.get("id"), "candidate"))
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
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


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
        f"{model} · {target} · {drafter} · ctx {context:,} · "
        f"winner {winner_label}"
    )
    path = _text(record.get("model_path"))
    runtime = _text(record.get("runtime_binary"), "—")
    build = record.get("runtime_build", "—")
    return (
        f'<details class="run" id="run-{index}"><summary>{escape(summary)}</summary>'
        '<div class="run-body">'
        '<div class="facts">'
        f'<div><b>Model path</b><span>{escape(path or "—")}</span></div>'
        f'<div><b>Workload</b><span>{fraction * 100:.4g}% context · '
        f'{_integer(record.get("generated_token_target"))} decode target</span></div>'
        f'<div><b>Elapsed</b><span>{elapsed:.1f}s</span></div>'
        f'<div><b>Strategy</b><span>{escape(_text(record.get("search_strategy"), "legacy/full"))} '
        f'· {escape(_text(record.get("profile_confidence"), "unknown confidence"))}</span></div>'
        f'<div><b>Runtime</b><span>{escape(runtime)} · build {escape(str(build))}</span></div>'
        f'<div><b>Decision</b><span>{escape(_text(record.get("reason"), "—"))}</span></div>'
        f'<div class="wide"><b>Frozen quality/placement</b><span>{escape(_quality_text(record))}</span></div>'
        "</div>"
        '<h4>All candidate runs</h4>'
        + _candidate_table(record)
        + '<h4>Throughput comparison</h4>'
        + _throughput_chart(record)
        + '<h4>MTP / speculative drafted-token acceptance</h4>'
        + _acceptance_chart(record)
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
            f'<td>{escape(_text(record.get("performance_target"), "unknown"))}</td>'
            f'<td>{escape(_text(record.get("drafter_label"), "No drafter"))}</td>'
            f'<td>{_integer(record.get("desired_context")):,}</td>'
            f'<td>{escape(_text(winner.get("label"), _text(winner.get("id"), "—")) if winner else "—")}</td>'
            f'<td>{_format_tps(winner.get("prompt_tps") if winner else None)}</td>'
            f'<td>{_format_tps(winner.get("generation_tps") if winner else None)}</td>'
            f'<td>{_format_tps(winner.get("overall_tps") if winner else None)}</td>'
            f'<td>{acceptance_text}</td>'
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="9">No saved benchmark runs.</td></tr>')
    return (
        '<div class="table-wrap"><table><thead><tr><th>Model</th><th>Mode</th>'
        "<th>Drafter</th><th>Context</th><th>Winner</th><th>PP tok/s</th>"
        "<th>Decode tok/s</th><th>End-to-end tok/s</th><th>Draft acceptance</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
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
            f'<section><h2>{escape(_TEST_TITLES[test_type])}</h2>'
            f'<p class="muted">{len(typed)} model/mode/drafter run(s).</p>'
            f'{details or "<p>No saved runs for this workload.</p>"}</section>'
        )

    css = """
:root{color-scheme:dark;--bg:#11141a;--panel:#1a2029;--panel2:#222a35;--text:#edf3fa;--muted:#aab7c6;--line:#344150;--accent:#62b4ff;--good:#64d98b;--warn:#ffcb6b;--bad:#ff7b86}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1,h2,h3,h4{line-height:1.2}h1{margin-bottom:4px}h2{margin-top:38px;border-bottom:1px solid var(--line);padding-bottom:8px}a{color:var(--accent)}.muted{color:var(--muted)}.callout{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);padding:12px 16px;border-radius:8px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;min-width:900px;background:var(--panel)}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--panel2);z-index:1}tr.winner{background:color-mix(in srgb,var(--good) 13%,transparent)}tr.failed{background:color-mix(in srgb,var(--bad) 10%,transparent)}details.run{margin:12px 0;border:1px solid var(--line);border-radius:9px;background:var(--panel)}details.run>summary{cursor:pointer;padding:13px 15px;font-weight:650}.run-body{padding:0 15px 18px}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:8px;margin:8px 0 18px}.facts div{background:var(--panel2);padding:9px;border-radius:6px;display:flex;flex-direction:column}.facts .wide{grid-column:1/-1}.facts span{color:var(--muted);overflow-wrap:anywhere}.chart{width:100%;height:auto;max-height:760px;background:var(--panel2);border:1px solid var(--line);border-radius:8px}.chart text{fill:var(--text);font-size:11px}.chart .chart-label{font-size:12px}.chart .chart-value{fill:var(--muted)}.bar-pp{fill:#62b4ff}.bar-decode{fill:#b58cff}.bar-overall{fill:#64d98b}.bar-track{fill:#303b49}.bar-accept{fill:#ffcb6b}@media print{body{background:#fff;color:#111}.run,.table-wrap,.chart{break-inside:avoid}.muted,.facts span{color:#444}}
"""
    generated_text = generated.astimezone(timezone.utc).isoformat(timespec="seconds")
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:; base-uri \'none\'; form-action \'none\'">'
        "<title>AutoTuner performance report</title>"
        f"<style>{css}</style></head><body><main>"
        "<h1>AutoTuner performance report</h1>"
        f'<p class="muted">Generated {escape(generated_text)} · {len(records)} saved run(s)</p>'
        '<p class="callout"><b>How to read this report:</b> each saved model/performance-mode/drafter run is expanded into every measured candidate. ★ marks the applied winner. Prompt processing, decode, end-to-end throughput and drafted-token acceptance use the native llama.cpp response timings/counters. Legacy 25% records are classified as Custom.</p>'
        "<h2>Winner overview</h2>"
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
