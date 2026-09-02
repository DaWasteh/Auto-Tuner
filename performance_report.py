"""Self-contained HTML performance report for AutoTuner benchmark evidence."""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import app_settings
from autotuner_version import VERSION

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


def _path_name(value: object) -> str:
    """Return one filename for either Windows or POSIX path text."""
    text = _text(value).replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def _model_name(record: dict) -> str:
    name = _text(record.get("model_name"))
    if name:
        return name
    filename = _path_name(record.get("model_path"))
    return Path(filename).stem if filename else "Unknown model"


_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![\w])(?:[a-z]:[\\/])[^<>\r\n]+")
_UNC_ABSOLUTE_PATH = re.compile(r"(?<![:\\/])(?:\\\\|//)(?![\\/])[^<>\r\n]+")
_FILE_URI = re.compile(r"(?i)\bfile:(?:/+|\\+)[^<>\r\n]+")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![\w:/])/(?!/)(?=[A-Za-z0-9._~-])[^<>\r\n]+")
_PUBLIC_PATH_PATTERNS = (
    _FILE_URI,
    _UNC_ABSOLUTE_PATH,
    _WINDOWS_ABSOLUTE_PATH,
    _POSIX_ABSOLUTE_PATH,
)
_PUBLIC_PATH_MARKER = "[local path omitted]"


def _redact_local_paths(value: object) -> str:
    text = _text(value)
    for pattern in _PUBLIC_PATH_PATTERNS:
        text = pattern.sub(_PUBLIC_PATH_MARKER, text)
    return text


def _public_text(value: object, *, public: bool = False, default: str = "") -> str:
    text = _text(value, default)
    return _redact_local_paths(text) if public else text


def _display_text(record: dict, value: object, *, public: bool = False) -> str:
    """Format free text and remove machine-local paths from public reports."""
    text = _text(value)
    if not public or not text:
        return text
    for field in ("model_path", "runtime_binary"):
        local_path = _text(record.get(field))
        if not local_path:
            continue
        replacement = _path_name(local_path) or _PUBLIC_PATH_MARKER
        variants = {
            local_path,
            local_path.replace("\\", "/"),
            local_path.replace("/", "\\"),
        }
        for variant in sorted(variants, key=len, reverse=True):
            text = text.replace(variant, replacement)
    return _redact_local_paths(text)


def _model_display(record: dict, *, public: bool = False) -> str:
    return _display_text(record, _model_name(record), public=public)


def _backend_display(record: dict, *, public: bool = False) -> str:
    return _display_text(record, _backend_text(record), public=public)


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


def _throughput_chart(record: dict, chart_id: str, *, public: bool = False) -> str:
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
        identifier = _display_text(
            record,
            _text(row.get("label"), _text(row.get("id"), "candidate")),
            public=public,
        )
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
        f'<div class="vertical-scroll chart" role="img" aria-labelledby="{title_id} {desc_id}" tabindex="0">'
        f'<span class="sr-only" id="{title_id}">Candidate throughput comparison</span>'
        f'<span class="sr-only" id="{desc_id}">Vertical prompt processing, decode, and end-to-end bars. Candidates are side by side; each metric is normalized only against the same metric.</span>'
        f'<div class="candidate-chart">{"".join(groups)}</div></div>'
    )


def _acceptance_chart(record: dict, chart_id: str, *, public: bool = False) -> str:
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
        label = _display_text(
            record,
            _text(candidate.get("label"), _text(candidate.get("id"), "candidate")),
            public=public,
        )
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
        f'<div class="vertical-scroll chart" role="img" aria-labelledby="{title_id} {desc_id}" tabindex="0">'
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


def _candidate_table(record: dict, *, public: bool = False) -> str:
    winner_id = _text(record.get("winner_id"))
    rows: List[str] = []
    for candidate in _candidate_rows(record):
        settings = _candidate_settings(candidate)
        drafted, accepted, acceptance = _acceptance(candidate)
        identifier = _display_text(
            record,
            _text(candidate.get("label"), _text(candidate.get("id"), "candidate")),
            public=public,
        )
        winner = _text(candidate.get("id")) == winner_id
        error = _display_text(record, candidate.get("error"), public=public)
        cls = ' class="winner"' if winner else (' class="failed"' if error else "")
        settings_text = _public_text(
            (
                f"threads {settings.get('threads', '—')} / batch threads {settings.get('batch_threads', '—')}; "
                f"batch {settings.get('batch', '—')} / ubatch {settings.get('ubatch', '—')}; "
                f"draft n-max {settings.get('draft_n_max', '—')}"
            ),
            public=public,
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
        '<div class="table-wrap" role="region" aria-label="All candidate runs" tabindex="0"><table>'
        '<caption class="sr-only">All benchmark candidates and their measured settings</caption><thead><tr>'
        '<th scope="col">Candidate</th><th scope="col">Runtime settings</th><th scope="col">PP tok/s</th>'
        '<th scope="col">Decode tok/s</th><th scope="col">End-to-end tok/s</th><th scope="col">Inference s</th>'
        '<th scope="col">Draft acceptance</th><th scope="col">Samples</th><th scope="col">Error</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _backend_text(record: dict) -> str:
    backend = app_settings.normalise_performance_backend(
        _text(record.get("benchmark_backend"))
    )
    label = app_settings.performance_backend_label(backend)
    runtime_label = _text(record.get("runtime_label"))
    return runtime_label or label


def _quality_text(record: dict, *, public: bool = False) -> str:
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
    return _public_text(
        ", ".join(f"{key}={quality.get(key)}" for key in fields if key in quality),
        public=public,
    )


def _record_detail(record: dict, index: int, *, public: bool = False) -> str:
    model = _model_display(record, public=public)
    target = _public_text(
        record.get("performance_target"), public=public, default="unknown"
    )
    backend = _backend_display(record, public=public)
    drafter = (
        _display_text(record, record.get("drafter_label"), public=public)
        or "No drafter"
    )
    context = _integer(record.get("desired_context"))
    fraction = _float(record.get("prompt_context_fraction"))
    elapsed = _float(record.get("elapsed_s"))
    winner = _winner(record)
    winner_label = _display_text(
        record,
        (
            _text(winner.get("label"), _text(winner.get("id"), "—"))
            if winner is not None
            else "—"
        ),
        public=public,
    )
    failed = bool(_text(record.get("error"))) or _text(record.get("status")) == "failed"
    status = "Failed" if failed else "Measured"
    summary = (
        f"{model} · {target} · {backend} · {drafter} · ctx {context:,} · "
        f"winner {winner_label}"
    )
    raw_path = _text(record.get("model_path"))
    path = _path_name(raw_path) if public else raw_path
    raw_runtime = _text(record.get("runtime_binary"))
    runtime = _path_name(raw_runtime) if public else raw_runtime
    build = _public_text(record.get("runtime_build"), public=public, default="—")
    strategy = (
        _display_text(record, record.get("search_strategy"), public=public)
        or "legacy/full"
    )
    confidence = (
        _display_text(record, record.get("profile_confidence"), public=public)
        or "unknown confidence"
    )
    decision = _display_text(record, record.get("reason"), public=public) or "—"
    return (
        f'<details class="run" id="run-{index}"><summary>'
        f'<span class="summary-copy">{escape(summary)}</span>'
        f'<span class="status-pill {"failed-pill" if failed else "measured-pill"}">{status}</span>'
        "</summary>"
        '<div class="run-body">'
        '<div class="facts">'
        f"<div><b>{'Model file' if public else 'Model path'}</b><span>{escape(path or '—')}</span></div>"
        f"<div><b>Workload</b><span>{fraction * 100:.4g}% context · "
        f"{_integer(record.get('generated_token_target'))} decode target</span></div>"
        f"<div><b>Elapsed</b><span>{elapsed:.1f}s</span></div>"
        f"<div><b>Strategy</b><span>{escape(strategy)} · {escape(confidence)}</span></div>"
        f"<div><b>Runtime</b><span>{escape(backend)} · {escape(runtime or '—')} · build {escape(str(build))}</span></div>"
        f"<div><b>Decision</b><span>{escape(decision)}</span></div>"
        f'<div class="wide"><b>Frozen quality/placement</b><span>{escape(_quality_text(record, public=public))}</span></div>'
        "</div>"
        "<h4>All candidate runs</h4>"
        + _candidate_table(record, public=public)
        + "</div></details>"
    )


def _summary_table(records: Sequence[dict], *, public: bool = False) -> str:
    rows: List[str] = []
    for index, record in enumerate(records, start=1):
        winner = _winner(record)
        drafted, accepted, acceptance = _acceptance(winner)
        acceptance_text = (
            f"{acceptance * 100:.1f}% ({accepted}/{drafted})" if drafted else "—"
        )
        model = _model_display(record, public=public)
        target = _public_text(
            record.get("performance_target"), public=public, default="unknown"
        )
        drafter = (
            _display_text(record, record.get("drafter_label"), public=public)
            or "No drafter"
        )
        winner_label = _display_text(
            record,
            (
                _text(winner.get("label"), _text(winner.get("id"), "—"))
                if winner
                else "—"
            ),
            public=public,
        )
        rows.append(
            "<tr>"
            f'<td><a href="#run-{index}">{escape(model)}</a></td>'
            f"<td>{escape(target)}</td>"
            f"<td>{escape(_backend_display(record, public=public))}</td>"
            f"<td>{escape(drafter)}</td>"
            f"<td>{_integer(record.get('desired_context')):,}</td>"
            f"<td>{escape(winner_label)}</td>"
            f"<td>{_format_tps(winner.get('prompt_tps') if winner else None)}</td>"
            f"<td>{_format_tps(winner.get('generation_tps') if winner else None)}</td>"
            f"<td>{_format_tps(winner.get('overall_tps') if winner else None)}</td>"
            f"<td>{acceptance_text}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="10">No saved benchmark runs.</td></tr>')
    return (
        '<div class="table-wrap" role="region" aria-label="Benchmark winner overview" tabindex="0"><table>'
        '<caption class="sr-only">Winning result for every saved benchmark run</caption><thead><tr>'
        '<th scope="col">Model</th><th scope="col">Mode</th>'
        '<th scope="col">Backend / build</th><th scope="col">Drafter</th><th scope="col">Context</th><th scope="col">Winner</th><th scope="col">PP tok/s</th>'
        '<th scope="col">Decode tok/s</th><th scope="col">End-to-end tok/s</th><th scope="col">Draft acceptance</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _representative_winners(records: Sequence[dict]) -> List[Tuple[dict, dict]]:
    """Select one fastest end-to-end winner lane for each model."""
    chosen: Dict[str, Tuple[dict, dict]] = {}
    for record in records:
        winner = _winner(record)
        if not isinstance(winner, dict) or winner.get("error"):
            continue
        model_key = _model_name(record).casefold()
        current = chosen.get(model_key)
        score = (
            _float(winner.get("overall_tps")),
            _float(winner.get("generation_tps")),
            _float(winner.get("prompt_tps")),
        )
        current_score = (
            (
                _float(current[1].get("overall_tps")),
                _float(current[1].get("generation_tps")),
                _float(current[1].get("prompt_tps")),
            )
            if current is not None
            else (-1.0, -1.0, -1.0)
        )
        if score > current_score:
            chosen[model_key] = (record, winner)
    return sorted(chosen.values(), key=lambda row: _model_name(row[0]).casefold())


def _winner_overview(
    records: Sequence[dict], chart_id: str, *, public: bool = False
) -> str:
    """Render one aligned, fastest winner lane per model."""
    rows = _representative_winners(records)
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
            model = _model_display(record, public=public)
            target = _public_text(
                record.get("performance_target"), public=public, default="unknown"
            )
            backend = _backend_display(record, public=public)
            drafter = (
                _display_text(record, record.get("drafter_label"), public=public)
                or "No drafter"
            )
            lane = f"{target} · {backend}"
            if drafter.casefold() != "no drafter":
                lane += f" · {drafter}"
            columns.append(
                '<div class="model-column">'
                f'<span class="bar-number">{value:.1f}</span>'
                '<div class="bar-stage overview-stage">'
                f'<div class="vertical-bar {css}" style="height:{height:.3f}%" '
                f'title="{escape(model)} · {escape(metric_label)}: {value:.2f} tok/s"></div>'
                "</div>"
                f'<div class="column-label"><b>{escape(model)}</b><br>'
                f"<span>{escape(lane)}</span></div></div>"
            )
        title_id = f"overview-{chart_id}-{metric_index}"
        desc_id = f"overview-{chart_id}-{metric_index}-desc"
        panels.append(
            '<section class="metric-panel">'
            f'<h4 id="{title_id}">{escape(metric_label)} <small>tok/s</small></h4>'
            f'<span class="sr-only" id="{desc_id}">The fastest end-to-end winner lane for each model, aligned across all three metric panels.</span>'
            f'<div class="overview-chart" role="img" aria-labelledby="{title_id} {desc_id}">'
            f'<div class="model-chart">{"".join(columns)}</div></div></section>'
        )
    # One overflow surface moves every metric panel together.  The same selected
    # lane stays in one column for all metrics, even when many runs exist per model.
    chart_width = max(1, len(rows)) * 154 + 8
    return (
        '<div class="overview-note"><b>Fastest lane per model.</b> The report picks '
        "each model’s highest end-to-end winner and keeps that exact lane aligned "
        "for PP, decode, and end-to-end. Scroll the stacked panels together; each "
        "model stays in the same column. Each panel has its own scale.</div>"
        '<div class="overview-scroll" role="region" '
        'aria-label="Synchronized model metric charts" tabindex="0">'
        f'<div class="overview-grid" style="width:max(100%,{chart_width}px)">'
        + "".join(panels)
        + "</div></div>"
    )


def _winner_acceptance_overview(
    records: Sequence[dict], chart_id: str, *, public: bool = False
) -> str:
    """Render the strongest winning drafted-token acceptance for each model."""
    chosen: Dict[str, Tuple[dict, dict, int, int, float]] = {}
    for record in records:
        winner = _winner(record)
        if not isinstance(winner, dict) or winner.get("error"):
            continue
        drafted, accepted, ratio = _acceptance(winner)
        if drafted <= 0:
            continue
        key = _model_name(record).casefold()
        current = chosen.get(key)
        if current is None or (ratio, accepted, drafted) > (
            current[4],
            current[3],
            current[2],
        ):
            chosen[key] = (record, winner, drafted, accepted, ratio)
    rows = sorted(chosen.values(), key=lambda row: _model_name(row[0]).casefold())
    if not rows:
        return ""
    columns: List[str] = []
    for record, _winner_row, drafted, accepted, ratio in rows:
        model = _model_display(record, public=public)
        target = _public_text(
            record.get("performance_target"), public=public, default="unknown"
        )
        backend = _backend_display(record, public=public)
        columns.append(
            '<div class="acceptance-column">'
            f'<span class="bar-number">{ratio * 100:.1f}%</span>'
            '<div class="bar-stage acceptance-stage">'
            f'<div class="vertical-bar bar-accept" style="height:{ratio * 100:.3f}%" '
            f'title="{escape(model)}: {accepted}/{drafted} accepted"></div></div>'
            f'<div class="column-label"><b>{escape(model)}</b><br>'
            f"<span>{escape(target)} · {escape(backend)} · {accepted}/{drafted}</span>"
            "</div></div>"
        )
    title_id = f"winner-acceptance-{chart_id}"
    desc_id = f"winner-acceptance-{chart_id}-desc"
    chart_width = max(1, len(rows)) * 154 + 8
    return (
        '<section class="acceptance-overview-panel">'
        f'<h4 id="{title_id}">Best winning speculative acceptance <small>%</small></h4>'
        f'<p class="muted" id="{desc_id}">Highest winning drafted-token acceptance per model. Compare it with throughput; acceptance alone is not speed.</p>'
        '<div class="acceptance-overview-scroll" role="region" '
        'aria-label="Winning drafted-token acceptance chart" tabindex="0">'
        f'<div class="acceptance-chart" style="width:max(100%,{chart_width}px)" role="img" aria-labelledby="{title_id} {desc_id}">'
        + "".join(columns)
        + "</div></div></section>"
    )


def _leader_cards(records: Sequence[dict], *, public: bool = False) -> str:
    valid: List[Tuple[dict, dict]] = []
    for record in records:
        winner = _winner(record)
        if isinstance(winner, dict) and not winner.get("error"):
            valid.append((record, winner))
    if not valid:
        return '<p class="muted">No successful winner metrics are available yet.</p>'

    cards: List[str] = []
    metrics = (
        ("PP", "Fastest prompt processing", "prompt_tps", "pp"),
        ("TG", "Fastest decode", "generation_tps", "decode"),
        ("E2E", "Fastest end-to-end", "overall_tps", "overall"),
    )
    for token, label, field, css in metrics:
        record, winner = max(valid, key=lambda row: _float(row[1].get(field)))
        value = _float(winner.get(field))
        cards.append(
            '<article class="leader-card">'
            f'<span class="metric-token {css}">{token}</span>'
            f'<div><span class="leader-label">{label}</span>'
            f"<strong>{value:.2f} <small>tok/s</small></strong>"
            f"<span>{escape(_model_display(record, public=public))}<br>{escape(_public_text(record.get('performance_target'), public=public, default='unknown'))} · {escape(_backend_display(record, public=public))}</span>"
            "</div></article>"
        )

    context_record, _context_winner = max(
        valid, key=lambda row: _integer(row[0].get("desired_context"))
    )
    cards.append(
        '<article class="leader-card">'
        '<span class="metric-token context">CTX</span><div>'
        '<span class="leader-label">Largest measured context</span>'
        f"<strong>{_integer(context_record.get('desired_context')):,} <small>tokens</small></strong>"
        f"<span>{escape(_model_display(context_record, public=public))}<br>{escape(_public_text(context_record.get('performance_target'), public=public, default='unknown'))} · {escape(_backend_display(context_record, public=public))}</span>"
        "</div></article>"
    )

    acceptance_rows: List[Tuple[dict, int, int, float]] = []
    for record, winner in valid:
        drafted, accepted, ratio = _acceptance(winner)
        if drafted > 0:
            acceptance_rows.append((record, drafted, accepted, ratio))
    if acceptance_rows:
        record, drafted, accepted, ratio = max(
            acceptance_rows, key=lambda row: (row[3], row[2], row[1])
        )
        cards.append(
            '<article class="leader-card">'
            '<span class="metric-token acceptance">MTP</span><div>'
            '<span class="leader-label">Highest winning acceptance</span>'
            f"<strong>{ratio * 100:.1f}% <small>{accepted}/{drafted}</small></strong>"
            f"<span>{escape(_model_display(record, public=public))}<br>{escape(_public_text(record.get('performance_target'), public=public, default='unknown'))} · {escape(_backend_display(record, public=public))}</span>"
            "</div></article>"
        )
    return '<div class="leader-grid">' + "".join(cards) + "</div>"


def _hardware_overview(records: Sequence[dict], *, public: bool = False) -> str:
    profiles: Dict[Tuple[Any, ...], Tuple[dict, int]] = {}
    for record in records:
        hardware = record.get("hardware")
        if not isinstance(hardware, dict):
            continue
        raw_gpus = hardware.get("gpus")
        gpus = (
            [gpu for gpu in raw_gpus if isinstance(gpu, dict)]
            if isinstance(raw_gpus, list)
            else []
        )
        signature = (
            _text(hardware.get("os")),
            _text(hardware.get("cpu")),
            _integer(hardware.get("physical_cores")),
            _integer(hardware.get("logical_cores")),
            round(_float(hardware.get("total_ram_gb") or hardware.get("ram_gb")), 2),
            tuple(
                (_text(gpu.get("name")), _integer(gpu.get("vram_mb"))) for gpu in gpus
            ),
        )
        existing = profiles.get(signature)
        profiles[signature] = (hardware, (existing[1] if existing else 0) + 1)
    if not profiles:
        return (
            '<section class="hardware-section" id="hardware"><div class="section-heading">'
            '<div><span class="section-kicker">Test platform</span><h2>Hardware snapshot</h2></div></div>'
            '<p class="muted">Older records did not include a hardware snapshot.</p></section>'
        )

    cards: List[str] = []
    for profile_index, (hardware, run_count) in enumerate(
        sorted(profiles.values(), key=lambda row: _text(row[0].get("cpu")).casefold()),
        start=1,
    ):
        raw_gpus = hardware.get("gpus")
        gpus = (
            [gpu for gpu in raw_gpus if isinstance(gpu, dict)]
            if isinstance(raw_gpus, list)
            else []
        )
        gpu_items = (
            "".join(
                '<li><span class="device-dot"></span><div>'
                f"<b>{escape(_public_text(gpu.get('name'), public=public, default='Unknown GPU'))}</b>"
                f"<span>{_integer(gpu.get('vram_mb')) / 1024:.1f} GiB VRAM</span>"
                "</div></li>"
                for gpu in gpus
            )
            or '<li><span class="device-dot cpu-dot"></span><div><b>CPU-only</b><span>No GPU snapshot stored</span></div></li>'
        )
        ram = _float(hardware.get("total_ram_gb") or hardware.get("ram_gb"))
        ram_html = (
            f'<div class="hardware-stat"><span>RAM</span><b>{ram:.1f} GiB</b></div>'
            if ram > 0.0
            else ""
        )
        cards.append(
            f'<article class="hardware-card" aria-labelledby="hardware-profile-{profile_index}">'
            '<div class="hardware-card-head">'
            f'<span class="hardware-index">{profile_index:02d}</span><div>'
            f'<h3 id="hardware-profile-{profile_index}">{escape(_public_text(hardware.get("os"), public=public, default="Unknown OS"))}</h3>'
            f"<p>{run_count} measured run(s) with this stored snapshot</p></div></div>"
            '<div class="hardware-cpu"><span>CPU</span>'
            f"<b>{escape(_public_text(hardware.get('cpu'), public=public, default='Unknown CPU'))}</b>"
            f"<small>{_integer(hardware.get('physical_cores'))} physical / {_integer(hardware.get('logical_cores'))} logical cores</small></div>"
            '<div class="hardware-stats">'
            f'<div class="hardware-stat"><span>GPU count</span><b>{len(gpus)}</b></div>{ram_html}'
            f'<div class="hardware-stat"><span>Recorded lanes</span><b>{run_count}</b></div></div>'
            f'<ul class="gpu-list">{gpu_items}</ul></article>'
        )
    return (
        '<section class="hardware-section" id="hardware"><div class="section-heading">'
        '<div><span class="section-kicker">Test platform</span><h2>Hardware snapshot</h2></div>'
        "<p>Static hardware captured with successful benchmark evidence.</p></div>"
        '<div class="hardware-grid">' + "".join(cards) + "</div></section>"
    )


def _has_chart_data(record: dict) -> bool:
    for candidate in _candidate_rows(record):
        if candidate.get("error"):
            continue
        if any(
            _float(candidate.get(field)) > 0.0
            for field in ("prompt_tps", "generation_tps", "overall_tps")
        ):
            return True
        if _acceptance(candidate)[0] > 0:
            return True
    return False


def _run_chart_card(record: dict, index: int, *, public: bool = False) -> str:
    model = _model_display(record, public=public)
    target = _public_text(
        record.get("performance_target"), public=public, default="unknown"
    )
    backend = _backend_display(record, public=public)
    drafter = (
        _display_text(record, record.get("drafter_label"), public=public)
        or "No drafter"
    )
    winner = _winner(record)
    winner_label = _display_text(
        record,
        (
            _text(winner.get("label"), _text(winner.get("id"), "—"))
            if isinstance(winner, dict)
            else "—"
        ),
        public=public,
    )
    title_id = f"run-chart-title-{index}"
    return (
        f'<article class="run-chart-card" aria-labelledby="{title_id}">'
        '<header class="run-chart-head"><div>'
        f'<h6 id="{title_id}"><a href="#run-{index}">{escape(model)}</a></h6>'
        f'<p>Winner: {escape(winner_label)}</p></div><div class="chips">'
        f"<span>{escape(target)}</span><span>{escape(backend)}</span><span>{escape(drafter)}</span>"
        "</div></header>"
        '<div class="chart-pair"><div><p class="chart-label">Candidate throughput</p>'
        + _throughput_chart(record, f"run-{index}", public=public)
        + '</div><div><p class="chart-label">Draft acceptance</p>'
        + _acceptance_chart(record, f"run-{index}", public=public)
        + "</div></div></article>"
    )


def _run_chart_groups(
    indexed_records: Sequence[Tuple[int, dict]],
    test_type: str,
    *,
    public: bool = False,
) -> str:
    chart_rows = [
        (index, record) for index, record in indexed_records if _has_chart_data(record)
    ]
    if not chart_rows:
        return (
            '<p class="muted">No successful candidate diagrams for this workload.</p>'
        )
    grouped: Dict[str, List[Tuple[int, dict]]] = {}
    display_names: Dict[str, str] = {}
    for index, record in chart_rows:
        key = _model_name(record).casefold()
        display_names.setdefault(key, _model_display(record, public=public))
        grouped.setdefault(key, []).append((index, record))
    groups: List[str] = []
    for group_index, key in enumerate(sorted(grouped), start=1):
        heading_id = f"run-chart-model-{test_type}-{group_index}"
        cards = "".join(
            _run_chart_card(record, index, public=public)
            for index, record in grouped[key]
        )
        groups.append(
            f'<section class="model-chart-group" aria-labelledby="{heading_id}">'
            f'<div class="model-group-heading"><h5 id="{heading_id}">{escape(display_names[key])}</h5>'
            f"<span>{len(grouped[key])} measured lane(s)</span></div>"
            f'<div class="run-chart-grid">{cards}</div></section>'
        )
    omitted = len(indexed_records) - len(chart_rows)
    omitted_note = (
        f'<p class="muted">{omitted} failed run(s) have no diagram and remain in the detailed evidence below.</p>'
        if omitted
        else ""
    )
    return omitted_note + "".join(groups)


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
    public: bool = False,
) -> str:
    """Render benchmark evidence as dependency-free, escaped HTML.

    ``public=True`` replaces machine-local paths and validates that none leaked
    into the final document before it can be published.
    """
    generated = generated_at or datetime.now(timezone.utc)
    records = _flatten_records(records_by_test)
    indexed_records = list(enumerate(records, start=1))
    visual_sections: List[str] = []
    detail_sections: List[str] = []
    for test_type in _TEST_ORDER:
        typed = [
            (index, record)
            for index, record in indexed_records
            if _text(record.get("benchmark_type")) == test_type
        ]
        raw_records = [record for _index, record in typed]
        visual_sections.append(
            f'<section class="visual-workload" id="charts-{test_type}" aria-labelledby="charts-{test_type}-title">'
            '<div class="workload-heading"><div>'
            f'<span class="section-kicker">{len(typed)} saved run(s)</span>'
            f'<h3 id="charts-{test_type}-title">{escape(_TEST_TITLES[test_type])}</h3>'
            '</div><a class="detail-jump" href="#details-'
            f'{test_type}">Jump to evidence ↓</a></div>'
            + (
                _winner_overview(raw_records, test_type, public=public)
                + _winner_acceptance_overview(raw_records, test_type, public=public)
                + '<div class="candidate-diagrams"><div class="subsection-heading">'
                '<div><span class="section-kicker">Every successful lane</span>'
                "<h4>Candidate comparison diagrams</h4></div>"
                "<p>Throughput and drafted-token acceptance stay together for each run.</p></div>"
                + _run_chart_groups(typed, test_type, public=public)
                + "</div>"
                if typed
                else '<p class="empty-state">No saved runs for this workload.</p>'
            )
            + "</section>"
        )
        details = "".join(
            _record_detail(record, index, public=public) for index, record in typed
        )
        detail_sections.append(
            f'<section class="detail-workload" id="details-{test_type}" aria-labelledby="details-{test_type}-title">'
            '<div class="workload-heading"><div>'
            f'<span class="section-kicker">{len(typed)} complete record(s)</span>'
            f'<h3 id="details-{test_type}-title">{escape(_TEST_TITLES[test_type])}</h3>'
            "</div></div>"
            + (details or '<p class="empty-state">No saved runs for this workload.</p>')
            + "</section>"
        )

    css = """
:root{color-scheme:dark;--bg:#0b1018;--panel:#151d29;--panel2:#1d2837;--panel3:#253448;--text:#f2f6fb;--muted:#a8b7c9;--line:#33445a;--accent:#66bbff;--good:#58d68d;--violet:#b794f6;--warn:#ffd166;--bad:#ff7f8a;--shadow:0 16px 45px #0006}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 15% -10%,#1d3958 0,transparent 32rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1,h2,h3,h4{line-height:1.2}h1{font-size:clamp(2rem,5vw,3.6rem);margin:.15em 0}h2{margin-top:46px;border-bottom:1px solid var(--line);padding-bottom:10px}h3 small{color:var(--muted);font-weight:500}a{color:var(--accent)}.muted{color:var(--muted)}.hero{padding:28px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,#1a2738dd,#111923dd);box-shadow:var(--shadow)}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.13em;text-transform:uppercase}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:22px}.kpi{padding:14px;border:1px solid var(--line);border-radius:12px;background:#0b111acc}.kpi b{display:block;font-size:1.6rem}.kpi span{color:var(--muted)}nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 24px}nav a{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:var(--panel);text-decoration:none}.callout,.overview-note{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);padding:12px 16px;border-radius:10px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px;box-shadow:0 8px 24px #0003}table{border-collapse:collapse;width:100%;min-width:980px;background:var(--panel)}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--panel3);z-index:1}tbody tr:hover{background:#ffffff08}tr.winner{background:color-mix(in srgb,var(--good) 13%,transparent)}tr.failed{background:color-mix(in srgb,var(--bad) 11%,transparent)}details.run{margin:14px 0;border:1px solid var(--line);border-radius:12px;background:var(--panel);box-shadow:0 7px 22px #0002}details.run>summary{cursor:pointer;padding:14px 16px;font-weight:700}details.run[open]>summary{border-bottom:1px solid var(--line);background:var(--panel2)}.run-body{padding:4px 16px 20px}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:8px;margin:12px 0 20px}.facts div{background:var(--panel2);padding:10px;border-radius:8px;display:flex;flex-direction:column}.facts .wide{grid-column:1/-1}.facts span{color:var(--muted);overflow-wrap:anywhere}.overview-scroll{overflow-x:auto;overflow-y:hidden;margin:12px 0 20px;padding:0 0 6px;scrollbar-gutter:stable}.overview-scroll:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.overview-grid{display:grid;grid-template-columns:1fr;gap:12px}.metric-panel{width:100%;min-width:0;margin:0;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}.metric-panel h3{position:sticky;left:12px;width:max-content;max-width:calc(100vw - 80px);margin:0 0 8px;padding:2px 8px;border-radius:6px;background:var(--panel);z-index:1}.overview-chart{padding:8px 4px 4px}.vertical-scroll{overflow-x:auto;overflow-y:hidden;padding:8px 4px 4px}.model-chart,.candidate-chart,.acceptance-chart{display:flex;align-items:flex-end;gap:12px;min-width:max-content}.model-column{width:126px;text-align:center}.candidate-column{width:220px;text-align:center}.acceptance-column{width:130px;text-align:center}.candidate-bars{display:flex;align-items:flex-end;justify-content:center;gap:10px}.mini-metric{display:flex;flex-direction:column;align-items:center;width:58px}.bar-stage{height:190px;width:32px;display:flex;align-items:flex-end;border-radius:7px 7px 3px 3px;background:linear-gradient(#ffffff0b,#ffffff03);border-bottom:2px solid var(--line);overflow:hidden}.overview-stage{height:220px;width:42px;margin:auto}.acceptance-stage{height:180px;width:46px;margin:auto}.vertical-bar{width:100%;min-height:2px;border-radius:6px 6px 2px 2px;box-shadow:inset 0 1px #fff5;transition:filter .15s}.vertical-bar:hover{filter:brightness(1.2)}.bar-number{display:block;color:var(--text);font-variant-numeric:tabular-nums;margin-bottom:4px}.bar-caption{color:var(--muted);font-size:.76rem;margin-top:5px}.column-label{margin-top:8px;overflow-wrap:anywhere}.column-label span{color:var(--muted);font-size:.82rem}.bar-pp{background:linear-gradient(#86ccff,#298ee3)}.bar-decode{background:linear-gradient(#ccb3ff,#805ad5)}.bar-overall{background:linear-gradient(#83e8aa,#2cab68)}.bar-accept{background:linear-gradient(#ffe49b,#e5a923)}.chart{width:100%;background:var(--panel2);border:1px solid var(--line);border-radius:10px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:1050px){.bar-stage{height:160px}.overview-stage{height:190px}}@media(max-width:640px){main{padding:14px}.hero{padding:18px}.candidate-column{width:190px}.mini-metric{width:50px}.bar-stage{height:140px}}@media print{body{background:#fff;color:#111}.hero,.run,.table-wrap,.metric-panel,.chart{break-inside:avoid;box-shadow:none}.muted,.facts span,.column-label span{color:#444}.vertical-scroll,.overview-scroll{overflow:visible}.bar-pp,.bar-decode,.bar-overall,.bar-accept{print-color-adjust:exact}}
"""
    css += """
:root{--panel-glass:#111a26e8;--cyan:#5bd6ff;--blue:#4b8dff;--green:#55d990;--amber:#ffd166;--purple:#b794f6;--radius:18px}
body{background:radial-gradient(circle at 12% -8%,#214f7566 0,transparent 30rem),radial-gradient(circle at 92% 12%,#4d2f7560 0,transparent 28rem),linear-gradient(180deg,#0a1018,#090d14 55%,#0b1119);background-attachment:fixed}
main{max-width:1720px;padding:32px clamp(18px,3vw,48px) 64px}section{scroll-margin-top:92px}h1,h2,h3,h4,h5,h6{line-height:1.14;text-wrap:balance}h2{font-size:clamp(1.6rem,3vw,2.4rem)}h3{font-size:clamp(1.35rem,2.2vw,1.85rem)}h4{font-size:1.15rem}h5{font-size:1.05rem}h6{font-size:1rem}a:focus-visible,summary:focus-visible,[tabindex="0"]:focus-visible{outline:3px solid var(--cyan);outline-offset:3px}.skip-link{position:fixed;z-index:100;left:16px;top:12px;transform:translateY(-160%);padding:9px 14px;border-radius:8px;background:#fff;color:#07101a;font-weight:800}.skip-link:focus{transform:none}.hero{position:relative;overflow:hidden;padding:clamp(24px,4vw,50px);border-color:#507095;background:linear-gradient(135deg,#172a40f2,#111a28f4 58%,#251c3cf0);isolation:isolate}.hero:after{content:"";position:absolute;z-index:-1;right:-8rem;top:-10rem;width:31rem;height:31rem;border-radius:50%;background:conic-gradient(from 40deg,#5bd6ff33,#b794f644,#55d99022,#5bd6ff33);filter:blur(8px)}.hero-top{display:flex;justify-content:space-between;align-items:center;gap:16px}.report-badge{display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border:1px solid #5bd6ff55;border-radius:999px;background:#07121dbb;color:#bceeff;font-size:.8rem;font-weight:750}.report-badge:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green)}.hero h1{max-width:900px;letter-spacing:-.035em}.hero-copy{max-width:850px;color:#c0ccdb;font-size:1rem}.kpis{grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px}.kpi{position:relative;overflow:hidden;padding:16px;background:linear-gradient(150deg,#0b141fcc,#111c2acc);border-color:#40566f}.kpi:after{content:"";position:absolute;inset:auto 0 0;height:3px;background:linear-gradient(90deg,var(--cyan),var(--purple))}.kpi b{font-variant-numeric:tabular-nums}.report-nav{position:sticky;z-index:20;top:10px;width:max-content;max-width:100%;margin:18px auto 28px;padding:7px;overflow-x:auto;flex-wrap:nowrap;border:1px solid #3d5068;border-radius:999px;background:#0a111be8;box-shadow:0 12px 32px #0008;backdrop-filter:blur(16px);scrollbar-width:none}.report-nav::-webkit-scrollbar{display:none}.report-nav a{flex:0 0 auto;border-color:transparent;background:transparent;color:#c6d3e2;font-weight:650}.report-nav a:hover{border-color:#4f6883;background:#182536;color:#fff}.section-heading,.subsection-heading,.workload-heading,.model-group-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:18px}.section-heading{margin:48px 0 18px}.section-heading h2,.workload-heading h3,.subsection-heading h4,.model-group-heading h5{margin:3px 0 0}.section-heading>p,.subsection-heading>p{max-width:620px;margin:0;color:var(--muted);text-align:right}.section-kicker{color:var(--cyan);font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.hardware-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:16px}.hardware-card{padding:20px;border:1px solid #3a506a;border-radius:var(--radius);background:linear-gradient(150deg,#152233,#101923);box-shadow:0 14px 34px #0004}.hardware-card-head{display:flex;align-items:center;gap:14px}.hardware-index{display:grid;place-items:center;width:48px;height:48px;border:1px solid #5bd6ff66;border-radius:14px;background:#07131f;color:var(--cyan);font-weight:850}.hardware-card h3{margin:0;font-size:1.15rem}.hardware-card p{margin:3px 0 0;color:var(--muted)}.hardware-cpu{display:flex;flex-direction:column;gap:3px;margin:18px 0;padding:14px;border-radius:12px;background:#0a121d}.hardware-cpu>span,.hardware-stat span{color:var(--muted);font-size:.72rem;font-weight:750;letter-spacing:.09em;text-transform:uppercase}.hardware-cpu small{color:var(--muted)}.hardware-stats{display:flex;flex-wrap:wrap;gap:8px}.hardware-stat{min-width:105px;flex:1;padding:10px 12px;border:1px solid #31445b;border-radius:10px;background:#192535}.hardware-stat b{display:block;margin-top:2px;font-size:1.12rem}.gpu-list{display:grid;gap:8px;margin:16px 0 0;padding:0;list-style:none}.gpu-list li{display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:10px;background:#0a121d}.gpu-list li div{display:flex;flex-direction:column}.gpu-list li span:last-child{color:var(--muted);font-size:.82rem}.device-dot{width:10px;height:10px;flex:0 0 auto;border-radius:50%;background:var(--purple);box-shadow:0 0 14px #b794f699}.cpu-dot{background:var(--cyan)}.visual-dashboard{margin-top:48px}.dashboard-head{margin-bottom:18px}.leader-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:18px 0 28px}.leader-card{display:flex;align-items:center;gap:14px;min-width:0;padding:15px;border:1px solid #354a62;border-radius:14px;background:linear-gradient(145deg,#142132,#101923)}.metric-token{display:grid;place-items:center;width:54px;height:54px;flex:0 0 auto;border-radius:16px;background:#142d42;color:#aee8ff;font-size:.76rem;font-weight:900;letter-spacing:.04em}.metric-token.decode{background:#2d2348;color:#ddccff}.metric-token.overall{background:#17372a;color:#a9efc2}.metric-token.context{background:#3a2c16;color:#ffe3a0}.metric-token.acceptance{background:#3e3118;color:#ffe7a6}.leader-card>div{display:flex;min-width:0;flex-direction:column}.leader-label{color:var(--muted);font-size:.78rem;font-weight:750;text-transform:uppercase}.leader-card strong{font-size:1.45rem;font-variant-numeric:tabular-nums}.leader-card strong small{font-size:.72rem;color:var(--muted)}.leader-card div>span:last-child{overflow:hidden;color:#c5d0dc;font-size:.82rem;text-overflow:ellipsis}.visual-workload{margin:22px 0 34px;padding:clamp(16px,2.5vw,28px);border:1px solid #354c65;border-radius:22px;background:linear-gradient(160deg,#111c29e8,#0d151fe8);box-shadow:0 18px 46px #0004}.workload-heading{align-items:center;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid var(--line)}.detail-jump{flex:0 0 auto;padding:8px 12px;border:1px solid #46617e;border-radius:999px;text-decoration:none}.overview-note{margin-top:14px;background:#101b28}.metric-panel{border-color:#354a62;background:linear-gradient(145deg,#152231,#111a25)}.metric-panel h4{position:sticky;left:12px;width:max-content;max-width:calc(100vw - 80px);margin:0 0 8px;padding:3px 9px;border-radius:7px;background:#152231;z-index:1}.metric-panel h4 small,.acceptance-overview-panel h4 small{color:var(--muted);font-weight:500}.acceptance-overview-panel{margin:18px 0;padding:16px;border:1px solid #4f452c;border-radius:14px;background:linear-gradient(145deg,#211d15,#171711)}.acceptance-overview-panel h4{margin:0}.acceptance-overview-panel p{margin:5px 0 12px}.acceptance-overview-scroll{overflow-x:auto;padding:6px 2px;scrollbar-gutter:stable}.candidate-diagrams{margin-top:24px;padding-top:18px;border-top:1px solid var(--line)}.model-chart-group{margin-top:22px}.model-group-heading{align-items:center;margin-bottom:9px}.model-group-heading span{color:var(--muted)}.run-chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,650px),1fr));gap:14px}.run-chart-card{min-width:0;padding:15px;border:1px solid #344a62;border-radius:15px;background:#0d1621;content-visibility:auto;contain-intrinsic-size:auto 640px}.run-chart-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px}.run-chart-head h6{margin:0}.run-chart-head p{margin:4px 0 0;color:var(--muted);font-size:.82rem}.chips{display:flex;max-width:55%;flex-wrap:wrap;justify-content:flex-end;gap:5px}.chips span,.status-pill{padding:4px 8px;border:1px solid #40566e;border-radius:999px;background:#172333;color:#c7d5e4;font-size:.72rem}.chart-pair{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,.8fr);gap:10px}.chart-pair>div{min-width:0}.chart-label{margin:0 0 6px;color:#dbe6f1;font-size:.78rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase}.chart{height:calc(100% - 25px)}.empty-state{padding:24px;border:1px dashed #425873;border-radius:14px;color:var(--muted);text-align:center}.results-section{margin-top:56px}.method-callout{margin:16px 0 22px}.table-wrap{border-color:#3b5068;border-radius:14px}.table-wrap table{font-variant-numeric:tabular-nums}.detail-workload{margin-top:34px}.detail-workload>.workload-heading{position:sticky;z-index:4;top:75px;padding:12px 14px;border:1px solid #354a61;border-radius:12px;background:#0d1620ed;backdrop-filter:blur(12px)}details.run{content-visibility:auto;contain-intrinsic-size:auto 420px;border-color:#344a60;border-radius:14px;background:#111a25}details.run>summary{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:15px 17px}.summary-copy{min-width:0;overflow-wrap:anywhere}.measured-pill{border-color:#39765a;background:#133326;color:#a7eac1}.failed-pill{border-color:#81454d;background:#3a1b20;color:#ffc0c5}.run-body{padding:8px 17px 22px}.facts div{border:1px solid #304257;background:#172331}.callout{background:#111b28}.privacy-note{display:inline-flex;margin-top:10px;padding:6px 10px;border:1px solid #37705b;border-radius:999px;background:#102b21;color:#a6e9c1;font-size:.78rem}.report-footer{margin-top:52px;padding:22px 0;border-top:1px solid var(--line);color:var(--muted);text-align:center}.report-footer a{font-weight:700}.sr-only{clip-path:inset(50%)}
@media(max-width:1000px){.chart-pair{grid-template-columns:1fr}.section-heading,.subsection-heading{align-items:flex-start;flex-direction:column}.section-heading>p,.subsection-heading>p{text-align:left}.detail-workload>.workload-heading{top:72px}}
@media(max-width:700px){main{padding:14px 12px 42px}.hero{padding:22px 18px}.hero-top,.workload-heading,.run-chart-head{align-items:flex-start;flex-direction:column}.report-nav{top:6px;margin:12px 0 22px;border-radius:14px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.kpi{padding:12px}.chips{max-width:100%;justify-content:flex-start}.visual-workload{padding:14px;border-radius:16px}.run-chart-grid{grid-template-columns:1fr}.detail-jump{align-self:flex-start}.detail-workload>.workload-heading{position:static}.status-pill{align-self:flex-start}details.run>summary{align-items:flex-start;flex-direction:column}.hardware-grid{grid-template-columns:1fr}}
@media(max-width:420px){.kpis{grid-template-columns:1fr}.leader-grid{grid-template-columns:1fr}.model-column{width:116px}.candidate-column{width:182px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.vertical-bar{transition:none}}
@media print{.skip-link,.report-nav,.detail-jump{display:none}.visual-workload,.hardware-card,.leader-card,.run-chart-card{break-inside:avoid;box-shadow:none}.detail-workload>.workload-heading{position:static}.run-chart-card{content-visibility:visible}.report-footer{color:#333}}
"""
    generated_text = generated.astimezone(timezone.utc).isoformat(timespec="seconds")
    model_count = len({_model_name(record).casefold() for record in records})
    backend_labels = {
        _backend_display(record, public=public)
        for record in records
        if _backend_display(record, public=public) != "Unknown"
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
    report_badge = "Public benchmark snapshot" if public else "Local benchmark evidence"
    hero_copy = (
        "Measured llama.cpp performance across models, modes, runtimes, and speculative heads on the hardware shown below."
        if public
        else "Measured llama.cpp performance across every saved model, mode, runtime, and speculative head."
    )
    privacy_note = (
        '<span class="privacy-note">Public-safe export · local filesystem paths omitted</span>'
        if public
        else ""
    )
    footer_source = (
        ' · <a href="https://github.com/DaWasteh/Auto-Tuner">AutoTuner source</a>'
        if public
        else ""
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="description" content="Measured AutoTuner llama.cpp benchmark results across models and runtimes on real hardware.">'
        '<meta name="theme-color" content="#0b1018">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'\">"
        "<title>AutoTuner benchmark dashboard</title>"
        f"<style>{css}</style></head><body>"
        '<a class="skip-link" href="#content">Skip to benchmark content</a>'
        '<main id="content"><header class="hero">'
        '<div class="hero-top"><div class="eyebrow">Measured llama.cpp evidence</div>'
        f'<span class="report-badge">{report_badge}</span></div>'
        "<h1>AutoTuner benchmark dashboard</h1>"
        f'<p class="hero-copy">{hero_copy}</p>'
        f'<p class="muted">Generated {escape(generated_text)} · AutoTuner v{escape(VERSION)} · dependency-free HTML</p>'
        + privacy_note
        + '<div class="kpis">'
        f'<div class="kpi"><b>{model_count}</b><span>models</span></div>'
        f'<div class="kpi"><b>{len(records)}</b><span>saved runs</span></div>'
        f'<div class="kpi"><b>{len(candidates)}</b><span>candidate runs</span></div>'
        f'<div class="kpi"><b>{failed_candidates}</b><span>failed candidates retained</span></div>'
        f'<div class="kpi"><b>{len(backend_labels)}</b><span>backend/build lanes</span></div>'
        "</div></header>"
        '<nav class="report-nav" aria-label="Report sections">'
        '<a href="#hardware">Hardware</a><a href="#visual-dashboard">Highlights</a>'
        '<a href="#charts-fast">Quick charts</a><a href="#charts-quick">Standard charts</a>'
        '<a href="#winner-overview">Winner table</a><a href="#run-details">Run details</a></nav>'
        + _hardware_overview(records, public=public)
        + '<section class="visual-dashboard" id="visual-dashboard" aria-labelledby="visual-dashboard-title">'
        '<div class="section-heading dashboard-head"><div><span class="section-kicker">Charts first</span>'
        '<h2 id="visual-dashboard-title">Visual performance overview</h2></div>'
        "<p>Fastest per-model lanes first, followed by every successful candidate comparison. Complete text and tables stay below.</p></div>"
        + _leader_cards(records, public=public)
        + "".join(visual_sections)
        + "</section>"
        '<section class="results-section" id="run-details" aria-labelledby="run-details-title">'
        '<div class="section-heading"><div><span class="section-kicker">Tables &amp; expandable evidence</span>'
        '<h2 id="run-details-title">Results and run details</h2></div>'
        "<p>Use the compact table for scanning, then open only the records you need.</p></div>"
        '<p class="callout method-callout"><b>How to read this report:</b> vertical bars place models and candidates side by side. Compare heights only within one metric panel. Each saved model/performance-mode/backend/drafter run expands below into every measured candidate, including failed settings. ★ marks the applied winner. Prompt processing, decode, end-to-end throughput, and drafted-token acceptance use native llama.cpp response timings/counters. Legacy 25% records are classified as Custom.</p>'
        '<h3 id="winner-overview">Winner overview table</h3>'
        + _summary_table(records, public=public)
        + "".join(detail_sections)
        + "</section>"
        f'<footer class="report-footer">Generated by AutoTuner v{escape(VERSION)}{footer_source}</footer>'
        "</main></body></html>"
    )
    if public:
        validate_public_report_html(html)
    return html


class _PublicReportPathScanner(HTMLParser):
    """Inspect rendered text/attributes without mistaking closing tags for paths."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.path_found = False

    def _scan(self, value: Optional[str]) -> None:
        if not value or self.path_found:
            return
        self.path_found = any(
            pattern.search(value) for pattern in _PUBLIC_PATH_PATTERNS
        )

    def handle_data(self, data: str) -> None:
        self._scan(data)

    def handle_comment(self, data: str) -> None:
        self._scan(data)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        for _name, value in attrs:
            self._scan(value)

    def handle_startendtag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        self.handle_starttag(tag, attrs)


def validate_public_report_html(html: str) -> None:
    """Fail closed if a publishable report contains active code or local paths."""
    lowered = html.casefold()
    if "<script" in lowered:
        raise ValueError("public benchmark report must not contain scripts")
    scanner = _PublicReportPathScanner()
    scanner.feed(html)
    scanner.close()
    if scanner.path_found:
        raise ValueError("public benchmark report contains a machine-local path")


def write_public_performance_report(
    records_by_test: Dict[str, List[dict]],
    destination: Path,
    *,
    generated_at: Optional[datetime] = None,
) -> Path:
    """Atomically write one path-redacted report intended for static hosting."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = build_performance_report_html(
        records_by_test,
        generated_at=generated_at or datetime.now(timezone.utc),
        public=True,
    )
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8", newline="\n")
    os.replace(tmp, destination)
    return destination


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
