from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from performance_report import (
    build_performance_report_html,
    validate_public_report_html,
    write_performance_report,
    write_public_performance_report,
)


def _record() -> dict:
    return {
        "schema": 2,
        "model_name": "Unsafe <Model>",
        "model_path": "C:/models/<unsafe>.gguf",
        "model_size": 123,
        "model_mtime_ns": 456,
        "performance_target": "throughput",
        "benchmark_type": "quick",
        "drafter_key": "external:q4",
        "drafter_label": "DFlash2 <Q4>",
        "desired_context": 32768,
        "prompt_context_fraction": 0.125,
        "generated_token_target": 256,
        "elapsed_s": 42.0,
        "runtime_binary": "llama-server",
        "runtime_build": 10590,
        "winner_id": "winner",
        "reason": "measured winner",
        "search_strategy": "full-staged-search",
        "profile_confidence": "validated",
        "quality_frozen": {"cache_k": "q4_0", "cache_v": "q4_0", "ngl": 99},
        "hardware": {
            "os": "Windows 11",
            "cpu": "Example CPU",
            "physical_cores": 8,
            "logical_cores": 16,
            "total_ram_gb": 48.0,
            "gpus": [{"name": "Example GPU", "vram_mb": 16384}],
        },
        "candidates": [
            {
                "id": "winner",
                "label": "Winner <fast>",
                "settings": {
                    "threads": 8,
                    "batch_threads": 8,
                    "batch": 1024,
                    "ubatch": 512,
                    "draft_n_max": 7,
                },
                "prompt_tps": 250.0,
                "generation_tps": 55.0,
                "overall_tps": 180.0,
                "inference_s": 2.0,
                "draft_tokens": 100,
                "draft_tokens_accepted": 73,
                "draft_acceptance": 0.73,
                "samples": [
                    {
                        "prompt_tps": 248.0,
                        "generation_tps": 54.0,
                        "elapsed_s": 2.1,
                    },
                    {
                        "prompt_tps": 252.0,
                        "generation_tps": 56.0,
                        "elapsed_s": 1.9,
                    },
                ],
                "error": "",
            },
            {
                "id": "baseline",
                "label": "Auto baseline",
                "settings": {
                    "threads": 6,
                    "batch_threads": 6,
                    "batch": 512,
                    "ubatch": 256,
                    "draft_n_max": 2,
                },
                "prompt_tps": 180.0,
                "generation_tps": 40.0,
                "overall_tps": 130.0,
                "inference_s": 3.0,
                "draft_tokens": 80,
                "draft_tokens_accepted": 40,
                "draft_acceptance": 0.5,
                "samples": [
                    {
                        "prompt_tps": 179.0,
                        "generation_tps": 39.5,
                        "elapsed_s": 3.1,
                    },
                    {
                        "prompt_tps": 181.0,
                        "generation_tps": 40.5,
                        "elapsed_s": 2.9,
                    },
                ],
                "error": "",
            },
        ],
    }


def test_html_report_contains_every_candidate_metric_and_escapes_content() -> None:
    html = build_performance_report_html(
        {"fast": [], "quick": [_record()], "custom": []},
        generated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert "Content-Security-Policy" in html
    assert "Quick pass · ≤3.125% context" in html
    assert "Standard benchmark · 12.5% context" in html
    assert "Custom context benchmark" in html
    assert "Legacy Normal" not in html
    assert "Winner &lt;fast&gt;" in html
    assert "Unsafe &lt;Model&gt;" in html
    assert "Unsafe <Model>" not in html
    assert "threads 8 / batch threads 8" in html
    assert "draft n-max 7" in html
    assert "73.0% (73/100)" in html
    assert "Candidate throughput comparison" in html
    assert "Drafted-token acceptance" in html
    assert "https://" not in html
    assert "<script" not in html


def test_html_uses_vertical_side_by_side_model_bars_and_unique_chart_ids() -> None:
    first = _record()
    first["benchmark_backend"] = "vulkan"
    first["runtime_label"] = "Vulkan · b10690"
    second = copy.deepcopy(first)
    second["model_name"] = "Second model"
    second["model_path"] = "C:/models/second.gguf"
    second["benchmark_backend"] = "hip"
    second["runtime_label"] = "HIP · b10690"
    second["winner_id"] = "baseline"
    second["candidates"][1]["settings"]["batch"] = 2048
    second["candidates"].append(
        {
            "id": "failed",
            "label": "Too large",
            "settings": {
                "threads": 16,
                "batch_threads": 16,
                "batch": 4096,
                "ubatch": 2048,
                "draft_n_max": 7,
            },
            "error": "allocation <failed>",
        }
    )

    html = build_performance_report_html(
        {"fast": [], "quick": [first, second], "custom": []}
    )
    assert html.count('class="overview-scroll"') == 1
    assert html.count('class="model-chart"') == 3
    assert 'class="overview-grid" style="width:max(100%,' in html
    assert ".overview-grid{display:grid;grid-template-columns:1fr" in html
    assert "grid-template-columns:repeat(3" not in html
    assert "Scroll the stacked panels together" in html
    assert not re.search(
        r'class="metric-panel".*?class="vertical-scroll"', html, re.DOTALL
    )
    metric_panels = re.findall(
        r'<section class="metric-panel">(.*?)</section>', html, re.DOTALL
    )
    assert len(metric_panels) == 3
    column_orders = [
        re.findall(r'class="column-label"><b>(.*?)</b>', panel)
        for panel in metric_panels
    ]
    assert column_orders[0] == column_orders[1] == column_orders[2]
    assert set(column_orders[0]) == {"Unsafe &lt;Model&gt;", "Second model"}

    assert 'class="model-chart"' in html
    assert 'class="vertical-bar bar-pp"' in html
    assert 'class="vertical-bar bar-decode"' in html
    assert 'class="vertical-bar bar-overall"' in html
    assert 'style="height:' in html
    assert "display:flex;align-items:flex-end" in html
    assert "Unsafe &lt;Model&gt;" in html and "Second model" in html
    assert "Vulkan · b10690" in html and "HIP · b10690" in html
    assert "batch 4096 / ubatch 2048" in html
    assert "allocation &lt;failed&gt;" in html

    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    for references in re.findall(r'aria-labelledby="([^"]+)"', html):
        for reference in references.split():
            assert reference in ids


def test_html_places_all_diagrams_before_tables_and_expandable_details() -> None:
    html = build_performance_report_html(
        {"fast": [], "quick": [_record()], "custom": []}
    )
    charts_at = html.index('id="visual-dashboard"')
    run_chart_at = html.index('class="run-chart-card"')
    table_at = html.index('id="winner-overview"')
    details_at = html.index('<details class="run"')
    assert charts_at < run_chart_at < table_at < details_at
    assert 'id="hardware"' in html
    assert "Example CPU" in html
    assert "48.0 GiB" in html
    assert "Example GPU" in html
    assert 'class="chart-pair"' not in html[details_at:]


def test_overview_collapses_many_lanes_to_fastest_lane_per_model() -> None:
    first = _record()
    second = copy.deepcopy(first)
    second["performance_target"] = "balanced"
    second["runtime_label"] = "HIP · b10743"
    second["candidates"][0]["overall_tps"] = 220.0
    html = build_performance_report_html(
        {"fast": [], "quick": [first, second], "custom": []}
    )
    metric_panels = re.findall(
        r'<section class="metric-panel">(.*?)</section>', html, re.DOTALL
    )
    assert len(metric_panels) == 3
    for panel in metric_panels:
        assert panel.count("Unsafe &lt;Model&gt;") == 2  # label + title attribute
        assert "balanced" in panel


def test_public_report_redacts_paths_and_keeps_static_hosting_metadata(
    tmp_path,
) -> None:
    record = _record()
    record["model_name"] += " /root/private/model.gguf"
    record["performance_target"] = "balanced /opt/private/target.txt"
    record["runtime_binary"] = r"C:\\Users\\Example\\llama builds\\llama-server.exe"
    record["runtime_label"] = "HIP /var/private/runtime.txt"
    record["runtime_build"] = "/etc/private/build.txt"
    record["reason"] = r"Selected after L:\\private\\benchmarks\\winner.json"
    record["search_strategy"] = "staged /mnt/private/search.txt"
    record["profile_confidence"] = "validated /run/private/confidence.txt"
    record["drafter_label"] = r"\\server\share\draft.gguf"
    record["quality_frozen"]["cache_k"] = "/usr/local/private/cache.txt"
    record["candidates"][0]["label"] = "file:///Volumes/private/winner.json"
    record["candidates"][0]["settings"]["threads"] = "/dev/private/thread.txt"
    record["candidates"][1]["id"] = "/proc/private/candidate.txt"
    record["candidates"][1]["label"] = ""
    record["candidates"][1]["error"] = r"Failed under /home/example/private/run.log"
    record["hardware"]["os"] = "/boot/private/os.txt"
    record["hardware"]["cpu"] = "/srv/private/cpu.txt"
    record["hardware"]["gpus"][0]["name"] = "/tmp/private/gpu.txt"
    destination = tmp_path / "site" / "index.html"
    path = write_public_performance_report(
        {"fast": [], "quick": [record], "custom": []}, destination
    )
    html = path.read_text(encoding="utf-8")
    assert path == destination
    assert "Public benchmark snapshot" in html
    assert "Public-safe export" in html
    assert "https://github.com/DaWasteh/Auto-Tuner" in html
    assert "Model file" in html
    assert "&lt;unsafe&gt;.gguf" in html
    assert "llama-server.exe" in html
    assert "C:/models" not in html
    assert "C:\\Users" not in html
    assert "L:\\private" not in html
    assert "/home/example" not in html
    for local_prefix in (
        "/root/private",
        "/opt/private",
        "/var/private",
        "/etc/private",
        "/mnt/private",
        "/run/private",
        "/usr/local/private",
        "/dev/private",
        "/proc/private",
        "/boot/private",
        "/srv/private",
        "/tmp/private",
    ):
        assert local_prefix not in html
    assert "file:///" not in html
    assert r"\\server\share" not in html
    assert "[local path omitted]" in html
    assert "<script" not in html


@pytest.mark.parametrize(
    "local_path",
    [
        "/root/private/run.log",
        "/tmp/private/run.log",
        "/srv/private/run.log",
        "/Volumes/private/run.log",
        "file:///private/run.log",
        "file:/private/run.log",
        r"\\server\share\run.log",
        "//server/share/run.log",
        r"C:\\private\\run.log",
    ],
)
def test_public_validator_rejects_every_absolute_path_form(local_path: str) -> None:
    with pytest.raises(ValueError, match="machine-local path"):
        validate_public_report_html(
            f"<!doctype html><html><body>{local_path}</body></html>"
        )


def test_committed_pages_snapshot_and_workflow_stay_public_safe() -> None:
    root = Path(__file__).resolve().parent
    page = root / "benchmark-site" / "index.html"
    workflow = root / ".github" / "workflows" / "pages.yml"
    html = page.read_text(encoding="utf-8")
    validate_public_report_html(html)
    assert html.startswith("<!doctype html>")
    assert "Public benchmark snapshot" in html
    assert "Content-Security-Policy" in html
    assert "<script" not in html

    workflow_text = workflow.read_text(encoding="utf-8")
    assert "pages: write" in workflow_text
    assert "id-token: write" in workflow_text
    assert "contents: read" in workflow_text
    assert "contents: write" not in workflow_text
    assert "path: benchmark-site" in workflow_text
    assert "actions/deploy-pages@v4" in workflow_text
    build_job, deploy_job = workflow_text.split("\n  deploy:", maxsplit=1)
    assert "pages: write" not in build_job
    assert "id-token: write" not in build_job
    assert "from performance_report" in build_job
    assert "actions/checkout" not in deploy_job
    assert "from performance_report" not in deploy_job


def test_report_writer_uses_shared_autotuner_report_folder(
    tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / ".autotuner"
    monkeypatch.setattr(
        "performance_report.app_settings.app_data_dir", lambda: data_dir
    )
    path = write_performance_report({"fast": [], "quick": [_record()], "custom": []})
    assert path.parent == data_dir / "reports"
    assert path.name.startswith("performance-report-")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
    latest = path.parent / "performance-report-latest.html"
    assert latest.is_file()
    assert latest.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_report_accepts_legacy_custom_records_without_candidate_details() -> None:
    legacy = {
        "model_name": "Legacy",
        "performance_target": "balanced",
        "benchmark_type": "custom",
        "prompt_context_fraction": 0.25,
        "desired_context": 16384,
    }
    html = build_performance_report_html({"fast": [], "quick": [], "custom": [legacy]})
    assert "Legacy" in html
    assert "No candidate details were stored." in html
    assert "Legacy 25% records are classified as Custom" in html
