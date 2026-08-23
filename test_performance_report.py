from __future__ import annotations

from datetime import datetime, timezone

from performance_report import build_performance_report_html, write_performance_report


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


def test_report_writer_uses_shared_autotuner_report_folder(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / ".autotuner"
    monkeypatch.setattr("performance_report.app_settings.app_data_dir", lambda: data_dir)
    path = write_performance_report(
        {"fast": [], "quick": [_record()], "custom": []}
    )
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
    html = build_performance_report_html(
        {"fast": [], "quick": [], "custom": [legacy]}
    )
    assert "Legacy" in html
    assert "No candidate details were stored." in html
    assert "Legacy 25% records are classified as Custom" in html
