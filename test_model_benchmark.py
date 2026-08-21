from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import app_settings
from model_benchmark import (
    BenchmarkCancelled,
    BenchmarkCandidate,
    BenchmarkLimits,
    BenchmarkRunner,
    BenchmarkSample,
    CandidateResult,
    batch_candidates,
    baseline_candidate,
    parse_timing_payload,
    thread_candidates,
)
from tuner import TunedConfig


def _config() -> TunedConfig:
    return TunedConfig(
        ctx=32768,
        ngl=99,
        threads=6,
        batch_threads=6,
        batch=512,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
        sampling={},
    )


def _runner(limits: BenchmarkLimits | None = None) -> BenchmarkRunner:
    return BenchmarkRunner(
        model=SimpleNamespace(path=Path("model.gguf")),
        profile=SimpleNamespace(),
        base_config=_config(),
        runtime_binary="llama-server",
        physical_cores=8,
        logical_cores=16,
        limits=limits,
    )


def _measured(candidate: BenchmarkCandidate, prompt: float, generation: float):
    sample = BenchmarkSample(
        prompt_tps=prompt,
        generation_tps=generation,
        prompt_tokens=1024,
        generated_tokens=64,
        elapsed_s=1.0,
    )
    return CandidateResult(candidate=candidate, samples=[sample, sample])


def test_candidate_generation_is_deterministic_and_valid() -> None:
    base = baseline_candidate(_config())
    first = thread_candidates(base, 8, 16)
    second = thread_candidates(base, 8, 16)
    assert first == second
    assert [item.threads for item in first] == [8, 4, 16]

    batches = batch_candidates(first[0])
    assert batches
    assert len({(item.batch, item.ubatch) for item in batches}) == len(batches)
    assert all(item.ubatch <= item.batch for item in batches)


def test_parse_timing_payload_supports_current_llama_fields() -> None:
    sample = parse_timing_payload(
        {
            "timings": {
                "prompt_n": 1024,
                "prompt_ms": 512.0,
                "predicted_n": 64,
                "predicted_ms": 1280.0,
            }
        },
        1.8,
    )
    assert sample.prompt_tps == pytest.approx(2000.0)
    assert sample.generation_tps == pytest.approx(50.0)

    with pytest.raises(Exception, match="too early"):
        parse_timing_payload(
            {
                "timings": {
                    "prompt_n": 1024,
                    "prompt_per_second": 1000,
                    "predicted_n": 2,
                    "predicted_per_second": 50,
                }
            },
            1.0,
        )


def test_staged_search_is_bounded_and_combines_best_axes(monkeypatch) -> None:
    limits = BenchmarkLimits(
        max_candidates=8,
        confirmation_runs=2,
        samples_per_candidate=2,
        total_timeout_s=60,
    )
    runner = _runner(limits)
    calls: list[BenchmarkCandidate] = []

    def fake_benchmark(candidate: BenchmarkCandidate) -> CandidateResult:
        calls.append(candidate)
        generation = 120.0 if candidate.threads == 8 else 100.0
        prompt = 100.0
        if (candidate.batch, candidate.ubatch) == (1024, 512):
            prompt = 125.0
        elif candidate.batch >= 1024:
            prompt = 110.0
        return _measured(candidate, prompt, generation)

    monkeypatch.setattr(runner, "_benchmark_candidate", fake_benchmark)
    monkeypatch.setattr(
        "model_benchmark.probe_binary_build_number", lambda _binary: 10572
    )
    result = runner.run()

    assert len(calls) <= limits.max_candidates + limits.confirmation_runs
    assert calls[0].id == "baseline"
    assert result.winner.candidate.threads == 8
    assert (result.winner.candidate.batch, result.winner.candidate.ubatch) == (
        1024,
        512,
    )
    assert result.runtime_build == 10572
    assert result.score(result.winner) > 1.03


def test_search_keeps_baseline_for_immaterial_gain(monkeypatch) -> None:
    runner = _runner(BenchmarkLimits(max_candidates=4, confirmation_runs=2))

    def fake_benchmark(candidate: BenchmarkCandidate) -> CandidateResult:
        gain = 1.01 if candidate.id != "baseline" else 1.0
        return _measured(candidate, 100.0 * gain, 100.0 * gain)

    monkeypatch.setattr(runner, "_benchmark_candidate", fake_benchmark)
    monkeypatch.setattr(
        "model_benchmark.probe_binary_build_number", lambda _binary: None
    )
    result = runner.run()
    assert result.winner_id == "baseline"
    assert "3%" in result.reason


def test_cancel_before_start_never_launches_baseline() -> None:
    runner = _runner()
    runner.cancel()
    with pytest.raises(BenchmarkCancelled):
        runner.run()


def test_benchmark_snapshot_reapplies_measured_runtime_values() -> None:
    from qt_launcher import MainWindow, expert_cfg_from_values

    cfg = _config()
    cfg.threads = 8
    cfg.batch_threads = 8
    cfg.batch = 1024
    cfg.ubatch = 512
    snapshot = MainWindow._benchmark_snapshot(cfg)

    assert snapshot["source"] == "measured-performance-test"
    assert snapshot["pins"] == {"user_ctx": 32768, "force_n_parallel": 1}
    restored = expert_cfg_from_values(_config(), snapshot["values"])
    assert (restored.threads, restored.batch_threads) == (8, 8)
    assert (restored.batch, restored.ubatch) == (1024, 512)
    assert restored.n_parallel == 1


def test_low_vram_performance_target_survives_restart(tmp_path, monkeypatch) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: settings_file)
    app_settings.set_performance_target("low_vram")
    assert app_settings.get_performance_target() == "low_vram"


def test_path_specific_tuning_save_is_atomic_and_legacy_compatible(
    tmp_path, monkeypatch
) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: settings_file)
    model_a = tmp_path / "a" / "same.gguf"
    model_b = tmp_path / "b" / "same.gguf"
    snapshot_a = {"mode": "auto", "pins": {}, "values": {"ctx": 32768}}
    snapshot_b = {"mode": "auto", "pins": {}, "values": {"ctx": 65536}}

    app_settings.set_expert_override("same", snapshot_b)  # legacy fallback
    assert app_settings.save_performance_tuning_result(
        "same", model_a, {"winner_id": "baseline"}, snapshot_a
    )

    assert app_settings.get_expert_override("same", model_a) == snapshot_a
    assert app_settings.get_expert_override("same", model_b) == snapshot_b
    assert app_settings.get_performance_tuning_result(model_a) == {
        "winner_id": "baseline"
    }

    app_settings.clear_expert_override("same", model_a)
    assert app_settings.get_expert_override("same", model_a) is None
