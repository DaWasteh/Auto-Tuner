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
    BenchmarkSuiteJob,
    BenchmarkSuiteRunner,
    CandidateResult,
    batch_candidates,
    baseline_candidate,
    draft_candidates,
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
    base = baseline_candidate(_config(), effective_draft_n_max=2)
    first = thread_candidates(base, 8, 16)
    second = thread_candidates(base, 8, 16)
    assert first == second
    assert [item.threads for item in first] == [8, 4, 16]
    assert all(item.draft_n_max == 2 for item in first)

    batches = batch_candidates(first[0])
    assert batches
    assert len({(item.batch, item.ubatch) for item in batches}) == len(batches)
    assert all(item.ubatch <= item.batch for item in batches)

    drafts = draft_candidates(batches[0], maximum=4)
    assert [item.draft_n_max for item in drafts] == [1, 3, 4]
    assert all(item.batch == batches[0].batch for item in drafts)


def test_measurement_payload_is_long_and_decode_heavy_enough() -> None:
    runner = _runner()
    assert runner._target_prompt_tokens(110592) == 27648
    payload = runner._measurement_payload(110592)
    assert payload["n_predict"] == 256
    assert len(payload["prompt"]) > 20_000
    assert payload["cache_prompt"] is False


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


def test_suite_runs_jobs_sequentially_and_keeps_individual_failures(monkeypatch) -> None:
    first = _runner()
    second = _runner()
    first._total_runs = 3
    second._total_runs = 4
    order: list[str] = []

    def first_run():
        order.append("first")
        first.progress(3, 3, "done")
        return SimpleNamespace(name="first-result")

    def second_run():
        order.append("second")
        raise Exception("unexpected")

    monkeypatch.setattr(first, "run", first_run)
    # Suite intentionally catches BenchmarkFailure only; infrastructure bugs
    # must still surface rather than being mislabeled as a model failure.
    monkeypatch.setattr(second, "run", second_run)
    suite = BenchmarkSuiteRunner(
        [
            BenchmarkSuiteJob("first", "Model [safe]", "safe", first),
            BenchmarkSuiteJob("second", "Model [balanced]", "balanced", second),
        ]
    )
    with pytest.raises(Exception, match="unexpected"):
        suite.run()
    assert order == ["first", "second"]


def test_suite_continues_after_bounded_model_failure(monkeypatch) -> None:
    from model_benchmark import BenchmarkFailure

    first = _runner()
    second = _runner()
    monkeypatch.setattr(first, "run", lambda: (_ for _ in ()).throw(BenchmarkFailure("OOM")))
    monkeypatch.setattr(second, "run", lambda: SimpleNamespace(name="ok"))
    suite = BenchmarkSuiteRunner(
        [
            BenchmarkSuiteJob("first", "A [safe]", "safe", first),
            BenchmarkSuiteJob("second", "B [safe]", "safe", second),
        ]
    )
    result = suite.run()
    assert len(result.failed) == 1
    assert result.failed[0].error == "OOM"
    assert len(result.successful) == 1


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


def test_performance_profiles_are_isolated_per_target(tmp_path, monkeypatch) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: settings_file)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    legacy = {"mode": "auto", "values": {"ctx": 8192}}
    safe = {"mode": "auto", "values": {"ctx": 65536}}
    balanced = {"mode": "auto", "values": {"ctx": 32768}}

    app_settings.set_expert_override("model", legacy, model)
    assert app_settings.get_expert_override("model", model, "safe") == legacy

    app_settings.set_expert_override("model", safe, model, "safe")
    app_settings.set_expert_override("model", balanced, model, "balanced")
    assert app_settings.get_expert_override("model", model, "safe") == safe
    assert app_settings.get_expert_override("model", model, "balanced") == balanced
    assert app_settings.get_expert_override("model", model, "throughput") is None

    app_settings.clear_expert_override("model", model, "safe")
    assert app_settings.get_expert_override("model", model, "safe") is None
    assert app_settings.get_expert_override("model", model, "balanced") == balanced


def test_mode_scoped_measured_result_round_trip(tmp_path, monkeypatch) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: settings_file)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"measured-model")
    snapshot = {"mode": "auto", "values": {"ctx": 110592}}
    record = {
        "performance_target": "low_vram",
        "model_size": model.stat().st_size,
        "winner_id": "baseline",
    }
    assert app_settings.save_performance_tuning_result(
        "model", model, record, snapshot, "low_vram"
    )
    assert app_settings.get_performance_tuning_result(model, "low_vram") == record
    assert app_settings.get_performance_tuning_result(model, "safe") is None
    assert app_settings.get_expert_override("model", model, "low_vram") == snapshot


def test_performance_profile_export_import_maps_moved_model(
    tmp_path, monkeypatch
) -> None:
    active_settings = [tmp_path / "old-settings.json"]
    monkeypatch.setattr(
        app_settings, "_settings_file", lambda: active_settings[0]
    )
    old_model = tmp_path / "old" / "model.gguf"
    old_model.parent.mkdir()
    old_model.write_bytes(b"same-model-bytes")
    for target, context in (("safe", 65536), ("throughput", 32768)):
        snapshot = {"mode": "auto", "values": {"ctx": context}}
        record = {
            "performance_target": target,
            "model_size": old_model.stat().st_size,
            "winner_id": f"winner-{target}",
        }
        assert app_settings.save_performance_tuning_result(
            "model", old_model, record, snapshot, target
        )
    app_settings.set_model_performance_target(old_model, "throughput")

    bundle = tmp_path / "profiles.json"
    ok, message, count = app_settings.export_performance_profiles(bundle)
    assert ok, message
    assert count == 2

    active_settings[0] = tmp_path / "new-settings.json"
    moved_model = tmp_path / "new" / "model.gguf"
    moved_model.parent.mkdir()
    moved_model.write_bytes(old_model.read_bytes())
    ok, message, count = app_settings.import_performance_profiles(
        bundle, [moved_model]
    )
    assert ok, message
    assert count == 2
    assert app_settings.get_expert_override(
        "model", moved_model, "safe"
    )["values"]["ctx"] == 65536
    assert app_settings.get_expert_override(
        "model", moved_model, "throughput"
    )["values"]["ctx"] == 32768
    assert app_settings.get_performance_tuning_result(
        moved_model, "safe"
    )["winner_id"] == "winner-safe"
    assert app_settings.get_model_performance_target(moved_model) == "throughput"


def test_performance_profile_import_rejects_unrelated_json(
    tmp_path, monkeypatch
) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: settings_file)
    source = tmp_path / "not-profiles.json"
    source.write_text('{"performance_target": "safe"}', encoding="utf-8")
    ok, message, count = app_settings.import_performance_profiles(source, [])
    assert not ok
    assert "not an AutoTuner" in message
    assert count == 0
    assert not settings_file.exists()


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
