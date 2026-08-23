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
    shortlist_candidates_from_record,
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


def test_standard_measurement_uses_twelve_and_a_half_percent() -> None:
    runner = _runner()
    assert runner._target_prompt_tokens(110592) == 13824
    payload = runner._measurement_payload(110592)
    assert payload["n_predict"] == 256
    assert len(payload["prompt"]) > 20_000
    assert payload["cache_prompt"] is False


def test_legacy_quick_measurement_still_accepts_twelve_percent() -> None:
    runner = _runner(BenchmarkLimits(prompt_context_fraction=0.12))
    assert runner._target_prompt_tokens(110592) == 13271
    payload = runner._measurement_payload(110592)
    assert payload["n_predict"] == 256
    assert len(payload["prompt"]) < len(_runner()._measurement_payload(110592)["prompt"])


def test_custom_context_fraction_ignores_65k_cap_and_4k_floor() -> None:
    full = _runner(
        BenchmarkLimits(
            prompt_context_fraction=1.0,
            min_prompt_tokens=1,
            max_prompt_tokens=None,
        )
    )
    assert full._target_prompt_tokens(110592) == 110208
    assert full._target_prompt_tokens(110592) > 65536

    tiny = _runner(
        BenchmarkLimits(
            prompt_context_fraction=0.0001,
            min_prompt_tokens=1,
            max_prompt_tokens=None,
        )
    )
    assert tiny._target_prompt_tokens(110592) == 11
    assert len(tiny._measurement_payload(110592)["prompt"]) < 1000


def test_prompt_is_calibrated_to_real_tokenizer_cap(monkeypatch) -> None:
    runner = _runner(
        BenchmarkLimits(
            prompt_context_fraction=1.0,
            min_prompt_tokens=1,
            max_prompt_tokens=1000,
        )
    )
    tokenize_calls = 0

    def fake_request(_port, _method, path, payload=None, **_kwargs):
        nonlocal tokenize_calls
        assert path == "/tokenize"
        tokenize_calls += 1
        # Deliberately model a tokenizer whose output differs from the text
        # builder's estimate; calibration must still stay under the real cap.
        count = max(1, len(str(payload["content"]).split()))
        return 200, {"tokens": list(range(count))}

    monkeypatch.setattr(runner, "_request_json", fake_request)
    payload = runner._measurement_payload(32768, port=12345)
    actual = len(payload["prompt"].split())
    assert 980 <= actual <= 1000
    assert tokenize_calls >= 2
    # Every candidate in the same runner gets byte-identical cached text and
    # does not repeat the potentially large /tokenize response.
    again = runner._measurement_payload(32768, port=23456)
    assert again["prompt"] == payload["prompt"]
    assert tokenize_calls >= 2
    cached_calls = tokenize_calls
    runner._measurement_payload(32768, port=34567)
    assert tokenize_calls == cached_calls

    # Repeated samples use distinct deterministic corpora so persistent n-gram
    # state cannot make sample 2 an artificial exact-prompt cache hit.
    variant = runner._measurement_payload(32768, port=45678, variant=1)
    assert variant["prompt"] != payload["prompt"]
    assert tokenize_calls > cached_calls
    variant_calls = tokenize_calls
    assert (
        runner._measurement_payload(32768, port=56789, variant=1)["prompt"]
        == variant["prompt"]
    )
    assert tokenize_calls == variant_calls


def test_parse_timing_payload_supports_current_llama_fields() -> None:
    sample = parse_timing_payload(
        {
            "timings": {
                "prompt_n": 1024,
                "prompt_ms": 512.0,
                "predicted_n": 64,
                "predicted_ms": 1280.0,
                "draft_n": 80,
                "draft_n_accepted": 52,
            }
        },
        1.8,
    )
    assert sample.prompt_tps == pytest.approx(2000.0)
    assert sample.generation_tps == pytest.approx(50.0)
    assert sample.draft_tokens == 80
    assert sample.draft_tokens_accepted == 52
    assert sample.draft_acceptance == pytest.approx(0.65)

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


def test_short_pass_shortlist_requires_stable_samples_and_keeps_coverage() -> None:
    seed = baseline_candidate(_config())

    def candidate(
        identifier: str, threads: int, batch: int, ubatch: int, score: float
    ) -> dict:
        return {
            "id": identifier,
            "label": identifier,
            "settings": {
                "threads": threads,
                "batch_threads": threads,
                "batch": batch,
                "ubatch": ubatch,
                "draft_n_max": 0,
            },
            "overall_tps": score,
            "samples": [
                {"prompt_tps": score * 2.0, "generation_tps": score},
                {"prompt_tps": score * 2.02, "generation_tps": score * 1.01},
            ],
            "error": "",
        }

    record = {
        "candidates": [
            candidate("baseline", 6, 512, 256, 100.0),
            candidate("winner", 8, 1024, 512, 140.0),
            candidate("runner-up", 8, 2048, 512, 135.0),
            candidate("thread-representative", 4, 1024, 512, 120.0),
            candidate("batch-representative", 8, 512, 512, 118.0),
        ]
    }
    shortlist = shortlist_candidates_from_record(record, seed)
    assert shortlist
    assert any(item.threads == 4 for item in shortlist)
    assert len({(item.batch, item.ubatch) for item in shortlist}) >= 2
    assert len({item.threads for item in shortlist}) >= 2

    noisy = __import__("copy").deepcopy(record)
    for item in noisy["candidates"]:
        item["samples"][1]["generation_tps"] *= 0.5
    assert shortlist_candidates_from_record(noisy, seed) == []


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


def test_speculative_runs_allow_expected_cross_prompt_acceptance_variance() -> None:
    regular = _runner(BenchmarkLimits(max_sample_spread=0.35))
    speculative = BenchmarkRunner(
        model=SimpleNamespace(path=Path("model.gguf")),
        profile=SimpleNamespace(draft_max=2),
        base_config=_config(),
        runtime_binary="llama-server",
        physical_cores=8,
        logical_cores=16,
        enable_speculative=True,
        limits=BenchmarkLimits(max_sample_spread=0.35),
    )
    assert regular._sample_spread_limit() == pytest.approx(0.35)
    assert speculative._sample_spread_limit() == pytest.approx(0.65)


def test_draft_depth_sweep_continues_until_first_decode_regression(
    monkeypatch,
) -> None:
    limits = BenchmarkLimits(
        max_candidates=1,
        confirmation_runs=0,
        samples_per_candidate=2,
        total_timeout_s=60,
        max_draft_tokens=10,
    )
    runner = BenchmarkRunner(
        model=SimpleNamespace(path=Path("model.gguf")),
        profile=SimpleNamespace(draft_max=2),
        base_config=_config(),
        runtime_binary="llama-server",
        physical_cores=8,
        logical_cores=16,
        enable_speculative=True,
        tune_draft_n_max=True,
        limits=limits,
    )
    calls: list[int] = []

    def fake_benchmark(candidate: BenchmarkCandidate) -> CandidateResult:
        depth = candidate.draft_n_max
        calls.append(depth)
        decode = float(depth * 10 if depth <= 7 else 60 - (depth - 8) * 10)
        return _measured(candidate, 100.0, decode)

    monkeypatch.setattr(runner, "_benchmark_candidate", fake_benchmark)
    monkeypatch.setattr(
        "model_benchmark.probe_binary_build_number", lambda _binary: 10590
    )
    result = runner.run()

    # Baseline/profile depth 2 is reused; every increasing value is measured,
    # including the user-reported 5 -> 6 -> 7 progression. Depth 8 regresses,
    # so 9 and 10 must never launch.
    assert calls == [2, 1, 3, 4, 5, 6, 7, 8]
    assert 7 in [item.candidate.draft_n_max for item in result.candidates]
    assert 8 in [item.candidate.draft_n_max for item in result.candidates]
    assert 9 not in calls


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


def test_suite_stop_after_mode_checkpoints_current_job(monkeypatch) -> None:
    first = _runner()
    second = _runner()
    checkpointed: list[str] = []
    suite = BenchmarkSuiteRunner(
        [
            BenchmarkSuiteJob(
                "model-a::safe",
                "A [safe]",
                "safe",
                first,
                metadata={"model_key": "model-a"},
            ),
            BenchmarkSuiteJob(
                "model-a::balanced",
                "A [balanced]",
                "balanced",
                second,
                metadata={"model_key": "model-a"},
            ),
        ],
        checkpoint=lambda outcome: checkpointed.append(outcome.job.key),
    )

    def first_run():
        suite.stop_after_performance_mode()
        return SimpleNamespace(name="first")

    monkeypatch.setattr(first, "run", first_run)
    monkeypatch.setattr(second, "run", lambda: SimpleNamespace(name="second"))
    result = suite.run()
    assert [item.job.key for item in result.jobs] == ["model-a::safe"]
    assert checkpointed == ["model-a::safe"]
    assert result.stopped_early
    assert "performance mode" in result.stop_reason


def test_suite_stop_after_model_finishes_remaining_modes(monkeypatch) -> None:
    first = _runner()
    second = _runner()
    third = _runner()
    checkpointed: list[str] = []
    suite = BenchmarkSuiteRunner(
        [
            BenchmarkSuiteJob(
                "model-a::safe", "A [safe]", "safe", first,
                metadata={"model_key": "model-a"},
            ),
            BenchmarkSuiteJob(
                "model-a::throughput", "A [throughput]", "throughput", second,
                metadata={"model_key": "model-a"},
            ),
            BenchmarkSuiteJob(
                "model-b::safe", "B [safe]", "safe", third,
                metadata={"model_key": "model-b"},
            ),
        ],
        checkpoint=lambda outcome: checkpointed.append(outcome.job.key),
    )

    def first_run():
        suite.stop_after_model()
        return SimpleNamespace(name="first")

    monkeypatch.setattr(first, "run", first_run)
    monkeypatch.setattr(second, "run", lambda: SimpleNamespace(name="second"))
    monkeypatch.setattr(third, "run", lambda: SimpleNamespace(name="third"))
    result = suite.run()
    assert [item.job.key for item in result.jobs] == [
        "model-a::safe",
        "model-a::throughput",
    ]
    assert checkpointed == ["model-a::safe", "model-a::throughput"]
    assert result.stopped_early
    assert "completed model" in result.stop_reason


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


def test_explicit_data_dir_override_is_isolated_from_legacy_locations(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AUTOTUNER_DATA_DIR", str(tmp_path / "isolated"))
    monkeypatch.delenv("AUTOTUNER_MIGRATE_LEGACY", raising=False)
    assert app_settings._legacy_settings_candidates() == []


def test_source_and_frozen_settings_merge_into_shared_user_data(
    tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "home" / ".autotuner"
    source_settings = tmp_path / "source" / "autotuner_settings.json"
    frozen_settings = tmp_path / "dist" / "autotuner_settings.json"
    source_settings.parent.mkdir()
    frozen_settings.parent.mkdir()
    source_settings.write_text(
        '{"models_path": "source-models", "nested": {"source": 1, "same": "old"}}',
        encoding="utf-8",
    )
    frozen_settings.write_text(
        '{"fork_path": "frozen-fork", "nested": {"frozen": 2, "same": "new"}}',
        encoding="utf-8",
    )
    # Deterministically make the frozen file the newer conflict winner.
    source_stat = source_settings.stat()
    frozen_mtime = source_stat.st_mtime_ns + 10_000_000
    source_settings.touch()
    import os

    os.utime(frozen_settings, ns=(frozen_mtime, frozen_mtime))
    monkeypatch.setenv("AUTOTUNER_DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        app_settings,
        "_legacy_settings_candidates",
        lambda: [source_settings, frozen_settings],
    )

    target = app_settings._settings_file()
    assert target == data_dir / "autotuner_settings.json"
    merged = app_settings.load_settings()
    assert merged["models_path"] == "source-models"
    assert merged["fork_path"] == "frozen-fork"
    assert merged["nested"] == {"source": 1, "frozen": 2, "same": "new"}
    assert len(list((data_dir / "migrations").glob("*.json"))) == 2

    # Exact source signatures are imported once; later shared writes remain
    # authoritative instead of being overwritten on every startup.
    merged["nested"]["same"] = "shared"
    assert app_settings.save_settings(merged)
    assert app_settings.load_settings()["nested"]["same"] == "shared"


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


def test_profile_bank_keeps_auto_perform_and_custom_drafter_variants(
    tmp_path, monkeypatch
) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: settings_file)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"profile-bank-model")
    custom = {"mode": "manual", "values": {"ctx": 16384, "draft_n_max": 3}}
    q4_perform = {"mode": "auto", "values": {"ctx": 32768, "draft_n_max": 7}}
    q8_perform = {"mode": "auto", "values": {"ctx": 32768, "draft_n_max": 5}}

    bank = app_settings.get_setting_profile_bank("model", model, "balanced")
    assert bank["selected"] == "auto"
    assert not app_settings.has_setting_profile_snapshot(
        "model", model, "balanced", "perform", "external:q4"
    )
    assert app_settings.set_setting_profile_snapshot(
        "model",
        model,
        "balanced",
        "custom1",
        custom,
        "external:q4",
        select=True,
    )
    assert app_settings.rename_custom_setting_profile(
        "model", model, "balanced", "custom1", "  My Q4 experiment  "
    )
    bank = app_settings.get_setting_profile_bank("model", model, "balanced")
    assert bank["selected"] == "custom1"
    assert bank["names"]["custom1"] == "My Q4 experiment"
    assert app_settings.get_setting_profile_snapshot(
        "model", model, "balanced", "custom1", "external:q4"
    ) == custom
    assert app_settings.get_setting_profile_snapshot(
        "model", model, "balanced", "custom1", "external:q8"
    ) is None

    for drafter, snapshot in (("external:q4", q4_perform), ("external:q8", q8_perform)):
        record = {
            "performance_target": "balanced",
            "model_size": model.stat().st_size,
            "winner_id": drafter,
            "drafter_key": drafter,
        }
        assert app_settings.save_performance_tuning_result(
            "model", model, record, snapshot, "balanced", "quick", drafter
        )
    assert app_settings.get_setting_profile_snapshot(
        "model", model, "balanced", "perform", "external:q4"
    ) == q4_perform
    assert app_settings.get_setting_profile_snapshot(
        "model", model, "balanced", "perform", "external:q8"
    ) == q8_perform
    grouped = app_settings.list_performance_run_results()
    assert {item["drafter_key"] for item in grouped["quick"]} == {
        "external:q4",
        "external:q8",
    }
    # Benchmark writes never replace the independent user-owned profile.
    assert app_settings.get_setting_profile_snapshot(
        "model", model, "balanced", "custom1", "external:q4"
    ) == custom


def test_custom_profile_bank_export_import_round_trip(tmp_path, monkeypatch) -> None:
    active_settings = [tmp_path / "source-settings.json"]
    monkeypatch.setattr(app_settings, "_settings_file", lambda: active_settings[0])
    model = tmp_path / "portable.gguf"
    model.write_bytes(b"portable-profile-model")
    snapshot = {"mode": "manual", "values": {"ctx": 49152, "draft_n_max": 6}}
    drafter = "external:portable-dflash-q4.gguf|123"
    assert app_settings.set_setting_profile_snapshot(
        "portable", model, "throughput", "custom2", snapshot, drafter, select=True
    )
    assert app_settings.rename_custom_setting_profile(
        "portable", model, "throughput", "custom2", "DFlash Q4 lab"
    )

    bundle = tmp_path / "profiles.json"
    ok, message, count = app_settings.export_performance_profiles(bundle)
    assert ok, message
    assert count == 1
    assert __import__("json").loads(bundle.read_text(encoding="utf-8"))["schema"] == 2

    active_settings[0] = tmp_path / "imported-settings.json"
    ok, message, count = app_settings.import_performance_profiles(bundle, [model])
    assert ok, message
    assert count == 1
    restored = app_settings.get_setting_profile_bank(
        "portable", model, "throughput"
    )
    assert restored["names"]["custom2"] == "DFlash Q4 lab"
    assert restored["selected_by_drafter"][drafter] == "custom2"
    assert app_settings.get_setting_profile_snapshot(
        "portable", model, "throughput", "custom2", drafter
    ) == snapshot


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


def test_quick_and_legacy_normal_results_map_to_standard_and_custom(
    tmp_path, monkeypatch
) -> None:
    active_settings = [tmp_path / "settings.json"]
    monkeypatch.setattr(app_settings, "_settings_file", lambda: active_settings[0])
    model = tmp_path / "model.gguf"
    model.write_bytes(b"measured-model")
    snapshot = {"mode": "auto", "values": {"ctx": 32768}}
    normal = {
        "model_name": "model",
        "model_path": str(model),
        "model_size": model.stat().st_size,
        "performance_target": "balanced",
        "benchmark_type": "normal",
        "winner_id": "normal-winner",
    }
    quick = {
        **normal,
        "benchmark_type": "quick",
        "winner_id": "quick-winner",
    }

    assert app_settings.save_performance_tuning_result(
        "model", model, normal, snapshot, "balanced", "normal"
    )
    assert app_settings.save_performance_tuning_result(
        "model", model, quick, snapshot, "balanced", "quick"
    )

    assert app_settings.get_performance_tuning_result(
        model, "balanced", "normal"
    )["winner_id"] == "normal-winner"
    assert app_settings.get_performance_tuning_result(
        model, "balanced", "quick"
    )["winner_id"] == "quick-winner"
    assert app_settings.get_performance_tuning_result(
        model, "balanced"
    )["winner_id"] == "quick-winner"
    grouped = app_settings.list_performance_run_results()
    assert [item["winner_id"] for item in grouped["custom"]] == ["normal-winner"]
    assert [item["winner_id"] for item in grouped["quick"]] == ["quick-winner"]
    assert "normal" not in grouped

    bundle = tmp_path / "profiles.json"
    ok, message, count = app_settings.export_performance_profiles(bundle)
    assert ok, message
    assert count == 1
    active_settings[0] = tmp_path / "imported-settings.json"
    ok, message, count = app_settings.import_performance_profiles(bundle, [model])
    assert ok, message
    assert count == 1
    assert app_settings.get_performance_tuning_result(
        model, "balanced", "normal"
    )["winner_id"] == "normal-winner"
    assert app_settings.get_performance_tuning_result(
        model, "balanced", "quick"
    )["winner_id"] == "quick-winner"


def test_custom_performance_result_round_trips_without_65k_cap(
    tmp_path, monkeypatch
) -> None:
    active_settings = [tmp_path / "settings.json"]
    monkeypatch.setattr(app_settings, "_settings_file", lambda: active_settings[0])
    model = tmp_path / "model.gguf"
    model.write_bytes(b"custom-model")
    snapshot = {"mode": "auto", "values": {"ctx": 262144}}
    record = {
        "model_name": "model",
        "model_path": str(model),
        "model_size": model.stat().st_size,
        "performance_target": "throughput",
        "benchmark_type": "custom",
        "prompt_context_fraction": 0.875,
        "prompt_token_cap": None,
        "winner_id": "custom-winner",
    }
    assert app_settings.save_performance_tuning_result(
        "model", model, record, snapshot, "throughput", "custom"
    )
    grouped = app_settings.list_performance_run_results()
    assert grouped["custom"][0]["winner_id"] == "custom-winner"
    assert grouped["custom"][0]["prompt_token_cap"] is None

    bundle = tmp_path / "profiles.json"
    ok, message, count = app_settings.export_performance_profiles(bundle)
    assert ok, message
    assert count == 1
    active_settings[0] = tmp_path / "imported.json"
    ok, message, count = app_settings.import_performance_profiles(bundle, [model])
    assert ok, message
    assert count == 1
    restored = app_settings.get_performance_tuning_result(
        model, "throughput", "custom"
    )
    assert restored is not None
    assert restored["prompt_context_fraction"] == pytest.approx(0.875)
    assert restored["prompt_token_cap"] is None


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
