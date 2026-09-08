"""Pinned b10863/profile/memory-contract regressions; no local models needed."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_smoke import (
    _fake_dual_gpu_system_with_vk_order,
    _fake_model,
    _fake_model_md,
    _fake_system,
)

ROOT = Path(__file__).parent


def test_minicpm5_sizes_use_distinct_official_sampling(tmp_path):
    profiles = load_profiles(ROOT / "settings")
    for name in ("MiniCPM5-2B-F16", "MiniCPM5-2.6B-Q8_0", "MiniCPM5 2.6B"):
        p = match_profile(name, profiles, "llama")
        assert p.source_file == "minicpm5-2b.yaml"
        assert p.max_context == 131072
        assert not p.arch_fallback
        for mode in ("chat", "coding"):
            cfg = tuner.compute_config(
                _fake_model(tmp_path, name, 5),
                _fake_system(),
                p,
                user_ctx=8192,
                mode=mode,
            )
            assert cfg.sampling["temperature"] == 1.0
            assert cfg.sampling["top_p"] == 0.95
            assert cfg.sampling["min_p"] == 0
    old = match_profile("MiniCPM5-1B-Q8_0", profiles, "llama")
    assert old.source_file == "minicpm5.yaml"
    assert old.sampling["coding"]["temperature"] == 0.9
    assert (
        match_profile("Llama-3.1-8B", profiles, "llama").source_file
        != "minicpm5-2b.yaml"
    )


@pytest.mark.parametrize("layers,kv_heads,size", [(28, 2, "1.7B"), (36, 4, "4B")])
def test_spark_sizes_use_metadata_swa_kv(layers, kv_heads, size):
    p = match_profile(
        f"Spark-X2.5-{size}", load_profiles(ROOT / "settings"), "spark2_5"
    )
    assert p.source_file == "spark2_5.yaml"
    assert p.min_llama_build == 10828
    md = {
        "general.architecture": "spark2_5",
        "spark2_5.block_count": layers,
        "spark2_5.embedding_length": 2560 if layers == 36 else 2048,
        "spark2_5.attention.head_count": 16 if layers == 36 else 8,
        "spark2_5.attention.head_count_kv": kv_heads,
        "spark2_5.attention.key_length": 256,
        "spark2_5.attention.value_length": 256,
        "spark2_5.attention.sliding_window": 512,
        "spark2_5.attention.sliding_window_pattern": [True, True, True, False]
        * (layers // 4),
    }
    # Full-context KV only grows in one quarter of layers; bounded SWA is
    # reserved separately by the tuner. Head counts must not be fixed to 1.7B.
    expected = (layers // 4) * kv_heads * 256 * 2 * 2 / 1024**2
    assert tuner.kv_per_token_mb_from_metadata(md) == pytest.approx(expected)


def test_k2_horizon_is_fork_gated_and_not_kimi(tmp_path, monkeypatch):
    profiles = load_profiles(ROOT / "settings")
    for name in ("K2-Horizon-MoVA-36B-A4B-Q8_0", "unknown-merge"):
        p = match_profile(name, profiles, "k2-horizon")
        assert p.source_file == "k2-horizon.yaml"
        assert p.max_context == 524288
        assert p.required_runtime_markers == ["k2-horizon"]
        assert not p.min_llama_build  # no invented upstream cutoff
        monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *args: False)
        assert not tuner.check_profile_build(p, "stock-b10863")[0]
        monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *args: True)
        assert tuner.check_profile_build(p, "IFM-fork")[0]
    assert match_profile("Kimi-K2", profiles, "deepseek2").source_file == "kimi-k2.yaml"


def test_k2_mova_uses_whole_layers_not_ffn_only_estimate(tmp_path, monkeypatch):
    md = {
        "general.architecture": "k2-horizon",
        "k2-horizon.block_count": 48,
        "k2-horizon.context_length": 524288,
        "k2-horizon.embedding_length": 2560,
        "k2-horizon.attention.head_count": 32,
        "k2-horizon.attention.head_count_kv": 8,
        "k2-horizon.attention.key_length": 128,
        "k2-horizon.attention.value_length": 128,
        "k2-horizon.expert_count": 100,
        "k2-horizon.expert_used_count": 8,
        "k2-horizon.attention.value_expert_count": 64,
        "k2-horizon.attention.value_expert_used_count": 4,
    }
    model = _fake_model_md(tmp_path, "K2-Horizon-MoVA-36B-A4B-Q8_0", 38, md)
    p = match_profile(model.name, load_profiles(ROOT / "settings"), model.architecture)

    def reject_ffn_estimate(*args, **kwargs):
        raise AssertionError("MoVA weights cannot use the 8%-shared FFN-only estimate")

    monkeypatch.setattr(tuner, "_decide_moe_offload", reject_ffn_estimate)
    cfg = tuner.compute_config(
        model, _fake_system(vram_total=16, vram_free=15), p, user_ctx=8192
    )
    assert cfg.is_moe and cfg.n_cpu_moe is None
    assert 0 < cfg.ngl < 48
    assert cfg.estimated_model_ram_gb > 0
    # Routing combines V before caching: KV does not grow by 4 or 64 experts.
    assert tuner.kv_per_token_mb_from_metadata(md) == pytest.approx(
        48 * 8 * 128 * 4 / 1024**2
    )
    assert "--n-cpu-moe" not in tuner.build_command(model, cfg, p)


def _lazy_model_config(tmp_path):
    model = _fake_model_md(
        tmp_path,
        "Qwen3.8-Flash-Next",
        50,
        {
            "general.architecture": "qwen4exp",
            "__read_lazy_tensor_bytes__": 28_800_138_240,
        },
    )
    p = match_profile(model.name, load_profiles(ROOT / "settings"), model.architecture)
    cfg = tuner.TunedConfig(
        ctx=4096,
        ngl=10,
        threads=4,
        batch_threads=4,
        batch=128,
        ubatch=64,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
        sampling={
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.95,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
        },
    )
    return model, cfg, p


@pytest.mark.parametrize("mode", ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"])
def test_lazy_ple_is_independent_of_ordinary_load_mode(tmp_path, mode):
    model, cfg, p = _lazy_model_config(tmp_path)
    cfg.load_mode = mode
    cmd = tuner.build_command(model, cfg, p)
    assert cmd[cmd.index("--lazy-mode") + 1] == "auto"
    if mode != "auto":
        assert cmd[cmd.index("--load-mode") + 1] == mode


@pytest.mark.parametrize("flag", ["--lazy-mode", "-lzm", "--tensor-read-lazy"])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("source", ["config", "profile", "extra"])
def test_lazy_override_requires_explicit_replan(tmp_path, flag, inline, source):
    model, cfg, p = _lazy_model_config(tmp_path)
    extras = [f"{flag}=off"] if inline else [flag, "off"]
    kwargs = {}
    if source == "config":
        cfg.extra_cli_flags = extras
    elif source == "profile":
        p.extra_args = extras
    else:
        kwargs["extra_args"] = extras
    with pytest.raises(ValueError, match="memory plan requires --lazy-mode auto"):
        tuner.build_command(model, cfg, p, **kwargs)


def test_lazy_alias_auto_is_deduplicated_and_kv_alias_cannot_override(tmp_path):
    model, cfg, p = _lazy_model_config(tmp_path)
    cmd = tuner.build_command(
        model,
        cfg,
        p,
        extra_args=["-lzm", "auto", "--cache-type-k", "f16", "--ctx-size=999999"],
    )
    assert "-lzm" not in cmd and "f16" not in cmd
    assert "--ctx-size=999999" not in cmd
    assert cmd.count("--lazy-mode") == 1


@pytest.mark.parametrize(
    "flag", ["--lazy-mode", "--tensor-read-lazy", "-lzm", "--spec-draft-device"]
)
def test_memory_contract_cannot_be_pruned_for_unsupported_runtime(flag):
    cmd = [
        "server",
        "-m",
        "model.gguf",
        flag,
        "auto" if "device" not in flag else "Vulkan1",
    ]
    kept, removed = tuner._filter_command_for_supported_flags(cmd, {"-m"})
    assert kept == cmd and not removed


@pytest.mark.parametrize("backend", ["Vulkan", "CUDA", "ROCm", "SYCL"])
@pytest.mark.parametrize("size,index", [(9, 0), (40, 1)])
def test_external_draft_pins_budgeted_post_visibility_gpu(
    tmp_path, size, index, backend
):
    expected = f"{backend}{index}"
    system = _fake_dual_gpu_system_with_vk_order()
    for gpu in system.gpus:
        gpu.runtime_backend = backend
        gpu.runtime_device = f"{backend}{gpu.hip_index}"
    model = _fake_model(tmp_path, "Llama-3.1-70B-Q4_K_M", size)
    draft = _fake_model(tmp_path, "Llama-3.1-1B-Q8_0", 1)
    p = match_profile(model.name, load_profiles(ROOT / "settings"))
    cfg = tuner.compute_config(model, system, p, draft_model=draft, user_ctx=8192)
    assert cfg.draft_device == expected
    assert cfg.draft_vram_gb == 1
    cmd = tuner.build_command(
        model, cfg, p, draft_model=draft, extra_args=["-devd", "Vulkan9"]
    )
    assert cmd[cmd.index("--spec-draft-device") + 1] == expected
    assert "Vulkan9" not in cmd


@pytest.mark.parametrize(
    "backend,env_key",
    [
        ("ROCm", "HIP_VISIBLE_DEVICES"),
        ("CUDA", "CUDA_VISIBLE_DEVICES"),
        ("Vulkan", "GGML_VK_VISIBLE_DEVICES"),
    ],
)
def test_external_draft_keeps_shared_output_on_primary_when_primary_was_first(
    tmp_path, backend, env_key
):
    system = _fake_dual_gpu_system_with_vk_order(large_vk_idx=0, small_vk_idx=1)
    for gpu in system.gpus:
        gpu.runtime_backend = backend
        gpu.runtime_device = f"{backend}{gpu.hip_index}"
    model = _fake_model(tmp_path, "Llama-3.1-70B", 40)
    draft = _fake_model(tmp_path, "draft-small", 1)
    p = match_profile(model.name, load_profiles(ROOT / "settings"))
    cfg = tuner.compute_config(model, system, p, draft_model=draft, user_ctx=8192)
    assert cfg.env_overrides[env_key] == "1,0"
    assert cfg.main_gpu == 1
    assert cfg.draft_device == f"{backend}1"
    split = [float(v) for v in cfg.tensor_split.split(",")]
    assert split[-1] > split[0]  # final layers/output and draft share the large GPU
    plain = tuner.compute_config(model, system, p, user_ctx=8192)
    assert (
        plain.env_overrides[env_key] == "0,1"
    )  # no unnecessary reorder without a draft


def test_small_gemma_does_not_require_lazy_capability(tmp_path):
    model, cfg, p = _lazy_model_config(tmp_path)
    model.metadata = {"general.architecture": "gemma4"}
    assert "--lazy-mode" not in tuner.build_command(model, cfg, p)


def test_force_mlock_applies_to_cpu_only_with_sufficient_ram(tmp_path, monkeypatch):
    system = _fake_system(ram_total=64, ram_free=60)
    system.gpus = []
    monkeypatch.setattr(tuner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tuner.os, "getuid", lambda: 0, raising=False)
    model = _fake_model(tmp_path, "Llama-3.1-8B-Q4", 4)
    p = match_profile(model.name, load_profiles(ROOT / "settings"))
    cfg = tuner.compute_config(model, system, p, user_ctx=4096, force_mlock=True)
    assert tuner.effective_load_mode(cfg) == "mlock"


@pytest.mark.parametrize(
    "build,warning", [(10852, True), (10853, False), (10863, False)]
)
def test_kimi_k3_rollback_notice_only_for_external_speculation(
    tmp_path, monkeypatch, build, warning
):
    model = _fake_model_md(tmp_path, "Kimi-K3", 10, {"general.architecture": "kimi-k3"})
    draft = _fake_model(tmp_path, "draft-small", 1)
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda binary: build)
    allowed, message, _ = tuner.check_draft_model_build(
        draft, "llama-server", target=model
    )
    assert allowed and bool(message) == warning
    if warning:
        assert "b10853" in message and "host checkpoints" in message
    assert tuner.check_draft_model_build(None, "llama-server", target=model) == (
        True,
        "",
        None,
    )


def test_shared_head_on_a_different_gpu_is_rejected(tmp_path):
    model, cfg, p = _lazy_model_config(tmp_path)
    model.metadata["qwen4exp.block_count"] = 48
    draft = _fake_model_md(
        tmp_path, "DFlash-draft", 1, {"general.architecture": "dflash"}
    )
    cfg.ngl = 999
    cfg.main_gpu = 0
    cfg.draft_device = "SYCL0"
    cfg.tensor_split = "0.7,0.3"
    with pytest.raises(ValueError, match="shares the target output tensor"):
        tuner.build_command(model, cfg, p, draft_model=draft)


def test_unresolved_multi_gpu_draft_is_rejected(tmp_path):
    model, cfg, p = _lazy_model_config(tmp_path)
    draft = _fake_model(tmp_path, "draft-small", 1)
    cfg.tensor_split = "0.3,0.7"
    with pytest.raises(ValueError, match="no exact backend device identity"):
        tuner.build_command(model, cfg, p, draft_model=draft)


def test_benchmark_memory_plan_error_is_reported_before_spawn(monkeypatch):
    import model_benchmark
    from test_model_benchmark import _runner, baseline_candidate

    runner = _runner()

    def reject(*args, **kwargs):
        raise ValueError("lazy override requires replanning")

    monkeypatch.setattr(model_benchmark, "build_command", reject)
    with pytest.raises(
        model_benchmark.BenchmarkFailure, match="Incompatible memory-plan settings"
    ):
        runner._benchmark_candidate(baseline_candidate(runner.base_config))


@pytest.mark.parametrize(
    "adapter,build,spec",
    [
        (tuner._adapt_nextn_regression_for_binary, 10743, "draft-mtp"),
        (tuner._adapt_spec_types_for_binary, 10150, "draft-dspark"),
    ],
)
def test_disabled_draft_removes_its_device_binding(monkeypatch, adapter, build, spec):
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda binary: build)
    cmd = [
        "server",
        "-m",
        "target.gguf",
        "--spec-type",
        f"{spec},ngram-map-k4v",
        "-md",
        "draft.gguf",
        "--spec-draft-device",
        "Vulkan1",
    ]
    kept, notes = adapter(cmd)
    assert notes
    assert "Vulkan1" not in kept and "-md" not in kept
    assert "ngram-map-k4v" in kept


def test_all_profile_flags_exist_in_b10863():
    flags = set(
        json.loads((ROOT / "docs/llama-b10863-server-flags.json").read_text())["flags"]
    )
    for p in load_profiles(ROOT / "settings"):
        for token in p.extra_args:
            if token.startswith("-"):
                assert token.split("=", 1)[0] in flags, (p.source_file, token)
