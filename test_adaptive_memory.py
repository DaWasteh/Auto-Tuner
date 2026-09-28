"""Regression coverage for the real two-pool Windows/Vulkan Auto fallback."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import app_settings
from adaptive_memory import eligible, fits
from hardware import GPUInfo, SystemInfo
from performance_target import PERFORMANCE_TARGETS
from scanner import ModelEntry, _moe_expert_spans
from settings_loader import ModelProfile, load_profiles
from tuner import (
    MOE_MIN_WORKING_CTX,
    MOE_VRAM_SAFETY_GB,
    _decide_moe_offload,
    _kv_headroom_reserve,
    build_command,
    compute_config,
    kv_quant_factor,
)

SETTINGS_DIR = Path(__file__).resolve().parent / "settings"


def model():
    return ModelEntry(
        name="gemma-4-26B-test",
        group="test",
        path=Path("model.gguf"),
        size_bytes=14439363584,
        mmproj=SimpleNamespace(stat=lambda: SimpleNamespace(st_size=1194828160)),
        metadata={
            "general.architecture": "gemma4",
            "gemma4.block_count": 30,
            "gemma4.context_length": 262144,
            "gemma4.expert_count": 128,
            "gemma4.attention.head_count": 16,
            "gemma4.attention.head_count_kv": ([8] * 5 + [2]) * 5,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": ([True] * 5 + [False]) * 5,
            "__moe_expert_bytes_by_layer__": [428212224] * 30,
            "__input_embedding_bytes__": 605552640,
            "__tied_output_embedding__": True,
        },
    )


def system(free=10):
    return SystemInfo(
        "Windows 11",
        "Ryzen",
        8,
        16,
        15.93,
        free,
        [
            GPUInfo(
                0,
                "Vega",
                "amd",
                8176,
                7350,
                runtime_backend="Vulkan",
                runtime_device="Vulkan0",
            )
        ],
    )


def profile():
    return ModelProfile(display_name="Gemma", max_context=262144)


def test_tensor_spans_count_only_movable_expert_weights():
    tensors = [
        ("token_embd.weight", 0),
        ("blk.0.ffn_up_exps.weight", 100),
        ("blk.0.ffn_up_shexp.weight", 300),
        ("blk.1.ffn_down_exps.weight", 400),
        ("output.weight", 700),
    ]
    assert _moe_expert_spans(tensors, 1024, 2024, 2) == [200, 300]
    assert _moe_expert_spans(tensors, 1024, 1025, 2) == []
    assert _moe_expert_spans(tensors, 1024, 2024, 99999) == []
    # llama.cpp's --n-cpu-moe regex also moves chunked expert tensors.
    chunked = [("blk.0.ffn_up_chexps.weight", 0), ("blk.0.attn_q.weight", 64)]
    assert _moe_expert_spans(chunked, 0, 128, 1) == [64]


@pytest.mark.parametrize("mode", PERFORMANCE_TARGETS)
def test_all_modes_fit_two_pools_with_visible_fallback(mode):
    m, s, p = model(), system(), profile()
    cfg = compute_config(m, s, p, perf_target=PERFORMANCE_TARGETS[mode])
    assert cfg.adaptive_memory
    assert 2048 <= cfg.ctx <= 262144
    assert cfg.memory_disable_vision and cfg.memory_disable_draft
    assert cfg.prompt_cache_ram_mib == 0
    assert cfg.batch <= 256 and cfg.ubatch <= 128
    assert cfg.ngl == 30  # tied output uses CPU input tensor, not a duplicate
    expected_cpu = (605552640 + cfg.n_cpu_moe * 428212224) / 2**30
    assert cfg.estimated_model_ram_gb == pytest.approx(expected_cpu)
    assert cfg.estimated_model_ram_gb + cfg.estimated_model_vram_gb == pytest.approx(
        m.size_gb
    )
    assert fits(cfg, s, PERFORMANCE_TARGETS[mode])
    assert cfg.memory_adjustments and "text-only" in cfg.warning
    # Planner does not edit the user's model/options.
    assert m.mmproj is not None
    assert "__prefer_cpu_tied_output__" not in m.metadata


def test_command_cannot_reenable_options_removed_by_auto():
    m, s, p = model(), system(), profile()
    cfg = compute_config(m, s, p)
    cfg.batch, cfg.ubatch = 4096, 4096  # stale snapshot / caller
    cfg.load_mode = "auto"
    cmd = build_command(
        m,
        cfg,
        p,
        enable_speculative=True,
        enable_ngram=True,
        enable_prompt_cache=True,
        prompt_cache_ram_mib=2048,
        extra_args=["--op-offload"],
    )
    assert "--mmproj" not in cmd and "--spec-type" not in cmd
    assert cmd[cmd.index("--cache-ram") + 1] == "0"
    assert cmd[cmd.index("--load-mode") + 1] == "none"
    assert cmd[cmd.index("-b") + 1] == "256"
    assert cmd[cmd.index("-ub") + 1] == "128"
    assert cmd.count("--no-op-offload") == 1 and "--op-offload" not in cmd


def test_saved_expert_overlay_keeps_the_adaptive_memory_contract():
    from qt_launcher import apply_expert_values

    cfg = compute_config(model(), system(), profile())
    apply_expert_values(
        cfg,
        {
            "batch": 4096,
            "ubatch": 4096,
            "load_mode": "auto",
            "extras": "--op-offload",
            "parallel_enabled": True,
            "parallel_count": 8,
        },
    )
    assert (cfg.batch, cfg.ubatch, cfg.load_mode, cfg.n_parallel) == (
        256,
        128,
        "none",
        1,
    )
    assert "--no-op-offload" in cfg.extra_cli_flags
    assert "--op-offload" not in cfg.extra_cli_flags


def test_explicit_quant_and_context_are_not_silently_replaced():
    cfg = compute_config(
        model(),
        system(),
        profile(),
        user_ctx=4096,
        force_cache_k="q8_0",
        force_cache_v="q8_0",
    )
    assert cfg.ctx == 4096
    assert cfg.cache_k == cfg.cache_v == "q8_0"


def test_insufficient_memory_is_refused_not_forced_to_2k():
    with pytest.raises(MemoryError, match="No safe Auto placement"):
        compute_config(model(), system(3), profile())


def test_unverified_hardware_and_incomplete_scans_keep_legacy_path():
    m, s = model(), system()
    assert eligible(m, s)
    assert not eligible(m, replace(s, os_name="Linux"))
    assert not eligible(m, replace(s, total_ram_gb=64))
    assert not eligible(m, replace(s, gpus=s.gpus * 2))
    assert not eligible(replace(m, metadata={}), s)
    lazy = replace(m, metadata={**m.metadata, "__read_lazy_tensor_bytes__": 6 * 2**30})
    assert not eligible(lazy, s)


# ---------------------------------------------------------------------------
# Integration review (v5.5.8): keep the validated behaviour inside its class.


def _plan_or_error(model_entry, system_info, target):
    try:
        cfg = compute_config(model_entry, system_info, profile(), perf_target=target)
    except MemoryError as exc:
        return str(exc)
    return (
        cfg.ctx,
        cfg.ngl,
        cfg.n_cpu_moe,
        round(cfg.estimated_model_vram_gb, 6),
        round(cfg.estimated_model_ram_gb, 6),
        cfg.adaptive_memory,
    )


@pytest.mark.parametrize("variant", ["linux", "ram64", "gpu16"])
def test_exact_expert_placement_stays_inside_the_adaptive_class(variant):
    # Exact prefix spans and their hard RAM refusal were validated only on the
    # adaptive hardware class. Everywhere else the scanned spans must not
    # change the historical heuristic plan (which warns about overcommit).
    base = system()
    systems = {
        "linux": replace(base, os_name="Linux"),
        "ram64": replace(base, total_ram_gb=64, free_ram_gb=40),
        "gpu16": replace(
            base,
            gpus=[
                GPUInfo(
                    0,
                    "RX 9070 XT",
                    "amd",
                    16368,
                    15500,
                    runtime_backend="Vulkan",
                    runtime_device="Vulkan0",
                )
            ],
        ),
    }
    sysinfo = systems[variant]
    scanned = model()
    assert not eligible(scanned, sysinfo)
    legacy_metadata = {
        key: value
        for key, value in scanned.metadata.items()
        if key
        not in (
            "__moe_expert_bytes_by_layer__",
            "__input_embedding_bytes__",
            "__tied_output_embedding__",
        )
    }
    legacy = replace(scanned, metadata=legacy_metadata)
    for target in PERFORMANCE_TARGETS.values():
        assert _plan_or_error(scanned, sysinfo, target) == _plan_or_error(
            legacy, sysinfo, target
        )


def test_ram_driven_expert_move_keeps_a_working_kv_floor():
    # Moving experts to the GPU to avoid host overcommit must not consume the
    # whole KV budget; that turned a usable 119k plan into a 2k refusal.
    args = dict(
        model_size_gb=12,
        free_vram_gb=8,
        free_ram_gb=5.0,
        n_layers=40,
        expert_count=128,
        params_billion=30,
        target_ctx=131072,
        base_kv_per_token_mb=0.0625,
    )
    _ngl, n_cpu_moe, vram, _ram, _full = _decide_moe_offload(**args)
    absolute, fraction = _kv_headroom_reserve(MOE_MIN_WORKING_CTX, 1, False)
    floor = (
        MOE_MIN_WORKING_CTX * 0.0625 * kv_quant_factor("q8_0") / 1024 + absolute
    ) / (1 - fraction)
    assert 8 - MOE_VRAM_SAFETY_GB - vram >= floor
    # With a little more RAM the move leaves a working context and happens.
    moved = _decide_moe_offload(**{**args, "free_ram_gb": 5.3})
    assert moved[1] < n_cpu_moe
    assert moved[3] <= 5.3


def test_heuristic_low_memory_plan_is_not_refused_at_the_floor():
    # Linux keeps the heuristic planner. Balanced previously planned ~119k
    # with an overcommit warning; the RAM-driven move must not refuse it.
    text_only = replace(model(), mmproj=None)
    sysinfo = replace(system(9.0), os_name="Linux")
    cfg = compute_config(
        text_only,
        sysinfo,
        profile(),
        perf_target=PERFORMANCE_TARGETS["balanced"],
        prompt_cache_ram_mib=0,
    )
    assert not cfg.adaptive_memory
    assert cfg.ctx >= MOE_MIN_WORKING_CTX


def test_planning_refusal_reaches_the_api_without_a_modal_dialog(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    # Keep a reference: an unreferenced QApplication is collected before the
    # window is built, and Qt aborts the whole test process.
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    assert app is qt_widgets.QApplication.instance()
    monkeypatch.setattr(
        app_settings, "_settings_file", lambda: tmp_path / "settings.json"
    )
    path = tmp_path / "models" / "Qwen.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF")
    entry = ModelEntry(path=path, name="Qwen", group="models", size_bytes=2**30)
    window = qt_launcher.MainWindow(
        tmp_path / "models", SETTINGS_DIR, start_background=False
    )
    fake_system = system()
    window._system = fake_system
    window._current_entry = entry
    window._profiles = load_profiles(SETTINGS_DIR)
    monkeypatch.setattr(qt_launcher, "detect_system", lambda *_a, **_k: fake_system)
    monkeypatch.setattr(window, "_resolve_binary", lambda *_a, **_k: "llama-server")

    def refuse(*_args, **_kwargs):
        window._config_error = "Insufficient KV/compute memory"
        return None

    def modal(*_args, **_kwargs):
        raise AssertionError("headless launch opened a modal dialog")

    monkeypatch.setattr(window, "_effective_config", refuse)
    monkeypatch.setattr(qt_launcher.QMessageBox, "warning", modal)
    try:
        assert window._launch_server(interactive=False) is None
        assert window._last_launch_error == "Insufficient KV/compute memory"
    finally:
        window.close()
