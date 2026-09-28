"""Low VRAM is not equivalent to abundant RAM; never invent a 2k budget."""

from pathlib import Path

import pytest

from hardware import GPUInfo, SystemInfo
from performance_target import PERFORMANCE_TARGETS
from scanner import ModelEntry
from settings_loader import ModelProfile
from tuner import _decide_moe_offload, build_command, compute_config


def model():
    return ModelEntry(
        name="llama-8B",
        group="test",
        path=Path("model.gguf"),
        size_bytes=8 * 1024**3,
        metadata={
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.context_length": 262144,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        },
    )


def system(ram):
    return SystemInfo(
        "Windows", "CPU", 8, 16, 16, ram, [GPUInfo(0, "Vega", "amd", 8192, 7168)]
    )


@pytest.mark.parametrize("ctx", [None, 131072])
def test_exhausted_host_budget_does_not_masquerade_as_2k(ctx):
    with pytest.raises(MemoryError, match="minimum 2,048-token"):
        compute_config(
            model(),
            system(3),
            ModelProfile(display_name="test"),
            perf_target=PERFORMANCE_TARGETS["low_vram"],
            user_ctx=ctx,
        )


def test_disabling_prompt_cache_recovers_real_capacity():
    profile = ModelProfile(display_name="test", max_context=262144)
    with pytest.raises(MemoryError):
        compute_config(
            model(), system(5), profile, perf_target=PERFORMANCE_TARGETS["low_vram"]
        )
    cfg = compute_config(
        model(),
        system(5),
        profile,
        perf_target=PERFORMANCE_TARGETS["low_vram"],
        prompt_cache_ram_mib=0,
    )
    assert cfg.ctx > 2048
    assert cfg.no_kv_offload
    assert cfg.estimated_model_ram_gb + cfg.kv_ram_gb + 1 < 5
    cmd = build_command(model(), cfg, profile)
    assert cmd[cmd.index("--cache-ram") + 1] == "0"
    assert "--no-kv-offload" in cmd or "-nkvo" in cmd


def test_moe_host_kv_does_not_reserve_device_kv():
    args = dict(
        model_size_gb=12,
        free_vram_gb=7,
        free_ram_gb=32,
        n_layers=40,
        expert_count=128,
        params_billion=30,
        target_ctx=131072,
        base_kv_per_token_mb=0.0625,
    )
    gpu = _decide_moe_offload(**args)
    host = _decide_moe_offload(**args, kv_to_ram=True)
    assert host[1] < gpu[1]  # more experts on GPU leaves RAM for host KV
    assert host[2] + 0.6 + 0.3 <= 7  # GPU workspace remains reserved


def test_moe_context_preference_cannot_force_avoidable_host_overcommit():
    args = dict(
        model_size_gb=12,
        free_vram_gb=8,
        free_ram_gb=6,
        n_layers=40,
        expert_count=128,
        params_billion=30,
        target_ctx=131072,
        base_kv_per_token_mb=0.0625,
    )
    result = _decide_moe_offload(**args)
    assert result[3] <= 6
    assert result[2] + 0.6 + 0.3 <= 8


def test_failed_preview_does_not_keep_previous_models_context():
    from types import SimpleNamespace
    from qt_launcher import MainWindow

    rendered = []
    window = SimpleNamespace(
        _system=object(),
        _effective_config=lambda *args: None,
        _config_error="Insufficient KV memory",
        _config_preview=SimpleNamespace(setPlainText=rendered.append),
    )
    MainWindow._update_config_text(window, model(), ModelProfile(display_name="test"))
    assert "No safe automatic configuration" in rendered[0]
    assert "Insufficient KV memory" in rendered[0]
