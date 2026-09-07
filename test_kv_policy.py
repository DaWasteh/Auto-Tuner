"""Q8-first KV policy and memory/compatibility regressions (v5.4.3)."""

from pathlib import Path

import pytest

from settings_loader import ModelProfile, load_profiles
from test_smoke import _fake_model_md, _fake_system as _system
from tuner import _auto_kv_requires_f16, _pick_kv_quant, build_command, compute_config


def _fake_system(vram_gb):
    return _system(vram_total=vram_gb, vram_free=vram_gb - 1)


def _model(tmp_path, arch="llama", key=128, value=128):
    return _fake_model_md(
        tmp_path,
        f"{arch}-1B",
        1.0,
        {
            "general.architecture": arch,
            f"{arch}.block_count": 8,
            f"{arch}.context_length": 32768,
            f"{arch}.embedding_length": 1024,
            f"{arch}.attention.head_count": 8,
            f"{arch}.attention.head_count_kv": 2,
            f"{arch}.attention.key_length": key,
            f"{arch}.attention.value_length": value,
        },
    )


@pytest.mark.parametrize("hint", ["f16", "bf16", "q4_0", "q5_0", "q8_0", ""])
def test_q8_is_ceiling_regardless_of_historical_hint(hint):
    assert _pick_kv_quant(hint, 32768, 0.0625, 100) == ("q8_0", "q8_0")
    assert _pick_kv_quant(hint, 32768, 0, 0) == ("q8_0", "q8_0")


def test_one_sided_pin_is_included_in_selection_budget():
    # At 32768 tokens: pinned K=F16 uses 1GiB; Q8 V uses .55GiB,
    # Q4 V uses .32GiB. A 1.4GiB budget holds only the latter pair.
    assert _pick_kv_quant(
        "q8_0",
        32768,
        0.0625,
        1.4,
        base_k_per_token_mb=0.03125,
        base_v_per_token_mb=0.03125,
        force_cache_k="f16",
    ) == ("f16", "q4_0")
    # Unequal K/V dimensions must not be averaged when a side is pinned.
    assert _pick_kv_quant(
        "q8_0",
        32768,
        0.0625,
        1.25,
        base_k_per_token_mb=0.046875,
        base_v_per_token_mb=0.015625,
        force_cache_v="f16",
    ) == ("q4_0", "f16")


@pytest.mark.parametrize("arch", ["llama", "grok", "deepseek2-ocr"])
def test_compute_non_fa_cache_matches_command(tmp_path, arch):
    model = _model(tmp_path, arch)
    profile = ModelProfile(display_name="non-FA", flash_attn=False, max_context=32768)
    cfg = compute_config(model, _fake_system(vram_gb=16), profile, user_ctx=8192)
    assert not cfg.flash_attn
    assert cfg.cache_k == cfg.cache_v == "f16"
    assert cfg.kv_quant_strategy == "compatibility-f16"
    assert "Flash Attention" in cfg.warning
    cmd = build_command(model, cfg, profile)
    assert cmd[cmd.index("-ctv") + 1] == "f16"
    assert cmd[cmd.index("-fa") + 1] == "off"


def test_flash_attention_override_recomputes_kv(tmp_path):
    model = _model(tmp_path)
    profile = ModelProfile(display_name="normal", max_context=32768)
    system = _fake_system(vram_gb=16)
    q8 = compute_config(model, system, profile, user_ctx=8192)
    f16 = compute_config(model, system, profile, user_ctx=8192, force_flash_attn=False)
    restored = compute_config(
        model, system, profile, user_ctx=8192, force_flash_attn=True
    )
    assert q8.cache_k == restored.cache_k == "q8_0"
    assert f16.cache_k == f16.cache_v == "f16"
    assert f16.estimated_kv_gb > q8.estimated_kv_gb
    # Explicit manual pairs remain authoritative, even if unsupported.
    pinned = compute_config(
        model,
        system,
        profile,
        user_ctx=8192,
        force_cache_k="bf16",
        force_cache_v="bf16",
    )
    assert pinned.cache_k == pinned.cache_v == "bf16"


@pytest.mark.parametrize("dims", [(72, 128), (128, 80), ([128, 72], 128)])
def test_non_block_aligned_heads_use_compatible_pair(tmp_path, dims):
    model = _model(tmp_path, key=dims[0], value=dims[1])
    assert _auto_kv_requires_f16(model, True)
    cfg = compute_config(
        model, _fake_system(vram_gb=16), ModelProfile(display_name="x")
    )
    assert cfg.cache_k == cfg.cache_v == "f16"
    assert "32-element" in cfg.warning


def test_picker_uses_per_slot_budget(tmp_path, monkeypatch):
    import tuner

    budgets = []
    original = tuner._pick_kv_quant

    def capture(*args, **kwargs):
        budgets.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(tuner, "_pick_kv_quant", capture)
    model = _model(tmp_path)
    profile = ModelProfile(display_name="x", max_context=32768)
    system = _fake_system(vram_gb=32)
    compute_config(model, system, profile, user_ctx=32768, force_n_parallel=1)
    compute_config(model, system, profile, user_ctx=32768, force_n_parallel=4)
    assert len(budgets) == 2
    assert 0 < budgets[1] < budgets[0] / 4  # extra per-slot compute headroom


def test_legacy_fa_snapshot_is_cascading_not_a_late_overlay(tmp_path):
    from qt_launcher import (
        _expert_cascading_pins,
        apply_expert_values,
        expert_cfg_from_values,
    )

    snap = {"pins": {"user_ctx": 8192}, "values": {"flash_attn": False}}
    assert _expert_cascading_pins(snap) == {"user_ctx": 8192, "force_flash_attn": False}
    model = _model(tmp_path)
    cfg = compute_config(
        model, _fake_system(vram_gb=16), ModelProfile(display_name="x")
    )
    assert apply_expert_values(cfg, snap["values"]).flash_attn is True
    assert expert_cfg_from_values(cfg, snap["values"]).flash_attn is False


@pytest.mark.parametrize("pair", [("f16", "f16"), ("q4_0", "q4_0"), ("f16", "q4_0")])
def test_manual_estimates_scale_separate_kv_sides(tmp_path, pair):
    from qt_launcher import expert_cfg_from_values
    from tuner import kv_quant_factor

    model = _model(tmp_path, key=256, value=64)
    cfg = compute_config(
        model, _fake_system(vram_gb=16), ModelProfile(display_name="x"), user_ctx=8192
    )
    assert cfg.kv_key_fraction == pytest.approx(0.8)
    manual = expert_cfg_from_values(
        cfg,
        {
            "ctx": 16384,
            "cache_k": pair[0],
            "cache_v": pair[1],
            "parallel_enabled": True,
            "parallel_count": 2,
        },
    )
    ratio = (0.8 * kv_quant_factor(pair[0]) + 0.2 * kv_quant_factor(pair[1])) / 0.55
    assert manual.estimated_kv_gb == pytest.approx(cfg.estimated_kv_gb * 4 * ratio)
    assert manual.kv_vram_gb == pytest.approx(cfg.kv_vram_gb * 4 * ratio)
    assert manual.recurrent_state_vram_gb == cfg.recurrent_state_vram_gb


def test_qsa_host_clamp_reconsiders_q8_at_final_context(tmp_path, monkeypatch):
    import tuner
    from performance_target import PERFORMANCE_TARGETS

    calls = []
    original = tuner._pick_kv_quant

    def capture(*args, **kwargs):
        pair = original(*args, **kwargs)
        calls.append(pair)
        return pair

    monkeypatch.setattr(tuner, "_pick_kv_quant", capture)
    model = _model(tmp_path, arch="qwen4exp", key=256, value=256)
    model.metadata["qwen4exp.context_length"] = 262144
    cfg = compute_config(
        model,
        _system(vram_total=4, vram_free=3, ram_total=9, ram_free=8),
        ModelProfile(display_name="QSA", max_context=262144),
        user_ctx=262144,
        prompt_cache_ram_mib=0,
        perf_target=PERFORMANCE_TARGETS["safe"],
        force_ngl=999,
    )
    assert calls == [("q4_0", "q4_0"), ("q8_0", "q8_0")]
    assert 32768 < cfg.ctx < 65536
    assert cfg.cache_k == cfg.cache_v == "q8_0"
    assert (
        cfg.estimated_model_vram_gb + cfg.runtime_vram_overhead_gb + cfg.kv_vram_gb < 3
    )


def test_shipped_normal_profile_hints_are_q8():
    profiles = load_profiles(Path(__file__).parent / "settings")
    for profile in profiles:
        if profile.recommended_kv_quant != "f16":
            assert profile.recommended_kv_quant == "q8_0", profile.source_file
