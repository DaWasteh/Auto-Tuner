"""Compute-buffer / host-footprint estimate added to finished plans (v5.6.1).

Coefficients come from b11371 ``-lv 4`` buffer reports compared with Windows
per-process GPU and host counters; see docs/v5.6.1-validation.md.
"""

from types import SimpleNamespace

import pytest

import tuner
from test_smoke import _fake_model_md, _fake_system
from settings_loader import load_profiles, match_profile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _config(**overrides):
    values = dict(
        ctx=262144,
        ngl=999,
        threads=8,
        batch_threads=8,
        batch=1024,
        ubatch=1024,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
        full_offload=True,
        estimated_model_vram_gb=16.0,
        kv_vram_gb=9.0,
        estimated_kv_gb=9.0,
    )
    values.update(overrides)
    return tuner.TunedConfig(**values)


def _system(os_name="Windows 11", free=(31.0, 15.0)):
    return SimpleNamespace(
        os_name=os_name,
        gpus=[SimpleNamespace(free_vram_mb=int(gb * 1024)) for gb in free],
    )


def _model(tmp_path, arch="qwen35", **metadata):
    return _fake_model_md(
        tmp_path, "Model-Q4_K_M", 16, {"general.architecture": arch, **metadata}
    )


def test_estimate_scales_with_context_ubatch_and_device_split():
    mask = 262144 * 1024 * 2 / 1024**3  # 0.5 GiB F16 attention mask
    single = tuner.compute_buffer_estimate_gb(262144, 1024, False)
    split = tuner.compute_buffer_estimate_gb(262144, 1024, True)
    act = 1024 * 4096 * 46 / 1024**3  # activations at the default width
    assert single == pytest.approx((2.0 * mask + act, 1.1 * mask))
    assert split == pytest.approx((10.0 * mask + act, 4.3 * mask))
    # Halving the physical batch halves the mask-driven part.
    half = tuner.compute_buffer_estimate_gb(262144, 512, True)
    assert half[0] == pytest.approx(10.0 * mask / 2 + act / 2)
    # A large physical batch is dominated by activations (Clef-Flash measured
    # 1.93 GiB at ubatch 8192); a 384-wide encoder needs a tenth of that.
    assert tuner.compute_buffer_estimate_gb(16384, 8192, False)[0] == pytest.approx(
        1.94, abs=0.02
    )
    assert tuner.compute_buffer_estimate_gb(8192, 8192, False, n_embd=384)[
        0
    ] == pytest.approx(0.38, abs=0.02)
    vram, ram = tuner.compute_buffer_estimate_gb(32768, 512, False, gpu=False)
    assert vram == 0.0 and ram > 0.0


def test_single_gpu_plan_reports_compute_and_host_footprint(tmp_path):
    config = _config()
    model = _model(tmp_path, __input_embedding_bytes__=int(0.67 * 1024**3))
    before = (config.runtime_vram_overhead_gb, config.runtime_ram_overhead_gb)
    tuner._finalize_runtime_buffers(config, model, _system())
    assert config.compute_vram_gb == pytest.approx(1.18, abs=0.01)
    # one host mask copy + fixed runtime + CPU-side input embeddings
    assert config.compute_ram_gb == pytest.approx(0.55 + 0.4 + 0.67, abs=0.01)
    assert config.ubatch == 1024
    # Without an exact scan the table is assumed to be 5 % of the file (<= 0.6).
    plain = _config()
    tuner._finalize_runtime_buffers(plain, _model(tmp_path), _system())
    assert plain.compute_ram_gb == pytest.approx(0.55 + 0.4 + 0.6, abs=0.01)
    # Placement-time reserves keep their meaning; the estimate is additive.
    assert (config.runtime_vram_overhead_gb, config.runtime_ram_overhead_gb) == before


def test_windows_full_offload_reads_without_mmap(tmp_path):
    model = _model(tmp_path)
    windows = _config()
    tuner._finalize_runtime_buffers(windows, model, _system("Windows 11"))
    assert (windows.load_mode, windows.no_mmap) == ("none", True)
    assert tuner.effective_load_mode(windows) == "none"

    linux = _config()
    tuner._finalize_runtime_buffers(linux, model, _system("Linux 6.17"))
    assert (linux.load_mode, linux.no_mmap) == ("auto", False)

    # CPU-resident weights, an explicit choice, mlock and lazy tables keep mmap.
    for overrides in (
        {"full_offload": False},
        {"n_cpu_moe": 8},
        {"load_mode": "mmap"},
        {"mlock": True},
        {"mapped_model_ram_gb": 26.8},
        {"unified_memory": True},
    ):
        config = _config(**overrides)
        tuner._finalize_runtime_buffers(config, model, _system("Windows 11"))
        assert config.no_mmap is False, overrides
        assert config.load_mode == overrides.get("load_mode", "auto")


def test_split_plan_lowers_ubatch_before_it_spills(tmp_path):
    model = _model(tmp_path)
    # 41.3 GiB of weights + KV on 47 GiB free: ten masks at ubatch 1024
    # (6.1 GiB at 312k context) do not fit beside the 3 GiB per-card reserve,
    # so the plan keeps its context and gives up physical batch size.
    tight = _config(
        ctx=312320,
        tensor_split="0.333,0.667",
        estimated_model_vram_gb=13.5,
        kv_vram_gb=26.2,
        vision_vram_gb=1.64,
    )
    tuner._finalize_runtime_buffers(tight, model, _system(free=(31.7, 15.4)))
    assert tight.ubatch == 256 and tight.batch == 1024
    assert tight.compute_vram_gb == pytest.approx(
        10 * 312320 * 256 * 2 / 1024**3 + 0.045, abs=0.01
    )

    # One halving is enough when 512 already fits.
    medium = _config(
        ctx=312320, tensor_split="1,2", estimated_model_vram_gb=13.5, kv_vram_gb=24.0
    )
    tuner._finalize_runtime_buffers(medium, model, _system(free=(31.7, 15.4)))
    assert medium.ubatch == 512

    # The same split with room to spare keeps the faster physical batch.
    roomy = _config(tensor_split="13,27", estimated_model_vram_gb=26.5, kv_vram_gb=2.75)
    tuner._finalize_runtime_buffers(roomy, model, _system(free=(31.7, 15.4)))
    assert roomy.ubatch == 1024
    assert roomy.compute_vram_gb == pytest.approx(5.18, abs=0.01)

    # Never below 256, and never below a non-causal projector's image budget.
    huge = _config(
        ctx=1048576, tensor_split="1,2", estimated_model_vram_gb=30, kv_vram_gb=12
    )
    tuner._finalize_runtime_buffers(huge, model, _system(free=(31.7, 15.4)))
    assert huge.ubatch == 256


def test_qwen4exp_and_adaptive_plans_are_left_alone(tmp_path):
    flash = _config(ubatch=64)
    tuner._finalize_runtime_buffers(flash, _model(tmp_path, "qwen4exp"), _system())
    assert flash.compute_vram_gb == 0.0 and flash.load_mode == "auto"
    adaptive = _config(adaptive_memory=True)
    tuner._finalize_runtime_buffers(adaptive, _model(tmp_path), _system())
    assert adaptive.compute_vram_gb == 0.0 and adaptive.load_mode == "auto"


def test_compute_config_attaches_the_estimate_and_command_follows(
    tmp_path, monkeypatch
):
    profiles = load_profiles(ROOT / "settings")
    model = _fake_model_md(
        tmp_path,
        "Qwen3.5-9B-Q4_K_M",
        6,
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 32,
            "qwen35.context_length": 262144,
        },
    )
    profile = match_profile(model.name, profiles, model.architecture)
    config = tuner.compute_config(
        model, _fake_system(), profile, user_ctx=32768, prompt_cache_ram_mib=0
    )
    assert config.compute_vram_gb > 0 and config.compute_ram_gb >= 0.4
    # The synthetic system is Linux: mmap stays the runtime default there.
    assert config.load_mode == "auto"

    windows = _fake_system()
    windows.os_name = "Windows 11"
    config = tuner.compute_config(
        model, windows, profile, user_ctx=32768, prompt_cache_ram_mib=0
    )
    assert config.full_offload and config.load_mode == "none"
    monkeypatch.setattr(tuner, "_probe_supported_flags", lambda _: None)
    command = tuner.build_command(model, config, profile)
    assert command[command.index("--load-mode") + 1] == "none"


def test_auto_context_shrinks_when_a_split_plan_still_overflows(tmp_path, monkeypatch):
    calls = []

    def fake_plan(*args, **kwargs):
        arguments = tuner._COMPUTE_SIGNATURE.bind(*args, **kwargs)
        arguments.apply_defaults()
        ctx = arguments.arguments["user_ctx"] or 262144
        calls.append(arguments.arguments["user_ctx"])
        # 30 GiB of weights, KV proportional to context (36 GiB at 262k).
        return _config(
            ctx=ctx,
            ubatch=1024,
            tensor_split="1,2",
            estimated_model_vram_gb=10.0,
            kv_vram_gb=30.0 * ctx / 262144,
        )

    monkeypatch.setattr(tuner, "_compute_config", fake_plan)
    profile = match_profile("Model-Q4_K_M", load_profiles(ROOT / "settings"), "llama")
    model = _model(tmp_path, "llama")
    system = _system(free=(31.7, 15.4))
    system.total_ram_gb = 64.0  # not an adaptive low-memory system

    config = tuner.compute_config(model, system, profile)
    assert calls[0] is None and len(calls) > 1
    assert config.ctx < 262144 and config.ctx % 1024 == 0
    budget = 31.7 + 15.4 - 6.0
    total = config.estimated_model_vram_gb + config.kv_vram_gb + config.compute_vram_gb
    assert total <= budget + 0.05
    assert config.ubatch == 256
    assert any(
        "multi-GPU compute buffers" in note for note in config.memory_adjustments
    )

    # An explicit context is never changed; the estimate still shows the cost.
    calls.clear()
    pinned = tuner.compute_config(model, system, profile, user_ctx=262144)
    assert calls == [262144] and pinned.ctx == 262144
    assert pinned.compute_vram_gb > 1.0 and not pinned.memory_adjustments


def test_mla_models_have_no_value_cache():
    # GLM-4.7-Flash (deepseek2 MLA): b11371 logs K 5561.79 MiB / V 0.00 MiB
    # at 202,752 cells and 47 layers with Q8_0.
    md = {
        "general.architecture": "deepseek2",
        "deepseek2.block_count": 47,
        "deepseek2.embedding_length": 2048,
        "deepseek2.attention.head_count": 20,
        "deepseek2.attention.head_count_kv": 1,
        "deepseek2.attention.key_length": 576,
        "deepseek2.attention.value_length": 512,
        "deepseek2.attention.key_length_mla": 256,
        "deepseek2.attention.value_length_mla": 256,
    }
    total = tuner.kv_per_token_mb_from_metadata(md)
    assert total == pytest.approx(47 * 576 * 2 / 1024**2)
    assert tuner.kv_per_token_parts_mb_from_metadata(md) == (total, 0.0)
    q8_gib = total * tuner.kv_quant_factor("q8_0") * 202752 / 1024
    assert q8_gib == pytest.approx(5561.79 / 1024, rel=0.06)

    # Without the MLA keys the same shape keeps a regular K + V cache.
    plain = {k: v for k, v in md.items() if not k.endswith("_mla")}
    assert tuner.kv_per_token_mb_from_metadata(plain) == pytest.approx(
        47 * (576 + 512) * 2 / 1024**2
    )


def test_gpt_oss_counts_only_the_dense_half_of_its_layers():
    # gpt-oss-20b: "131072 cells, 12 layers ... K (q8_0): 816.00 MiB, V: 816.00".
    md = {
        "general.architecture": "gpt-oss",
        "gpt-oss.block_count": 24,
        "gpt-oss.embedding_length": 2880,
        "gpt-oss.attention.head_count": 64,
        "gpt-oss.attention.head_count_kv": 8,
        "gpt-oss.attention.key_length": 64,
        "gpt-oss.attention.value_length": 64,
        "gpt-oss.attention.sliding_window": 128,
    }
    total = tuner.kv_per_token_mb_from_metadata(md)
    assert total == pytest.approx(12 * 8 * 128 * 2 / 1024**2)
    q8_gib = total * tuner.kv_quant_factor("q8_0") * 131072 / 1024
    assert q8_gib == pytest.approx(1632.0 / 1024, rel=0.06)
    # No window means every layer is dense.
    dense = dict(md)
    del dense["gpt-oss.attention.sliding_window"]
    assert tuner.kv_per_token_mb_from_metadata(dense) == pytest.approx(2 * total)


def test_hybrid_split_plans_use_the_single_device_factor(tmp_path):
    # Ling 3.0 flash, two GPUs + CPU experts, 131072 x 4096: llama.cpp reserved
    # 4.0 GiB of GPU compute, not ten 1 GiB masks.
    hybrid = _config(
        ctx=131072,
        ubatch=4096,
        tensor_split="15,32",
        n_cpu_moe=24,
        full_offload=False,
        estimated_model_ram_gb=16.5,
        batch_vram_overhead_gb=0.9,
    )
    model = _model(tmp_path, "bailingmoe3", **{"bailingmoe3.embedding_length": 2048})
    tuner._finalize_runtime_buffers(hybrid, model, _system())
    assert hybrid.ubatch == 4096
    assert hybrid.compute_vram_gb == pytest.approx(2.0 + 0.36, abs=0.02)
    assert hybrid.compute_vram_gb + hybrid.batch_vram_overhead_gb < 4.04
    assert (hybrid.load_mode, hybrid.no_mmap) == ("auto", False)
