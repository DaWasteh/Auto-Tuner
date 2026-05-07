"""Smoke tests for AutoTuner.

These tests don't need real GGUF models and run on any GitHub Actions
runner. They cover:
  - profile loading + pattern matching against real-world model names
  - mmproj pairing (longest-prefix) on a synthetic models tree
  - compute_config produces sensible values across hardware shapes
  - hardware detection doesn't crash on a runner without GPUs
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

# Make the project root importable when tests are run from the repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hardware import GPUInfo, SystemInfo, detect_system, format_system  # noqa: E402
from scanner import group_entries, scan_models  # noqa: E402
from settings_loader import load_profiles, match_profile  # noqa: E402
from tuner import build_command, compute_config, extract_params_billion  # noqa: E402


SETTINGS_DIR = ROOT / "settings"


# ---------------------------------------------------------------------------
# Hardware detection

def test_detect_system_does_not_raise():
    info = detect_system()
    assert info.total_ram_gb > 0
    assert info.cpu_cores_logical >= 1
    # GitHub runners may have no detected GPUs — that's fine
    assert isinstance(info.gpus, list)


def test_format_system_produces_text():
    info = detect_system()
    out = format_system(info)
    assert "CPU:" in out and "RAM:" in out


# ---------------------------------------------------------------------------
# Profile loading + pattern matching

def test_all_profiles_load():
    profiles = load_profiles(SETTINGS_DIR)
    assert len(profiles) >= 8, "expected the 8 shipped YAML profiles"
    files = {p.source_file for p in profiles}
    assert "_default.yaml" in files
    assert "qwen2_5-3.yaml" in files
    assert "gemma-4.yaml" in files
    assert "devstral.yaml" in files

@pytest.mark.parametrize("filename, expected_display", [
    ("Qwen3.5-9B-Q8_0",                       "Qwen3 (Alibaba)"),
    ("Qwen3.6-27B-UD-Q3_K_XL",                "Qwen3 (Alibaba)"),
    ("Qwen3.6-35B-A3B-UD-IQ3_S",              "Qwen3 (Alibaba)"),
    ("Gemma-4-26B-A4B-IQ3_M",                 "Gemma 4 (Google)"),
    ("gemma-4-E2B-it-BF16",                   "Gemma 4 (Google)"),
    ("Devstral-Small-2-24B-Instruct-2512-Q3_K_L", "Devstral (Mistral, code)"),
    ("Ministral-3-14B-Reasoning-2512-Q6_K",   "Ministral 3 (Mistral, reasoning)"),
    ("Mistral-Medium-3.5-128B-UD-IQ3_XXS",    "Mistral Medium 3.x"),
    ("Bonsai-8B",                             "Bonsai 8B (PrismML, 1-bit)"),
    ("Ternary-Bonsai-8B-Q2_0",                "Ternary-Bonsai (PrismML, 1.58-bit)"),
    ("Archon-14B.Q6_K",                       "Frankenmerger / community merge"),
    ("voldemort-10b-dpo.Q8_0",                "Frankenmerger / community merge"),
    ("Some-Random-LLM.gguf",                  "Generic / fallback"),
])
def test_pattern_matching(filename, expected_display):
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile(filename, profiles)
    assert p.display_name == expected_display, (
        f"{filename!r} matched {p.display_name!r}, "
        f"expected {expected_display!r}")


def test_ministral_does_not_collide_with_mistral_medium():
    """The ministral pattern has no overlap with mistral-medium."""
    profiles = load_profiles(SETTINGS_DIR)
    assert match_profile("Ministral-3-14B", profiles).display_name \
        == "Ministral 3 (Mistral, reasoning)"
    assert match_profile("Mistral-Medium-3.5-128B", profiles).display_name \
        == "Mistral Medium 3.x"


# ---------------------------------------------------------------------------
# Param extraction

@pytest.mark.parametrize("name, expected", [
    ("Qwen3.5-9B-Q8_0", 9.0),
    ("Qwen3.6-35B-A3B-UD-IQ3_S", 35.0),       # MoE: total params, not active
    ("Mistral-Medium-3.5-128B-UD-IQ3_XXS", 128.0),
    ("gemma-4-E2B-it-BF16", 2.0),             # Gemma "effective" size
    ("gemma-4-E4B-it-Q8_0", 4.0),
    ("Qwen3.5-0.8B-Q8_0", 0.8),
])
def test_extract_params_billion(name, expected):
    assert extract_params_billion(name) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Scanner + mmproj pairing

def _write_minimal_gguf(path: Path) -> None:
    """Write a valid empty-metadata GGUF header so the scanner accepts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))   # version
        f.write(struct.pack("<Q", 0))   # tensor count
        f.write(struct.pack("<Q", 0))   # kv count


def test_scanner_pairs_mmproj_by_size(tmp_path):
    """Each Qwen3.5 sub-model must get its own size-matched mmproj."""
    folder = tmp_path / "Alibaba" / "Qwen3.5"
    files = [
        "mmproj-Qwen3.5-0.8B-BF16.gguf", "Qwen3.5-0.8B-Q8_0.gguf",
        "mmproj-Qwen3.5-2B-BF16.gguf",   "Qwen3.5-2B-Q8_0.gguf",
        "mmproj-Qwen3.5-9B-BF16.gguf",   "Qwen3.5-9B-Q8_0.gguf",
    ]
    for f in files:
        _write_minimal_gguf(folder / f)

    entries = scan_models(tmp_path)
    by_name = {e.name: e for e in entries}

    expected = [
        ("Qwen3.5-0.8B-Q8_0", "mmproj-Qwen3.5-0.8B-BF16.gguf"),
        ("Qwen3.5-2B-Q8_0",   "mmproj-Qwen3.5-2B-BF16.gguf"),
        ("Qwen3.5-9B-Q8_0",   "mmproj-Qwen3.5-9B-BF16.gguf"),
    ]
    for stem, expected_name in expected:
        mmproj = by_name[stem].mmproj
        assert mmproj is not None, f"{stem} should have been paired with an mmproj"
        assert mmproj.name == expected_name


def test_scanner_skips_mmproj_from_main_list(tmp_path):
    _write_minimal_gguf(tmp_path / "MyModel-Q8_0.gguf")
    _write_minimal_gguf(tmp_path / "mmproj-MyModel-F16.gguf")
    entries = scan_models(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "MyModel-Q8_0"
    assert entries[0].has_vision


def test_scanner_handles_empty_folder(tmp_path):
    assert scan_models(tmp_path) == []


def test_group_entries_buckets_by_folder(tmp_path):
    _write_minimal_gguf(tmp_path / "Vendor1" / "ModelA.gguf")
    _write_minimal_gguf(tmp_path / "Vendor2" / "ModelB.gguf")
    entries = scan_models(tmp_path)
    groups = group_entries(entries)
    assert "Vendor1" in groups
    assert "Vendor2" in groups


# ---------------------------------------------------------------------------
# Tuner

def _fake_system(ram_total: float = 64, ram_free: float = 48,
                 vram_total: float = 24, vram_free: float = 22):
    """Build a synthetic SystemInfo for tuner tests."""
    return SystemInfo(
        os_name="Linux test",
        cpu_name="Test CPU",
        cpu_cores_physical=16,
        cpu_cores_logical=32,
        total_ram_gb=ram_total,
        free_ram_gb=ram_free,
        gpus=[GPUInfo(
            index=0, name="Test GPU", vendor="amd",
            total_vram_mb=int(vram_total * 1024),
            free_vram_mb=int(vram_free * 1024),
        )] if vram_total > 0 else [],
    )


def _fake_model(tmp_path, name, size_gb):
    p = tmp_path / f"{name}.gguf"
    _write_minimal_gguf(p)
    from scanner import ModelEntry
    return ModelEntry(
        path=p, name=name, group=".",
        size_bytes=int(size_gb * 1024 ** 3),
    )


def test_small_model_full_offload(tmp_path):
    """A small model on a big GPU → full offload, generous context."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    assert cfg.full_offload is True
    assert cfg.ngl == 999
    assert cfg.ctx >= 32768


def test_huge_model_falls_back_to_partial_or_cpu(tmp_path):
    """A 50 GB model on a 24 GB GPU → not a full offload."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS",
                        size_gb=50.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(ram_total=96, ram_free=80),
                         profile)
    assert cfg.full_offload is False
    assert cfg.ctx >= 2048   # always at least the floor


def test_no_gpu_falls_back_to_cpu(tmp_path):
    """No GPU → ngl=0, no full_offload."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(vram_total=0, vram_free=0),
                         profile)
    assert cfg.ngl == 0
    assert cfg.full_offload is False


def test_user_ctx_override_wins(tmp_path):
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile, user_ctx=8192)
    assert cfg.ctx == 8192


def test_devstral_uses_high_context_when_ram_is_plenty(tmp_path):
    """Regression test for the v1 bug: Devstral was capped at 16k context
    even with tons of free RAM. With a roomy system it must now reach far
    above that."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path,
                        "Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL",
                        size_gb=13.5)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model,
                         _fake_system(ram_total=96, ram_free=71,
                                      vram_total=24, vram_free=22.8),
                         profile)
    assert cfg.ctx > 16384, f"v1 bug regression — got only {cfg.ctx}"


# ---------------------------------------------------------------------------
# Command builder

def test_build_command_includes_essentials(tmp_path):
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, port=12345)
    assert "-m" in cmd and str(model.path) in cmd
    assert "-c" in cmd and str(cfg.ctx) in cmd
    assert "-ngl" in cmd
    assert "--port" in cmd and "12345" in cmd
    assert "-ctk" in cmd and "-ctv" in cmd


def test_build_command_passes_extra_args(tmp_path):
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile,
                        extra_args=["--metrics", "--log-disable"])
    assert "--metrics" in cmd
    assert "--log-disable" in cmd


# ---------------------------------------------------------------------------
# GPU detection / iGPU filtering

def test_filter_drops_igpu_next_to_dgpu():
    """User's exact scenario: Intel iGPU + AMD RX 9070 XT.
    The iGPU must be ignored so the tuner doesn't underuse the dGPU."""
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(index=0, name="Intel(R) Graphics", vendor="intel",
                total_vram_mb=2048, free_vram_mb=1900),
        GPUInfo(index=1, name="AMD Radeon RX 9070 XT", vendor="amd",
                total_vram_mb=16 * 1024, free_vram_mb=int(15.2 * 1024)),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 1
    assert used[0].vendor == "amd"
    assert len(ignored) == 1
    assert ignored[0].vendor == "intel"


def test_filter_keeps_matched_dual_gpus():
    """Two equal dGPUs (e.g. 2x RTX 4090) must both be kept for tensor-split."""
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(index=0, name="RTX 4090", vendor="nvidia",
                total_vram_mb=24 * 1024, free_vram_mb=23 * 1024),
        GPUInfo(index=1, name="RTX 4090", vendor="nvidia",
                total_vram_mb=24 * 1024, free_vram_mb=23 * 1024),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 2 and not ignored


def test_vendor_inference():
    from hardware import _vendor_from_name

    assert _vendor_from_name("AMD Radeon RX 9070 XT") == "amd"
    assert _vendor_from_name("NVIDIA GeForce RTX 4090") == "nvidia"
    assert _vendor_from_name("Intel(R) UHD Graphics 770") == "intel"
    assert _vendor_from_name("Intel(R) Arc(TM) A770") == "intel"
    assert _vendor_from_name("some-mystery-card") == "unknown"


# ---------------------------------------------------------------------------
# llama-server resolver

def test_resolver_returns_input_when_nothing_matches(tmp_path, monkeypatch):
    """When the binary can't be found anywhere, the resolver echoes the
    original input — `launch()` then prints a clean 'not found' error
    rather than us silently swallowing the failure."""
    from auto_tuner import _resolve_server_binary

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    result = _resolve_server_binary("definitely-not-a-binary-anywhere")
    assert result == "definitely-not-a-binary-anywhere"


def test_resolver_finds_binary_in_sibling_llama_cpp(tmp_path, monkeypatch):
    """Simulate the user's layout: auto_tuner sits beside an ai-local/
    folder that contains the llama.cpp checkout."""
    from auto_tuner import _resolve_server_binary

    # Build the tree: tmp/Auto Tuner/, tmp/ai-local/llama.cpp/build/...
    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    server = (tmp_path / "ai-local" / "llama.cpp"
              / "build" / "bin" / "Release" / "llama-server.exe")
    server.parent.mkdir(parents=True)
    server.write_text("")

    monkeypatch.chdir(auto_dir)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

    resolved = _resolve_server_binary("llama-server")
    assert Path(resolved).resolve() == server.resolve(), (
        f"expected {server}, got {resolved}")


def test_resolver_distinguishes_between_llama_and_1b_llama(tmp_path,
                                                          monkeypatch):
    """The Bonsai-Ternary profile uses a relative path starting with the
    fork's directory name. The resolver must respect that and pick the
    1b_llama.cpp checkout, not the regular one sitting next to it."""
    from auto_tuner import _resolve_server_binary
    
    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    regular = (tmp_path / "ai-local" / "llama.cpp" / "build" / "bin"
               / "Release" / "llama-server.exe")
    bitnet = (tmp_path / "ai-local" / "1b_llama.cpp" / "build" / "bin"
              / "Release" / "llama-server.exe")
    for s in (regular, bitnet):
        s.parent.mkdir(parents=True)
        s.write_text("")

    monkeypatch.chdir(auto_dir)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

    # Default resolves to the regular fork
    res1 = _resolve_server_binary("llama-server")
    assert Path(res1).resolve() == regular.resolve()

    # Profile-style relative path must hit the BitNet fork
    res2 = _resolve_server_binary(
        "1b_llama.cpp/build/bin/Release/llama-server.exe")
    assert Path(res2).resolve() == bitnet.resolve()

def test_turbo_quant_mode_selection(tmp_path, monkeypatch):
    """Test that the turbo mode selection logic works (mocking input)."""
    from auto_tuner import main
    import sys
    from unittest.mock import patch

    # Mocking sys.argv to avoid command line arguments
    sys.argv = ["auto_tuner.py"]
    
    # We can't easily test the interactive input in a pure unit test
    # without complex mocking, but we can verify the logic if we
    # were to refactor main. For now, we ensure no crashes occur
    # when simulating different inputs.
    with patch('builtins.input', side_effect=["2", KeyboardInterrupt]):
        try:
            main([])
        except (KeyboardInterrupt, SystemExit):
            pass # Expected behavior for testing exit


# ---------------------------------------------------------------------------
# Profile schema (server_binary field)

def test_profile_supports_server_binary_field():
    """Bonsai-Ternary should declare its preferred server binary."""
    profiles = load_profiles(SETTINGS_DIR)
    by_name = {p.source_file: p for p in profiles}
    assert "bonsai-ternary.yaml" in by_name
    p = by_name["bonsai-ternary.yaml"]
    assert p.server_binary, "Bonsai-Ternary profile must set server_binary"
    assert "1b_llama" in p.server_binary.lower()


def test_ternary_bonsai_pattern_beats_regular_bonsai():
    """Longest-pattern-wins: a Ternary-Bonsai filename must match the
    BitNet profile, not the generic Bonsai one."""
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile("Ternary-Bonsai-8B-Q2_0", profiles)
    assert p.source_file is not None, "matched profile must come from a YAML file"
    assert "ternary" in (p.display_name + " " + p.source_file).lower()
    assert p.server_binary, "Ternary profile must override the server"