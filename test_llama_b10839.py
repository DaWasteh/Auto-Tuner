"""Portable b10839 model and optional-flag compatibility checks."""

import json
from pathlib import Path

import pytest

from settings_loader import load_profiles, match_profile
from test_smoke import _fake_diffusion_config, _fake_diffusion_model
from tuner import (
    build_diffusion_command,
    _filter_command_for_supported_flags as filter_unsupported_optional_flags,
)

ROOT = Path(__file__).parent


def test_spark25_mainline_profile():
    profiles = load_profiles(ROOT / "settings")
    for name in ("Spark-X2.5-1.7B-Q8_0", "unknown-merge"):
        profile = match_profile(name, profiles, "spark2_5")
        assert profile.source_file == "spark2_5.yaml"
        assert profile.min_llama_build == 10828
        assert profile.max_context == 1048576
        assert profile.recommended_kv_quant == "q8_0"
        assert profile.sampling["chat"]["temperature"] == 1.0
        assert profile.sampling["chat"]["top_k"] == 0
        assert "--jinja" in profile.extra_args


def test_hy4_metadata_disambiguates_identical_filenames():
    profiles = load_profiles(ROOT / "settings")
    name = "Hy4-preview-Q4_K_M.gguf"
    stock = match_profile(name, profiles, "hy_v4")
    fork = match_profile(name, profiles, "hyv4")
    assert stock.source_file == "hy4-mainline.yaml"
    assert stock.min_llama_build == 10813
    assert not stock.required_runtime_markers
    assert fork.source_file == "hy4-preview.yaml"
    assert fork.required_runtime_markers == ["hyv4"]
    assert match_profile(name, profiles).source_file == "hy4-preview.yaml"


@pytest.mark.parametrize("flag", ["--log-jsonl", "--no-log-jsonl"])
def test_jsonl_boolean_flags_supported_and_old_build_pruned(flag):
    flags = set(
        json.loads((ROOT / "docs/llama-b10839-server-flags.json").read_text())["flags"]
    )
    assert flag in flags
    cmd = ["llama-server", "-m", "model.gguf", flag, "--port", "1234"]
    kept, removed = filter_unsupported_optional_flags(cmd, flags)
    assert kept == cmd and not removed
    kept, removed = filter_unsupported_optional_flags(cmd, {"-m", "--port"})
    assert kept == ["llama-server", "-m", "model.gguf", "--port", "1234"]
    assert flag in removed


@pytest.mark.parametrize("method", ["simple", "map-k"])
@pytest.mark.parametrize("suffix", ["size-n", "size-m", "min-hits"])
def test_all_ngram_value_options_prune_their_value(method, suffix):
    flag = f"--spec-ngram-{method}-{suffix}"
    cmd = ["llama-server", "-m", "model.gguf", flag, "8", "--port", "1234"]
    kept, removed = filter_unsupported_optional_flags(cmd, {"-m", "--port"})
    assert kept == ["llama-server", "-m", "model.gguf", "--port", "1234"]
    assert removed == [f"{flag} 8"]


def test_cache_controls_are_not_silently_pruned_for_old_runtimes():
    cmd = [
        "llama-server",
        "-m",
        "model.gguf",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-fa",
        "on",
    ]
    kept, removed = filter_unsupported_optional_flags(cmd, {"-m"})
    assert kept == cmd  # unsupported runtime must reject, not allocate default F16
    assert not removed


def test_diffusion_cli_enacts_planned_kv_and_fa():
    profile = match_profile("unknown", load_profiles(ROOT / "settings"))
    cfg = _fake_diffusion_config()
    cfg.cache_k = cfg.cache_v = "q8_0"
    cfg.flash_attn = True
    cmd = build_diffusion_command(_fake_diffusion_model(), cfg, profile)
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"
    assert cmd[cmd.index("-fa") + 1] == "on"
