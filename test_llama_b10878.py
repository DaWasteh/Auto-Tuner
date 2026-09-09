"""Pinned b10878 removed-option and lazy residency regressions."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles
from test_llama_b10863 import _lazy_model_config

ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize("flag,mode", tuner._LEGACY_LOAD_MODES.items())
def test_removed_load_switches_migrate_before_pruning(monkeypatch, flag, mode):
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"-m", "--load-mode"}
    )
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10878)
    cmd, notes = tuner.prepare_command_for_binary(["server", "-m", "test.gguf", flag])
    assert cmd == ["server", "-m", "test.gguf", "--load-mode", mode]
    assert any(flag + " ->" in note for note in notes)


@pytest.mark.parametrize("flag", tuner._LEGACY_LOAD_MODES)
def test_removed_extra_switch_cannot_override_explicit_load_mode(tmp_path, flag):
    model, cfg, profile = _lazy_model_config(tmp_path)
    cfg.load_mode = "mmap+mlock"
    cmd = tuner.build_command(model, cfg, profile, extra_args=[flag])
    assert flag not in cmd
    assert cmd.count("--load-mode") == 1
    assert cmd[cmd.index("--load-mode") + 1] == "mmap+mlock"


def test_load_migration_preserves_values_and_order():
    args = ["--chat-template", "--mmap", "--no-mmap", "--load-mode=dio", "--mmap"]
    normalized, notes = tuner._normalize_legacy_load_flags(args)
    assert normalized == ["--chat-template", "--mmap", "--load-mode", "mmap"]
    assert len(notes) == 2


@pytest.mark.parametrize("flag", tuner._LEGACY_LOAD_MODES)
def test_legacy_boolean_inline_is_not_silently_reinterpreted(flag):
    with pytest.raises(ValueError, match="value-less switch"):
        tuner._normalize_legacy_load_flags([flag + "=false"])


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("legacy_mode", ["auto", "on"])
def test_giant_table_never_relies_on_device_dependent_lazy_auto(
    tmp_path, unified, legacy_mode
):
    model, cfg, profile = _lazy_model_config(tmp_path)
    cfg.unified_memory = unified
    cfg.extra_cli_flags = ["--tensor-read-lazy", legacy_mode]
    cmd = tuner.build_command(model, cfg, profile)
    assert cmd.count("--lazy-mode") == 1
    assert cmd[cmd.index("--lazy-mode") + 1] == "on"
    assert "--tensor-read-lazy" not in cmd


def test_old_runtime_keeps_supported_legacy_switch(monkeypatch):
    monkeypatch.setattr(tuner, "_probe_supported_flags", lambda _: {"-m", "--mlock"})
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 9000)
    original = ["server", "-m", "model.gguf", "--mlock"]
    assert tuner.prepare_command_for_binary(original) == (original, [])


def test_legacy_extras_keep_upstream_last_wins(tmp_path):
    model, cfg, profile = _lazy_model_config(tmp_path)
    cmd = tuner.build_command(model, cfg, profile, extra_args=["--mmap", "--no-mmap"])
    assert cmd.count("--load-mode") == 1
    assert cmd[cmd.index("--load-mode") + 1] == "none"


@pytest.mark.parametrize(
    "extras", [["--mlock"], ["--load-mode", "mmap+mlock"], ["-lm=mlock"]]
)
@pytest.mark.parametrize("source", ["config", "profile", "extra"])
def test_free_form_locking_cannot_bypass_safety_veto(tmp_path, extras, source):
    model, cfg, profile = _lazy_model_config(tmp_path)
    kwargs = {}
    if source == "config":
        cfg.extra_cli_flags = extras
    elif source == "profile":
        profile.extra_args = extras
    else:
        kwargs["extra_args"] = extras
    with pytest.raises(ValueError, match="bypasses safety checks"):
        tuner.build_command(model, cfg, profile, **kwargs)


def test_old_mlock_does_not_acquire_non_mmap_semantics(monkeypatch):
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"-m", "--mlock", "--load-mode"}
    )
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10114)
    cmd = ["server", "-m", "model.gguf", "--mlock"]
    assert tuner.prepare_command_for_binary(cmd) == (cmd, [])


def test_failed_help_is_not_a_capability_list(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        tuner.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, "Usage: -m --help"),
    )
    # Bypass the lru cache to isolate this failed probe from other tests.
    assert tuner._probe_supported_flags_cached.__wrapped__("server", 0, 0) is None


def test_partial_help_cannot_adapt_load_modes(monkeypatch):
    monkeypatch.setattr(tuner, "_probe_supported_flags", lambda _: {"--load-mode"})
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10000)
    cmd = ["server", "-m", "model.gguf", "--load-mode", "mlock"]
    assert tuner.prepare_command_for_binary(cmd) == (cmd, [])


def test_explicit_lazy_on_survives_legacy_option_rename():
    cmd, _ = tuner._adapt_lazy_mode_for_binary(
        ["server", "--lazy-mode", "on", "-m", "model.gguf"],
        {"--tensor-read-lazy", "-m"},
    )
    assert cmd[1:3] == ["--tensor-read-lazy", "on"]


def test_b10878_manifest_removes_old_loading_switches():
    flags = set(
        json.loads((ROOT / "docs/llama-b10878-server-flags.json").read_text())["flags"]
    )
    assert not (flags & tuner._LEGACY_LOAD_MODES.keys())
    assert {"--load-mode", "--lazy-mode", "--cache-type-k", "--cache-type-v"} <= flags
    for profile in load_profiles(ROOT / "settings"):
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)
