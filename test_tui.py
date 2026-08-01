"""Focused tests for AutoTuner's dependency-free terminal UI."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import auto_tuner
from autotuner_version import VERSION
from hardware import GPUInfo, SystemInfo
from scanner import ModelEntry


def _entry(*, name: str = "Example-MTP", draft: bool = False) -> ModelEntry:
    return ModelEntry(
        path=Path("example.gguf"),
        name=name,
        group="Examples",
        size_bytes=1024,
        draft=Path("draft.gguf") if draft else None,
    )


def test_banner_and_version_are_shared(capsys) -> None:
    auto_tuner._configure_renderer(auto_tuner._parse_args(["--plain"]))
    auto_tuner._print_banner()
    assert f"AutoTuner v{VERSION}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exited:
        auto_tuner.main(["--version"])
    assert exited.value.code == 0
    assert f"AutoTuner v{VERSION}" in capsys.readouterr().out


def test_plain_help_is_ascii_and_documents_passthrough(capsys) -> None:
    with pytest.raises(SystemExit) as exited:
        auto_tuner._parse_args(["--plain", "--help"])
    assert exited.value.code == 0
    output = capsys.readouterr().out
    assert output.isascii()
    assert "arguments after --" in output


def test_parser_supports_tui_options_and_real_passthrough() -> None:
    args = auto_tuner._parse_args(
        [
            "--model",
            "Example",
            "--mode",
            "coding",
            "--cache-ram-mib",
            "4096",
            "--debug-category",
            "config",
            "--plain",
            "--non-interactive",
            "--",
            "--threads",
            "8",
        ]
    )
    assert args.mode == "coding"
    assert args.cache_ram_mib == 4096
    assert args.debug_category == ["config"]
    assert args.plain and args.non_interactive
    assert args.passthrough == ["--threads", "8"]
    assert auto_tuner._parse_args([]).cache_ram_mib is None
    assert auto_tuner._server_was_explicit(["--server", "custom"])
    assert auto_tuner._server_was_explicit(["--server=custom"])
    assert not auto_tuner._server_was_explicit(["--", "--server", "custom"])


def test_mode_and_cache_defaults_share_gui_settings(monkeypatch) -> None:
    monkeypatch.setattr(auto_tuner.app_settings, "get_mode", lambda: "coding")
    monkeypatch.setattr(
        auto_tuner.app_settings, "get_prompt_cache_ram_mib", lambda: 3072
    )
    args = auto_tuner._parse_args([])
    assert auto_tuner._effective_mode(args) == "coding"
    assert auto_tuner._effective_prompt_cache_mib(args) == 3072
    args = auto_tuner._parse_args(["--mode", "chat", "--cache-ram-mib", "512"])
    assert auto_tuner._effective_mode(args) == "chat"
    assert auto_tuner._effective_prompt_cache_mib(args) == 512


def test_plain_renderer_is_ascii_and_compact(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        auto_tuner.shutil,
        "get_terminal_size",
        lambda fallback=(80, 24): types.SimpleNamespace(columns=30),
    )
    auto_tuner._configure_renderer(auto_tuner._parse_args(["--plain"]))
    entry = _entry(name="Long 模型 name with a very long suffix")
    auto_tuner._print_banner()
    auto_tuner._print_menu({"Very Long Group": [entry]})
    system = SystemInfo(
        os_name="Test — OS",
        cpu_name="CPU 模型",
        cpu_cores_physical=1,
        cpu_cores_logical=2,
        total_ram_gb=8,
        free_ram_gb=4,
        ignored_gpus=[
            GPUInfo(
                index=1, name="Ignored ⚠", vendor="AMD", total_vram_mb=1, free_vram_mb=1
            )
        ],
    )
    auto_tuner._print_system(system)
    output = capsys.readouterr().out
    assert output.isascii()
    assert all(len(line) <= 30 for line in output.splitlines())
    assert "..." in output


def test_windows_ansi_gate_and_cp65001(monkeypatch) -> None:
    class Stream:
        encoding = "cp65001"

        def isatty(self):
            return True

    monkeypatch.setattr(auto_tuner.sys, "stdout", Stream())
    monkeypatch.setattr(auto_tuner.os, "name", "nt")
    monkeypatch.setattr(auto_tuner, "_windows_vt_enabled", lambda: False)
    renderer = auto_tuner._TerminalRenderer()
    assert not renderer.plain
    assert not renderer.color


def test_non_interactive_features_include_embedded_mtp() -> None:
    model = _entry()
    model.mmproj = Path("vision.gguf")
    model.metadata["tokenizer.chat_template"] = "<think>"
    vision, draft, thinking, ngram, effective = auto_tuner._ask_interactive_features(
        model, None, Path("settings"), non_interactive=True, draft_available=True
    )
    assert vision and draft and thinking and not ngram
    assert effective is None


def test_disabled_thinking_skips_interactive_prompt(monkeypatch) -> None:
    model = _entry()
    model.metadata["tokenizer.chat_template"] = "<think>"
    monkeypatch.setattr(
        auto_tuner, "_confirm", lambda *_args, **_kwargs: pytest.fail("must not prompt")
    )
    _vision, _draft, thinking, ngram, _effective = auto_tuner._ask_interactive_features(
        model,
        None,
        Path("settings"),
        force_ngram=True,
        draft_available=False,
        thinking_available=False,
    )
    assert not thinking
    assert ngram


def test_non_interactive_without_model_does_not_probe_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        auto_tuner, "detect_system", lambda: pytest.fail("must not run")
    )
    assert auto_tuner.main(["--non-interactive", "--plain"]) == 2


def test_diagnose_skips_irrelevant_fork_discovery(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        auto_tuner,
        "_discover_llama_forks",
        lambda: pytest.fail("diagnostics do not need a server fork"),
    )
    monkeypatch.setattr(
        auto_tuner,
        "detect_system",
        lambda: SystemInfo(
            os_name="Test",
            cpu_name="CPU",
            cpu_cores_physical=1,
            cpu_cores_logical=1,
            total_ram_gb=8,
            free_ram_gb=8,
        ),
    )
    monkeypatch.setattr(auto_tuner, "scan_models", lambda _: [])
    assert (
        auto_tuner.main(["--diagnose", "--models-path", str(tmp_path), "--plain"]) == 2
    )


def test_noninteractive_and_yes_never_call_confirm(monkeypatch, tmp_path) -> None:
    entry = _entry(name="Only")
    profile = types.SimpleNamespace(
        draft_max=0,
        display_name="Default",
        source_file="",
        notes="",
        runner="",
        server_binary="",
        performance_target="",
    )
    system = SystemInfo(
        os_name="Test",
        cpu_name="CPU",
        cpu_cores_physical=1,
        cpu_cores_logical=1,
        total_ram_gb=8,
        free_ram_gb=8,
    )
    monkeypatch.setattr(auto_tuner, "_discover_llama_forks", lambda: [])
    monkeypatch.setattr(auto_tuner, "detect_system", lambda: system)
    monkeypatch.setattr(auto_tuner, "scan_models", lambda _: [entry])
    monkeypatch.setattr(auto_tuner, "load_profiles", lambda _: [profile])
    monkeypatch.setattr(auto_tuner, "match_profile", lambda *_: profile)
    monkeypatch.setattr(
        auto_tuner,
        "compute_config",
        lambda *a, **k: types.SimpleNamespace(
            full_offload=False,
            ngl=0,
            performance_target=types.SimpleNamespace(name="balanced"),
            ctx=1024,
            cache_k="q8_0",
            cache_v="q8_0",
            threads=1,
            batch_threads=1,
            batch=1,
            ubatch=1,
            flash_attn=False,
            mlock=False,
            no_mmap=False,
            numa=None,
            no_context_shift=False,
            tensor_split=None,
            main_gpu=None,
            sampling={},
            estimated_model_vram_gb=0,
            estimated_model_ram_gb=0,
            vision_vram_gb=0,
            vision_ram_gb=0,
            runtime_vram_overhead_gb=0,
            estimated_kv_gb=0,
            prompt_cache_ram_gb=0,
            env_overrides={},
        ),
    )
    monkeypatch.setattr(auto_tuner, "_resolve_server_binary", lambda _: "server")
    monkeypatch.setattr(
        auto_tuner, "prepare_command_for_binary", lambda command: (command, [])
    )
    monkeypatch.setattr(auto_tuner, "build_command", lambda **_: ["server"])
    monkeypatch.setattr(auto_tuner, "launch", lambda *a, **k: 0)
    monkeypatch.setattr(
        auto_tuner, "_confirm", lambda *a, **k: pytest.fail("must not prompt")
    )
    for flag in ("--non-interactive", "--yes"):
        assert (
            auto_tuner.main(
                [
                    flag,
                    "--model",
                    "Only",
                    "--server",
                    "server",
                    "--models-path",
                    str(tmp_path),
                    "--plain",
                ]
            )
            == 0
        )
