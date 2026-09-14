"""v5.4.8 regressions: GPU merge, KV floor, repeatable flags, process I/O, fallbacks."""

from __future__ import annotations

import http.client
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import hardware
import localization
import ocr_workflow
import scanner
import theme_manager
import tuner
from hardware import GPUInfo
from server_process import ServerProcess
from settings_loader import ModelProfile, load_profiles
from test_kv_policy import _model
from test_smoke import _fake_model_md, _fake_system

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# hardware.detect_system: two cards from one detector are never one card


def _isolate_detectors(monkeypatch, nvidia):
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: list(nvidia))
    monkeypatch.setattr(hardware, "_detect_amd_rocm", lambda: [])
    monkeypatch.setattr(hardware, "_detect_apple", lambda: [])
    monkeypatch.setattr(hardware, "_detect_windows_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_drm_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_other_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_get_pci_device_ids", lambda: {})
    monkeypatch.setattr(hardware, "_probe_llama_devices", lambda _binary: (False, []))


def test_substring_named_cards_from_one_detector_stay_separate(monkeypatch):
    ti = GPUInfo(0, "NVIDIA GeForce RTX 3060 Ti", "nvidia", 8192, 8000)
    plain = GPUInfo(1, "NVIDIA GeForce RTX 3060", "nvidia", 12288, 12000)
    _isolate_detectors(monkeypatch, [ti, plain])
    monkeypatch.setattr(hardware, "_assign_hip_indices", lambda *a, **k: None)
    system = hardware.detect_system("llama-server")
    names = sorted(g.name for g in system.gpus)
    assert names == ["NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 3060 Ti"]
    by_name = {g.name: g for g in system.gpus}
    assert by_name["NVIDIA GeForce RTX 3060 Ti"].total_vram_mb == 8192
    assert by_name["NVIDIA GeForce RTX 3060"].total_vram_mb == 12288


def test_cross_detector_duplicates_still_merge(monkeypatch):
    rocm = GPUInfo(
        0, "AMD Radeon RX 9070 XT", "amd", 16304, 15000, pci_device_id=0x7550
    )
    drm = GPUInfo(0, "Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]", "amd", 0, 0)
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: [])
    monkeypatch.setattr(hardware, "_detect_amd_rocm", lambda: [rocm])
    monkeypatch.setattr(hardware, "_detect_apple", lambda: [])
    monkeypatch.setattr(hardware, "_detect_windows_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_drm_gpus", lambda **kwargs: [drm])
    monkeypatch.setattr(hardware, "_detect_linux_other_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_get_pci_device_ids", lambda: {})
    monkeypatch.setattr(hardware, "_probe_llama_devices", lambda _binary: (False, []))
    monkeypatch.setattr(hardware, "_assign_hip_indices", lambda *a, **k: None)
    system = hardware.detect_system("llama-server")
    assert len(system.gpus) == 1
    assert system.gpus[0].total_vram_mb == 16304


def test_different_pci_ids_veto_the_name_merge(monkeypatch):
    xt = GPUInfo(0, "AMD Radeon RX 9070 XT", "amd", 16304, 15000, pci_device_id=0x7550)
    plain = GPUInfo(0, "AMD Radeon RX 9070", "amd", 16304, 15000, pci_device_id=0x7551)
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: [])
    monkeypatch.setattr(hardware, "_detect_amd_rocm", lambda: [xt])
    monkeypatch.setattr(hardware, "_detect_apple", lambda: [])
    monkeypatch.setattr(hardware, "_detect_windows_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_drm_gpus", lambda **kwargs: [plain])
    monkeypatch.setattr(hardware, "_detect_linux_other_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_get_pci_device_ids", lambda: {})
    monkeypatch.setattr(hardware, "_probe_llama_devices", lambda _binary: (False, []))
    monkeypatch.setattr(hardware, "_assign_hip_indices", lambda *a, **k: None)
    system = hardware.detect_system("llama-server")
    assert len(system.gpus) == 2


# ---------------------------------------------------------------------------
# tuner: an exhausted VRAM budget must clamp, not fall back to the 32k floor


def _dense_14b(tmp_path, size_gb):
    return _fake_model_md(
        tmp_path,
        f"llama-{size_gb}",
        size_gb,
        {
            "general.architecture": "llama",
            "llama.block_count": 40,
            "llama.context_length": 131072,
            "llama.embedding_length": 5120,
            "llama.attention.head_count": 40,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128,
            "llama.attention.value_length": 128,
        },
    )


@pytest.mark.parametrize("user_ctx", [None, 131072])
def test_exhausted_kv_budget_clamps_to_floor(tmp_path, user_ctx):
    model = _dense_14b(tmp_path, 14.9)
    system = _fake_system(vram_total=16, vram_free=15.2, ram_total=64, ram_free=48)
    profile = ModelProfile(display_name="dense", max_context=131072)
    cfg = tuner.compute_config(model, system, profile, user_ctx=user_ctx, force_ngl=999)
    # Weights alone leave < 0.6 GiB of the card: the known per-token cost
    # yields max_fit_ctx == 0, which is a clamp, not "unknown".
    assert cfg.ctx == 2048, cfg
    if user_ctx is not None:
        assert "clamped" in (cfg.warning or "").lower()


# ---------------------------------------------------------------------------
# tuner: repeatable value flags in Extras keep every value, prune with it


def _cfg_with_extras(tmp_path, extras):
    model = _model(tmp_path)
    profile = ModelProfile(display_name="extras", max_context=32768)
    cfg = tuner.compute_config(
        model, _fake_system(vram_total=16), profile, user_ctx=4096
    )
    cfg.extra_cli_flags = extras
    return model, profile, cfg


def test_repeated_override_tensor_keeps_both_values(tmp_path):
    model, profile, cfg = _cfg_with_extras(
        tmp_path,
        [
            "--override-tensor",
            r"blk\.(0|1)\.ffn_.*=CPU",
            "--override-tensor",
            r"blk\.2\.ffn_.*=CPU",
        ],
    )
    cmd = tuner.build_command(model, cfg, profile)
    positions = [i for i, tok in enumerate(cmd) if tok == "--override-tensor"]
    assert len(positions) == 2
    assert cmd[positions[0] + 1] == r"blk\.(0|1)\.ffn_.*=CPU"
    assert cmd[positions[1] + 1] == r"blk\.2\.ffn_.*=CPU"
    # No value token may be left behind as a positional argument.
    assert cmd[-1] == r"blk\.2\.ffn_.*=CPU" and cmd[-2] == "--override-tensor"


def test_identical_repeated_lora_is_deduplicated(tmp_path):
    model, profile, cfg = _cfg_with_extras(
        tmp_path,
        ["--lora", "adapter.gguf", "--lora", "adapter.gguf", "--lora", "other.gguf"],
    )
    cmd = tuner.build_command(model, cfg, profile)
    loras = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--lora"]
    assert loras == ["adapter.gguf", "other.gguf"]


def test_pruning_an_unsupported_override_tensor_drops_its_value():
    cmd = ["srv", "-m", "m.gguf", "--override-tensor", "a=CPU", "-c", "4096"]
    kept, removed = tuner._filter_command_for_supported_flags(cmd, {"-m", "-c"})
    assert kept == ["srv", "-m", "m.gguf", "-c", "4096"]
    assert "a=CPU" not in kept and removed


# ---------------------------------------------------------------------------
# server_process: UTF-8 output must not end the reader thread


def test_server_process_decodes_utf8_output():
    child = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'BOS <\\xe2\\x96\\x81sentence> \\x81\\n'); "
        "sys.stdout.flush()",
    ]
    proc = ServerProcess(child)
    proc.start()
    assert proc.wait(timeout=30) == 0
    deadline = time.monotonic() + 5
    lines: list[str] = []
    while time.monotonic() < deadline and not lines:
        lines = proc.get_logs()
        time.sleep(0.05)
    assert lines and "▁sentence" in lines[0]
    proc.stop()
    assert proc.proc is None  # graceful exit still resets the wrapper


# ---------------------------------------------------------------------------
# ocr_workflow: cancel aborts an in-flight socket without blocking


@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="macOS does not wake a recv blocked in another thread on shutdown/close; "
    "cancel falls back to the request timeout there",
    strict=False,
)
def test_abort_connection_wakes_a_blocked_reader():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted: list[socket.socket] = []

    def serve():
        conn, _ = listener.accept()
        accepted.append(conn)
        time.sleep(30)  # never answer

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    outcome: dict[str, object] = {}

    def reader():
        try:
            connection.request("GET", "/")
            connection.getresponse()
            outcome["result"] = "answered"
        except (OSError, http.client.HTTPException) as exc:
            outcome["result"] = type(exc).__name__

    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    time.sleep(0.5)
    started = time.monotonic()
    ocr_workflow._abort_connection(connection)
    worker.join(timeout=5)
    elapsed = time.monotonic() - started
    assert not worker.is_alive(), "reader still blocked after abort"
    assert elapsed < 3, elapsed
    assert outcome.get("result") != "answered"
    listener.close()
    for conn in accepted:
        conn.close()


def test_convert_office_cancel_terminates_soffice(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(
        ocr_workflow, "_terminate_process", lambda proc: calls.append("terminated")
    )

    class FakeProc:
        returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("soffice", timeout)

    monkeypatch.setattr(ocr_workflow, "find_libreoffice", lambda: "soffice")
    monkeypatch.setattr(ocr_workflow.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(ocr_workflow.shutil, "rmtree", lambda *a, **k: None)
    runner = ocr_workflow.OcrJobRunner.__new__(ocr_workflow.OcrJobRunner)
    runner.cancel_event = threading.Event()
    runner._state_lock = threading.Lock()
    runner._active_process = None
    runner._emit = lambda *a, **k: None
    runner._check_cancelled = lambda: (_ for _ in ()).throw(
        ocr_workflow.OcrCancelled("cancelled")
    )
    source = tmp_path / "doc.docx"
    source.write_bytes(b"x")
    with pytest.raises(ocr_workflow.OcrCancelled):
        runner._convert_office(source, tmp_path / "converted")
    assert calls == ["terminated"]


# ---------------------------------------------------------------------------
# scanner: giant string arrays are skipped by seeking


def _write_gguf_with_vocab(path: Path, n_tokens: int) -> None:
    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 0)  # n_tensors
    out += struct.pack("<Q", 3)  # n_kv
    out += string("general.architecture") + struct.pack("<I", 8) + string("llama")
    out += string("tokenizer.ggml.tokens") + struct.pack("<I", 9)
    out += struct.pack("<I", 8) + struct.pack("<Q", n_tokens)
    for i in range(n_tokens):
        out += string(f"tok▁{i}")
    out += string("llama.block_count") + struct.pack("<I", 4) + struct.pack("<I", 12)
    path.write_bytes(bytes(out))


def test_string_array_skip_keeps_following_keys(tmp_path):
    path = tmp_path / "vocab.gguf"
    _write_gguf_with_vocab(path, 1000)
    md = scanner.read_gguf_metadata(path)
    assert md["general.architecture"] == "llama"
    assert md["llama.block_count"] == 12
    assert not md.get("tokenizer.ggml.tokens")


# ---------------------------------------------------------------------------
# settings_loader: one malformed profile must not take the others down


def test_malformed_profile_is_skipped_not_fatal(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text("display_name: Good\npatterns: ['good']\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text("display_name: Bad\nmax_context:\n", encoding="utf-8")
    profiles = load_profiles(tmp_path)
    assert [p.display_name for p in profiles] == ["Good"]
    assert "bad.yaml" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# localization / theme_manager fallbacks


def test_missing_builtin_english_pack_degrades_to_source_text(tmp_path):
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    user = tmp_path / "user"
    manager = localization.LanguageManager(builtin_dir=builtin, user_dir=user)
    assert manager.current_id == localization.DEFAULT_LANGUAGE_ID
    assert manager.translate("Start server") == "Start server"
    assert any("falling back" in e for e in manager.errors)


def test_user_theme_save_survives_missing_hard_links(tmp_path, monkeypatch):
    manager = theme_manager.ThemeManager(
        ROOT / "assets" / "themes", tmp_path / "themes"
    )
    base = manager.get(theme_manager.SYSTEM_THEME_ID)
    saved = theme_manager.ThemeDefinition(
        "no-hardlink", "No hard link", "test", dict(base.colors), source="user"
    )

    def refuse_link(*args, **kwargs):
        raise OSError(1, "Incorrect function")

    monkeypatch.setattr(theme_manager.os, "link", refuse_link)
    target = manager.save_user_theme(saved)
    assert target.is_file()
    assert manager.get("user:no-hardlink").name == "No hard link"
    with pytest.raises(FileExistsError):
        manager.save_user_theme(saved)


# ---------------------------------------------------------------------------
# hardware: the DXGI PowerShell fallback is lazy


@pytest.mark.skipif(os.name != "nt", reason="Windows registry detector")
def test_windows_detector_skips_dxgi_when_wmi_covers_every_card(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        hardware, "_get_vram_from_registry", lambda: {"Card A": 16 * 1024**3}
    )
    monkeypatch.setattr(
        hardware, "_get_gpu_vram_used_via_wmi", lambda: {"card a": 1024.0}
    )
    monkeypatch.setattr(
        hardware,
        "_get_gpu_vram_via_dxgi_powershell",
        lambda: calls.append("dxgi") or {},
    )
    monkeypatch.setattr(
        hardware, "_get_gpu_vram_free_via_wmi", lambda: calls.append("free") or {}
    )
    monkeypatch.setattr(hardware, "_get_nvidia_gpu_utilization", lambda: {})
    gpus = hardware._detect_windows_gpus()
    assert [g.free_vram_mb for g in gpus] == [16 * 1024 - 1024]
    assert calls == []
    monkeypatch.setattr(hardware, "_get_gpu_vram_used_via_wmi", lambda: {})
    gpus = hardware._detect_windows_gpus()
    assert calls == ["dxgi", "free"]


# ---------------------------------------------------------------------------
# qt_launcher / control_api: shutdown and listener hardening


def test_terminal_process_stop_is_awaited_before_exit(tmp_path):
    qt_launcher = pytest.importorskip("qt_launcher")
    ready = tmp_path / "ready"
    # The child ignores SIGTERM (POSIX) and never sees CTRL_BREAK (Windows,
    # separate console), so stop() has to run its 10 s kill escalation. The
    # marker file guarantees the handler is installed before stop() runs.
    child = [
        sys.executable,
        "-c",
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)",
        str(ready),
    ]
    proc = qt_launcher._TerminalProcess(child)
    proc.start()
    deadline = time.monotonic() + 20
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists() and proc.is_running()
    proc.stop()
    assert proc in qt_launcher._TerminalProcess._pending_stops
    pending = qt_launcher._TerminalProcess.wait_pending_stops(20.0)
    assert pending == []
    assert proc.has_stopped()


@pytest.mark.skipif(os.name != "nt", reason="Windows exclusive-bind semantics")
def test_control_listener_refuses_a_second_bind_on_windows():
    import control_api

    first = control_api._ControlHTTPServer(("127.0.0.1", 0), api=None)
    try:
        port = first.server_address[1]
        with pytest.raises(OSError):
            control_api._ControlHTTPServer(("127.0.0.1", port), api=None)
    finally:
        first.server_close()


def test_scalar_patterns_and_non_mapping_roots(tmp_path):
    (tmp_path / "scalar.yaml").write_text(
        "display_name: Scalar\npatterns: qwen\narch_fallback: qwen3\n", encoding="utf-8"
    )
    (tmp_path / "list.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    profiles = load_profiles(tmp_path)
    assert [p.display_name for p in profiles] == ["Scalar"]
    assert profiles[0].patterns == ["qwen"]
    assert profiles[0].arch_fallback == ["qwen3"]


def test_detect_system_is_serialized(monkeypatch):
    order: list[str] = []
    real = hardware._detect_system_locked

    def slow(binary=None):
        order.append("enter")
        time.sleep(0.3)
        order.append("leave")
        return real(binary)

    monkeypatch.setattr(hardware, "_detect_system_locked", slow)
    _isolate_detectors(monkeypatch, [])
    monkeypatch.setattr(hardware, "_assign_hip_indices", lambda *a, **k: None)
    threads = [
        threading.Thread(target=hardware.detect_system, args=("llama-server",))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert order == ["enter", "leave", "enter", "leave"]
