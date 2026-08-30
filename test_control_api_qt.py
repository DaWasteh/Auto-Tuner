from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import app_settings
from control_api import ControlApiError, ControlRequest
from hardware import GPUInfo, SystemInfo
from scanner import ModelEntry
from settings_loader import load_profiles


ROOT = Path(__file__).resolve().parent
SETTINGS_DIR = ROOT / "settings"


def _entry(path: Path, name: str, metadata=None) -> ModelEntry:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    return ModelEntry(
        path=path,
        name=name,
        group=str(path.parent),
        size_bytes=1024**3,
        metadata=dict(metadata or {}),
    )


def _system() -> SystemInfo:
    return SystemInfo(
        os_name="test",
        cpu_name="test",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=64,
        free_ram_gb=48,
        gpus=[
            GPUInfo(
                index=0,
                name="Test GPU",
                vendor="amd",
                total_vram_mb=16 * 1024,
                free_vram_mb=15 * 1024,
            )
        ],
    )


def test_qt_catalogue_uses_path_stable_ids_capabilities_and_routeability(
    tmp_path, monkeypatch
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_launcher = pytest.importorskip("qt_launcher")
    monkeypatch.setattr(
        app_settings, "_settings_file", lambda: tmp_path / "settings.json"
    )
    first = _entry(tmp_path / "a" / "Same Model.gguf", "Same Model")
    first.mmproj = tmp_path / "a" / "mmproj.gguf"
    second = _entry(tmp_path / "b" / "Same Model.gguf", "Same Model")
    diffusion = _entry(
        tmp_path / "d" / "Dream-7B.gguf",
        "Dream-7B",
        {"general.architecture": "dream"},
    )
    drafter = _entry(
        tmp_path / "m" / "draft.gguf",
        "Gemma draft",
        {"general.architecture": "gemma4-assistant"},
    )
    profiles = load_profiles(SETTINGS_DIR)

    models = qt_launcher._control_api_catalogue(
        [first, second, diffusion, drafter], profiles
    )
    by_path = {Path(model.path): model for model in models}
    assert by_path[first.path].id != by_path[second.path].id
    assert by_path[first.path].id == qt_launcher._control_model_id(first)
    assert by_path[first.path].input_types == ("text", "image")
    assert by_path[first.path].runnable is True
    assert by_path[diffusion.path].runnable is False
    assert "single-shot" in by_path[diffusion.path].unavailable_reason
    assert by_path[drafter.path].runnable is False
    assert "draft" in by_path[drafter.path].unavailable_reason.casefold()

    app_settings.set_mmproj_selection(
        first.name, app_settings.MMPROJ_NONE_SENTINEL
    )
    disabled = qt_launcher._control_api_catalogue([first], profiles)[0]
    assert disabled.input_types == ("text",)


def test_qt_control_requests_wait_for_health_and_stop_only_managed_server(
    tmp_path, monkeypatch
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    assert app is qt_widgets.QApplication.instance()
    monkeypatch.setattr(
        app_settings, "_settings_file", lambda: tmp_path / "settings.json"
    )
    model = _entry(tmp_path / "models" / "Qwen.gguf", "Qwen")
    window = qt_launcher.MainWindow(
        tmp_path / "models", SETTINGS_DIR, start_background=False
    )
    window._system = _system()
    window._all_entries = [model]
    window._profiles = load_profiles(SETTINGS_DIR)
    window._refresh_control_api_catalogue()
    model_id = next(iter(window._control_model_paths))

    monkeypatch.setattr(
        window, "_show_config", lambda entry: setattr(window, "_current_entry", entry)
    )
    monkeypatch.setattr(window, "_select_model_path", lambda _path, _view: None)

    launched: list[dict] = []

    class FakeProcess:
        proc = None

        def is_running(self) -> bool:
            return True

        def stop(self) -> None:
            return

        def returncode(self):
            return None

    def launch(*, interactive: bool = True):
        assert interactive is False
        record = {
            "id": 100 + len(launched),
            "proc": FakeProcess(),
            "port": 1234 + len(launched),
            "base_url": f"http://127.0.0.1:{1234 + len(launched)}",
            "client_base_url": f"http://127.0.0.1:{1234 + len(launched)}",
            "ready": False,
            "model": "Qwen",
            "alias": "Qwen alias",
            "command": ["llama-server", "--api-key", "backend-secret"],
        }
        launched.append(record)
        window._servers.append(record)
        return record

    monkeypatch.setattr(window, "_launch_server", launch)
    request = ControlRequest("switch", model_id, timeout_s=1)
    window._handle_control_request(request)
    assert launched and not request._event.is_set()
    launched[0]["ready"] = True
    window._complete_control_record(launched[0])
    assert request.wait() == {
        "backend_url": "http://127.0.0.1:1234",
        "backend_api_key": "backend-secret",
        "alias": "Qwen alias",
    }

    again = ControlRequest("switch", model_id, timeout_s=1)
    window._handle_control_request(again)
    assert again.wait()["backend_url"] == "http://127.0.0.1:1234"
    assert len(launched) == 1

    unrelated = {
        "id": 999,
        "proc": FakeProcess(),
        "port": 9999,
        "base_url": "http://127.0.0.1:9999",
        "ready": True,
        "model": "Manual server",
    }
    window._servers.append(unrelated)
    window._refresh_server_combo()
    stop = ControlRequest("stop", timeout_s=1)
    window._handle_control_request(stop)
    assert stop.wait() == {"status": "stopped"}
    assert unrelated in window._servers
    assert launched[0] not in window._servers
    window._servers.remove(unrelated)
    window.close()


def test_qt_control_transition_waits_for_real_process_exit(tmp_path, monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    qt_core = pytest.importorskip("PyQt6.QtCore")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        app_settings, "_settings_file", lambda: tmp_path / "settings.json"
    )
    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    process = qt_launcher._TerminalProcess(["unused"])
    process._stopped_event.clear()
    request = ControlRequest("switch", "model", timeout_s=1)
    completed: list[bool] = []
    window._wait_for_control_process(
        request,
        process,
        time.monotonic() + 1,
        lambda: completed.append(True),
    )
    app.processEvents()
    assert completed == []
    qt_core.QTimer.singleShot(20, process._stopped_event.set)
    deadline = time.monotonic() + 1
    while not completed and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert completed == [True]
    window.close()


def test_qt_control_health_timeout_stops_alive_unready_backend(
    tmp_path, monkeypatch
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        app_settings, "_settings_file", lambda: tmp_path / "settings.json"
    )
    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)

    class AliveProcess:
        proc = None

        def __init__(self) -> None:
            self.stopped = False

        def is_running(self) -> bool:
            return not self.stopped

        def stop(self) -> None:
            self.stopped = True

        def returncode(self):
            return None

    process = AliveProcess()
    request = ControlRequest("switch", "model", timeout_s=10)
    record = {
        "id": 1,
        "proc": process,
        "port": 65530,
        "base_url": "http://127.0.0.1:65530",
        "client_base_url": "http://127.0.0.1:65530",
        "ready": False,
        "model": "never-ready",
        "control_model_id": "model",
        "control_requests": [request],
        "control_deadline": time.monotonic() - 1,
    }
    window._servers = [record]
    window._control_api_record = record
    window._refresh_server_combo()
    window._poll_server()
    app.processEvents()
    with pytest.raises(ControlApiError, match="did not become ready") as error:
        request.wait()
    assert getattr(error.value, "status", None) == 504
    assert process.stopped is True
    assert record not in window._servers
    assert window._control_api_record is None
    window.close()


def test_launch_command_credentials_are_redacted_but_recoverable_for_proxy(
    tmp_path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_launcher = pytest.importorskip("qt_launcher")
    key_file = tmp_path / "keys.txt"
    key_file.write_text("# comment\nfile-secret\n", encoding="utf-8")
    direct = ["llama-server", "--api-key", "direct-secret", "--port", "1234"]
    from_file = ["llama-server", "--api-key-file", str(key_file)]
    assert qt_launcher._extract_server_api_key(direct) == "direct-secret"
    assert qt_launcher._extract_server_api_key(from_file) == "file-secret"
    rendered = qt_launcher._redacted_command(direct)
    assert "direct-secret" not in rendered
    assert "--api-key <redacted>" in rendered
    assert str(key_file) not in qt_launcher._redacted_command(from_file)
