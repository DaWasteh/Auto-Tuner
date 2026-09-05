"""Single-instance guard and clean-shutdown regression tests.

Background (v5.4.2): with *Hide on close* enabled, clicking X parks AutoTuner
in the notification area. Windows 11 hides new tray icons in the overflow
menu, so the next double-click on ``AutoTuner.exe`` started a second copy and
the first one looked like a leftover zombie. These tests pin down:

* the per-user/per-data-folder instance lock (in-process and cross-process),
* that a redirected launch activates the running window,
* that Quit ends the event loop even when the window sits in the tray,
* and, opt-in on a real Windows desktop, that the GUI process really exits
  and that a second launch restores a hidden first instance
  (``AUTOTUNER_GUI_PROCESS_TESTS=1``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
SETTINGS_DIR = ROOT / "settings"

import app_settings  # noqa: E402
from single_instance import (  # noqa: E402
    ALLOW_MULTIPLE_ENV,
    SingleInstanceGuard,
    instance_key,
    multiple_instances_allowed,
)


def _app():
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    return qt_widgets.QApplication.instance() or qt_widgets.QApplication([])


def _pump(app, condition, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    app.processEvents()
    return condition()


# ---------------------------------------------------------------------------
# Instance key


def test_instance_key_is_stable_per_data_dir_and_safe_as_socket_name(tmp_path) -> None:
    first = instance_key(tmp_path / "a")
    assert first == instance_key(tmp_path / "a")
    assert first != instance_key(tmp_path / "b")
    assert first.startswith("AutoTuner-")
    assert first.replace("-", "").isalnum()
    # A relative spelling of the same folder must not create a second lock.
    assert instance_key(tmp_path / "a" / ".." / "a") == first


def test_multiple_instances_override_parses_like_a_boolean(monkeypatch) -> None:
    for value in ("", "0", "false", "No", "off"):
        monkeypatch.setenv(ALLOW_MULTIPLE_ENV, value)
        assert not multiple_instances_allowed()
    for value in ("1", "true", "yes"):
        monkeypatch.setenv(ALLOW_MULTIPLE_ENV, value)
        assert multiple_instances_allowed()


# ---------------------------------------------------------------------------
# Guard in one process


def test_second_guard_is_refused_and_activation_reaches_the_primary(tmp_path) -> None:
    app = _app()
    key = instance_key(tmp_path / f"inproc-{os.getpid()}")
    primary = SingleInstanceGuard(key)
    activations: list[int] = []
    primary.activate_requested.connect(lambda: activations.append(1))
    try:
        assert primary.try_acquire()
        assert primary.is_primary
        assert primary.try_acquire(), "re-acquire by the owner must be idempotent"

        secondary = SingleInstanceGuard(key)
        assert not secondary.try_acquire()
        assert not secondary.is_primary
        assert secondary.notify_running_instance()
        assert _pump(app, lambda: bool(activations)), "primary never saw the activate request"
        assert activations == [1]

        # Only the documented command activates; noise must be ignored.
        secondary_noise = SingleInstanceGuard(key)
        assert not secondary_noise.try_acquire()
        assert not _pump(app, lambda: len(activations) > 1, timeout_s=0.3)
    finally:
        primary.release()
    assert not primary.is_primary

    # After release() the name is free again — also on Unix, where a stale
    # socket file would otherwise block listen().
    successor = SingleInstanceGuard(key)
    try:
        assert successor.try_acquire()
    finally:
        successor.release()


def test_notify_without_a_running_instance_fails_fast(tmp_path) -> None:
    _app()
    guard = SingleInstanceGuard(instance_key(tmp_path / "nobody-home"))
    started = time.monotonic()
    assert not guard.notify_running_instance()
    assert time.monotonic() - started < 5.0


_CHILD_PRIMARY = r"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, sys.argv[1])
from PyQt6.QtCore import QCoreApplication, QTimer
from single_instance import SingleInstanceGuard
app = QCoreApplication([])
guard = SingleInstanceGuard(sys.argv[2])
if not guard.try_acquire():
    print("busy", flush=True)
    raise SystemExit(3)
def activated():
    print("activated", flush=True)
    guard.release()
    app.quit()
guard.activate_requested.connect(activated)
QTimer.singleShot(20000, lambda: (print("timeout", flush=True), app.exit(4)))
print("listening", flush=True)
raise SystemExit(app.exec())
"""


def test_guard_is_honoured_across_processes(tmp_path) -> None:
    _app()
    key = instance_key(tmp_path / f"xproc-{os.getpid()}")
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_PRIMARY, str(ROOT), key],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )
    try:
        assert child.stdout is not None
        first = child.stdout.readline().strip()
        assert first == "listening", first

        secondary = SingleInstanceGuard(key)
        assert not secondary.try_acquire(), "the lock must be visible from another process"
        assert secondary.notify_running_instance()
        out, _ = child.communicate(timeout=30)
        assert "activated" in out, out
        assert child.returncode == 0

        # The child released its lock on the way out.
        successor = SingleInstanceGuard(key)
        try:
            assert successor.try_acquire()
        finally:
            successor.release()
    finally:
        if child.poll() is None:
            child.kill()


# ---------------------------------------------------------------------------
# MainWindow close paths


@pytest.fixture
def window(tmp_path, monkeypatch):
    qt_launcher = pytest.importorskip("qt_launcher")
    app = _app()
    monkeypatch.setattr(app_settings, "_settings_file", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(app_settings, "app_data_dir", lambda: tmp_path)
    # Never let a modal box block the offscreen test process.
    qt_widgets = sys.modules["PyQt6.QtWidgets"]
    monkeypatch.setattr(
        qt_widgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        qt_widgets.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    win = qt_launcher.MainWindow(tmp_path / "models", SETTINGS_DIR, start_background=False)
    yield qt_launcher, app, win
    try:
        win._force_quit = True
        win.close()
        win.deleteLater()
    except RuntimeError:
        pass
    app.processEvents()


def test_close_without_tray_option_closes_the_window(window) -> None:
    _qt_launcher, _app_, win = window
    win.show()
    assert not app_settings.get_minimize_on_close()
    assert win.close()
    assert not win.isVisible()
    assert win._tray_icon is None


def test_close_with_tray_option_hides_and_quit_from_tray_ends_the_event_loop(
    window, monkeypatch
) -> None:
    qt_launcher, app, win = window
    monkeypatch.setattr(qt_launcher, "_system_tray_supported", lambda: True)
    app_settings.set_minimize_on_close(True)
    win.show()

    # X → hidden into the notification area, process keeps running.
    assert not win.close()
    assert not win.isVisible()
    assert win._tray_icon is not None
    assert not win._force_quit

    # Quit from the tray menu must end the Qt event loop even though the
    # window was no longer a *visible* last window for Qt.
    from PyQt6.QtCore import QTimer

    outcome: list[str] = []
    QTimer.singleShot(0, win._request_quit)
    QTimer.singleShot(8000, lambda: (outcome.append("watchdog"), app.quit()))
    app.exec()
    assert outcome == [], "Quit from the tray did not stop the event loop"
    assert win._tray_icon is None
    assert not win.isVisible()


def test_activation_restores_a_tray_hidden_window(window, monkeypatch) -> None:
    qt_launcher, app, win = window
    monkeypatch.setattr(qt_launcher, "_system_tray_supported", lambda: True)
    app_settings.set_minimize_on_close(True)
    win.show()
    assert not win.close()
    assert win.isHidden()

    win._activate_from_other_instance()
    app.processEvents()
    assert win.isVisible()
    assert not win.isMinimized()

    # Visible + maximised must survive activation without being un-maximised.
    win.showMaximized()
    app.processEvents()
    win._activate_from_other_instance()
    app.processEvents()
    assert win.isVisible()
    assert win.isMaximized()


# ---------------------------------------------------------------------------
# Opt-in: real GUI processes on the Windows desktop


_GUI_PROCESS_TESTS = os.environ.get("AUTOTUNER_GUI_PROCESS_TESTS", "") == "1"
pytestmark_process = pytest.mark.skipif(
    not (_GUI_PROCESS_TESTS and sys.platform == "win32"),
    reason="set AUTOTUNER_GUI_PROCESS_TESTS=1 on a Windows desktop session",
)


def _process_tree(root_pid: int) -> set[int]:
    import ctypes
    import ctypes.wintypes as wt

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD),
            ("cntUsage", wt.DWORD),
            ("th32ProcessID", wt.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wt.DWORD),
            ("cntThreads", wt.DWORD),
            ("th32ParentProcessID", wt.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wt.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    pairs = []
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    if k32.Process32FirstW(snap, ctypes.byref(entry)):
        while True:
            pairs.append((entry.th32ParentProcessID, entry.th32ProcessID))
            if not k32.Process32NextW(snap, ctypes.byref(entry)):
                break
    k32.CloseHandle(snap)
    tree = {root_pid}
    grown = True
    while grown:
        grown = False
        for parent, pid in pairs:
            if parent in tree and pid not in tree:
                tree.add(pid)
                grown = True
    return tree


def _find_main_window(pids: set[int]):
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    found: list[tuple[int, str]] = []

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids:
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value.startswith("AutoTuner v"):
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(proc_type(cb), 0)
    return found[0][0] if found else None


class _GuiProcess:
    """One real ``qt_launcher.py`` process with an isolated data folder."""

    def __init__(self, data_dir: Path, *, hide_on_close: bool) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        models = data_dir / "models"
        models.mkdir(exist_ok=True)
        (data_dir / "autotuner_settings.json").write_text(
            json.dumps(
                {
                    "minimize_on_close": hide_on_close,
                    "control_api_enabled": False,
                    "models_path": str(models),
                }
            ),
            encoding="utf-8",
        )
        env = dict(os.environ, AUTOTUNER_DATA_DIR=str(data_dir))
        env.pop("QT_QPA_PLATFORM", None)
        # Keep the GUI's own output (and a faulthandler dump, should the
        # process abort) next to its data folder for post-mortem reading.
        self.stderr_path = data_dir / "gui-stderr.log"
        self._stderr = self.stderr_path.open("w", encoding="utf-8")
        # AUTOTUNER_GUI_PROCESS_BINARY=<path to AutoTuner.exe> runs the same
        # scenarios against the frozen build instead of the source launcher.
        binary = os.environ.get("AUTOTUNER_GUI_PROCESS_BINARY", "").strip()
        if binary:
            command = [binary, "--models-path", str(models)]
        else:
            command = [
                sys.executable,
                "-X",
                "faulthandler",
                str(ROOT / "qt_launcher.py"),
                "--models-path",
                str(models),
            ]
        self.proc = subprocess.Popen(
            command,
            env=env,
            cwd=str(ROOT),
            stdout=self._stderr,
            stderr=subprocess.STDOUT,
        )

    def output(self) -> str:
        try:
            self._stderr.flush()
            return self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def wait_for_window(self, timeout_s: float = 60.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            hwnd = _find_main_window(_process_tree(self.proc.pid))
            if hwnd:
                return hwnd
            if self.proc.poll() is not None:
                raise AssertionError(f"GUI exited early with {self.proc.returncode}")
            time.sleep(0.25)
        raise AssertionError("main window did not appear")

    def close_window(self, hwnd) -> None:
        import ctypes

        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE

    def window_visible(self, hwnd) -> bool:
        import ctypes

        user32 = ctypes.windll.user32
        return bool(user32.IsWindow(hwnd) and user32.IsWindowVisible(hwnd))

    def wait_exit(self, timeout_s: float):
        try:
            return self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self._stderr.close()
        except OSError:
            pass


@pytestmark_process
def test_gui_process_exits_after_closing_the_window(tmp_path) -> None:
    gui = _GuiProcess(tmp_path / "close", hide_on_close=False)
    try:
        hwnd = gui.wait_for_window()
        time.sleep(4)  # let startup work (scan, hardware probe) get going
        gui.close_window(hwnd)
        code = gui.wait_exit(30)
        assert code == 0, f"exit code {code!r} after the window closed:\n{gui.output()}"
        assert _process_tree(gui.proc.pid) == {gui.proc.pid}
    finally:
        gui.kill()


@pytestmark_process
def test_second_launch_restores_a_tray_hidden_instance_and_exits(tmp_path) -> None:
    data_dir = tmp_path / "tray"
    first = _GuiProcess(data_dir, hide_on_close=True)
    try:
        hwnd = first.wait_for_window()
        time.sleep(3)
        first.close_window(hwnd)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and first.window_visible(hwnd):
            time.sleep(0.2)
        assert not first.window_visible(hwnd), "X did not hide the window"
        assert first.proc.poll() is None, "hide-on-close must keep the process alive"

        second = _GuiProcess(data_dir, hide_on_close=True)
        try:
            assert second.wait_exit(30) == 0, "second launch did not exit"
        finally:
            second.kill()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not first.window_visible(hwnd):
            time.sleep(0.2)
        assert first.window_visible(hwnd), "hidden instance was not restored"
        assert first.proc.poll() is None
    finally:
        first.kill()
