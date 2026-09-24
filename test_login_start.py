"""v5.5.6: login autostart marker and "Start minimized after login"."""

import plistlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

import app_settings
import startup_manager

ROOT = Path(__file__).resolve().parent
SETTINGS_DIR = ROOT / "settings"


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "autotuner_settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: path)
    monkeypatch.setattr(app_settings, "app_data_dir", lambda: tmp_path)
    return path


def _fake_winreg(values):
    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _query_value(_key, name):
        if name not in values:
            raise FileNotFoundError(name)
        return values[name], 1

    return types.SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        KEY_SET_VALUE=2,
        REG_SZ=1,
        OpenKey=lambda *_args: _Key(),
        CreateKey=lambda *_args: _Key(),
        QueryValueEx=_query_value,
        SetValueEx=lambda _key, name, _reserved, _kind, value: values.__setitem__(
            name, value
        ),
        DeleteValue=lambda _key, name: values.pop(name),
    )


# ---------------------------------------------------------------------------
# Setting: follows "Hide on close" until an explicit choice is saved


def test_start_minimized_follows_hide_on_close_until_explicit(settings_file) -> None:
    assert app_settings.get_start_minimized_at_login() is False
    assert not app_settings.start_minimized_at_login_is_explicit()
    app_settings.set_minimize_on_close(True)
    assert app_settings.get_start_minimized_at_login() is True

    app_settings.set_start_minimized_at_login(False)
    assert app_settings.start_minimized_at_login_is_explicit()
    assert app_settings.get_start_minimized_at_login() is False
    app_settings.set_minimize_on_close(False)
    app_settings.set_start_minimized_at_login(True)
    assert app_settings.get_start_minimized_at_login() is True


# ---------------------------------------------------------------------------
# Registration marker and in-place upgrade of pre-v5.5.6 entries


def test_windows_legacy_entry_for_this_install_is_upgraded(monkeypatch) -> None:
    values = {}
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(values))
    monkeypatch.setattr(startup_manager.sys, "platform", "win32")
    exe = r"L:\GitHub\Auto Tuner\dist\AutoTuner.exe"
    monkeypatch.setattr(startup_manager, "launch_arguments", lambda: [exe])

    assert startup_manager.refresh_autostart_registration() is False  # none
    values["AutoTuner"] = subprocess.list2cmdline([exe])  # v5.5.5 format
    assert startup_manager.refresh_autostart_registration() is True
    assert values["AutoTuner"] == f'"{exe}" --autostart'
    assert startup_manager.refresh_autostart_registration() is False  # current

    # A source run must not hijack the frozen EXE's entry, and vice versa.
    values["AutoTuner"] = subprocess.list2cmdline([r"C:\Other\AutoTuner.exe"])
    assert startup_manager.refresh_autostart_registration() is False
    assert values["AutoTuner"] == r"C:\Other\AutoTuner.exe"


def test_linux_legacy_desktop_entry_is_upgraded(tmp_path, monkeypatch) -> None:
    desktop = tmp_path / "autostart" / "AutoTuner.desktop"
    args = ["/opt/Auto Tuner/python3", "/opt/Auto Tuner/qt_launcher.py"]
    monkeypatch.setattr(startup_manager.sys, "platform", "linux")
    monkeypatch.setattr(startup_manager, "_linux_autostart_path", lambda: desktop)
    monkeypatch.setattr(startup_manager, "launch_arguments", lambda: args)

    startup_manager.set_autostart_enabled(True)
    legacy = desktop.read_text(encoding="utf-8").replace(' "--autostart"', "")
    desktop.write_text(legacy, encoding="utf-8")
    assert startup_manager.refresh_autostart_registration() is True
    assert '"--autostart"' in desktop.read_text(encoding="utf-8")
    assert startup_manager.refresh_autostart_registration() is False

    desktop.write_text(legacy.replace("/opt/", "/srv/"), encoding="utf-8")
    assert startup_manager.refresh_autostart_registration() is False


def test_macos_legacy_launch_agent_is_upgraded(tmp_path, monkeypatch) -> None:
    agent = tmp_path / "com.dawasteh.autotuner.plist"
    args = ["/Applications/AutoTuner.app/Contents/MacOS/AutoTuner"]
    monkeypatch.setattr(startup_manager.sys, "platform", "darwin")
    monkeypatch.setattr(startup_manager, "_macos_launch_agent_path", lambda: agent)
    monkeypatch.setattr(startup_manager, "launch_arguments", lambda: args)

    with agent.open("wb") as fh:
        plistlib.dump({"Label": "x", "ProgramArguments": args}, fh)
    assert startup_manager.refresh_autostart_registration() is True
    with agent.open("rb") as fh:
        assert plistlib.load(fh)["ProgramArguments"] == [*args, "--autostart"]
    assert startup_manager.refresh_autostart_registration() is False


# ---------------------------------------------------------------------------
# Initial window mode


class _FakeWindow:
    def __init__(self, tray_ok: bool) -> None:
        self.tray_ok = tray_ok
        self.calls: list[str] = []

    def _start_in_notification_area(self) -> bool:
        self.calls.append("tray")
        return self.tray_ok

    def show(self) -> None:
        self.calls.append("show")

    def showMinimized(self) -> None:  # noqa: N802
        self.calls.append("showMinimized")

    def _finish_native_window_setup(self) -> None:
        self.calls.append("native")


@pytest.mark.parametrize(
    ("autostart", "start_minimized", "hide_on_close", "tray_ok", "mode", "calls"),
    [
        (False, True, True, True, "normal", ["show", "native"]),
        (True, False, True, True, "normal", ["show", "native"]),
        (True, True, True, True, "tray", ["tray"]),
        (True, True, True, False, "minimized", ["tray", "showMinimized", "native"]),
        (True, True, False, True, "minimized", ["showMinimized", "native"]),
    ],
)
def test_initial_window_mode(
    settings_file, autostart, start_minimized, hide_on_close, tray_ok, mode, calls
) -> None:
    qt_launcher = pytest.importorskip("qt_launcher")
    app_settings.set_minimize_on_close(hide_on_close)
    app_settings.set_start_minimized_at_login(start_minimized)
    window = _FakeWindow(tray_ok)
    assert qt_launcher._show_initial_window(window, autostart) == mode
    assert window.calls == calls


def test_login_start_in_tray_restores_and_sets_up_native_window(
    settings_file, monkeypatch
) -> None:
    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(qt_launcher, "_system_tray_supported", lambda: True)
    app_settings.set_minimize_on_close(True)
    win = qt_launcher.MainWindow(
        settings_file.parent / "models", SETTINGS_DIR, start_background=False
    )
    try:
        assert qt_launcher._show_initial_window(win, True) == "tray"
        app.processEvents()
        assert win.isHidden()
        assert win._tray_icon is not None
        assert not win._native_window_setup_done

        win._activate_from_other_instance()
        for _ in range(5):
            app.processEvents()
        assert win.isVisible()
        assert win._native_window_setup_done
    finally:
        win._force_quit = True
        win.close()
        win.deleteLater()
        app.processEvents()


def test_settings_dialog_login_option_mirrors_hide_on_close(
    settings_file, monkeypatch
) -> None:
    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(qt_launcher, "_system_tray_supported", lambda: True)
    monkeypatch.setattr(
        qt_launcher.startup_manager, "is_autostart_enabled", lambda: False
    )
    parent = qt_widgets.QWidget()
    dialog = qt_launcher._ApplicationSettingsDialog(parent)
    box = dialog.start_minimized_checkbox
    assert not box.isEnabled() and not box.isChecked()
    dialog.autostart_checkbox.setChecked(True)
    assert box.isEnabled()
    dialog.minimize_checkbox.setChecked(True)
    assert box.isChecked()  # follows Hide on close while not chosen
    box.click()  # the user decides explicitly
    assert dialog.start_minimized_touched and not box.isChecked()
    dialog.minimize_checkbox.setChecked(False)
    dialog.minimize_checkbox.setChecked(True)
    assert not box.isChecked()  # an explicit choice is no longer overridden

    app_settings.set_start_minimized_at_login(False)
    app_settings.set_minimize_on_close(True)
    explicit = qt_launcher._ApplicationSettingsDialog(parent)
    assert explicit.start_minimized_explicit
    assert not explicit.start_minimized_checkbox.isChecked()
