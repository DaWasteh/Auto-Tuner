"""Focused tests for the declarative AutoTuner theme system."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import app_settings
from theme_manager import (
    COLOR_ROLES,
    SYSTEM_THEME_ID,
    ThemeDefinition,
    ThemeLoadError,
    ThemeManager,
    parse_theme,
)


ROOT = __import__("pathlib").Path(__file__).resolve().parent


def _raw(theme_id="sample"):
    system = json.loads(
        (ROOT / "assets" / "themes" / "system.json").read_text(encoding="utf-8")
    )
    system["id"] = theme_id
    system["name"] = "Sample"
    return system


def test_builtins_are_valid_and_namespaced(tmp_path):
    manager = ThemeManager(ROOT / "assets" / "themes", tmp_path / "user")
    assert {theme.qualified_id for theme in manager.available()} >= {
        "builtin:system",
        "builtin:dark",
        "builtin:light",
        "builtin:high-contrast",
    }
    assert manager.errors == []


def test_parser_rejects_untrusted_or_incomplete_schema():
    raw = _raw()
    raw["colors"].pop("text")
    with pytest.raises(ThemeLoadError):
        parse_theme(json.dumps(raw), "user")
    raw = _raw()
    raw["qss"] = "QWidget { display:none; }"
    with pytest.raises(ThemeLoadError):
        parse_theme(json.dumps(raw), "user")
    raw = _raw("../../escape")
    with pytest.raises(ThemeLoadError):
        parse_theme(json.dumps(raw), "user")


def test_invalid_json_is_isolated_and_system_fallback(tmp_path):
    user = tmp_path / "user"
    user.mkdir()
    (user / "broken.json").write_text("{ nope", encoding="utf-8")
    manager = ThemeManager(ROOT / "assets" / "themes", user)
    assert manager.get("user:missing").qualified_id == SYSTEM_THEME_ID
    assert manager.errors and "broken.json" in manager.errors[0]


def test_atomic_user_save_reload_and_traversal_guard(tmp_path):
    manager = ThemeManager(ROOT / "assets" / "themes", tmp_path / "user")
    base = manager.get(SYSTEM_THEME_ID)
    saved = ThemeDefinition(
        "my-theme", "My theme", "test", dict(base.colors), source="user"
    )
    path = manager.save_user_theme(saved)
    assert path.exists() and manager.get("user:my-theme").name == "My theme"
    with pytest.raises(FileExistsError):
        manager.save_user_theme(saved)
    with pytest.raises(ThemeLoadError):
        manager.save_user_theme(
            ThemeDefinition("../bad", "x", "x", dict(base.colors), source="user")
        )


def test_apply_palette_qss_and_mono_font(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = ThemeManager(ROOT / "assets" / "themes", tmp_path / "user")
    assert manager.apply(app, "builtin:dark", 13) == "builtin:dark"
    assert "QPushButton" in app.styleSheet()
    assert app.font().pointSize() == 13
    assert manager.mono_font(11).pointSize() == 11


def test_settings_json_root_and_theme_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: path)
    path.write_text("[]", encoding="utf-8")
    assert app_settings.load_settings() == {}
    assert app_settings.get_theme_id() == SYSTEM_THEME_ID
    app_settings.set_theme_id("user:ocean")
    assert app_settings.get_theme_id() == "user:ocean"


def test_all_roles_present_in_example():
    raw = _raw()
    assert set(raw["colors"]) == set(COLOR_ROLES)


def test_bad_utf8_deep_json_and_missing_system_are_safe(tmp_path):
    builtins = tmp_path / "builtins"
    builtins.mkdir()
    (builtins / "bad-utf8.json").write_bytes(b"\xff\xfe")
    (builtins / "deep.json").write_text("[" * 20_000 + "]" * 20_000, encoding="utf-8")
    manager = ThemeManager(builtins, tmp_path / "user")
    assert manager.get(SYSTEM_THEME_ID).qualified_id == SYSTEM_THEME_ID
    assert len(manager.errors) == 2


def test_save_validates_full_document_and_duplicate_semantic_id(tmp_path):
    user = tmp_path / "user"
    manager = ThemeManager(ROOT / "assets" / "themes", user)
    base = manager.get(SYSTEM_THEME_ID)
    invalid = ThemeDefinition("valid", "x" * 161, "", dict(base.colors), source="user")
    with pytest.raises(ThemeLoadError):
        manager.save_user_theme(invalid)
    other = ThemeDefinition(
        "duplicate", "Existing", "", dict(base.colors), source="user"
    )
    (user / "different-name.json").parent.mkdir()
    (user / "different-name.json").write_text(
        json.dumps({**_raw("duplicate"), "name": "Existing"}), encoding="utf-8"
    )
    manager.reload()
    with pytest.raises(ThemeLoadError):
        manager.save_user_theme(other)


def test_system_keeps_native_palette_and_other_themes_have_focus_qss(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = ThemeManager(ROOT / "assets" / "themes", tmp_path / "user")
    manager.apply(app, SYSTEM_THEME_ID, 10)
    assert "QPushButton:focus" not in app.styleSheet()
    manager.apply(app, "builtin:dark", 10)
    assert "QPushButton:focus" in app.styleSheet()
    assert "border: 2px solid" in app.styleSheet()


def test_promised_builtin_preflight_set(tmp_path):
    manager = ThemeManager(ROOT / "assets" / "themes", tmp_path / "user")
    assert manager.is_valid_builtin_set()
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "system.json").write_text(
        (ROOT / "assets" / "themes" / "system.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert not ThemeManager(incomplete, tmp_path / "other").is_valid_builtin_set()


def test_theme_folder_limit_rejects_save_without_orphan(tmp_path):
    user = tmp_path / "user"
    user.mkdir()
    raw = _raw()
    for index in range(128):
        (user / f"{index:03}.json").write_text(json.dumps(raw), encoding="utf-8")
    manager = ThemeManager(ROOT / "assets" / "themes", user)
    theme = ThemeDefinition(
        "new-theme", "New", "", dict(_raw()["colors"]), source="user"
    )
    with pytest.raises(ThemeLoadError, match="limited"):
        manager.save_user_theme(theme)
    assert not (user / "new-theme.json").exists()


def test_application_uses_one_theme_manager_and_22pt_dialogs_fit(tmp_path):
    """Run the dialog probe in isolation to avoid PyQt teardown races on Windows."""
    script = textwrap.dedent(
        f"""
        import os, sys
        from pathlib import Path
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        from PyQt6.QtWidgets import QApplication, QWidget
        import qt_launcher
        from theme_dialog import ThemeEditorDialog
        from theme_manager import SYSTEM_THEME_ID, ThemeManager

        app = QApplication([])
        font = app.font()
        font.setPointSize(22)
        app.setFont(font)
        manager = ThemeManager(Path({str(ROOT / "assets" / "themes")!r}), Path({str(tmp_path / "user")!r}))
        setattr(app, 'theme_manager', manager)
        parent = QWidget()
        settings = qt_launcher._ApplicationSettingsDialog(parent)
        editor = ThemeEditorDialog(manager.get(SYSTEM_THEME_ID), parent)
        assert qt_launcher._application_theme_manager(app) is manager
        assert settings.theme_combo.findData(SYSTEM_THEME_ID) >= 0
        for dialog in (settings, editor):
            dialog.resize(800, 800)
            dialog.show()
            app.processEvents()
            # Font metrics and advisory size hints vary across Qt's Windows,
            # macOS, and Linux platform plugins. The real acceptance criterion
            # is the rendered, maximum-bounded window, not a native style's
            # unbounded preferred width.
            assert dialog.isVisible()
            assert dialog.width() <= 800
            assert dialog.height() <= 800
        assert settings.theme_combo.width() >= 80
        assert editor.ui_font.width() >= 80
        assert editor.mono_font.width() >= 80
        print('22pt dialogs and singleton OK', flush=True)
        os._exit(0)
        """
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
