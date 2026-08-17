"""Safe, declarative application themes for AutoTuner.

Theme files are JSON data, never QSS. This module validates their small schema
and generates the complete internal stylesheet itself.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

import app_settings

SCHEMA_VERSION = 1
SYSTEM_THEME_ID = "builtin:system"
REQUIRED_BUILTIN_IDS = {
    "system",
    "dark",
    "dark-gray",
    "light",
    "high-contrast",
    "midnight-rose",
}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
MAX_THEME_BYTES = 64 * 1024
MAX_THEMES = 128
COLOR_ROLES = (
    "window_bg",
    "panel_bg",
    "control_bg",
    "control_hover",
    "control_pressed",
    "text",
    "muted_text",
    "accent",
    "accent_text",
    "border",
    "selection_bg",
    "selection_text",
    "disabled_text",
    "success",
    "warning",
    "error",
    "favorite_active",
    "favorite_inactive",
    "sysbar_bg",
    "sysbar_text",
    "section_text",
)

# This is deliberately embedded: a damaged/missing resource must not prevent
# the launcher from opening, including in a partially extracted frozen build.
_EMERGENCY_SYSTEM_COLORS = {
    "window_bg": "#202124",
    "panel_bg": "#292a2d",
    "control_bg": "#35363a",
    "control_hover": "#44464d",
    "control_pressed": "#55575e",
    "text": "#f1f3f4",
    "muted_text": "#b8bcc2",
    "accent": "#8ab4f8",
    "accent_text": "#101214",
    "border": "#5f6368",
    "selection_bg": "#3c6eae",
    "selection_text": "#ffffff",
    "disabled_text": "#8a8d91",
    "success": "#66cc88",
    "warning": "#f0c75e",
    "error": "#f07878",
    "favorite_active": "#ffd54f",
    "favorite_inactive": "#777777",
    "sysbar_bg": "#161625",
    "sysbar_text": "#88bbee",
    "section_text": "#88bbee",
}


class ThemeLoadError(ValueError):
    """A theme file is malformed or unsupported."""


@dataclass(frozen=True)
class ThemeDefinition:
    id: str
    name: str
    description: str
    colors: Dict[str, str]
    ui_family: str = ""
    mono_family: str = ""
    source: str = "builtin"

    @property
    def qualified_id(self) -> str:
        return f"{self.source}:{self.id}"


def emergency_system_theme() -> ThemeDefinition:
    return ThemeDefinition(
        "system",
        "System",
        "Native system palette and font",
        dict(_EMERGENCY_SYSTEM_COLORS),
    )


def _json_object(text: str) -> dict:
    def no_duplicates(pairs: List[Tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ThemeLoadError(f"Duplicate key: {key}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise ThemeLoadError(f"Unsupported JSON value: {value}")

    try:
        value = json.loads(
            text, object_pairs_hook=no_duplicates, parse_constant=no_constant
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ThemeLoadError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ThemeLoadError("Theme root must be a JSON object")
    return value


def parse_theme(text: str, source: str) -> ThemeDefinition:
    raw = _json_object(text)
    required = {"schema_version", "id", "name", "description", "colors", "font"}
    if set(raw) != required:
        raise ThemeLoadError(
            "Theme must contain exactly schema_version, id, name, description, colors and font"
        )
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        raise ThemeLoadError(f"Only schema_version {SCHEMA_VERSION} is supported")
    theme_id = raw["id"]
    if not isinstance(theme_id, str) or not ID_RE.fullmatch(theme_id):
        raise ThemeLoadError("id must match [a-z0-9][a-z0-9_-]{0,63}")
    for key in ("name", "description"):
        if not isinstance(raw[key], str) or len(raw[key]) > 160:
            raise ThemeLoadError(f"{key} must be text no longer than 160 characters")
    colors = raw["colors"]
    if not isinstance(colors, dict) or set(colors) != set(COLOR_ROLES):
        raise ThemeLoadError(
            "colors must contain every documented color role and no others"
        )
    for role, color in colors.items():
        if (
            not isinstance(color, str)
            or not COLOR_RE.fullmatch(color)
            or not QColor(color).isValid()
        ):
            raise ThemeLoadError(f"{role} must be an opaque #RRGGBB color")
    font = raw["font"]
    if not isinstance(font, dict) or set(font) != {"ui_family", "mono_family"}:
        raise ThemeLoadError("font must contain ui_family and mono_family")
    for value in font.values():
        if not isinstance(value, str) or len(value) > 120 or "\x00" in value:
            raise ThemeLoadError("font families must be short text")
    return ThemeDefinition(
        theme_id,
        raw["name"],
        raw["description"],
        dict(colors),
        font["ui_family"],
        font["mono_family"],
        source,
    )


def theme_to_json(theme: ThemeDefinition) -> str:
    return (
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "id": theme.id,
                "name": theme.name,
                "description": theme.description,
                "colors": theme.colors,
                "font": {
                    "ui_family": theme.ui_family,
                    "mono_family": theme.mono_family,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


class ThemeManager:
    def __init__(
        self, builtin_dir: Optional[Path] = None, user_dir: Optional[Path] = None
    ) -> None:
        root = Path(__file__).resolve().parent
        self.builtin_dir = builtin_dir or root / "assets" / "themes"
        self.user_dir = user_dir or app_settings.app_data_dir() / "autotuner_themes"
        self.errors: List[str] = []
        self.themes: Dict[str, ThemeDefinition] = {}
        self.builtin_resource_ids: set[str] = set()
        self.current_id = SYSTEM_THEME_ID
        self.current_definition = emergency_system_theme()
        self._base_palette: Optional[QPalette] = None
        self._base_font: Optional[QFont] = None
        self.reload()

    def _discover(self, directory: Path, source: str) -> Iterable[ThemeDefinition]:
        try:
            paths = sorted(
                directory.glob("*.json"), key=lambda path: path.name.casefold()
            )[:MAX_THEMES]
        except OSError as exc:
            self.errors.append(f"Could not read {source} themes: {exc}")
            return []
        result: List[ThemeDefinition] = []
        seen = set()
        for path in paths:
            try:
                if path.stat().st_size > MAX_THEME_BYTES:
                    raise ThemeLoadError("file is larger than 64 KiB")
                theme = parse_theme(path.read_text(encoding="utf-8"), source)
                if theme.id.casefold() in seen:
                    raise ThemeLoadError("duplicate id in theme directory")
                seen.add(theme.id.casefold())
                result.append(theme)
            except (OSError, UnicodeError, RecursionError, ThemeLoadError) as exc:
                self.errors.append(f"{path.name}: {exc}")
        return result

    def reload(self) -> None:
        self.errors.clear()
        self.builtin_resource_ids.clear()
        themes: Dict[str, ThemeDefinition] = {SYSTEM_THEME_ID: emergency_system_theme()}
        for theme in self._discover(self.builtin_dir, "builtin"):
            self.builtin_resource_ids.add(theme.id)
            themes[theme.qualified_id] = theme
        for theme in self._discover(self.user_dir, "user"):
            themes[theme.qualified_id] = theme
        self.themes = themes

    def available(self) -> List[ThemeDefinition]:
        return sorted(
            self.themes.values(),
            key=lambda theme: (theme.source != "builtin", theme.name.casefold()),
        )

    def get(self, qualified_id: str) -> ThemeDefinition:
        return self.themes.get(qualified_id, self.themes[SYSTEM_THEME_ID])

    def is_valid_builtin_set(self) -> bool:
        return self.builtin_resource_ids == REQUIRED_BUILTIN_IDS and not self.errors

    def stylesheet(self, theme: ThemeDefinition) -> str:
        c = theme.colors
        if theme.qualified_id == SYSTEM_THEME_ID:
            return f"""QLabel[themeRole="saved"] {{ color: {c["success"]}; font-style: italic; }}
QLabel[themeRole="muted"] {{ color: {c["muted_text"]}; }}
QLabel[themeRole="section"] {{ color: {c["section_text"]}; padding-top: 4px; }}
QWidget[themeRole="sysbar"] {{ background: {c["sysbar_bg"]}; }}
QLabel[themeRole="sysbar"] {{ color: {c["sysbar_text"]}; padding: 0 12px; }}"""
        return f"""QWidget {{ background: {c["window_bg"]}; color: {c["text"]}; }}
QToolBar, QStatusBar, QMenuBar, QMenu {{ background: {c["panel_bg"]}; border: 1px solid {c["border"]}; }}
QGroupBox {{ background: {c["panel_bg"]}; border: 1px solid {c["border"]}; border-radius: 3px; margin-top: 0.8em; padding-top: 0.35em; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 4px; color: {c["text"]}; }}
QTextEdit, QListWidget, QTreeWidget, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {c["panel_bg"]}; border: 1px solid {c["border"]}; border-radius: 3px; }}
QPushButton {{ background: {c["control_bg"]}; border: 1px solid {c["border"]}; padding: 4px; border-radius: 3px; }}
QPushButton:hover {{ background: {c["control_hover"]}; }}
QPushButton:pressed {{ background: {c["control_pressed"]}; }}
QPushButton:checked, QPushButton:default {{ background: {c["accent"]}; color: {c["accent_text"]}; }}
QWidget:disabled {{ color: {c["disabled_text"]}; }}
QListWidget::item:selected, QTreeWidget::item:selected, QComboBox QAbstractItemView::item:selected, QMenu::item:selected {{ background: {c["selection_bg"]}; color: {c["selection_text"]}; }}
QPushButton:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus, QTreeWidget:focus, QTextEdit:focus {{ border: 2px solid {c["accent"]}; }}
QToolTip {{ background: {c["panel_bg"]}; color: {c["text"]}; border: 1px solid {c["border"]}; }}
QScrollBar:vertical {{ background: {c["panel_bg"]}; width: 12px; }}
QScrollBar::handle:vertical {{ background: {c["control_hover"]}; min-height: 20px; border: 1px solid {c["border"]}; }}
QLabel[themeRole="saved"] {{ color: {c["success"]}; font-style: italic; }}
QLabel[themeRole="muted"] {{ color: {c["muted_text"]}; }}
QLabel[themeRole="section"] {{ color: {c["section_text"]}; padding-top: 4px; }}
QWidget[themeRole="sysbar"] {{ background: {c["sysbar_bg"]}; }}
QLabel[themeRole="sysbar"] {{ color: {c["sysbar_text"]}; padding: 0 12px; }}"""

    def apply(
        self, app: QApplication, qualified_id: str, font_size: Optional[int] = None
    ) -> str:
        return self.apply_definition(app, self.get(qualified_id), font_size)

    def apply_definition(
        self, app: QApplication, theme: ThemeDefinition, font_size: Optional[int] = None
    ) -> str:
        if self._base_palette is None:
            self._base_palette = QPalette(app.palette())
            self._base_font = QFont(app.font())
        if theme.qualified_id == SYSTEM_THEME_ID:
            app.setPalette(QPalette(self._base_palette))
            font = QFont(self._base_font)
        else:
            c = theme.colors
            palette = QPalette(self._base_palette)
            for role, color in (
                (QPalette.ColorRole.Window, c["window_bg"]),
                (QPalette.ColorRole.Base, c["panel_bg"]),
                (QPalette.ColorRole.Button, c["control_bg"]),
                (QPalette.ColorRole.Text, c["text"]),
                (QPalette.ColorRole.WindowText, c["text"]),
                (QPalette.ColorRole.ButtonText, c["text"]),
                (QPalette.ColorRole.Highlight, c["selection_bg"]),
                (QPalette.ColorRole.HighlightedText, c["selection_text"]),
            ):
                palette.setColor(role, QColor(color))
            palette.setColor(
                QPalette.ColorGroup.Disabled,
                QPalette.ColorRole.Text,
                QColor(c["disabled_text"]),
            )
            palette.setColor(
                QPalette.ColorGroup.Disabled,
                QPalette.ColorRole.WindowText,
                QColor(c["disabled_text"]),
            )
            app.setPalette(palette)
            font = QFont(self._base_font)
            if theme.ui_family:
                font.setFamily(theme.ui_family)
        if font_size is not None:
            font.setPointSize(font_size)
        app.setStyleSheet(self.stylesheet(theme))
        app.setFont(font)
        self.current_id = theme.qualified_id
        self.current_definition = theme
        return self.current_id

    def mono_font(self, point_size: int) -> QFont:
        if self.current_definition.mono_family:
            font = QFont(self.current_definition.mono_family)
        else:
            font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(point_size)
        return font

    def favorite_color(self, favorite: bool) -> str:
        """Return the exact configured star color for the requested state."""
        return self.current_definition.colors[
            "favorite_active" if favorite else "favorite_inactive"
        ]

    def save_user_theme(
        self, theme: ThemeDefinition, *, replace_id: Optional[str] = None
    ) -> Path:
        """Atomically create a user theme or replace the selected user theme.

        ``replace_id`` is the original id of the user theme being edited. It
        deliberately identifies the selected theme instead of granting a
        general "overwrite any id" permission when the editor's ID field is
        changed.
        """
        if theme.source != "user":
            raise ThemeLoadError("Only a user theme can be saved")
        # The parser is the single schema authority. A successful save can now
        # never be rejected by reload for shape/content reasons.
        validated = parse_theme(theme_to_json(theme), "user")
        if replace_id is not None and not ID_RE.fullmatch(replace_id):
            raise ThemeLoadError("Invalid replacement theme id")
        self.user_dir.mkdir(parents=True, exist_ok=True)
        root = self.user_dir.resolve()
        canonical_target = root / f"{validated.id}.json"
        if canonical_target.resolve().parent != root:
            raise ThemeLoadError("Invalid theme id")

        theme_files = list(root.glob("*.json"))
        existing: List[Tuple[Path, ThemeDefinition]] = []
        for path in theme_files:
            try:
                if path.is_symlink():
                    raise ThemeLoadError("symbolic-link theme files cannot be replaced")
                if path.resolve().parent != root:
                    raise ThemeLoadError("theme file resolves outside the theme folder")
                if path.stat().st_size > MAX_THEME_BYTES:
                    continue
                existing.append(
                    (path, parse_theme(path.read_text(encoding="utf-8"), "user"))
                )
            except (OSError, UnicodeError, RecursionError, ThemeLoadError):
                continue

        replacement: Optional[Path] = None
        if replace_id is not None:
            replacement = next(
                (
                    path
                    for path, current in existing
                    if current.id.casefold() == replace_id.casefold()
                ),
                None,
            )
            if replacement is None:
                raise ThemeLoadError(
                    "The user theme being edited no longer exists; reload themes"
                )

        for path, current in existing:
            if current.id.casefold() != validated.id.casefold():
                continue
            if replacement is None or path != replacement:
                raise FileExistsError(f"A user theme with id {validated.id!r} exists")

        target = replacement or canonical_target
        if replacement is None:
            if target.exists():
                raise FileExistsError(target)
            if len(theme_files) >= MAX_THEMES:
                raise ThemeLoadError(f"Theme folder is limited to {MAX_THEMES} files")

        data = theme_to_json(validated)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{validated.id}-", suffix=".tmp", dir=root, text=True
        )
        tmp = Path(tmp_name)
        original_data: Optional[bytes] = None
        if replacement is not None:
            try:
                original_data = target.read_bytes()
            except OSError as exc:
                try:
                    os.close(fd)
                    tmp.unlink()
                except OSError:
                    pass
                raise ThemeLoadError(
                    f"Could not back up the user theme: {exc}"
                ) from exc
        created_target = False
        replaced_target = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if replacement is not None:
                os.replace(tmp, target)
                replaced_target = True
            else:
                try:
                    # Atomic no-replace operation supported by Windows and POSIX.
                    os.link(tmp, target)
                    created_target = True
                except FileExistsError:
                    raise FileExistsError(target) from None
                finally:
                    if tmp.exists():
                        tmp.unlink()
            self.reload()
            loaded = self.themes.get(validated.qualified_id)
            if loaded != validated:
                raise ThemeLoadError("Saved theme could not be loaded")
            return target
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            if created_target:
                try:
                    target.unlink()
                except OSError:
                    pass
                self.reload()
            elif replaced_target and original_data is not None:
                restore_fd = -1
                restore_tmp: Optional[Path] = None
                try:
                    restore_fd, restore_name = tempfile.mkstemp(
                        prefix=f".{validated.id}-restore-",
                        suffix=".tmp",
                        dir=root,
                    )
                    restore_tmp = Path(restore_name)
                    with os.fdopen(restore_fd, "wb") as handle:
                        restore_fd = -1
                        handle.write(original_data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(restore_tmp, target)
                    self.reload()
                except Exception:
                    # Preserve the original save exception. A failed rollback is
                    # still visible on the next explicit reload/startup.
                    pass
                finally:
                    if restore_fd >= 0:
                        try:
                            os.close(restore_fd)
                        except OSError:
                            pass
                    if restore_tmp is not None:
                        try:
                            restore_tmp.unlink()
                        except OSError:
                            pass
            raise
