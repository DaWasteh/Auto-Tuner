"""Small declarative-theme editor used by AutoTuner settings."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Optional

from PyQt6.QtCore import QSize, QTimer
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFontComboBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from theme_manager import (
    COLOR_ROLES,
    ThemeDefinition,
    ThemeLoadError,
    parse_theme,
    theme_to_json,
)


def _label(role: str) -> str:
    return role.replace("_", " ").title()


def _relative_luminance(color: QColor) -> float:
    channels = (color.redF(), color.greenF(), color.blueF())
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _swatch_text_color(color: QColor) -> str:
    luminance = _relative_luminance(color)
    black = (luminance + 0.05) / 0.05
    white = 1.05 / (luminance + 0.05)
    return "#000000" if black >= white else "#ffffff"


def editable_theme_copy(theme: ThemeDefinition) -> ThemeDefinition:
    """Return the user-theme draft shown by the editor."""
    if theme.source == "builtin":
        editable_id = f"{theme.id}-user"
        editable_name = f"{theme.name}-user"
    else:
        editable_id = theme.id
        editable_name = theme.name
    return replace(
        theme,
        id=editable_id,
        name=editable_name,
        colors=dict(theme.colors),
        source="user",
    )


class ThemeEditorDialog(QDialog):
    """Edit a copy of a theme; callers decide when/how to save it."""

    def __init__(
        self,
        theme: ThemeDefinition,
        parent=None,
        preview: Optional[Callable[[ThemeDefinition], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Customize theme")
        self.setMaximumWidth(800)
        self._theme = editable_theme_copy(theme)
        self._preview = preview
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.name_edit = QLineEdit(self._theme.name)
        self.id_edit = QLineEdit(self._theme.id)
        self.description_edit = QLineEdit(self._theme.description)
        for edit in (self.name_edit, self.id_edit, self.description_edit):
            edit.setMaximumWidth(400)
        self.ui_default = QCheckBox("Use system font")
        self.mono_default = QCheckBox("Use fixed-width font")
        self.ui_font = QFontComboBox()
        self.mono_font = QFontComboBox()
        for combo in (self.ui_font, self.mono_font):
            combo.setMinimumWidth(160)
            combo.setMaximumWidth(500)
            combo.setSizeAdjustPolicy(QFontComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            combo.currentFontChanged.connect(
                lambda font, combo=combo: combo.setToolTip(font.family())
            )
        self.mono_font.setFontFilters(QFontComboBox.FontFilter.MonospacedFonts)
        self.ui_default.setChecked(not theme.ui_family)
        self.mono_default.setChecked(not theme.mono_family)
        if theme.ui_family:
            self.ui_font.setCurrentFont(QFont(theme.ui_family))
        if theme.mono_family:
            self.mono_font.setCurrentFont(QFont(theme.mono_family))
        else:
            self.mono_font.setCurrentFont(
                QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            )
        self.ui_font.setToolTip(self.ui_font.currentFont().family())
        self.mono_font.setToolTip(self.mono_font.currentFont().family())
        self.ui_font.setEnabled(not self.ui_default.isChecked())
        self.mono_font.setEnabled(not self.mono_default.isChecked())
        form.addRow("Name", self.name_edit)
        form.addRow("ID", self.id_edit)
        form.addRow("Description", self.description_edit)
        form.addRow("UI font", self.ui_font)
        form.addRow("", self.ui_default)
        form.addRow("Monospace font", self.mono_font)
        form.addRow("", self.mono_default)
        layout.addLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(760)
        holder = QWidget()
        holder.setMaximumWidth(720)
        colors = QFormLayout(holder)
        colors.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.color_buttons = {}
        for role in COLOR_ROLES:
            button = QPushButton(self._theme.colors[role])
            button.setMaximumWidth(240)
            button.clicked.connect(
                lambda _=False, selected_role=role: self._pick(selected_role)
            )
            colors.addRow(_label(role), button)
            self.color_buttons[role] = button
            self._set_swatch(role)
        scroll.setWidget(holder)
        layout.addWidget(scroll, 1)
        self.ui_default.toggled.connect(self.ui_font.setDisabled)
        self.mono_default.toggled.connect(self.mono_font.setDisabled)
        self.ui_default.toggled.connect(lambda _: self._emit_preview())
        self.mono_default.toggled.connect(lambda _: self._emit_preview())
        self.ui_font.currentFontChanged.connect(lambda _: self._emit_preview())
        self.mono_font.currentFontChanged.connect(lambda _: self._emit_preview())
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        QTimer.singleShot(0, self._limit_height_to_screen)

    def _limit_height_to_screen(self) -> None:
        """Keep the editor usable on short displays while its colors scroll."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available_height = screen.availableGeometry().height()
        if available_height > 0:
            self.setMaximumHeight(max(240, available_height - 80))

    def sizeHint(self) -> QSize:  # noqa: N802
        """Keep the editor usable on an 800px-wide display; colors scroll."""
        hint = super().sizeHint()
        return QSize(min(hint.width(), 800), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """Let native Linux styles compress the form to the same 800px bound."""
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 800), hint.height())

    def _set_swatch(self, role: str) -> None:
        color = QColor(self._theme.colors[role])
        self.color_buttons[role].setStyleSheet(
            f"background:{color.name()}; color:{_swatch_text_color(color)}"
        )

    def _pick(self, role: str) -> None:
        color = QColorDialog.getColor(
            QColor(self._theme.colors[role]), self, f"Choose {_label(role)}"
        )
        if color.isValid() and color.alpha() == 255:
            self._theme.colors[role] = color.name(QColor.NameFormat.HexRgb)
            self._set_swatch(role)
            self._emit_preview()

    def theme(self) -> ThemeDefinition:
        raw_id = self.id_edit.text().strip().lower().replace(" ", "-")
        raw_id = re.sub(r"[^a-z0-9_-]", "-", raw_id).strip("-")
        return replace(
            self._theme,
            id=raw_id,
            name=self.name_edit.text().strip(),
            description=self.description_edit.text().strip(),
            ui_family=""
            if self.ui_default.isChecked()
            else self.ui_font.currentFont().family(),
            mono_family=""
            if self.mono_default.isChecked()
            else self.mono_font.currentFont().family(),
            source="user",
        )

    def _validate_and_accept(self) -> None:
        try:
            parse_theme(theme_to_json(self.theme()), "user")
        except ThemeLoadError as exc:
            QMessageBox.warning(self, "Invalid theme", str(exc))
            return
        self.accept()

    def _emit_preview(self) -> None:
        if self._preview:
            self._preview(self.theme())
