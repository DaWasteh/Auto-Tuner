"""JSON language packs and live Qt widget translation for AutoTuner.

Built-in packs live under ``assets/languages``.  User packs are loaded from
``~/.autotuner/languages`` and use the same deliberately small, human-editable
schema.  Strings are keyed by the English (UK) source text so a custom pack can
override any exposed label without compiling Qt ``.qm`` resources.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QTextEdit,
    QWidget,
)

LANGUAGE_PACK_SCHEMA = 1
DEFAULT_LANGUAGE_ID = "builtin:en-GB"
CUSTOM_LANGUAGE_ACTION = "__custom_language_pack__"
_MAX_PACK_BYTES = 2 * 1024 * 1024
_MAX_STRINGS = 2000
#: Translated model-profile ``notes`` keyed by the profile's YAML file name.
_MAX_PROFILE_NOTES = 500
_MAX_NOTE_CHARS = 20_000
_PROFILE_NOTE_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}\.ya?ml$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Section labels of the two-level hover help built by :func:`setting_tooltip_html`.
#: They are translation keys like any other English (UK) source string.
TOOLTIP_SUMMARY_LABEL = "In short:"
TOOLTIP_TECHNICAL_LABEL = "Technical details:"
_TOOLTIP_HTML_RE = re.compile(
    r"^<html><body style='max-width:520px'>"
    r"<p><b>(?P<summary_label>[^<]*)</b> (?P<summary>.*?)</p>"
    r"<p><b>(?P<technical_label>[^<]*)</b> (?P<technical>.*?)</p>"
    r"</body></html>(?P<suffix>.*)$",
    re.DOTALL,
)
_BULLET_LINE_RE = re.compile(r"^(?P<prefix>\s*[•\-\*]\s+[^:]{1,40}:\s+)(?P<rest>.+)$")
_PREFIX_LINE_RE = re.compile(r"^(?P<prefix>[^:\n]{1,60}:)\s+(?P<rest>.+)$")


def setting_tooltip_html(
    summary: str,
    technical: str,
    *,
    summary_label: str = TOOLTIP_SUMMARY_LABEL,
    technical_label: str = TOOLTIP_TECHNICAL_LABEL,
) -> str:
    """Build the consistent two-level hover help for beginner and expert users.

    The result is deliberately regular so :meth:`LanguageManager.translate_tooltip`
    can take it apart again, translate each plain-text section, and rebuild it.
    """
    summary_html = html.escape(summary).replace("\n", "<br>")
    technical_html = html.escape(technical).replace("\n", "<br>")
    return (
        "<html><body style='max-width:520px'>"
        f"<p><b>{html.escape(summary_label)}</b> {summary_html}</p>"
        f"<p><b>{html.escape(technical_label)}</b> {technical_html}</p>"
        "</body></html>"
    )


def _tooltip_html_to_text(fragment: str) -> str:
    return html.unescape(fragment.replace("<br>", "\n"))


#: The manager most recently installed on the running QApplication.
_ACTIVE_MANAGER: Optional["LanguageManager"] = None


def translate(source_text: str) -> str:
    """Translate *source_text* through the installed manager (English otherwise).

    Text that the application composes at runtime (message boxes, dynamic
    labels, item tooltips) cannot rely on the widget event filter, so it is
    translated at construction time through this module-level helper.
    """
    manager = _ACTIVE_MANAGER
    if manager is None:
        return source_text
    return manager.translate(source_text)


class LanguagePackError(ValueError):
    """Raised when a language-pack file is malformed or unsafe to import."""


@dataclass(frozen=True)
class LanguagePack:
    """One validated built-in or user language pack."""

    id: str
    name: str
    locale: str
    strings: Dict[str, str]
    source: str
    path: Path
    # Optional translated explanations for model profiles. Keys are YAML file
    # names from ``settings/`` (for example ``gemma-4.yaml``); values replace
    # the English ``notes`` text while this pack is active.
    profile_notes: Dict[str, str] = field(default_factory=dict)

    @property
    def qualified_id(self) -> str:
        return f"{self.source}:{self.id}"


def _read_language_pack(path: Path, source: str) -> LanguagePack:
    """Read and strictly validate one JSON language pack."""
    try:
        if path.stat().st_size > _MAX_PACK_BYTES:
            raise LanguagePackError("language pack is larger than 2 MiB")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except LanguagePackError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LanguagePackError(f"could not read JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise LanguagePackError("top-level value must be an object")
    if payload.get("schema_version") != LANGUAGE_PACK_SCHEMA:
        raise LanguagePackError(
            f"schema_version must be {LANGUAGE_PACK_SCHEMA}"
        )

    pack_id = payload.get("id")
    if not isinstance(pack_id, str) or not _ID_RE.fullmatch(pack_id):
        raise LanguagePackError(
            "id must start with a letter or digit and contain only letters, "
            "digits, dots, underscores, or hyphens"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise LanguagePackError("name must be a non-empty string up to 80 characters")
    locale = payload.get("locale", pack_id)
    if not isinstance(locale, str) or not locale.strip() or len(locale.strip()) > 40:
        raise LanguagePackError("locale must be a non-empty string up to 40 characters")

    raw_strings = payload.get("strings")
    if not isinstance(raw_strings, dict):
        raise LanguagePackError("strings must be an object")
    if len(raw_strings) > _MAX_STRINGS:
        raise LanguagePackError(f"strings may contain at most {_MAX_STRINGS} entries")

    strings: Dict[str, str] = {}
    for key, value in raw_strings.items():
        if not isinstance(key, str) or not key or len(key) > 10_000:
            raise LanguagePackError("every translation key must be a non-empty string")
        if not isinstance(value, str) or not value or len(value) > 10_000:
            raise LanguagePackError(
                f"translation for {key!r} must be a non-empty string"
            )
        strings[key] = value

    raw_notes = payload.get("profile_notes", {})
    if raw_notes is None:
        raw_notes = {}
    if not isinstance(raw_notes, dict):
        raise LanguagePackError("profile_notes must be an object")
    if len(raw_notes) > _MAX_PROFILE_NOTES:
        raise LanguagePackError(
            f"profile_notes may contain at most {_MAX_PROFILE_NOTES} entries"
        )
    profile_notes: Dict[str, str] = {}
    for key, value in raw_notes.items():
        if not isinstance(key, str) or not _PROFILE_NOTE_KEY_RE.fullmatch(key):
            raise LanguagePackError(
                "every profile_notes key must be a profile file name such as "
                "'gemma-4.yaml'"
            )
        if not isinstance(value, str) or not value.strip():
            raise LanguagePackError(
                f"profile note for {key!r} must be a non-empty string"
            )
        if len(value) > _MAX_NOTE_CHARS:
            raise LanguagePackError(
                f"profile note for {key!r} exceeds {_MAX_NOTE_CHARS} characters"
            )
        profile_notes[key] = value.strip()

    return LanguagePack(
        id=pack_id,
        name=name.strip(),
        locale=locale.strip(),
        strings=strings,
        source=source,
        path=path,
        profile_notes=profile_notes,
    )


class LanguageManager(QObject):
    """Load packs, persist a selected ID externally, and translate live Qt UI.

    AutoTuner historically constructed widgets with plain string literals.  The
    event filter records each widget's original English source text the first
    time it sees it, so changing from one non-English pack to another can always
    retranslate from English instead of trying to translate an already
    translated label.  Dynamic text that the application changes later is
    detected and becomes the new source value.
    """

    def __init__(
        self,
        builtin_dir: Path,
        user_dir: Path,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.builtin_dir = Path(builtin_dir)
        self.user_dir = Path(user_dir)
        self.packs: Dict[str, LanguagePack] = {}
        self.errors: List[str] = []
        self.current_id = DEFAULT_LANGUAGE_ID
        self._installed_app: Optional[QApplication] = None
        self.reload()

    def reload(self) -> None:
        """Reload built-in and custom JSON files without discarding a valid choice."""
        packs: Dict[str, LanguagePack] = {}
        errors: List[str] = []
        for source, root in (("builtin", self.builtin_dir), ("user", self.user_dir)):
            try:
                paths = sorted(root.glob("*.json"), key=lambda item: item.name.casefold())
            except OSError as exc:
                errors.append(f"{root}: {exc}")
                continue
            for path in paths:
                try:
                    pack = _read_language_pack(path, source)
                except LanguagePackError as exc:
                    errors.append(f"{path.name}: {exc}")
                    continue
                qualified = pack.qualified_id
                if qualified in packs:
                    errors.append(f"{path.name}: duplicate language id {qualified}")
                    continue
                packs[qualified] = pack

        if DEFAULT_LANGUAGE_ID not in packs:
            raise LanguagePackError(
                f"required default pack {DEFAULT_LANGUAGE_ID!r} is missing or invalid"
            )
        self.packs = packs
        self.errors = errors
        if self.current_id not in packs:
            self.current_id = DEFAULT_LANGUAGE_ID

    def available(self) -> List[LanguagePack]:
        """Return built-ins in file order followed by alphabetised user packs."""
        builtins = [pack for pack in self.packs.values() if pack.source == "builtin"]
        users = [pack for pack in self.packs.values() if pack.source == "user"]
        builtins.sort(key=lambda pack: pack.path.name.casefold())
        users.sort(key=lambda pack: pack.name.casefold())
        return [*builtins, *users]

    @property
    def current(self) -> LanguagePack:
        return self.packs.get(self.current_id, self.packs[DEFAULT_LANGUAGE_ID])

    def select(self, qualified_id: str) -> str:
        """Select a known pack, safely falling back to English (UK)."""
        self.current_id = (
            qualified_id if qualified_id in self.packs else DEFAULT_LANGUAGE_ID
        )
        return self.current_id

    def translate(self, source_text: str) -> str:
        """Translate an exact English source string with English fallback."""
        if not isinstance(source_text, str) or not source_text:
            return source_text
        return self.current.strings.get(source_text, source_text)

    def translate_text(self, text: str) -> str:
        """Translate plain (possibly multi-line) text with line-level fallbacks.

        Whole-text matches win.  Otherwise every line is translated on its
        own, and lines that the application composes at runtime, such as
        ``• balanced: <description>`` or ``Active build: <name>``, are
        matched by their translatable part so the dynamic remainder survives.
        """
        if not isinstance(text, str) or not text:
            return text
        strings = self.current.strings
        direct = strings.get(text)
        if direct is not None:
            return direct
        if "\n" not in text:
            return self._translate_line(text, strings)
        return "\n".join(self._translate_line(line, strings) for line in text.split("\n"))

    @staticmethod
    def _translate_line(line: str, strings: Dict[str, str]) -> str:
        if not line:
            return line
        direct = strings.get(line)
        if direct is not None:
            return direct
        stripped = line.strip()
        direct = strings.get(stripped)
        if direct is not None:
            return line.replace(stripped, direct, 1)
        bullet = _BULLET_LINE_RE.match(line)
        if bullet is not None:
            translated = strings.get(bullet.group("rest"))
            if translated is not None:
                return f"{bullet.group('prefix')}{translated}"
        prefixed = _PREFIX_LINE_RE.match(line)
        if prefixed is not None:
            translated_prefix = strings.get(prefixed.group("prefix"))
            translated_rest = strings.get(prefixed.group("rest"))
            if translated_prefix is not None or translated_rest is not None:
                return (
                    f"{translated_prefix or prefixed.group('prefix')} "
                    f"{translated_rest or prefixed.group('rest')}"
                )
        return line

    def translate_tooltip(self, text: str) -> str:
        """Translate a widget tooltip, including the two-level HTML help.

        Tooltips built by :func:`setting_tooltip_html` are split into their
        "In short" and "Technical details" sections, translated as plain
        text, and reassembled with translated section labels.  Any other
        tooltip is translated as plain text.
        """
        if not isinstance(text, str) or not text:
            return text
        match = _TOOLTIP_HTML_RE.match(text)
        if match is None:
            return self.translate_text(text)
        summary = self.translate_text(_tooltip_html_to_text(match.group("summary")))
        technical = self.translate_text(
            _tooltip_html_to_text(match.group("technical"))
        )
        rebuilt = setting_tooltip_html(
            summary,
            technical,
            summary_label=self.translate(match.group("summary_label")),
            technical_label=self.translate(match.group("technical_label")),
        )
        suffix = match.group("suffix")
        if suffix:
            rebuilt += "<br>".join(
                html.escape(self.translate_text(html.unescape(part))) if part else ""
                for part in suffix.split("<br>")
            )
        return rebuilt

    def profile_notes(self, profile: object, fallback: str = "") -> str:
        """Return the active language's explanation for a model profile.

        *profile* may be a ``ModelProfile`` (its ``source_file`` and
        ``notes`` are used), a file name such as ``gemma-4.yaml``, or
        ``None``. Missing translations fall back to the English (UK) pack and
        finally to the YAML ``notes`` text so no model loses its explanation.
        """
        source_file = ""
        default_notes = str(fallback or "")
        if isinstance(profile, str):
            source_file = profile
        elif profile is not None:
            source_file = str(getattr(profile, "source_file", "") or "")
            if not default_notes:
                default_notes = str(getattr(profile, "notes", "") or "")
        if source_file:
            translated = self.current.profile_notes.get(source_file)
            if translated:
                return translated
            english = self.packs.get(DEFAULT_LANGUAGE_ID)
            if english is not None:
                translated = english.profile_notes.get(source_file)
                if translated:
                    return translated
        return default_notes

    def install(self, app: QApplication) -> None:
        """Translate newly shown dialogs as well as the already-built main window."""
        global _ACTIVE_MANAGER
        _ACTIVE_MANAGER = self
        if self._installed_app is app:
            return
        if self._installed_app is not None:
            self._installed_app.removeEventFilter(self)
        self._installed_app = app
        app.installEventFilter(self)

    def uninstall(self) -> None:
        global _ACTIVE_MANAGER
        if _ACTIVE_MANAGER is self:
            _ACTIVE_MANAGER = None
        if self._installed_app is not None:
            self._installed_app.removeEventFilter(self)
        self._installed_app = None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget):
            # Show can arrive before native styles finish creating standard
            # dialog buttons.  A zero-delay second pass catches those children.
            self.apply_to(watched)
            QTimer.singleShot(0, lambda root=watched: self._apply_if_alive(root))
        return False

    def _apply_if_alive(self, root: QWidget) -> None:
        try:
            self.apply_to(root)
        except RuntimeError:
            # The dialog may have been accepted/deleted during the event turn.
            pass

    def apply_all(self) -> None:
        app = self._installed_app
        if app is None:
            return
        for widget in app.topLevelWidgets():
            self.apply_to(widget)

    def _translate_property(
        self,
        obj: QObject,
        key: str,
        getter: Callable[[], str],
        setter: Callable[[str], None],
        translator: Optional[Callable[[str], str]] = None,
    ) -> None:
        try:
            current = getter()
        except RuntimeError:
            return
        if not isinstance(current, str) or not current:
            return
        # Keep state on the QObject wrapper itself.  Using ``id(obj)`` in a
        # process-wide dictionary would let Python reuse a deleted dialog's ID
        # for an unrelated widget and apply the wrong original text.
        state = getattr(obj, "_autotuner_i18n_state", None)
        if not isinstance(state, dict):
            state = {}
            setattr(obj, "_autotuner_i18n_state", state)
        source, previous_translation = state.get(key, (current, None))
        if previous_translation is not None and current != previous_translation:
            source = current
        translated = (translator or self.translate)(source)
        if current != translated:
            try:
                setter(translated)
            except RuntimeError:
                return
        state[key] = (source, translated)

    def apply_to(self, root: QWidget) -> None:
        """Retranslate static text on *root* and all current QObject children."""
        objects: Iterable[QObject] = [root, *root.findChildren(QObject)]
        for obj in objects:
            if isinstance(obj, QWidget):
                self._translate_property(
                    obj, "windowTitle", obj.windowTitle, obj.setWindowTitle
                )
                self._translate_property(
                    obj,
                    "toolTip",
                    obj.toolTip,
                    obj.setToolTip,
                    self.translate_tooltip,
                )
                self._translate_property(
                    obj, "accessibleName", obj.accessibleName, obj.setAccessibleName
                )
            if isinstance(obj, QAbstractButton):
                self._translate_property(obj, "text", obj.text, obj.setText)
            elif isinstance(obj, QLabel):
                self._translate_property(obj, "text", obj.text, obj.setText)
            elif isinstance(obj, QGroupBox):
                self._translate_property(obj, "title", obj.title, obj.setTitle)
            elif isinstance(obj, QLineEdit):
                self._translate_property(
                    obj,
                    "placeholderText",
                    obj.placeholderText,
                    obj.setPlaceholderText,
                )
            elif isinstance(obj, QTextEdit):
                self._translate_property(
                    obj,
                    "placeholderText",
                    obj.placeholderText,
                    obj.setPlaceholderText,
                )
            elif isinstance(obj, QMenu):
                self._translate_property(obj, "title", obj.title, obj.setTitle)
            if isinstance(obj, QAction):
                self._translate_property(obj, "text", obj.text, obj.setText)
                self._translate_property(
                    obj, "toolTip", obj.toolTip, obj.setToolTip, self.translate_tooltip
                )

    def import_pack(self, source_path: Path, *, replace: bool = False) -> LanguagePack:
        """Validate and atomically copy a custom pack into the user directory."""
        source_path = Path(source_path)
        pack = _read_language_pack(source_path, "user")
        destination = self.user_dir / f"{pack.id}.json"
        try:
            same_path = source_path.resolve(strict=False) == destination.resolve(
                strict=False
            )
        except (OSError, RuntimeError):
            same_path = False
        if destination.exists() and not same_path and not replace:
            raise FileExistsError(destination)

        self.user_dir.mkdir(parents=True, exist_ok=True)
        if not same_path:
            temp_path: Optional[Path] = None
            try:
                fd, temp_name = tempfile.mkstemp(
                    dir=self.user_dir,
                    prefix=f".{pack.id}.",
                    suffix=".tmp",
                )
                os.close(fd)
                temp_path = Path(temp_name)
                shutil.copyfile(source_path, temp_path)
                # Parse the exact staged bytes before replacing an existing pack.
                _read_language_pack(temp_path, "user")
                os.replace(temp_path, destination)
            finally:
                if temp_path is not None and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

        self.reload()
        qualified = f"user:{pack.id}"
        imported = self.packs.get(qualified)
        if imported is None:
            raise LanguagePackError("imported pack could not be reloaded")
        return imported

    def ensure_custom_template(self) -> Path:
        """Create an editable English-based custom-pack template once."""
        self.user_dir.mkdir(parents=True, exist_ok=True)
        destination = self.user_dir / "custom-language-template.json"
        if destination.exists():
            return destination
        english = self.packs[DEFAULT_LANGUAGE_ID]
        payload = {
            "schema_version": LANGUAGE_PACK_SCHEMA,
            "id": "my-language",
            "name": "My language",
            "locale": "xx-XX",
            "strings": english.strings,
            "profile_notes": english.profile_notes,
        }
        fd, temp_name = tempfile.mkstemp(
            dir=self.user_dir, prefix=".custom-language-template.", suffix=".tmp"
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp, destination)
        finally:
            if temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass
        return destination
