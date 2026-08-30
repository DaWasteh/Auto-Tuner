"""JSON language packs and live Qt widget translation for AutoTuner.

Built-in packs live under ``assets/languages``.  User packs are loaded from
``~/.autotuner/languages`` and use the same deliberately small, human-editable
schema.  Strings are keyed by the English (UK) source text so a custom pack can
override any exposed label without compiling Qt ``.qm`` resources.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
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
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


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

    return LanguagePack(
        id=pack_id,
        name=name.strip(),
        locale=locale.strip(),
        strings=strings,
        source=source,
        path=path,
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

    def install(self, app: QApplication) -> None:
        """Translate newly shown dialogs as well as the already-built main window."""
        if self._installed_app is app:
            return
        if self._installed_app is not None:
            self._installed_app.removeEventFilter(self)
        self._installed_app = app
        app.installEventFilter(self)

    def uninstall(self) -> None:
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
        translated = self.translate(source)
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
                self._translate_property(obj, "toolTip", obj.toolTip, obj.setToolTip)
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
                self._translate_property(obj, "toolTip", obj.toolTip, obj.setToolTip)

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
