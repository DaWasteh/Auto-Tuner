"""Persistent app settings for the AutoTuner GUI.

Stores last-used paths so that a manually picked models folder or
llama.cpp fork is remembered across launches. JSON file lives next to
the script when writable (portable), otherwise in the user's home dir.

Public API:
    load_settings()        -> dict
    save_settings(dict)    -> bool
    get_models_path()      -> Optional[Path]
    set_models_path(Path)
    get_fork_path()        -> Optional[Path]
    set_fork_path(Path)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_FILENAME = "autotuner_settings.json"


def _settings_file() -> Path:
    """Resolve the settings file location.

    Preference: alongside the script (portable install). Fallback: the
    user's home directory if the script directory is read-only.
    """
    here = Path(__file__).resolve().parent
    portable = here / _FILENAME
    # Try a write probe to catch read-only installs (e.g. Program Files).
    try:
        probe = here / ".autotuner_write_probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return portable
    except (OSError, PermissionError):
        return Path.home() / _FILENAME


def load_settings() -> Dict[str, Any]:
    """Load settings from disk; return {} on missing file or parse error."""
    f = _settings_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(data: Dict[str, Any]) -> bool:
    """Atomically save settings; return True on success, False otherwise.

    Writes to a temp file in the same directory then renames, so a
    crash mid-write never leaves a half-written settings file.
    """
    f = _settings_file()
    tmp = f.with_suffix(f.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, f)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _update(key: str, value: Any) -> None:
    s = load_settings()
    s[key] = value
    save_settings(s)


# ---------------------------------------------------------------------------
# Convenience accessors

def get_models_path() -> Optional[Path]:
    p = load_settings().get("models_path")
    if not p:
        return None
    pp = Path(p)
    return pp if pp.exists() else None


def set_models_path(path: Path) -> None:
    _update("models_path", str(path.resolve()))


def get_fork_path() -> Optional[Path]:
    p = load_settings().get("fork_path")
    if not p:
        return None
    pp = Path(p)
    return pp if pp.exists() else None


def set_fork_path(path: Path) -> None:
    _update("fork_path", str(path.resolve()))


def get_performance_target() -> Optional[str]:
    """Return the persisted GUI performance-target choice, or None.

    Empty string and unknown values are treated as None so the GUI
    falls back to whatever the active profile (or global default)
    recommends.
    """
    val = load_settings().get("performance_target")
    if not val:
        return None
    val = str(val).lower().strip()
    return val if val in ("safe", "balanced", "throughput") else None


def set_performance_target(name: str) -> None:
    """Persist the GUI performance-target choice. Empty string clears it."""
    name = (name or "").lower().strip()
    if name in ("safe", "balanced", "throughput", ""):
        _update("performance_target", name)


def settings_file_location() -> Path:
    """Where settings are (or would be) written. For diagnostic logging."""
    return _settings_file()