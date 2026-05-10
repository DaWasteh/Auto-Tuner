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


# ---------------------------------------------------------------------------
# Fork-container path
#
# When the user picks a folder via "📂 Fork", they often pick a *parent*
# directory that holds several llama.cpp builds (e.g. C:\LAB\ai-local with
# `1b_llama.cpp/`, `atq_llama.cpp/`, `ik_llama.cpp/` inside). We must
# remember that container — not just the currently-selected build — so
# the next launch still shows ALL siblings instead of forcing the user
# to re-navigate up one level.
#
# `fork_path` keeps tracking the *currently active* build for things
# like `LLAMA_CPP_DIR`; `fork_container_path` is the root the GUI
# expanded the combo from.

def get_fork_container_path() -> Optional[Path]:
    p = load_settings().get("fork_container_path")
    if not p:
        return None
    pp = Path(p)
    return pp if pp.exists() else None


def set_fork_container_path(path: Path) -> None:
    _update("fork_container_path", str(path.resolve()))


def clear_fork_container_path() -> None:
    s = load_settings()
    if "fork_container_path" in s:
        s.pop("fork_container_path", None)
        save_settings(s)


# ---------------------------------------------------------------------------
# Per-model option overrides (vision / draft / thinking)
#
# Once a user toggles vision/draft/thinking for a specific model they
# expect that choice to stick — across performance-target changes,
# across selecting a different model and coming back, and across app
# restarts. We persist a small dict keyed by `entry.name` (the GGUF
# filename stem, which is stable for a given file on disk).
#
# Schema:
#   "model_overrides": {
#       "Qwen3.5-30B-A3B-UD-Q4_K_XL": {
#           "vision":   true,
#           "draft":    false,
#           "thinking": true
#       },
#       ...
#   }
#
# Absent keys mean "use the model's default capability detection" so
# turning a feature back on is just a matter of clearing the override.

_OVERRIDE_KEYS = ("vision", "draft", "thinking")


def get_model_overrides(model_name: str) -> Dict[str, bool]:
    """Return the per-model checkbox overrides, or {} when nothing stored."""
    if not model_name:
        return {}
    overrides = load_settings().get("model_overrides") or {}
    raw = overrides.get(model_name) or {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, bool] = {}
    for k in _OVERRIDE_KEYS:
        if k in raw:
            out[k] = bool(raw[k])
    return out


def set_model_override(model_name: str, key: str, value: bool) -> None:
    """Persist a single (model, option) → bool override.

    `key` must be one of "vision", "draft", "thinking"; anything else
    is silently ignored to keep the JSON file uncluttered.
    """
    if not model_name or key not in _OVERRIDE_KEYS:
        return
    s = load_settings()
    overrides = s.get("model_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    cur = overrides.get(model_name)
    if not isinstance(cur, dict):
        cur = {}
    cur[key] = bool(value)
    overrides[model_name] = cur
    s["model_overrides"] = overrides
    save_settings(s)


def clear_model_overrides(model_name: str) -> None:
    """Drop all stored overrides for a single model (e.g. on uninstall)."""
    if not model_name:
        return
    s = load_settings()
    overrides = s.get("model_overrides") or {}
    if isinstance(overrides, dict) and model_name in overrides:
        overrides.pop(model_name, None)
        s["model_overrides"] = overrides
        save_settings(s)


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