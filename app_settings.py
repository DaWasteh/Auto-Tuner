"""Persistent app settings for the AutoTuner GUI.

Stores last-used paths so that a manually picked models folder or
llama.cpp fork is remembered across launches. JSON file lives next to
the script when writable (portable), otherwise in the user's home dir.

Public API:
    load_settings()        -> dict
    save_settings(dict)    -> bool
    get_models_path()      -> Optional[Path]
    set_models_path(Path)
    get_model_paths()      -> list[(Path, enabled)]
    set_model_paths(list[(Path, enabled)])
    get_fork_path()        -> Optional[Path]
    set_fork_path(Path)
    get_llama_build_paths() -> list[(Path, enabled)]
    set_llama_build_paths(list[(Path, enabled)])
    get_window_geometry()  -> Optional[str]   # base64 of QByteArray
    set_window_geometry(str)
    get_window_state()     -> Optional[str]   # base64 of QByteArray (toolbars/docks)
    set_window_state(str)
    get_splitter_state(name) -> Optional[str]  # base64 of a QSplitter's saveState()
    set_splitter_state(name, str)
    get_mmproj_selection(model_name) -> Optional[str]  # chosen projector filename
    set_mmproj_selection(model_name, filename)
    get_font_size()        -> Optional[int]
    set_font_size(int)
    get_model_view_mode()  -> str
    set_model_view_mode(str)
    get_model_tree_collapsed_paths() -> set[str]
    set_model_tree_collapsed_paths(set[str])
    get_theme_id()         -> str
    set_theme_id(str)
    get_minimize_on_close() -> bool
    set_minimize_on_close(bool)
    get_base_port()        -> int
    set_base_port(int)
    get_port_offset()      -> int
    set_port_offset(int)
    get_prompt_cache_ram_mib() -> int
    set_prompt_cache_ram_mib(int)
    get_turbo_kv_warning_suppressed() -> bool
    set_turbo_kv_warning_suppressed(bool)
    get_reasoning_effort(model_name) -> Optional[str]
    set_reasoning_effort(model_name, value)
    favorite_model_key(model_path) -> str
    get_favorite_models() -> set[str]
    set_model_favorite(model_path, favorite)
    get_expert_override(model_name, model_path=None) -> Optional[dict]
    set_expert_override(model_name, snapshot: dict, model_path=None)
    clear_expert_override(model_name, model_path=None)
    get_performance_tuning_result(model_path, performance_target=None, benchmark_type=None) -> Optional[dict]
    list_performance_run_results() -> dict[str, list[dict]]
    save_performance_tuning_result(..., performance_target=None, benchmark_type=None) -> bool
    get_model_performance_target(model_path) -> Optional[str]
    set_model_performance_target(model_path, target)
    export_performance_profiles(path) -> (ok, message, count)
    import_performance_profiles(path, available_model_paths=()) -> (ok, message, count)
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_FILENAME = "autotuner_settings.json"


def app_data_dir() -> Path:
    r"""Persistent, user-writable directory for settings / logs / update state.

    Two regimes:
      * **Source install** — the script directory (portable, same folder the
        user runs ``qt_launcher.py`` from).
      * **Frozen build** (PyInstaller onefile) — the directory that contains
        the compiled ``AutoTuner.exe`` / Linux binary. The bundled code runs
        from a throw-away ``_MEIPASS`` temp folder that is DELETED on exit, so
        any user state written there (``autotuner_settings.json``, logs, …)
        would silently vanish between launches. Routing all user-writable
        state through the EXE folder keeps it stable across runs.

    If the resolved directory is read-only (e.g. the EXE sits in
    ``C:\Program Files``), fall back to the user's home directory so the
    app never crashes trying to persist state. Works on Windows and Linux.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    try:
        probe = base / ".autotuner_write_probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return base
    except (OSError, PermissionError):
        return Path.home()


def _settings_file() -> Path:
    """Resolve the settings file location.

    Preference: a portable install (alongside the script when running from
    source, or next to the .exe when frozen). Fallback: the user's home
    directory if that location is read-only (e.g. Program Files).
    """
    return app_data_dir() / _FILENAME


def load_settings() -> Dict[str, Any]:
    """Load settings from disk; return {} on missing file or parse error."""
    f = _settings_file()
    if not f.exists():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        # A valid JSON scalar/list is not a settings document.  Treat it like
        # corrupt data so all accessors remain safe.
        return data if isinstance(data, dict) else {}
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
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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

# OS-namespaced path keys. The settings JSON is portable and on dual-boot
# machines SHARED between the Windows and the Linux boot (it lives next to
# the script on the data partition). An absolute path saved on one OS is
# invalid on the other, so a single key ping-ponged: every boot the "other"
# OS lost its models/fork selection ("/run/media/…" read on Windows, "L:\…"
# read on Ubuntu) and overwrote the entry when the user re-picked it.
# Path-valued settings are therefore stored per-OS ("models_path.windows" /
# "models_path.linux"); the plain legacy key is still read as a fallback so
# existing files migrate seamlessly, and it is mirrored on write so an older
# AutoTuner version on the same OS keeps working.

_OS_KEY_SUFFIX = (
    "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux"
)


def _os_path_key(key: str) -> str:
    return f"{key}.{_OS_KEY_SUFFIX}"


def _get_os_path(key: str) -> Optional[Path]:
    """Read a per-OS path setting (legacy plain key as fallback); must exist."""
    s = load_settings()
    for k in (_os_path_key(key), key):
        p = s.get(k)
        if p:
            pp = Path(p)
            if pp.exists():
                return pp
    return None


def _set_os_path(key: str, value: str) -> None:
    s = load_settings()
    s[_os_path_key(key)] = value
    s[key] = value  # legacy mirror for older AutoTuner versions
    save_settings(s)


def get_models_path() -> Optional[Path]:
    return _get_os_path("models_path")


def set_models_path(path: Path) -> None:
    _set_os_path("models_path", str(path.resolve()))


PathEnabled = Tuple[Path, bool]


def _read_path_list(key: str) -> List[PathEnabled]:
    """Read a persisted multi-folder list as ``[(Path, enabled), ...]``.

    The JSON schema is intentionally small and human-editable:
    ``[{"path": "...", "enabled": true}, ...]``. Invalid and duplicate
    paths are skipped; missing folders are kept so the GUI can still show and
    edit a removable stale entry.
    """
    s = load_settings()
    raw = s.get(_os_path_key(key))
    if not isinstance(raw, list) or not raw:
        # Legacy fallback: plain key written before per-OS namespacing (or by
        # an older version). Stale other-OS entries surface as editable
        # missing folders in the GUI, exactly like before.
        raw = s.get(key)
    if not isinstance(raw, list):
        return []
    out: List[PathEnabled] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            p_raw = item.get("path")
            enabled = bool(item.get("enabled", True))
        else:
            p_raw = item
            enabled = True
        if not p_raw:
            continue
        try:
            p = Path(str(p_raw)).expanduser()
            # ``resolve(strict=False)`` normalises duplicates without requiring
            # the directory to still exist.
            rp = p.resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        key_text = os.path.normcase(str(rp))
        if key_text in seen:
            continue
        seen.add(key_text)
        out.append((rp, enabled))
    return out


def _write_path_list(key: str, paths: List[PathEnabled]) -> None:
    clean = []
    seen: set[str] = set()
    for path, enabled in paths:
        try:
            rp = Path(path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        key_text = os.path.normcase(str(rp))
        if key_text in seen:
            continue
        seen.add(key_text)
        clean.append({"path": str(rp), "enabled": bool(enabled)})
    s = load_settings()
    s[_os_path_key(key)] = clean
    s[key] = clean  # legacy mirror for older AutoTuner versions
    save_settings(s)


def get_model_paths() -> List[PathEnabled]:
    """Return configured model folders with their enabled state.

    Empty means the multi-folder setting has never been written; callers should
    fall back to ``models_path`` / ``AUTOTUNER_MODELS`` / defaults and then save
    through ``set_model_paths`` once the user edits the list.
    """
    return _read_path_list("model_paths")


def set_model_paths(paths: List[PathEnabled]) -> None:
    _write_path_list("model_paths", paths)
    first_enabled = next((p for p, enabled in paths if enabled), None)
    if first_enabled is None and paths:
        first_enabled = paths[0][0]
    if first_enabled is not None:
        set_models_path(first_enabled)


def get_fork_path() -> Optional[Path]:
    return _get_os_path("fork_path")


def set_fork_path(path: Path) -> None:
    _set_os_path("fork_path", str(path.resolve()))


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
    return _get_os_path("fork_container_path")


def set_fork_container_path(path: Path) -> None:
    _set_os_path("fork_container_path", str(path.resolve()))


def clear_fork_container_path() -> None:
    s = load_settings()
    changed = False
    for k in (_os_path_key("fork_container_path"), "fork_container_path"):
        if k in s:
            s.pop(k, None)
            changed = True
    if changed:
        save_settings(s)


def get_llama_build_paths() -> List[PathEnabled]:
    """Return configured llama.cpp build/container folders with enabled state."""
    return _read_path_list("llama_build_paths")


def set_llama_build_paths(paths: List[PathEnabled]) -> None:
    """Persist llama.cpp build/container folders.

    The selected active fork remains stored separately in ``fork_path`` because
    this list represents scan roots (containers or individual builds), not the
    combo-box selection.
    """
    _write_path_list("llama_build_paths", paths)


# ---------------------------------------------------------------------------
# Per-model option overrides (vision / mmproj CPU / draft / thinking)
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
#           "vision":       true,
#           "mmproj_cpu":   false,
#           "draft":        false,
#           "thinking":     true,
#           "ngram":        false,
#           "prompt_cache": true
#       },
#       ...
#   }
#
# Absent keys mean "use the model's default capability detection" so
# turning a feature back on is just a matter of clearing the override.

_OVERRIDE_KEYS = (
    "vision",
    "mmproj_cpu",
    "draft",
    "thinking",
    "ngram",
    "prompt_cache",
)


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

    `key` must be one of "vision", "mmproj_cpu", "draft", "thinking",
    "ngram", "prompt_cache"; anything else is silently ignored to keep the JSON
    file uncluttered.
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


# ---------------------------------------------------------------------------
# Favorite models
#
# Favorites are keyed by normalized absolute GGUF paths so identically named
# models in different configured roots can be marked independently. Like all
# persisted paths, the compact list is OS-namespaced because one settings file
# may be shared across Windows, Linux, and macOS installations.


def favorite_model_key(model_path: Path) -> str:
    """Return the normalized identity used for one favorite GGUF path."""
    try:
        path = Path(model_path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return ""
    return os.path.normcase(str(path))


def get_favorite_models() -> set[str]:
    """Return normalized favorite GGUF identities for the current OS."""
    settings = load_settings()
    raw = settings.get(_os_path_key("favorite_models"))
    if not isinstance(raw, list):
        return set()
    favorites: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            continue
        key = favorite_model_key(Path(value))
        if key:
            favorites.add(key)
    return favorites


def set_model_favorite(model_path: Path, favorite: bool) -> None:
    """Persist or clear the favorite marker for one GGUF path."""
    key = favorite_model_key(model_path)
    if not key:
        return
    favorites = get_favorite_models()
    if favorite:
        favorites.add(key)
    else:
        favorites.discard(key)
    settings = load_settings()
    settings[_os_path_key("favorite_models")] = sorted(favorites, key=str.casefold)
    save_settings(settings)


# ---------------------------------------------------------------------------
# Expert-panel state (per model)
#
# The Expert panel lets a user override the AutoTuner's decisions for a
# single model — context length, KV quants, layer placement, threads,
# sampling, flags, reasoning, … Before this existed the panel started
# from the auto defaults every time the user (re)opened it, so a low-VRAM
# user had to re-enter the same hand-tuned settings on every launch.
#
# We now persist the full panel state per model so it is restored the
# next time that model is selected — and applied at launch just like the
# checkbox overrides above, completing the "remembers everything" story.
#
# New writes can also use the OS-namespaced ``expert_overrides_by_path`` map,
# keyed by normalized absolute GGUF path. The legacy name map remains a fallback
# so existing settings migrate without losing a user's configuration.
#
# Schema (legacy map stored under "expert_overrides", keyed by model name):
#   "expert_overrides": {
#       "Qwen3.5-30B-A3B-UD-Q4_K_XL": {
#           "mode": "auto",            # "auto" | "manual"
#           "pins": {                   # auto-mode cascade pins
##               "user_ctx": 32768,
#               "force_cache_k": "q8_0"
#           },
#           "values": {                 # full widget snapshot (both modes)
#               "ctx": 32768, "cache_k": "q8_0", …
#           },
#           "saved_at": "2026-06-30T12:00:00"
#       }
#   }
#
# Reset (the new button next to Auto/Manual) simply clears the entry.


_VALID_PERFORMANCE_TARGETS = ("safe", "balanced", "throughput", "low_vram")
_VALID_BENCHMARK_TYPES = ("quick", "normal")


def _normalise_performance_target(value: Optional[str]) -> str:
    target = str(value or "").lower().strip()
    return target if target in _VALID_PERFORMANCE_TARGETS else ""


def _normalise_benchmark_type(value: Optional[str]) -> str:
    benchmark_type = str(value or "").lower().strip()
    return benchmark_type if benchmark_type in _VALID_BENCHMARK_TYPES else ""


def _valid_expert_snapshot(snapshot: Any) -> bool:
    return (
        isinstance(snapshot, dict)
        and "mode" in snapshot
        and isinstance(snapshot.get("values"), dict)
    )


def portable_model_key(model_path: Path, model_size: Optional[int] = None) -> str:
    """Return a path-independent filename+size identity for profile transfer."""
    try:
        path = Path(model_path).expanduser()
        size = int(model_size) if model_size is not None else int(path.stat().st_size)
    except (OSError, TypeError, ValueError):
        try:
            path = Path(model_path)
        except (TypeError, ValueError):
            return ""
        size = max(0, int(model_size or 0))
    name = path.name.strip().casefold()
    return f"{name}|{max(0, size)}" if name else ""


def _target_map(settings: Dict[str, Any], storage_key: str) -> Dict[str, Any]:
    value = settings.get(storage_key)
    return value if isinstance(value, dict) else {}


def _set_target_value(
    settings: Dict[str, Any], storage_key: str, identity: str, target: str, value: Any
) -> None:
    outer = _target_map(settings, storage_key)
    per_model = outer.get(identity)
    if not isinstance(per_model, dict):
        per_model = {}
    per_model[target] = value
    outer[identity] = per_model
    settings[storage_key] = outer


def _get_target_value(
    settings: Dict[str, Any], storage_key: str, identity: str, target: str
) -> Tuple[bool, Any]:
    outer = settings.get(storage_key)
    if not identity or not isinstance(outer, dict):
        return False, None
    per_model = outer.get(identity)
    if not isinstance(per_model, dict):
        return False, None
    # Once a model has any mode-scoped state, a missing target deliberately
    # means "pure Auto" rather than leaking a legacy winner across all modes.
    return True, per_model.get(target)


def _set_benchmark_result_value(
    settings: Dict[str, Any],
    storage_key: str,
    identity: str,
    benchmark_type: str,
    target: str,
    value: Any,
) -> None:
    outer = _target_map(settings, storage_key)
    per_model = outer.get(identity)
    if not isinstance(per_model, dict):
        per_model = {}
    per_test = per_model.get(benchmark_type)
    if not isinstance(per_test, dict):
        per_test = {}
    per_test[target] = value
    per_model[benchmark_type] = per_test
    outer[identity] = per_model
    settings[storage_key] = outer


def _get_benchmark_result_value(
    settings: Dict[str, Any],
    storage_key: str,
    identity: str,
    benchmark_type: str,
    target: str,
) -> Tuple[bool, Any]:
    outer = settings.get(storage_key)
    if not identity or not isinstance(outer, dict):
        return False, None
    per_model = outer.get(identity)
    if not isinstance(per_model, dict):
        return False, None
    per_test = per_model.get(benchmark_type)
    if not isinstance(per_test, dict):
        return False, None
    return True, per_test.get(target)


def get_expert_override(
    model_name: str,
    model_path: Optional[Path] = None,
    performance_target: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the Expert snapshot for one model and performance target.

    v5.2.4 stores independent snapshots for safe/balanced/throughput/low_vram.
    Exact path wins, then the portable filename+size identity used by profile
    import, then legacy path/name state for backward compatibility.
    """
    settings = load_settings()
    target = _normalise_performance_target(performance_target)
    key = favorite_model_key(model_path) if model_path is not None else ""
    if target:
        if key:
            found, snap = _get_target_value(
                settings,
                _os_path_key("expert_overrides_by_performance"),
                key,
                target,
            )
            if found:
                return snap if _valid_expert_snapshot(snap) else None
        portable = portable_model_key(model_path) if model_path is not None else ""
        if portable:
            found, snap = _get_target_value(
                settings,
                _os_path_key("expert_overrides_portable_by_performance"),
                portable,
                target,
            )
            if found:
                return snap if _valid_expert_snapshot(snap) else None
        if model_name:
            found, snap = _get_target_value(
                settings,
                "expert_overrides_by_name_by_performance",
                model_name,
                target,
            )
            if found:
                return snap if _valid_expert_snapshot(snap) else None

    if key:
        by_path = settings.get(_os_path_key("expert_overrides_by_path")) or {}
        if isinstance(by_path, dict):
            snap = by_path.get(key)
            if _valid_expert_snapshot(snap):
                return snap
    if not model_name:
        return None
    raw = settings.get("expert_overrides") or {}
    if not isinstance(raw, dict):
        return None
    snap = raw.get(model_name)
    return snap if _valid_expert_snapshot(snap) else None


def set_expert_override(
    model_name: str,
    snapshot: Dict[str, Any],
    model_path: Optional[Path] = None,
    performance_target: Optional[str] = None,
) -> None:
    """Persist a validated Expert snapshot, scoped by performance target."""
    if not model_name or not _valid_expert_snapshot(snapshot):
        return
    settings = load_settings()
    key = favorite_model_key(model_path) if model_path is not None else ""
    target = _normalise_performance_target(performance_target)
    if target:
        if key:
            _set_target_value(
                settings,
                _os_path_key("expert_overrides_by_performance"),
                key,
                target,
                snapshot,
            )
            portable = portable_model_key(model_path) if model_path is not None else ""
            if portable:
                _set_target_value(
                    settings,
                    _os_path_key("expert_overrides_portable_by_performance"),
                    portable,
                    target,
                    snapshot,
                )
        else:
            _set_target_value(
                settings,
                "expert_overrides_by_name_by_performance",
                model_name,
                target,
                snapshot,
            )
    elif key:
        storage_key = _os_path_key("expert_overrides_by_path")
        overrides = settings.get(storage_key)
        if not isinstance(overrides, dict):
            overrides = {}
        overrides[key] = snapshot
        settings[storage_key] = overrides
    else:
        overrides = settings.get("expert_overrides")
        if not isinstance(overrides, dict):
            overrides = {}
        overrides[model_name] = snapshot
        settings["expert_overrides"] = overrides
    save_settings(settings)


def clear_expert_override(
    model_name: str,
    model_path: Optional[Path] = None,
    performance_target: Optional[str] = None,
) -> None:
    """Reset one mode, or all legacy state when no target is supplied."""
    if not model_name:
        return
    settings = load_settings()
    target = _normalise_performance_target(performance_target)
    key = favorite_model_key(model_path) if model_path is not None else ""
    if target:
        if key:
            _set_target_value(
                settings,
                _os_path_key("expert_overrides_by_performance"),
                key,
                target,
                None,
            )
            portable = portable_model_key(model_path) if model_path is not None else ""
            if portable:
                _set_target_value(
                    settings,
                    _os_path_key("expert_overrides_portable_by_performance"),
                    portable,
                    target,
                    None,
                )
        else:
            _set_target_value(
                settings,
                "expert_overrides_by_name_by_performance",
                model_name,
                target,
                None,
            )
        save_settings(settings)
        return

    changed = False
    if key:
        storage_key = _os_path_key("expert_overrides_by_path")
        by_path = settings.get(storage_key) or {}
        if isinstance(by_path, dict) and key in by_path:
            by_path.pop(key, None)
            settings[storage_key] = by_path
            changed = True
    overrides = settings.get("expert_overrides") or {}
    if isinstance(overrides, dict) and model_name in overrides:
        overrides.pop(model_name, None)
        settings["expert_overrides"] = overrides
        changed = True
    if changed:
        save_settings(settings)


def get_performance_tuning_result(
    model_path: Path,
    performance_target: Optional[str] = None,
    benchmark_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return measured evidence for one exact model/target/test-type tuple.

    Without ``benchmark_type`` this retains the v5.2.4 contract and returns the
    latest profile evidence. Supplying ``quick`` or ``normal`` reads the
    independently retained analysis result so the two workloads never mix.
    """
    settings = load_settings()
    key = favorite_model_key(model_path)
    target = _normalise_performance_target(performance_target)
    test_type = _normalise_benchmark_type(benchmark_type)
    if target and key and test_type:
        found, record = _get_benchmark_result_value(
            settings,
            _os_path_key("performance_run_results_by_test"),
            key,
            test_type,
            target,
        )
        if found:
            return record if isinstance(record, dict) else None
        portable = portable_model_key(model_path)
        found, record = _get_benchmark_result_value(
            settings,
            _os_path_key("performance_run_results_portable_by_test"),
            portable,
            test_type,
            target,
        )
        if found:
            return record if isinstance(record, dict) else None

    record: Any = None
    found_current = False
    if target and key:
        found_current, record = _get_target_value(
            settings,
            _os_path_key("performance_tuning_results_by_performance"),
            key,
            target,
        )
        if not found_current:
            portable = portable_model_key(model_path)
            found_current, record = _get_target_value(
                settings,
                _os_path_key("performance_tuning_results_portable_by_performance"),
                portable,
                target,
            )
    if not found_current:
        records = settings.get(_os_path_key("performance_tuning_results")) or {}
        if key and isinstance(records, dict):
            record = records.get(key)

    if not isinstance(record, dict):
        return None
    if test_type:
        stored_type = _normalise_benchmark_type(record.get("benchmark_type")) or "normal"
        return record if stored_type == test_type else None
    return record


def list_performance_run_results() -> Dict[str, List[Dict[str, Any]]]:
    """List latest quick and normal evidence for every tested model/mode.

    The test-specific store is authoritative. Legacy v5.2.4 evidence is
    classified as ``normal`` because that release always used 25% context.
    Portable mirror entries are listed only when no exact-path result represents
    the same filename+size identity, preserving unmapped imports without duplicates.
    """
    settings = load_settings()
    grouped: Dict[str, List[Dict[str, Any]]] = {"quick": [], "normal": []}
    seen: set[Tuple[str, str, str]] = set()

    exact = settings.get(_os_path_key("performance_run_results_by_test"))
    if isinstance(exact, dict):
        for identity, per_model in exact.items():
            if not isinstance(identity, str) or not isinstance(per_model, dict):
                continue
            for raw_type, per_test in per_model.items():
                test_type = _normalise_benchmark_type(str(raw_type))
                if not test_type or not isinstance(per_test, dict):
                    continue
                for raw_target, raw_record in per_test.items():
                    target = _normalise_performance_target(str(raw_target))
                    if not target or not isinstance(raw_record, dict):
                        continue
                    item = dict(raw_record)
                    item.setdefault("model_path", identity)
                    item["performance_target"] = target
                    item["benchmark_type"] = test_type
                    grouped[test_type].append(item)
                    seen.add((identity, test_type, target))

    legacy = settings.get(_os_path_key("performance_tuning_results_by_performance"))
    if isinstance(legacy, dict):
        for identity, per_model in legacy.items():
            if not isinstance(identity, str) or not isinstance(per_model, dict):
                continue
            for raw_target, raw_record in per_model.items():
                target = _normalise_performance_target(str(raw_target))
                if not target or not isinstance(raw_record, dict):
                    continue
                test_type = (
                    _normalise_benchmark_type(raw_record.get("benchmark_type"))
                    or "normal"
                )
                marker = (identity, test_type, target)
                if marker in seen:
                    continue
                item = dict(raw_record)
                item.setdefault("model_path", identity)
                item["performance_target"] = target
                item["benchmark_type"] = test_type
                grouped[test_type].append(item)
                seen.add(marker)

    legacy_single = settings.get(_os_path_key("performance_tuning_results"))
    if isinstance(legacy_single, dict):
        for identity, raw_record in legacy_single.items():
            if not isinstance(identity, str) or not isinstance(raw_record, dict):
                continue
            target = _normalise_performance_target(raw_record.get("performance_target"))
            if not target:
                continue
            test_type = (
                _normalise_benchmark_type(raw_record.get("benchmark_type")) or "normal"
            )
            marker = (identity, test_type, target)
            if marker in seen:
                continue
            item = dict(raw_record)
            item.setdefault("model_path", identity)
            item["performance_target"] = target
            item["benchmark_type"] = test_type
            grouped[test_type].append(item)
            seen.add(marker)

    represented_portable: set[str] = set()
    for records in grouped.values():
        for record in records:
            raw_path = str(record.get("model_path", "") or "")
            try:
                model_size = int(record.get("model_size", 0) or 0)
            except (TypeError, ValueError):
                model_size = 0
            if raw_path:
                portable = portable_model_key(Path(raw_path), model_size)
                if portable:
                    represented_portable.add(portable)

    portable_exact = settings.get(
        _os_path_key("performance_run_results_portable_by_test")
    )
    if isinstance(portable_exact, dict):
        for portable, per_model in portable_exact.items():
            if portable in represented_portable or not isinstance(per_model, dict):
                continue
            filename = str(portable).rpartition("|")[0]
            for raw_type, per_test in per_model.items():
                test_type = _normalise_benchmark_type(str(raw_type))
                if not test_type or not isinstance(per_test, dict):
                    continue
                for raw_target, raw_record in per_test.items():
                    target = _normalise_performance_target(str(raw_target))
                    marker = (f"portable:{portable}", test_type, target)
                    if (
                        not target
                        or not isinstance(raw_record, dict)
                        or marker in seen
                    ):
                        continue
                    item = dict(raw_record)
                    item.setdefault("model_name", Path(filename).stem)
                    item["performance_target"] = target
                    item["benchmark_type"] = test_type
                    grouped[test_type].append(item)
                    seen.add(marker)
            represented_portable.add(str(portable))

    portable_legacy = settings.get(
        _os_path_key("performance_tuning_results_portable_by_performance")
    )
    if isinstance(portable_legacy, dict):
        for portable, per_model in portable_legacy.items():
            if portable in represented_portable or not isinstance(per_model, dict):
                continue
            filename = str(portable).rpartition("|")[0]
            for raw_target, raw_record in per_model.items():
                target = _normalise_performance_target(str(raw_target))
                if not target or not isinstance(raw_record, dict):
                    continue
                test_type = (
                    _normalise_benchmark_type(raw_record.get("benchmark_type"))
                    or "normal"
                )
                marker = (f"portable:{portable}", test_type, target)
                if marker in seen:
                    continue
                item = dict(raw_record)
                item.setdefault("model_name", Path(filename).stem)
                item["performance_target"] = target
                item["benchmark_type"] = test_type
                grouped[test_type].append(item)
                seen.add(marker)
    return grouped


def save_performance_tuning_result(
    model_name: str,
    model_path: Path,
    result: Dict[str, Any],
    snapshot: Dict[str, Any],
    performance_target: Optional[str] = None,
    benchmark_type: Optional[str] = None,
) -> bool:
    """Atomically save the active profile and test-type-isolated evidence."""
    key = favorite_model_key(model_path)
    if (
        not model_name
        or not key
        or not isinstance(result, dict)
        or not _valid_expert_snapshot(snapshot)
    ):
        return False
    settings = load_settings()
    target = _normalise_performance_target(
        performance_target or str(result.get("performance_target", ""))
    )
    test_type = (
        _normalise_benchmark_type(
            benchmark_type or str(result.get("benchmark_type", ""))
        )
        or "normal"
    )
    history_record = dict(result)
    history_record.setdefault("benchmark_type", test_type)
    if target:
        _set_target_value(
            settings,
            _os_path_key("performance_tuning_results_by_performance"),
            key,
            target,
            result,
        )
        _set_target_value(
            settings,
            _os_path_key("expert_overrides_by_performance"),
            key,
            target,
            snapshot,
        )
        _set_benchmark_result_value(
            settings,
            _os_path_key("performance_run_results_by_test"),
            key,
            test_type,
            target,
            history_record,
        )
        portable = portable_model_key(model_path, result.get("model_size"))
        if portable:
            _set_target_value(
                settings,
                _os_path_key("performance_tuning_results_portable_by_performance"),
                portable,
                target,
                result,
            )
            _set_target_value(
                settings,
                _os_path_key("expert_overrides_portable_by_performance"),
                portable,
                target,
                snapshot,
            )
            _set_benchmark_result_value(
                settings,
                _os_path_key("performance_run_results_portable_by_test"),
                portable,
                test_type,
                target,
                history_record,
            )
    else:
        result_key = _os_path_key("performance_tuning_results")
        results = settings.get(result_key)
        if not isinstance(results, dict):
            results = {}
        results[key] = result
        settings[result_key] = results
        expert_key = _os_path_key("expert_overrides_by_path")
        experts = settings.get(expert_key)
        if not isinstance(experts, dict):
            experts = {}
        experts[key] = snapshot
        settings[expert_key] = experts
    return save_settings(settings)


_PROFILE_BUNDLE_FORMAT = "autotuner-performance-profiles"
_PROFILE_BUNDLE_SCHEMA = 1
_PROFILE_BUNDLE_MAX_BYTES = 32 * 1024 * 1024


def _profile_bundle_entry(
    *,
    filename: str,
    model_size: int,
    source_path: str,
    model_name: str,
    modes: Dict[str, Any],
    preferred_target: str = "",
) -> Optional[Dict[str, Any]]:
    clean_modes: Dict[str, Any] = {}
    for target, payload in modes.items():
        target_name = _normalise_performance_target(target)
        if not target_name or not isinstance(payload, dict):
            continue
        snapshot = payload.get("snapshot")
        result = payload.get("result")
        if not _valid_expert_snapshot(snapshot):
            continue
        clean_runs: Dict[str, Dict[str, Any]] = {}
        raw_runs = payload.get("runs")
        if isinstance(raw_runs, dict):
            for raw_type, raw_record in raw_runs.items():
                test_type = _normalise_benchmark_type(str(raw_type))
                if test_type and isinstance(raw_record, dict):
                    clean_runs[test_type] = raw_record
        clean_mode = {
            "snapshot": snapshot,
            "result": result if isinstance(result, dict) else {},
        }
        if clean_runs:
            clean_mode["runs"] = clean_runs
        clean_modes[target_name] = clean_mode
    clean_filename = Path(str(filename or "")).name
    if not clean_filename or not clean_modes:
        return None
    return {
        "model_name": str(model_name or Path(clean_filename).stem),
        "filename": clean_filename,
        "model_size": max(0, int(model_size or 0)),
        "source_path": str(source_path or ""),
        "preferred_target": _normalise_performance_target(preferred_target),
        "modes": clean_modes,
    }


def export_performance_profiles(destination: Path) -> Tuple[bool, str, int]:
    """Export only measured/expert performance profiles, never broad settings."""
    settings = load_settings()
    experts_by_path = _target_map(
        settings, _os_path_key("expert_overrides_by_performance")
    )
    results_by_path = _target_map(
        settings, _os_path_key("performance_tuning_results_by_performance")
    )
    runs_by_path = _target_map(
        settings, _os_path_key("performance_run_results_by_test")
    )
    legacy_experts = settings.get(_os_path_key("expert_overrides_by_path"))
    legacy_results = settings.get(_os_path_key("performance_tuning_results"))
    preferred_by_path = settings.get(_os_path_key("model_performance_targets"))
    if not isinstance(preferred_by_path, dict):
        preferred_by_path = {}
    if not isinstance(legacy_experts, dict):
        legacy_experts = {}
    if not isinstance(legacy_results, dict):
        legacy_results = {}

    def runs_for_target(per_model: Any, target: str) -> Dict[str, Dict[str, Any]]:
        collected: Dict[str, Dict[str, Any]] = {}
        if not isinstance(per_model, dict):
            return collected
        for raw_type, per_test in per_model.items():
            test_type = _normalise_benchmark_type(str(raw_type))
            if not test_type or not isinstance(per_test, dict):
                continue
            record = per_test.get(target)
            if isinstance(record, dict):
                collected[test_type] = record
        return collected

    entries: List[Dict[str, Any]] = []
    represented_portable: set[str] = set()
    path_keys = (
        set(experts_by_path)
        | set(results_by_path)
        | set(runs_by_path)
        | set(legacy_experts)
    )
    for path_key in sorted(path_keys, key=str.casefold):
        path = Path(path_key)
        per_expert = experts_by_path.get(path_key)
        per_result = results_by_path.get(path_key)
        per_runs = runs_by_path.get(path_key)
        modes: Dict[str, Any] = {}
        if isinstance(per_expert, dict):
            for target, snapshot in per_expert.items():
                result = per_result.get(target) if isinstance(per_result, dict) else {}
                modes[target] = {
                    "snapshot": snapshot,
                    "result": result,
                    "runs": runs_for_target(per_runs, str(target)),
                }
        legacy_snapshot = legacy_experts.get(path_key)
        if _valid_expert_snapshot(legacy_snapshot):
            legacy_record = legacy_results.get(path_key)
            inferred = _normalise_performance_target(
                str(legacy_record.get("performance_target", ""))
                if isinstance(legacy_record, dict)
                else ""
            ) or "balanced"
            modes.setdefault(
                inferred,
                {
                    "snapshot": legacy_snapshot,
                    "result": legacy_record if isinstance(legacy_record, dict) else {},
                    "runs": runs_for_target(per_runs, inferred),
                },
            )
        size = 0
        try:
            size = int(path.stat().st_size)
        except OSError:
            if isinstance(per_result, dict):
                for record in per_result.values():
                    if isinstance(record, dict) and record.get("model_size") is not None:
                        try:
                            size = int(record["model_size"])
                        except (TypeError, ValueError):
                            pass
                        if size > 0:
                            break
            legacy_record = legacy_results.get(path_key)
            if size <= 0 and isinstance(legacy_record, dict):
                try:
                    size = int(legacy_record.get("model_size", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
        item = _profile_bundle_entry(
            filename=path.name,
            model_size=size,
            source_path=path_key,
            model_name=path.stem,
            modes=modes,
            preferred_target=str(preferred_by_path.get(path_key, "")),
        )
        if item is not None:
            entries.append(item)
            portable = portable_model_key(path, size)
            if portable:
                represented_portable.add(portable)

    # Imported profiles may intentionally have no currently-scanned local path.
    # Preserve them in subsequent exports via their portable identity.
    portable_experts = _target_map(
        settings, _os_path_key("expert_overrides_portable_by_performance")
    )
    portable_results = _target_map(
        settings, _os_path_key("performance_tuning_results_portable_by_performance")
    )
    portable_runs = _target_map(
        settings, _os_path_key("performance_run_results_portable_by_test")
    )
    preferred_portable = settings.get(
        _os_path_key("model_performance_targets_portable")
    )
    if not isinstance(preferred_portable, dict):
        preferred_portable = {}
    for portable in sorted(
        set(portable_experts) | set(portable_results) | set(portable_runs),
        key=str.casefold,
    ):
        if portable in represented_portable:
            continue
        filename, separator, raw_size = portable.rpartition("|")
        if not separator:
            continue
        try:
            size = max(0, int(raw_size))
        except (TypeError, ValueError):
            continue
        per_expert = portable_experts.get(portable)
        per_result = portable_results.get(portable)
        per_runs = portable_runs.get(portable)
        if not isinstance(per_expert, dict):
            continue
        modes = {
            target: {
                "snapshot": snapshot,
                "result": per_result.get(target) if isinstance(per_result, dict) else {},
                "runs": runs_for_target(per_runs, str(target)),
            }
            for target, snapshot in per_expert.items()
        }
        item = _profile_bundle_entry(
            filename=filename,
            model_size=size,
            source_path="",
            model_name=Path(filename).stem,
            modes=modes,
            preferred_target=str(preferred_portable.get(portable, "")),
        )
        if item is not None:
            entries.append(item)

    profile_count = sum(len(item["modes"]) for item in entries)
    bundle = {
        "format": _PROFILE_BUNDLE_FORMAT,
        "schema": _PROFILE_BUNDLE_SCHEMA,
        "profile_count": profile_count,
        "models": entries,
    }
    destination = Path(destination)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, destination)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False, f"Could not export profiles: {exc}", 0
    return (
        True,
        f"Exported {profile_count} performance profile(s).",
        profile_count,
    )


def import_performance_profiles(
    source: Path, available_model_paths: Optional[List[Path]] = None
) -> Tuple[bool, str, int]:
    """Validate and merge a portable profile bundle into current settings."""
    source = Path(source)
    try:
        if source.stat().st_size > _PROFILE_BUNDLE_MAX_BYTES:
            return False, "Profile bundle is larger than 32 MiB.", 0
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"Could not read profile bundle: {exc}", 0
    if not isinstance(payload, dict):
        return False, "Profile bundle root must be a JSON object.", 0
    if payload.get("format") != _PROFILE_BUNDLE_FORMAT:
        return False, "This is not an AutoTuner performance-profile bundle.", 0
    if payload.get("schema") != _PROFILE_BUNDLE_SCHEMA:
        return False, f"Unsupported profile bundle schema: {payload.get('schema')!r}.", 0
    models = payload.get("models")
    if not isinstance(models, list) or len(models) > 10_000:
        return False, "Profile bundle has an invalid models list.", 0

    available = [Path(path) for path in (available_model_paths or [])]
    by_filename: Dict[str, List[Path]] = {}
    for path in available:
        by_filename.setdefault(path.name.casefold(), []).append(path)

    settings = load_settings()
    imported = 0
    mapped = 0
    for item in models:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", "") or "").strip()
        if not filename or Path(filename).name != filename:
            continue
        try:
            model_size = max(0, int(item.get("model_size", 0) or 0))
        except (TypeError, ValueError):
            continue
        model_name = str(item.get("model_name", "") or Path(filename).stem)
        modes = item.get("modes")
        if not isinstance(modes, dict):
            continue
        portable = portable_model_key(Path(filename), model_size)
        if not portable:
            continue

        candidates = by_filename.get(filename.casefold(), [])
        local_path: Optional[Path] = None
        exact_size_matches: List[Path] = []
        for candidate in candidates:
            try:
                if model_size > 0 and candidate.stat().st_size == model_size:
                    exact_size_matches.append(candidate)
            except OSError:
                continue
        if len(exact_size_matches) == 1:
            local_path = exact_size_matches[0]
        elif model_size == 0 and len(candidates) == 1:
            local_path = candidates[0]
        else:
            source_path = str(item.get("source_path", "") or "")
            if source_path:
                candidate = Path(source_path)
                try:
                    if candidate.is_file() and (
                        model_size == 0 or candidate.stat().st_size == model_size
                    ):
                        local_path = candidate
                except OSError:
                    pass

        preferred_target = _normalise_performance_target(
            str(item.get("preferred_target", ""))
        )
        if preferred_target:
            portable_pref_key = _os_path_key("model_performance_targets_portable")
            portable_prefs = settings.get(portable_pref_key)
            if not isinstance(portable_prefs, dict):
                portable_prefs = {}
            portable_prefs[portable] = preferred_target
            settings[portable_pref_key] = portable_prefs
            if local_path is not None:
                local_key = favorite_model_key(local_path)
                if local_key:
                    path_pref_key = _os_path_key("model_performance_targets")
                    path_prefs = settings.get(path_pref_key)
                    if not isinstance(path_prefs, dict):
                        path_prefs = {}
                    path_prefs[local_key] = preferred_target
                    settings[path_pref_key] = path_prefs

        for target, mode_payload in modes.items():
            target_name = _normalise_performance_target(str(target))
            if not target_name or not isinstance(mode_payload, dict):
                continue
            snapshot = mode_payload.get("snapshot")
            result = mode_payload.get("result")
            if not _valid_expert_snapshot(snapshot):
                continue
            record = result if isinstance(result, dict) else {}
            imported_runs: Dict[str, Dict[str, Any]] = {}
            raw_runs = mode_payload.get("runs")
            if isinstance(raw_runs, dict):
                for raw_type, raw_record in raw_runs.items():
                    test_type = _normalise_benchmark_type(str(raw_type))
                    if test_type and isinstance(raw_record, dict):
                        imported_runs[test_type] = raw_record
            if not imported_runs and record:
                test_type = (
                    _normalise_benchmark_type(record.get("benchmark_type"))
                    or "normal"
                )
                imported_runs[test_type] = record
            _set_target_value(
                settings,
                _os_path_key("expert_overrides_portable_by_performance"),
                portable,
                target_name,
                snapshot,
            )
            _set_target_value(
                settings,
                _os_path_key("performance_tuning_results_portable_by_performance"),
                portable,
                target_name,
                record,
            )
            for test_type, run_record in imported_runs.items():
                _set_benchmark_result_value(
                    settings,
                    _os_path_key("performance_run_results_portable_by_test"),
                    portable,
                    test_type,
                    target_name,
                    run_record,
                )
            if local_path is not None:
                local_key = favorite_model_key(local_path)
                if local_key:
                    _set_target_value(
                        settings,
                        _os_path_key("expert_overrides_by_performance"),
                        local_key,
                        target_name,
                        snapshot,
                    )
                    _set_target_value(
                        settings,
                        _os_path_key("performance_tuning_results_by_performance"),
                        local_key,
                        target_name,
                        record,
                    )
                    for test_type, run_record in imported_runs.items():
                        _set_benchmark_result_value(
                            settings,
                            _os_path_key("performance_run_results_by_test"),
                            local_key,
                            test_type,
                            target_name,
                            run_record,
                        )
            elif model_name:
                # Name fallback is only used until a matching filename+size is
                # scanned; portable identity remains the authoritative mapping.
                _set_target_value(
                    settings,
                    "expert_overrides_by_name_by_performance",
                    model_name,
                    target_name,
                    snapshot,
                )
            imported += 1
        if local_path is not None:
            mapped += 1

    if imported <= 0:
        return False, "The bundle contained no valid performance profiles.", 0
    if not save_settings(settings):
        return False, "Profiles were valid, but settings could not be saved.", 0
    return (
        True,
        f"Imported {imported} performance profile(s); mapped {mapped} model file(s).",
        imported,
    )


def get_model_performance_target(model_path: Path) -> Optional[str]:
    """Return the remembered fastest/manual target for one exact GGUF."""
    settings = load_settings()
    key = favorite_model_key(model_path)
    by_path = settings.get(_os_path_key("model_performance_targets"))
    if key and isinstance(by_path, dict):
        target = _normalise_performance_target(by_path.get(key))
        if target:
            return target
    portable = portable_model_key(model_path)
    by_portable = settings.get(_os_path_key("model_performance_targets_portable"))
    if portable and isinstance(by_portable, dict):
        target = _normalise_performance_target(by_portable.get(portable))
        if target:
            return target
    return None


def set_model_performance_target(model_path: Path, performance_target: str) -> None:
    """Remember the selected/fastest target for this model and portable identity."""
    target = _normalise_performance_target(performance_target)
    key = favorite_model_key(model_path)
    if not target or not key:
        return
    settings = load_settings()
    storage_key = _os_path_key("model_performance_targets")
    by_path = settings.get(storage_key)
    if not isinstance(by_path, dict):
        by_path = {}
    by_path[key] = target
    settings[storage_key] = by_path
    portable = portable_model_key(model_path)
    if portable:
        portable_key = _os_path_key("model_performance_targets_portable")
        by_portable = settings.get(portable_key)
        if not isinstance(by_portable, dict):
            by_portable = {}
        by_portable[portable] = target
        settings[portable_key] = by_portable
    save_settings(settings)


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
    return val if val in ("safe", "balanced", "throughput", "low_vram") else None


def set_performance_target(name: str) -> None:
    """Persist the GUI performance-target choice. Empty string clears it."""
    name = (name or "").lower().strip()
    if name in ("safe", "balanced", "throughput", "low_vram", ""):
        _update("performance_target", name)


# ---------------------------------------------------------------------------
# Sampling mode (chat / coding)
#
# Each YAML profile (new format) carries two sampling sub-blocks:
#   sampling:
#     chat:   { temperature: 1.0, top_k: 64, ... }
#     coding: { temperature: 1.5, top_k: 64, ... }
#
# The active mode is a global GUI choice (not per-model) — most users
# stay in one mode for hours, switch to coding when they pair-program,
# switch back. Per-model overrides would only add UI clutter without
# matching the actual workflow.

_VALID_MODES = ("chat", "coding")


def get_mode() -> Optional[str]:
    """Return the persisted sampling mode ("chat" / "coding"), or None."""
    val = load_settings().get("mode")
    if not val:
        return None
    val = str(val).lower().strip()
    return val if val in _VALID_MODES else None


def set_mode(name: str) -> None:
    """Persist the GUI sampling-mode choice. Empty string clears it."""
    name = (name or "").lower().strip()
    if name in _VALID_MODES + ("",):
        _update("mode", name)


# ---------------------------------------------------------------------------
# Model browser view (flat list / folder tree)

_VALID_MODEL_VIEW_MODES = ("list", "tree")


def get_model_view_mode() -> str:
    """Return the persisted model-browser mode, defaulting safely to list."""
    value = str(load_settings().get("model_view_mode", "list")).lower().strip()
    return value if value in _VALID_MODEL_VIEW_MODES else "list"


def set_model_view_mode(mode: str) -> None:
    """Persist the model-browser mode when it is one of the supported values."""
    value = str(mode or "").lower().strip()
    if value in _VALID_MODEL_VIEW_MODES:
        _update("model_view_mode", value)


def _valid_model_tree_path(value: Any) -> bool:
    """Return whether *value* is one of the tree's stable branch identities."""
    return isinstance(value, str) and (
        value == "favorites" or (value.startswith("folder:") and len(value) > 7)
    )


def get_model_tree_collapsed_paths() -> set[str]:
    """Return folder-tree branches the user explicitly collapsed on this OS."""
    settings = load_settings()
    raw = settings.get(_os_path_key("model_tree_collapsed_paths"))
    if not isinstance(raw, list):
        # Plain-key fallback keeps hand-written/early-development settings valid.
        raw = settings.get("model_tree_collapsed_paths")
    if not isinstance(raw, list):
        return set()
    return {value for value in raw if _valid_model_tree_path(value)}


def set_model_tree_collapsed_paths(paths: set[str]) -> None:
    """Persist the exact set of manually collapsed model-tree branches."""
    clean = {value for value in paths if _valid_model_tree_path(value)}
    settings = load_settings()
    settings[_os_path_key("model_tree_collapsed_paths")] = sorted(
        clean, key=str.casefold
    )
    save_settings(settings)


# ---------------------------------------------------------------------------
# Window geometry & state
#
# Qt's QMainWindow can hand us two opaque QByteArrays:
#   * saveGeometry()  → size, position, screen, maximize/fullscreen state
#   * saveState()     → toolbar/dock/splitter positions
#
# We persist them as base64 strings (the only safe round-trip for
# arbitrary bytes inside JSON). On restart the GUI passes the bytes
# back to restoreGeometry/restoreState; if anything is corrupted or
# from an incompatible Qt version, those calls just return False and
# the GUI falls back to the hard-coded default size.


def _get_b64(key: str) -> Optional[str]:
    val = load_settings().get(key)
    if not isinstance(val, str) or not val:
        return None
    # Defensive: ignore obviously-broken payloads so a corrupt JSON
    # never crashes the GUI launch path.
    try:
        base64.b64decode(val, validate=True)
    except (ValueError, TypeError):
        return None
    return val


def get_window_geometry() -> Optional[str]:
    """Return the persisted QMainWindow.saveGeometry() blob (base64)."""
    return _get_b64("window_geometry")


def set_window_geometry(b64_value: str) -> None:
    """Persist the base64-encoded saveGeometry() output."""
    if isinstance(b64_value, str):
        _update("window_geometry", b64_value)


def get_window_state() -> Optional[str]:
    """Return the persisted QMainWindow.saveState() blob (base64)."""
    return _get_b64("window_state")


def set_window_state(b64_value: str) -> None:
    """Persist the base64-encoded saveState() output."""
    if isinstance(b64_value, str):
        _update("window_state", b64_value)


# ---------------------------------------------------------------------------
# Inner splitter layout
#
# QMainWindow.saveState() only round-trips toolbars and dock widgets — it
# does NOT capture the position of plain QSplitter handles that live inside
# the central widget. The AutoTuner GUI arranges its panes with two named
# QSplitters (top horizontal: model-list | config, and the vertical split
# between that row and the log panel). To remember the *inner* arrangement
# (not just the outer window size), each splitter's own saveState() blob is
# stored here under a stable object name.
#
# Schema:
#   "splitter_states": { "top_split": "<b64>", "main_split": "<b64>", ... }


def get_splitter_state(name: str) -> Optional[str]:
    """Return the persisted QSplitter.saveState() blob (base64) for *name*."""
    if not name:
        return None
    bucket = load_settings().get("splitter_states")
    if not isinstance(bucket, dict):
        return None
    val = bucket.get(name)
    if not isinstance(val, str) or not val:
        return None
    try:
        base64.b64decode(val, validate=True)
    except (ValueError, TypeError):
        return None
    return val


def set_splitter_state(name: str, b64_value: str) -> None:
    """Persist the base64-encoded saveState() of the QSplitter *name*."""
    if not name or not isinstance(b64_value, str):
        return
    s = load_settings()
    bucket = s.get("splitter_states")
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[name] = b64_value
    s["splitter_states"] = bucket
    save_settings(s)


# ---------------------------------------------------------------------------
# Per-model mmproj (vision projector) selection
#
# A model can ship several projector precisions side by side (bf16 / f16 /
# f32). The scanner auto-picks one, but the user may prefer a different
# precision. We remember the chosen projector *filename* per model so the
# choice sticks across restarts. Stored as the bare filename (not full
# path) because models move between drives; the GUI matches it back against
# the freshly-scanned candidate list and falls back to the auto pick when
# the remembered file is no longer present.
#
# Schema:
#   "mmproj_selection": { "<model_name>": "mmproj-…-f32.gguf", ... }

MMPROJ_NONE_SENTINEL = "<none>"


def get_mmproj_selection(model_name: str) -> Optional[str]:
    """Return the remembered mmproj filename for *model_name*.

    Returns the literal ``"<none>"`` sentinel when the user explicitly chose
    no projector, the chosen filename when one was picked, or ``None`` when
    there is no stored preference (caller uses the scanner's auto pick).
    """
    if not model_name:
        return None
    bucket = load_settings().get("mmproj_selection")
    if not isinstance(bucket, dict):
        return None
    val = bucket.get(model_name)
    return val if isinstance(val, str) and val else None


def set_mmproj_selection(model_name: str, filename: Optional[str]) -> None:
    """Persist (or clear) the chosen mmproj filename for *model_name*.

    Pass ``None`` / empty to drop the override (model falls back to the
    scanner's automatic best pick).
    """
    if not model_name:
        return
    s = load_settings()
    bucket = s.get("mmproj_selection")
    if not isinstance(bucket, dict):
        bucket = {}
    if not filename:
        bucket.pop(model_name, None)
    else:
        bucket[model_name] = str(filename)
    s["mmproj_selection"] = bucket
    save_settings(s)


# ---------------------------------------------------------------------------
# Per-model draft (speculative-decoding head) selection.
#
# Mirrors mmproj_selection. The GUI exposes an always-on dropdown listing
# every draft/assistant GGUF in the model's folder; the chosen filename is
# remembered here. A sentinel empty string is NOT used — absence of a key
# means "use the scanner's auto pick", and the explicit literal "<none>"
# means "the user deliberately chose no draft" (so we don't silently
# re-enable the auto draft on the next launch).
#
# Schema:
#   "draft_selection": {
#       "<model_name>": "…-assistant-Q4_K_M.gguf" | "<embedded-mtp>" | "<none>"
#   }

DRAFT_NONE_SENTINEL = "<none>"
DRAFT_EMBEDDED_SENTINEL = "<embedded-mtp>"


def get_draft_selection(model_name: str) -> Optional[str]:
    """Return the remembered draft filename for *model_name*.

    Returns ``"<none>"`` when the user explicitly chose no draft,
    ``"<embedded-mtp>"`` for the main GGUF's internal head, the chosen
    filename for an external draft, or ``None`` when there is no preference.
    """
    if not model_name:
        return None
    bucket = load_settings().get("draft_selection")
    if not isinstance(bucket, dict):
        return None
    val = bucket.get(model_name)
    return val if isinstance(val, str) and val else None


def set_draft_selection(model_name: str, filename: Optional[str]) -> None:
    """Persist (or clear) the chosen draft filename for *model_name*.

    Pass the filename to remember an external draft, ``"<embedded-mtp>"``
    for the internal head, ``"<none>"`` for no draft, or ``None`` / empty to
    drop the override entirely (model reverts to the scanner's automatic pick).
    """
    if not model_name:
        return
    s = load_settings()
    bucket = s.get("draft_selection")
    if not isinstance(bucket, dict):
        bucket = {}
    if not filename:
        bucket.pop(model_name, None)
    else:
        bucket[model_name] = str(filename)
    s["draft_selection"] = bucket
    save_settings(s)


# ---------------------------------------------------------------------------
# Global font size
#
# The A+/A- toolbar buttons should affect the whole UI, not just the
# config preview and the log panel. We persist the chosen point size
# so a user who picked size 14 keeps size 14 across restarts.

_FONT_SIZE_MIN = 7
_FONT_SIZE_MAX = 22
_FONT_SIZE_DEFAULT = 10


# ---------------------------------------------------------------------------
# Server base port + offset
#
# The "Base port" field in the launcher toolbar selects the port the FIRST
# llama-server binds to (subsequent concurrent servers get base+1, base+2…).
# Persisting it means a user who switched away from the 1234 default — e.g.
# to avoid clashing with another local service — does not have to re-enter it
# on every restart. The manual port offset (0..10) is persisted alongside so
# the whole port-selection state round-trips. Both fall back to the hardcoded
# defaults when nothing is stored yet.

_BASE_PORT_MIN = 1
_BASE_PORT_MAX = 65535
_BASE_PORT_DEFAULT = 1234

_PORT_OFFSET_MIN = 0
_PORT_OFFSET_MAX = 10
_PORT_OFFSET_DEFAULT = 0

_PROMPT_CACHE_RAM_MIB_MIN = -1
_PROMPT_CACHE_RAM_MIB_MAX = 65536
_PROMPT_CACHE_RAM_MIB_DEFAULT = 2048


def get_base_port() -> int:
    """Return the persisted server base port (default 1234, clamped to 1..65535)."""
    val = load_settings().get("base_port")
    try:
        n = int(val) if val is not None else _BASE_PORT_DEFAULT
    except (TypeError, ValueError):
        return _BASE_PORT_DEFAULT
    return max(_BASE_PORT_MIN, min(_BASE_PORT_MAX, n))


def set_base_port(port: int) -> None:
    """Persist the server base port (clamped to the valid range)."""
    try:
        n = int(port)
    except (TypeError, ValueError):
        return
    _update("base_port", max(_BASE_PORT_MIN, min(_BASE_PORT_MAX, n)))


def get_port_offset() -> int:
    """Return the persisted manual port offset (default 0, clamped to 0..10)."""
    val = load_settings().get("port_offset")
    try:
        n = int(val) if val is not None else _PORT_OFFSET_DEFAULT
    except (TypeError, ValueError):
        return _PORT_OFFSET_DEFAULT
    return max(_PORT_OFFSET_MIN, min(_PORT_OFFSET_MAX, n))


def set_port_offset(offset: int) -> None:
    """Persist the manual port offset (clamped to the valid range)."""
    try:
        n = int(offset)
    except (TypeError, ValueError):
        return
    _update("port_offset", max(_PORT_OFFSET_MIN, min(_PORT_OFFSET_MAX, n)))


def get_prompt_cache_ram_mib() -> int:
    """Return the global host prompt-cache limit in MiB (default 2048).

    ``-1`` preserves llama.cpp's unlimited mode, while ``0`` disables the
    cache. The launch checkbox remains the authoritative on/off control.
    """
    val = load_settings().get("prompt_cache_ram_mib")
    try:
        n = int(val) if val is not None else _PROMPT_CACHE_RAM_MIB_DEFAULT
    except (TypeError, ValueError):
        return _PROMPT_CACHE_RAM_MIB_DEFAULT
    return max(_PROMPT_CACHE_RAM_MIB_MIN, min(_PROMPT_CACHE_RAM_MIB_MAX, n))


def set_prompt_cache_ram_mib(value: int) -> None:
    """Persist the global host prompt-cache limit in MiB."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return
    _update(
        "prompt_cache_ram_mib",
        max(_PROMPT_CACHE_RAM_MIB_MIN, min(_PROMPT_CACHE_RAM_MIB_MAX, n)),
    )


def get_turbo_kv_warning_suppressed() -> bool:
    """Return whether the user chose ``Never Show Again`` for Turbo KV."""
    return load_settings().get("turbo_kv_warning_suppressed") is True


def set_turbo_kv_warning_suppressed(suppressed: bool) -> None:
    """Persist the global TurboQuant special-fork warning preference."""
    _update("turbo_kv_warning_suppressed", bool(suppressed))


def get_font_size() -> int:
    """Return the persisted GUI point size; clamped to a sane range."""
    val = load_settings().get("font_size")
    try:
        n = int(val) if val is not None else _FONT_SIZE_DEFAULT
    except (TypeError, ValueError):
        return _FONT_SIZE_DEFAULT
    return max(_FONT_SIZE_MIN, min(_FONT_SIZE_MAX, n))


def set_font_size(size: int) -> None:
    """Persist the GUI point size (clamped to the safe range)."""
    try:
        n = int(size)
    except (TypeError, ValueError):
        return
    n = max(_FONT_SIZE_MIN, min(_FONT_SIZE_MAX, n))
    _update("font_size", n)


# ---------------------------------------------------------------------------
# Appearance theme


def get_theme_id() -> str:
    """Return the selected namespaced theme, defaulting safely to System."""
    value = load_settings().get("theme_id")
    return value if isinstance(value, str) and value else "builtin:system"


def set_theme_id(theme_id: str) -> None:
    """Persist a namespaced ThemeManager id; validation happens at load time."""
    if isinstance(theme_id, str) and theme_id:
        _update("theme_id", theme_id)


# ---------------------------------------------------------------------------
# Application behaviour


def get_minimize_on_close() -> bool:
    """Return whether title-bar X should hide in the notification area.

    This is deliberately opt-in: missing settings and non-boolean legacy
    values both resolve to ``False``.
    """
    return load_settings().get("minimize_on_close") is True


def set_minimize_on_close(enabled: bool) -> None:
    """Persist the opt-in X-to-notification-area behaviour."""
    _update("minimize_on_close", bool(enabled))


# ---------------------------------------------------------------------------
# Reasoning effort (per model)
#
# Some models (gpt-oss, certain Nemotron / Qwen3.5+ variants) honour a
# ``reasoning_effort`` kwarg that controls how much the model "thinks"
# before answering. Llama-server passes the value through to the chat
# template via ``--chat-template-kwargs '{"reasoning_effort":"high"}'``.
#
# Officially recognised values across the ecosystem:
#   * "low" / "medium" / "high"  — gpt-oss + Qwen3.5+ canonical set
#   * "minimal"                  — some Qwen3.6 builds
#   * "auto"                     — sentinel meaning "no flag, let the
#                                   chat template / model decide"
#
# "extra high" is not standardised upstream but several recent Qwen3.6
# community builds accept it; we keep it as an option and let the user
# discover whether their build supports it.
#
# Storage: per-model, alongside vision/draft/thinking overrides.

_VALID_REASONING = ("auto", "off", "minimal", "low", "medium", "high", "extra_high")


def get_reasoning_effort(model_name: str) -> Optional[str]:
    """Return the persisted reasoning_effort for ``model_name`` or None."""
    if not model_name:
        return None
    val = (load_settings().get("reasoning_effort") or {}).get(model_name)
    if not isinstance(val, str):
        return None
    val = val.lower().strip()
    return val if val in _VALID_REASONING else None


def set_reasoning_effort(model_name: str, value: Optional[str]) -> None:
    """Persist (or clear) the reasoning_effort for ``model_name``.

    Pass ``None`` or an empty string to drop the override (model falls
    back to "auto" — i.e. no CLI flag at all).
    """
    if not model_name:
        return
    s = load_settings()
    bucket = s.get("reasoning_effort")
    if not isinstance(bucket, dict):
        bucket = {}
    if not value:
        bucket.pop(model_name, None)
    else:
        v = value.lower().strip()
        if v not in _VALID_REASONING:
            return
        bucket[model_name] = v
    s["reasoning_effort"] = bucket
    save_settings(s)


def settings_file_location() -> Path:
    """Where settings are (or would be) written. For diagnostic logging."""
    return _settings_file()


# ---------------------------------------------------------------------------
# GPU priority overrides
#
# The user can mark each GPU with a priority (≥1) via the "gpu_overrides"
# section of autotuner_settings.json:
#
#   "gpu_overrides": {
#       "AMD Radeon AI PRO R9700":   { "enabled": true, "priority": 2 },
#       "AMD Radeon RX 9070 XT":     { "enabled": true, "priority": 1 }
#   }
#
# Higher priority → that GPU is preferred as the primary compute device
# (main_gpu in --tensor-split / --main-gpu).  When two GPUs have the same
# VRAM size the priority breaks the tie.  When VRAM sizes differ (e.g.
# 32 GB vs 16 GB) the score = priority × vram_mb already gives the larger
# GPU a comfortable lead, so the user can rely on VRAM winning naturally
# unless they explicitly want to invert the preference.


def get_gpu_priorities() -> Dict[str, int]:
    """Return a mapping of GPU name → user-assigned priority for all GPUs
    that have a priority entry in gpu_overrides.  Missing keys default to 1.
    """
    overrides = load_settings().get("gpu_overrides") or {}
    if not isinstance(overrides, dict):
        return {}
    result: Dict[str, int] = {}
    for gpu_name, entry in overrides.items():
        if not isinstance(entry, dict):
            continue
        try:
            result[str(gpu_name)] = max(1, int(entry.get("priority", 1)))
        except (TypeError, ValueError):
            result[str(gpu_name)] = 1
    return result


def get_gpu_priority(gpu_name: str) -> int:
    """Return the user-assigned priority for *gpu_name* (default 1)."""
    if not gpu_name:
        return 1
    overrides = load_settings().get("gpu_overrides") or {}
    entry = overrides.get(gpu_name) if isinstance(overrides, dict) else None
    if not isinstance(entry, dict):
        return 1
    try:
        return max(1, int(entry.get("priority", 1)))
    except (TypeError, ValueError):
        return 1


def set_gpu_priority(gpu_name: str, priority: int) -> None:
    """Persist *priority* for *gpu_name* in gpu_overrides.

    Creates the entry if it doesn't exist; leaves other fields (e.g.
    ``enabled``) untouched.
    """
    if not gpu_name:
        return
    s = load_settings()
    overrides = s.get("gpu_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    entry = overrides.get(gpu_name)
    if not isinstance(entry, dict):
        entry = {}
    try:
        entry["priority"] = max(1, int(priority))
    except (TypeError, ValueError):
        entry["priority"] = 1
    overrides[gpu_name] = entry
    s["gpu_overrides"] = overrides
    save_settings(s)


# ---------------------------------------------------------------------------
# Forced GPU (hard pin for the next server launch)
#
# Stored as a top-level string under "forced_gpu". When set to a GPU name
# (or a distinctive substring of it, e.g. "R9700"), compute_config pins the
# server to that single card and hides the others — the manual "boot only on
# the GPU I choose" control used when launching a second server so it lands
# on the still-empty card instead of piling onto an already-full one. An
# empty string / missing key means "auto" (free-VRAM-aware selection).


def get_forced_gpu() -> Optional[str]:
    """Return the GPU name the next launch is pinned to, or None for auto."""
    val = load_settings().get("forced_gpu")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def set_forced_gpu(gpu_name: Optional[str]) -> None:
    """Pin launches to *gpu_name* exclusively, or clear the pin when None/empty."""
    s = load_settings()
    if gpu_name and gpu_name.strip():
        s["forced_gpu"] = gpu_name.strip()
    else:
        s.pop("forced_gpu", None)
    save_settings(s)
