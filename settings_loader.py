"""Load YAML profiles from the settings/ folder and match them
against model filenames."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "PyYAML is required. Install with:  pip install -r requirements.txt"
    ) from e


@dataclass
class ModelProfile:
    display_name: str
    patterns: List[str] = field(default_factory=list)
    max_context: int = 8192
    sampling: Dict[str, Any] = field(default_factory=dict)
    chat_template: Optional[str] = None
    recommended_kv_quant: str = "q8_0"
    extra_args: List[str] = field(default_factory=list)
    notes: str = ""
    source_file: Optional[str] = None  # which YAML this came from
    # Optional: override the llama-server binary for this model family.
    # Used e.g. by Ternary-Bonsai (BitNet) to invoke a 1bllama build.
    server_binary: Optional[str] = None


def load_profiles(settings_dir: Path) -> List[ModelProfile]:
    """Load every *.yaml / *.yml file in settings_dir."""
    profiles: List[ModelProfile] = []
    if not settings_dir.exists():
        return profiles

    files = sorted(list(settings_dir.glob("*.yaml"))
                   + list(settings_dir.glob("*.yml")))
    for yml in files:
        try:
            with yml.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"[AutoTuner] Warning: failed to load {yml.name}: {e}")
            continue

        sampling = data.get("sampling") or {}
        if not isinstance(sampling, dict):
            sampling = {}
        extra = data.get("extra_args") or []
        if not isinstance(extra, list):
            extra = []

        profiles.append(ModelProfile(
            display_name=str(data.get("display_name", yml.stem)),
            patterns=[str(p).lower() for p in (data.get("patterns") or [])],
            max_context=int(data.get("max_context", 8192)),
            sampling=sampling,
            chat_template=data.get("chat_template"),
            recommended_kv_quant=str(data.get("recommended_kv_quant", "q8_0")),
            extra_args=[str(x) for x in extra],
            notes=str(data.get("notes", "") or ""),
            source_file=yml.name,
            server_binary=(str(data["server_binary"])
                           if data.get("server_binary") else None),
        ))
    return profiles


def match_profile(
    model_name: str,
    profiles: List[ModelProfile],
) -> ModelProfile:
    """Pick the best-matching profile for the given model filename.

    Rule: case-insensitive substring match on each pattern; the longest
    pattern wins. Profiles with empty `patterns:` are treated as fallback.
    """
    name_lower = model_name.lower()
    best: Optional[ModelProfile] = None
    best_len = -1
    fallback: Optional[ModelProfile] = None

    for p in profiles:
        if not p.patterns:
            if fallback is None:
                fallback = p
            continue
        for pat in p.patterns:
            if pat and pat in name_lower and len(pat) > best_len:
                best = p
                best_len = len(pat)
                # Don't break — a later pattern in the same file might be longer

    return best or fallback or ModelProfile(
        display_name="builtin-default",
        max_context=8192,
        sampling={
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.05,
            "repeat_penalty": 1.05,
        },
    )
