"""Performance target presets for the AutoTuner.

Each target is a coherent bundle of safety/headroom values that controls
how aggressively the AutoTuner uses GPU VRAM and system RAM. The goal
is to give users one knob ("safe / balanced / throughput") instead of
forcing them to tune four interacting numbers individually.

Three tiers
-----------
- **safe**: conservative reservations. Best for long-context sessions
  (>64k tokens) and users who prefer "it just works" over peak speed.
  Equivalent to the AutoTuner's pre-perf-target behaviour.
- **balanced** (default): moderate reservations. Mild VRAM
  optimisation that benefits everyone on most workloads.
- **throughput**: aggressive reservations. Optimised for short-context
  inference (~32k) where you want every available expert layer to sit
  in VRAM. Trades context headroom for tokens-per-second.

Resolution priority (highest wins)
----------------------------------
1. Explicit user choice (CLI flag, GUI dropdown).
2. YAML profile field ``performance_target:``. Lets a model author
   recommend a target appropriate for the architecture (e.g. a tiny
   3B dense model rarely benefits from "safe").
3. Module default (``balanced``).

Unknown values fall back silently to the default — we never want a
typo in YAML to crash the tuner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PerformanceTarget:
    """Coherent bundle of placement parameters."""

    name: str
    moe_vram_safety_gb: float
    moe_placement_ctx_target: int
    dense_vram_safety_gb: float
    ram_safety_gb: float
    description: str

    def __str__(self) -> str:  # pragma: no cover — trivial
        return self.name


# ---------------------------------------------------------------------------
# Registry. Add new tiers here; nothing else needs to change.

PERFORMANCE_TARGETS: Dict[str, PerformanceTarget] = {
    "safe": PerformanceTarget(
        name="safe",
        moe_vram_safety_gb=0.30,
        moe_placement_ctx_target=131072,   # 128k — full long-context budget
        dense_vram_safety_gb=0.30,
        ram_safety_gb=1.50,
        description=(
            "Conservative. KV reserved for 128k context, generous "
            "safety bands. Pick this for long-context sessions."
        ),
    ),
    "balanced": PerformanceTarget(
        name="balanced",
        moe_vram_safety_gb=0.25,
        moe_placement_ctx_target=65536,    # 64k — typical working ceiling
        dense_vram_safety_gb=0.25,
        ram_safety_gb=1.25,
        description=(
            "Default. KV reserved for 64k context — enough headroom "
            "for most chats while letting more expert layers fit on GPU."
        ),
    ),
    "throughput": PerformanceTarget(
        name="throughput",
        moe_vram_safety_gb=0.15,
        moe_placement_ctx_target=32768,    # 32k — short reasoning / coding
        dense_vram_safety_gb=0.15,
        ram_safety_gb=1.00,
        description=(
            "Aggressive. KV reserved for only 32k — every spare GB "
            "of VRAM goes to expert layers. Best tokens/s on tight "
            "MoE setups; not recommended above ~32k context."
        ),
    ),
}

DEFAULT_TARGET_NAME = "balanced"


# ---------------------------------------------------------------------------
# Public API

def list_target_names() -> List[str]:
    """Return target names in display order (safe → balanced → throughput)."""
    return ["safe", "balanced", "throughput"]


def get_target(name: str) -> Optional[PerformanceTarget]:
    """Return target by name (case-insensitive); ``None`` if unknown."""
    if not name:
        return None
    return PERFORMANCE_TARGETS.get(name.lower().strip())


def resolve_performance_target(
    cli_choice: Optional[str] = None,
    profile_choice: Optional[str] = None,
    default: str = DEFAULT_TARGET_NAME,
) -> PerformanceTarget:
    """Resolve which target to use.

    Tries ``cli_choice`` first, then ``profile_choice``, then ``default``.
    Unknown / empty values are silently skipped so a single bad source
    never breaks the chain. Always returns a valid ``PerformanceTarget``.
    """
    for choice in (cli_choice, profile_choice, default):
        target = get_target(choice) if choice else None
        if target is not None:
            return target
    # Defensive fallback. ``DEFAULT_TARGET_NAME`` must exist in the registry.
    return PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME]


def describe_targets() -> str:
    """Multiline summary for ``--help`` text and GUI tooltips."""
    lines = []
    for name in list_target_names():
        t = PERFORMANCE_TARGETS[name]
        lines.append(f"  {name:<11} {t.description}")
    return "\n".join(lines)