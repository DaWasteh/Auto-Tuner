from __future__ import annotations

import math
import re
import platform
import os
import shutil
import subprocess
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from hardware import SystemInfo, GPUInfo
from scanner import ModelEntry
from settings_loader import ModelProfile
from performance_target import (
    PerformanceTarget,
    PERFORMANCE_TARGETS,
    resolve_performance_target,
    DEFAULT_TARGET_NAME,
)

ctypes: Any = None
try:
    import ctypes as _ctypes

    ctypes = _ctypes
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Tunables. Kept at module scope so tests / callers can override them.
#
# These are now thin compat shims — the real values come from the active
# PerformanceTarget. Keeping the module constants means external callers
# (tests, scripts) that monkey-patched them in the past keep working, and
# reading the constants still gives the "balanced" defaults.

DEFAULT_VRAM_SAFETY_GB = PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME].dense_vram_safety_gb
DEFAULT_RAM_SAFETY_GB = PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME].ram_safety_gb

# MoE-specific knobs. Read from the "balanced" preset for back-compat.
MOE_VRAM_SAFETY_GB = PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME].moe_vram_safety_gb
MOE_PLACEMENT_CTX_TARGET = PERFORMANCE_TARGETS[
    DEFAULT_TARGET_NAME
].moe_placement_ctx_target
MOE_KV_RESERVE_FRAC = 0.06

# Capacity-first stock llama.cpp KV default. Q4_0/Q4_0 is supported by the
# current Vulkan and HIP FlashAttention paths and avoids mixed-quant backend
# fallbacks. Expert/manual pins remain authoritative, and runner-specific
# exceptions (currently DiffusionGemma) still report the cache they really use.
DEFAULT_KV_CACHE_TYPE = "q4_0"

# Default host-RAM prompt-cache size in MiB (``--cache-ram``). A bounded,
# computed default replaces the previous unlimited/uncomputed value so the
# cache no longer silently consumes all available RAM. -1 keeps the legacy
# "unlimited" semantics (planned conservatively as a 2 GiB reservation).
PROMPT_CACHE_RAM_MIB_DEFAULT = 2048
# Conservative reservation used when the legacy -1 (unlimited) cache is
# requested, so the RAM budget never under-provisions against an unbounded
# host-RAM cache.
PROMPT_CACHE_UNLIMITED_RESERVE_GB = 2.0
# Extra VRAM the DiffusionGemma diffusion-server (PR #24427) reserves for its
# diffusion-runtime buffers beyond weights + KV. The PR server parses common
# args but ignores cache_type_k/v and n_cpu_moe overrides, so this overhead
# plus a forced-F16 KV estimate hardens the config against Vulkan OOM.
DIFFUSION_GEMMA_RUNTIME_VRAM_OVERHEAD_GB = 1.5

# Compute-buffer headroom a model needs ON TOP of its weights before the
# AutoTuner will commit to a FULL GPU offload (ngl 999). llama.cpp allocates
# a flash-attention / compute workspace on top of the weights (~0.3–0.8 GB
# for a dense model). Without this reserve a model that fits the WEIGHTS but
# not weights+compute was marked full_off=True, and the GUI pre-launch check
# then refused it on low-VRAM cards (8 GB box: a 7 GB model + 1 GB margin >
# 7.4 GB free → "Not enough free VRAM" for a model that genuinely fits).
# Reserving this headroom makes such models fall through to PARTIAL offload,
# where the overflow layers spill to CPU and the server runs fine.
FULL_OFF_HEADROOM_GB = 0.5

# Qwen3.8-Flash-Next (qwen4exp) uses a context×ubatch QSA graph that is much
# larger than an ordinary FlashAttention workspace. Exact b10666 Windows/
# Vulkan measurements at 90,112 context are linear across ubatch 256/1024:
# ~64 B of device buffers and ~204 B of host buffers per ctx×ubatch element.
# Round upward for allocator drift. ubatch 256 retained 98.7% of decode speed
# and ~89.6% of prompt speed versus 1024; Safe/Balanced step down to 64/128
# before sacrificing requested context because the scaling is linear.
_QWEN4EXP_UBATCH = 256
_QWEN4EXP_GPU_COMPUTE_BYTES = 72
_QWEN4EXP_HOST_COMPUTE_BYTES = 215
_QWEN4EXP_FIXED_HOST_RUNTIME_GB = 5.5

# ``--lazy-mode auto`` (``--tensor-read-lazy`` before b10700) maps the complete
# row table into the process address space, but untouched file-backed pages are
# reclaimable and do not consume
# committed/physical RAM merely because MapViewOfFile/mmap spans them. Budget
# the measured active-row working set (upstream observed about 4.4%) rather
# than the full virtual mapping. Keep the full mapping visible separately so
# users can still see the paging/I/O pressure it may create.
_READ_LAZY_RESIDENT_FRACTION = 0.05
_READ_LAZY_RESIDENT_MIN_GB = 0.5


def _qwen4exp_ubatch_for_target(target_name: str) -> int:
    """Bound QSA graph memory by tier before sacrificing requested context."""
    name = str(target_name or "").strip().lower()
    if name in {"safe", "low_vram"}:
        return 64
    if name == "balanced":
        return 128
    return _QWEN4EXP_UBATCH


def qwen4exp_compute_buffers_gb(
    ctx: int, ubatch: int = _QWEN4EXP_UBATCH, n_parallel: int = 1
) -> Tuple[float, float]:
    """Return conservative b10666 qwen4exp ``(VRAM, RAM)`` graph buffers."""
    elements = max(0, int(ctx)) * max(1, int(ubatch)) * max(1, int(n_parallel))
    return (
        elements * _QWEN4EXP_GPU_COMPUTE_BYTES / (1024.0**3),
        elements * _QWEN4EXP_HOST_COMPUTE_BYTES / (1024.0**3),
    )


# ---------------------------------------------------------------------------
# llama.cpp CLI compatibility helpers

# Flags for which the next argv token is a value. Used only when pruning a
# command for an older llama.cpp binary whose --help output does not advertise
# a flag the current AutoTuner knows about.
_ARG_FLAGS_WITH_VALUES: Set[str] = {
    "-a",
    "-b",
    "-c",
    "-cram",
    "-ctk",
    "-ctv",
    "-dev",
    "-fa",
    "-m",
    "-md",
    "-mmdev",
    "-lm",
    "-n",
    "-ngl",
    "-np",
    "-p",
    "-rea",
    "-sm",
    "-t",
    "-tb",
    "-ub",
    "--alias",
    "--batch-size",
    "--cache-ram",
    "--cache-type-k",
    "--cache-type-v",
    "--chat-template",
    "--chat-template-file",
    "--chat-template-kwargs",
    "--ctx-size",
    "--diffusion-algorithm",
    "--diffusion-block-length",
    "--diffusion-eps",
    "--diffusion-steps",
    "--device",
    "--dry-allowed-length",
    "--dry-base",
    "--dry-multiplier",
    "--dry-penalty-last-n",
    "--dry-sequence-breaker",
    "--fit",
    "--flash-attn",
    "--frequency-penalty",
    "--gpu-layers",
    "--host",
    "--image-max-tokens",
    "--image-min-tokens",
    "--load-mode",
    "--main-gpu",
    "--media-path",
    "--mcp-servers-config",
    "--mcp-servers-json",
    "--min-p",
    "--mmproj",
    "--mmproj-device",
    "--models-dir",
    "--models-max",
    "--models-preset",
    "--mtmd-batch-max-tokens",
    "--model",
    "--model-draft",
    "--n-cpu-moe",
    "--n-gpu-layers",
    "--numa",
    "--parallel",
    "--pooling",
    "--port",
    "--predict",
    "--presence-penalty",
    "--prompt",
    "--repeat-last-n",
    "--reasoning",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--reasoning-effort",
    "--repeat-penalty",
    "--rope-scale",
    "--rope-scaling",
    "--samplers",
    "--spec-draft-model",
    "--spec-draft-n-max",
    "--spec-draft-n-min",
    "--spec-draft-ngl",
    "--spec-draft-p-min",
    "--spec-draft-p-split",
    "--spec-ngram-map-k4v-min-hits",
    "--spec-ngram-map-k4v-size-m",
    "--spec-ngram-map-k4v-size-n",
    "--spec-ngram-mod-n-match",
    "--spec-ngram-mod-n-max",
    "--spec-ngram-mod-n-min",
    "--spec-type",
    "--slot-prompt-similarity",
    "--split-mode",
    "--temp",
    "--lazy-mode",
    "--tensor-read-lazy",
    "--tensor-split",
    "--threads",
    "--threads-batch",
    "--tools",
    "--tools-runtime",
    "--top-k",
    "--top-p",
    "--ubatch-size",
    "-lzm",
}

_FLAG_ALIAS_GROUPS: Tuple[Set[str], ...] = (
    {"-a", "--alias"},
    {"-b", "--batch-size"},
    {"-c", "--ctx-size"},
    {"-ctk", "--cache-type-k"},
    {"-ctv", "--cache-type-v"},
    {"-dev", "--device"},
    {"-fa", "--flash-attn"},
    {"-m", "--model"},
    {"-md", "--model-draft"},
    {"-mmdev", "--mmproj-device"},
    {"-lm", "--load-mode"},
    {"-lzm", "--lazy-mode"},
    {"-n", "--predict"},
    {"-ngl", "--gpu-layers", "--n-gpu-layers"},
    {"-np", "--parallel"},
    {"-p", "--prompt"},
    {"-sm", "--split-mode"},
    {"-t", "--threads"},
    {"-tb", "--threads-batch"},
    {"-ub", "--ubatch-size"},
    {"-cram", "--cache-ram"},
    {"-rea", "--reasoning"},
)

_FLAG_RE = re.compile(r"(?<![\w-])-{1,2}[A-Za-z][A-Za-z0-9_-]*")


def _expand_supported_flag_aliases(flags: Set[str]) -> Set[str]:
    """Add known llama.cpp short/long aliases to a parsed --help flag set."""
    expanded = set(flags)
    changed = True
    while changed:
        changed = False
        for group in _FLAG_ALIAS_GROUPS:
            if expanded & group and not group <= expanded:
                expanded.update(group)
                changed = True
    return expanded


def _flag_name(token: str) -> str:
    """Return the flag part of an argv token, stripping an inline '=value'."""
    return token.split("=", 1)[0]


def _filter_command_for_supported_flags(
    cmd: List[str], supported_flags: Set[str]
) -> Tuple[List[str], List[str]]:
    """Drop argv flags absent from a llama.cpp binary's advertised --help.

    New AutoTuner releases intentionally use recent llama.cpp flags
    (``--fit``, ``--cache-ram``, ``--perf``, ``--metrics``, newer
    speculative-decoding knobs). Older or forked binaries abort on unknown
    flags before they even load the model, which looks like a launch crash on
    Ubuntu/macOS where there may be no separate terminal. This helper keeps the
    command runnable by removing only arguments the selected binary does not
    advertise. The caller logs the removed chunks so the user knows when an old
    build is limiting features.
    """
    if not cmd or not supported_flags:
        return list(cmd), []

    supported = _expand_supported_flag_aliases(supported_flags)
    # If help parsing failed badly (not even -m/--model appeared), do not risk
    # mangling the command. Returning it unchanged is safer than stripping core
    # model-loading arguments.
    if "-m" not in supported and "--model" not in supported:
        return list(cmd), []

    def _is_flag_token(token: str) -> bool:
        # A value can start with "-" too (e.g. --cache-ram -1, -n -1), so only
        # treat tokens that look like a named option as flags.
        return bool(token not in ("-", "--") and _FLAG_RE.match(token))

    filtered: List[str] = [cmd[0]]
    removed: List[str] = []
    i = 1
    while i < len(cmd):
        tok = cmd[i]
        if _is_flag_token(tok):
            flag = _flag_name(tok)
            takes_value = "=" not in tok and flag in _ARG_FLAGS_WITH_VALUES
            if flag not in supported:
                chunk = [tok]
                i += 1
                # Consume the flag's value as well: for known value-flags
                # always, otherwise when the next token is not a flag itself
                # (llama-server has no positional arguments, so a left-behind
                # value would abort the launch just like the unknown flag).
                if (
                    "=" not in tok
                    and i < len(cmd)
                    and (takes_value or not _is_flag_token(cmd[i]))
                ):
                    chunk.append(cmd[i])
                    i += 1
                removed.append(" ".join(chunk))
                continue
            filtered.append(tok)
            i += 1
            # Keep the kept flag's value verbatim — never scan it as a flag
            # (it may be negative, e.g. --cache-ram -1).
            if takes_value and i < len(cmd):
                filtered.append(cmd[i])
                i += 1
            continue
        filtered.append(tok)
        i += 1
    return filtered, removed


def _resolve_probe_binary(binary: str) -> Optional[str]:
    """Resolve a command's argv[0] to something we can safely run --help on."""
    if not binary:
        return None
    p = Path(binary).expanduser()
    if p.is_file():
        if os.name != "nt" and not os.access(p, os.X_OK):
            return None
        return str(p)
    return shutil.which(binary)


@lru_cache(maxsize=32)
def _probe_supported_flags_cached(
    binary_path: str, mtime_ns: int, size: int
) -> Optional[frozenset[str]]:
    del mtime_ns, size  # cache key only; values are intentionally unused
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        cp = subprocess.run(
            [binary_path, "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=6,
            errors="replace",
            **kwargs,
        )
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        return None
    out = cp.stdout or ""
    flags = set(_FLAG_RE.findall(out))
    return frozenset(flags) if flags else None


def _probe_supported_flags(binary: str) -> Optional[Set[str]]:
    resolved = _resolve_probe_binary(binary)
    if not resolved:
        return None
    try:
        st = Path(resolved).stat()
        probed = _probe_supported_flags_cached(resolved, st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    return set(probed) if probed is not None else None


_MIN_VISION_PROMPT_CACHE_BUILD = 10045
# b10151 split the old mmap-backed ``mlock`` mode into two explicit choices:
# ``mlock`` (normal reads + lock) and ``mmap+mlock`` (mapped + lock).
_MIN_DISTINCT_MLOCK_BUILD = 10151
# DSpark landed in b10164. Older binaries advertise --spec-type but reject
# the new enum value, so ordinary flag-name probing cannot catch this case.
_MIN_DSPARK_BUILD = 10164
# DFlash2 reached stock mainline in b10658. Retain the reviewed PR commits as
# compatible legacy fallbacks for users who deliberately keep an older fork.
_DFLASH2_PR_COMMITS = ("5ecbe1ac", "1deefcca")
_MIN_MAINLINE_DFLASH2_BUILD = 10658
# b10741 hoisted NextN metadata before per-layer array loading, but the matching
# array-length and all-NextN fixes did not land until b10749. Builds in between
# abort on Qwen MTP graphs and Gemma 4 assistant heads (PRs #28173/#28183).
_BROKEN_NEXTN_BUILD_START = 10741
_FIXED_NEXTN_BUILD = 10749


def _parse_llama_build_number(version_output: str) -> Optional[int]:
    """Parse legacy and semantic-version llama.cpp build identifiers safely.

    Current llama.cpp prints ``version: 0.2.0-dev (build 10572, commit ...)``;
    older builds printed ``version: 10056 (...)``.  Prefer the explicit build
    field so the semantic major version is never mistaken for build ``b0``.
    Keeping both matches anchored to the version line also avoids compiler
    versions such as ``MSVC 19.51``.
    """
    output = version_output or ""
    match = re.search(r"(?im)^\s*version:[^\r\n]*?\(\s*build\s+b?(\d+)\b", output)
    if match is None:
        match = re.search(r"(?im)^\s*version:\s*b?(\d+)(?=\s|$)", output)
    return int(match.group(1)) if match else None


def _parse_llama_commit(version_output: str) -> Optional[str]:
    """Return the hexadecimal commit token from current/legacy banners."""
    output = version_output or ""
    match = re.search(r"(?i)\bcommit\s+([0-9a-f]{7,40})\b", output)
    if match is None:
        # Legacy banners used ``version: N (abcdef123)`` without the word
        # "commit". Anchor this fallback to the version line so compiler or
        # backend hashes elsewhere in the output are never mistaken for it.
        match = re.search(r"(?im)^\s*version:[^\r\n]*?\(([0-9a-f]{7,40})\)\s*$", output)
    return match.group(1).lower() if match else None


@lru_cache(maxsize=32)
def _probe_version_output_cached(
    binary_path: str, mtime_ns: int, size: int
) -> Optional[str]:
    """Return the selected binary's bounded ``--version`` output."""
    del mtime_ns, size  # cache key only; values are intentionally unused
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        cp = subprocess.run(
            [binary_path, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=6,
            errors="replace",
            **kwargs,
        )
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        return None
    return (cp.stdout or "")[:4096]


@lru_cache(maxsize=32)
def _probe_build_number_cached(
    binary_path: str, mtime_ns: int, size: int
) -> Optional[int]:
    """Return the numeric llama.cpp build reported by ``--version``."""
    return _parse_llama_build_number(
        _probe_version_output_cached(binary_path, mtime_ns, size) or ""
    )


def _probe_binary_version_output(binary: str) -> Optional[str]:
    resolved = _resolve_probe_binary(binary)
    if not resolved:
        return None
    try:
        st = Path(resolved).stat()
        return _probe_version_output_cached(resolved, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _probe_binary_build_number(binary: str) -> Optional[int]:
    """Best-effort llama.cpp build number, or ``None`` when unprobeable."""
    resolved = _resolve_probe_binary(binary)
    if not resolved:
        return None
    try:
        st = Path(resolved).stat()
        return _probe_build_number_cached(resolved, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def probe_binary_build_number(binary: str) -> Optional[int]:
    """Public, cached build probe shared by GUI/TUI feature workflows."""
    return _probe_binary_build_number(binary)


def probe_binary_build_info(binary: str) -> str:
    """Return ``b<build>-<commit>`` (or ``b<build>``) for external clients.

    Uses the same bounded, cached ``--version`` probe as the build gate, so
    the control API never spawns an extra process per request.
    """
    output = _probe_binary_version_output(binary) or ""
    build = _parse_llama_build_number(output)
    commit = _parse_llama_commit(output)
    if build is None:
        return ""
    return f"b{build}-{commit}" if commit else f"b{build}"


@lru_cache(maxsize=32)
def _runtime_markers_present_cached(
    files: Tuple[Tuple[str, int, int], ...], markers: Tuple[bytes, ...]
) -> bool:
    remaining = set(markers)
    max_marker = max((len(marker) for marker in remaining), default=1)
    for filename, _mtime_ns, _size in files:
        try:
            with Path(filename).open("rb") as handle:
                overlap = b""
                while remaining:
                    chunk = handle.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    data = (overlap + chunk).lower()
                    remaining = {marker for marker in remaining if marker not in data}
                    overlap = data[-max_marker:]
        except OSError:
            continue
        if not remaining:
            return True
    return not remaining


def _runtime_has_required_markers(binary: str, markers: List[str]) -> bool:
    resolved = _resolve_probe_binary(binary)
    if not resolved:
        return False
    executable = Path(resolved)
    candidates = [executable]
    try:
        for candidate in executable.parent.iterdir():
            name = candidate.name.lower()
            if not candidate.is_file() or candidate == executable:
                continue
            if (
                (name.startswith("llama") and name.endswith(".dll"))
                or (name.startswith("libllama") and ".so" in name)
                or (name.startswith("libllama") and name.endswith(".dylib"))
            ):
                candidates.append(candidate)
    except OSError:
        pass
    signatures: List[Tuple[str, int, int]] = []
    for candidate in candidates:
        try:
            stat = candidate.stat()
        except OSError:
            continue
        signatures.append((str(candidate), stat.st_mtime_ns, stat.st_size))
    encoded = tuple(
        marker.strip().lower().encode("ascii", errors="ignore")
        for marker in markers
        if marker.strip().encode("ascii", errors="ignore")
    )
    return bool(encoded) and _runtime_markers_present_cached(tuple(signatures), encoded)


def check_profile_build(
    profile: ModelProfile, binary: str
) -> Tuple[bool, str, Optional[int]]:
    """Validate numeric and fork-capability requirements against ``binary``.

    Returns ``(allowed, message, detected_build)``. Numeric-only profiles keep
    allowing unprobeable wrappers with a warning. Fork-only architecture
    markers fail closed because a stock runtime is known not to load them.
    This is shared by GUI, TUI, and OCR launch paths.
    """
    required = max(0, int(getattr(profile, "min_llama_build", 0) or 0))
    markers = [
        str(marker).strip().lower()
        for marker in getattr(profile, "required_runtime_markers", [])
        if str(marker).strip()
    ]
    detected: Optional[int] = None
    build_warning = ""
    if required > 0:
        detected = probe_binary_build_number(binary)
        if detected is None:
            build_warning = (
                f"Could not verify the selected binary's llama.cpp build; "
                f"{profile.display_name} requires b{required}+."
            )
        elif detected < required:
            return (
                False,
                f"{profile.display_name} requires llama.cpp b{required}+; "
                f"the selected binary reports b{detected}.",
                detected,
            )
    if markers and not _runtime_has_required_markers(binary, markers):
        return (
            False,
            f"{profile.display_name} requires a patched llama runtime with "
            f"capability marker(s): {', '.join(markers)}. The selected binary "
            "and its sibling llama library do not contain them.",
            detected,
        )
    return True, build_warning, detected


def check_model_build(
    model: ModelEntry, binary: str
) -> Tuple[bool, str, Optional[int]]:
    """Reject target GGUFs that b10741-b10748 cannot load safely.

    PR #28159 made ``n_layer()`` exclude NextN before the generic per-layer
    arrays were read. Standard GGUFs that store FF/head metadata as arrays of
    ``block_count`` entries then fail during hparam loading because those builds
    expect only the main-layer count. Scalar-metadata Qwen targets remain usable
    with MTP disabled by :func:`prepare_command_for_binary`; array-backed
    targets must use b10749+ (PRs #28173/#28183) or a pre-regression build.
    """
    if not model.has_embedded_mtp:
        return True, "", None

    detected = probe_binary_build_number(binary)
    if (
        detected is None
        or detected < _BROKEN_NEXTN_BUILD_START
        or detected >= _FIXED_NEXTN_BUILD
    ):
        return True, "", detected

    metadata = model.metadata or {}
    arch = str(metadata.get("general.architecture", "") or "").strip()
    if not arch:
        return True, "", detected
    try:
        block_count = int(metadata.get(f"{arch}.block_count", 0) or 0)
        nextn_count = int(metadata.get(f"{arch}.nextn_predict_layers", 0) or 0)
    except (TypeError, ValueError):
        return True, "", detected
    main_count = block_count - nextn_count
    if block_count <= 0 or nextn_count <= 0 or main_count < 0:
        return True, "", detected

    for suffix in (
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
    ):
        key = f"{arch}.{suffix}"
        value = metadata.get(key)
        if isinstance(value, list) and len(value) != main_count:
            return (
                False,
                f"{model.name} cannot load on llama.cpp b{detected}: its "
                f"{key} array has {len(value)} entries while that regressed "
                f"runtime expects {main_count}. Use b{_FIXED_NEXTN_BUILD}+ or "
                f"b{_BROKEN_NEXTN_BUILD_START - 1} and earlier (upstream PRs "
                "#28159/#28173).",
                detected,
            )
    return True, "", detected


def resolve_draft_n_max(
    profile: ModelProfile,
    draft_model: Optional[ModelEntry] = None,
    forced: Optional[int] = None,
) -> int:
    """Resolve an effective speculative depth from override/model/profile.

    DFlash checkpoints train a fixed noise block.  llama.cpp's documented
    value is one less than ``dflash.block_size`` because the block includes
    the anchor token (for Qwen3.8 DFlash2: 8 -> ``n-max 7``).  Falling back to
    Qwen3.8's generic MTP profile value of 2 both wastes the DFlash2 head and
    hits known early-PR loader paths.  Explicit Expert/benchmark overrides
    remain authoritative.
    """
    if forced is not None and int(forced) > 0:
        return max(1, int(forced))
    if draft_model is not None and draft_model.drafter_spec_type == "dflash":
        try:
            block_size = int(
                (draft_model.metadata or {}).get("dflash.block_size", 0) or 0
            )
        except (TypeError, ValueError):
            block_size = 0
        if block_size > 1:
            return block_size - 1
    return max(1, int(getattr(profile, "draft_max", 0) or 2))


def check_draft_model_build(
    draft_model: Optional[ModelEntry], binary: str
) -> Tuple[bool, str, Optional[int]]:
    """Preflight draft formats whose support is not visible in ``--help``.

    b10741-b10748 contain an upstream NextN loader regression that aborts on
    standalone MTP heads. DFlash2 also intentionally reuses the ``dflash``
    architecture and ``draft-dflash`` enum, so pre-b10658 stock builds
    advertise the right CLI while instantiating the older 58-tensor graph.
    Reject either known-incompatible runtime before it can crash; retain the
    existing warning-only behavior for wrappers whose version cannot be read.
    """
    if draft_model is None:
        return True, "", None

    if draft_model.drafter_spec_type == "mtp":
        detected = probe_binary_build_number(binary)
        if (
            detected is not None
            and _BROKEN_NEXTN_BUILD_START <= detected < _FIXED_NEXTN_BUILD
        ):
            return (
                False,
                "This MTP draft head cannot load on llama.cpp "
                f"b{detected}: b{_BROKEN_NEXTN_BUILD_START}-b{_FIXED_NEXTN_BUILD - 1} "
                "contain the upstream NextN loader regression from PR #28159. "
                f"Use b{_FIXED_NEXTN_BUILD}+ or b{_BROKEN_NEXTN_BUILD_START - 1} "
                "and earlier.",
                detected,
            )

    if not draft_model.is_dflash2_drafter:
        return True, "", None

    detected = probe_binary_build_number(binary)
    version_output = _probe_binary_version_output(binary) or ""
    commit = _parse_llama_commit(version_output)
    if commit and any(commit.startswith(item) for item in _DFLASH2_PR_COMMITS):
        return True, "", detected
    if "dflash2" in version_output.casefold():
        return True, "", detected
    if detected is not None and detected >= _MIN_MAINLINE_DFLASH2_BUILD:
        return True, "", detected
    if detected is None:
        return (
            True,
            "Could not verify DFlash2 support in the selected llama.cpp binary. "
            f"Use mainline b{_MIN_MAINLINE_DFLASH2_BUILD}+ or a reviewed "
            "PR #27342 build for this 81-tensor sidecar.",
            None,
        )
    commit_label = f" ({commit})" if commit else ""
    return (
        False,
        "Qwen3.8 DFlash2 requires mainline llama.cpp "
        f"b{_MIN_MAINLINE_DFLASH2_BUILD}+ (or reviewed PR #27342); selected "
        f"b{detected}{commit_label} only has the older DFlash graph. Update "
        "llama.cpp or choose the model's embedded MTP head.",
        detected,
    )


def _memlock_limit_gb() -> Optional[float]:
    """Soft RLIMIT_MEMLOCK in GB on POSIX; ``None`` = unlimited/not applicable.

    Desktop Linux distros (Ubuntu/Debian/Arch/Fedora) default to a tiny
    limit (8 MiB), so a non-root process can never mlock a model there.
    """
    if platform.system() == "Windows":
        return None
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    if soft in (resource.RLIM_INFINITY, -1):
        return None
    return soft / (1024**3)


def _adapt_load_mode_for_binary(cmd: List[str]) -> Tuple[List[str], List[str]]:
    """Adapt b10151's split locking modes for older versioned binaries.

    Before b10151, ``--load-mode mlock`` meant mmap + mlock and there was no
    way to request non-mmap mlock. On those builds ``mmap+mlock`` is therefore
    translated to the legacy equivalent, while the new non-mmap ``mlock``
    choice is removed rather than silently changing its meaning.
    """
    if not cmd:
        return [], []
    build = _probe_binary_build_number(cmd[0])
    if build is None or build >= _MIN_DISTINCT_MLOCK_BUILD:
        return list(cmd), []

    adapted = list(cmd)
    notes: List[str] = []
    i = 1
    while i < len(adapted):
        token = adapted[i]
        flag = _flag_name(token)
        if flag not in ("-lm", "--load-mode"):
            i += 1
            continue
        # Inline form (--load-mode=MODE / -lm=MODE) keeps the value in the
        # same token after '='. The space-separated form holds it in the
        # following argv slot.
        inline = "=" in token
        if inline:
            mode = token.split("=", 1)[1].strip().lower()
        elif i + 1 < len(adapted):
            mode = adapted[i + 1].strip().lower()
        else:
            i += 1
            continue
        if mode == "mmap+mlock":
            replacement = flag + "=mlock" if inline else "mlock"
            if inline:
                adapted[i] = replacement
            else:
                adapted[i + 1] = replacement
            notes.append(
                "--load-mode mmap+mlock -> --load-mode mlock "
                "(legacy equivalent before b10151)"
            )
        elif mode == "mlock":
            if inline:
                del adapted[i]
            else:
                del adapted[i : i + 2]
            notes.append("--load-mode mlock (non-mmap mlock requires b10151+)")
            continue
        i += 1 if inline else 2
    return adapted, notes


def _adapt_lazy_mode_for_binary(
    cmd: List[str], supported_flags: Set[str]
) -> Tuple[List[str], List[str]]:
    """Translate the b10700 lazy-row-table option rename in either direction.

    ``--tensor-read-lazy`` was replaced (not aliased) by ``--lazy-mode`` /
    ``-lzm``. Generic alias expansion cannot handle that across builds because
    retaining the wrong spelling would still abort. Rewrite to an option the
    selected binary actually advertises so memory planning and runtime loading
    stay aligned on both old and current llama.cpp versions.
    """
    if not cmd or not supported_flags:
        return list(cmd), []

    spellings = ("--lazy-mode", "-lzm", "--tensor-read-lazy")
    target = next((flag for flag in spellings if flag in supported_flags), None)
    if target is None:
        return list(cmd), []

    adapted = list(cmd)
    notes: List[str] = []
    for index in range(1, len(adapted)):
        token = adapted[index]
        flag = _flag_name(token)
        if flag not in spellings or flag in supported_flags:
            continue
        suffix = token[len(flag) :] if token.startswith(flag) else ""
        adapted[index] = target + suffix
        notes.append(f"{flag} -> {target} (llama.cpp b10700 option rename)")
    return adapted, notes


def _adapt_spec_types_for_binary(cmd: List[str]) -> Tuple[List[str], List[str]]:
    """Remove DSpark's whole external-draft path on pre-b10164 builds.

    ``--help`` probing only discovers option *names*. A b10151 binary therefore
    appears to support ``--spec-type`` even though it rejects the newer
    ``draft-dspark`` enum value. Falling back to plain ``-md`` would be worse:
    DSpark carries a Markov head and must not silently run as ordinary DFlash.
    Keep any comma-separated n-gram method, but remove the DSpark model and its
    draft-only tuning arguments with an explicit compatibility note.
    """
    if not cmd:
        return [], []
    build = _probe_binary_build_number(cmd[0])
    if build is None or build >= _MIN_DSPARK_BUILD:
        return list(cmd), []

    adapted = list(cmd)
    notes: List[str] = []
    found_dspark = False
    i = 1
    while i < len(adapted):
        token = adapted[i]
        flag = _flag_name(token)
        if flag != "--spec-type":
            i += 1
            continue
        inline = "=" in token
        if inline:
            raw = token.split("=", 1)[1]
        elif i + 1 < len(adapted):
            raw = adapted[i + 1]
        else:
            i += 1
            continue
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if "draft-dspark" not in values:
            i += 1 if inline else 2
            continue
        found_dspark = True
        kept = [v for v in values if v != "draft-dspark"]
        if kept:
            replacement = ",".join(kept)
            if inline:
                adapted[i] = f"--spec-type={replacement}"
            else:
                adapted[i + 1] = replacement
            i += 1 if inline else 2
        else:
            del adapted[i : i + (1 if inline else 2)]

    if not found_dspark:
        return adapted, notes

    # Remove the external DSpark model and draft-only parameters. Preserve any
    # surviving n-gram method and its --spec-ngram-* tuning flags.
    draft_value_flags = {
        "-md",
        "--model-draft",
        "--spec-draft-model",
        "--spec-draft-ngl",
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--spec-draft-p-min",
        "--spec-draft-p-split",
    }
    cleaned: List[str] = [adapted[0]]
    i = 1
    while i < len(adapted):
        token = adapted[i]
        flag = _flag_name(token)
        if flag in draft_value_flags:
            i += 1
            if "=" not in token and i < len(adapted):
                i += 1
            continue
        cleaned.append(token)
        i += 1
    notes.append(
        "draft-dspark and its external draft model disabled "
        f"(requires llama.cpp b{_MIN_DSPARK_BUILD}+, selected b{build})"
    )
    return cleaned, notes


def _adapt_nextn_regression_for_binary(cmd: List[str]) -> Tuple[List[str], List[str]]:
    """Disable only MTP drafting on llama.cpp's b10741-b10748 regression.

    Those builds abort while constructing integrated Qwen MTP graphs and while
    loading standalone NextN heads. Preserve the target model and any compatible
    n-gram method, but remove ``draft-mtp`` plus its model/tuning arguments.
    The upstream fixes are PRs #28173 and #28183, first tagged in b10749.
    """
    if not cmd:
        return [], []
    build = _probe_binary_build_number(cmd[0])
    if (
        build is None
        or build < _BROKEN_NEXTN_BUILD_START
        or build >= _FIXED_NEXTN_BUILD
    ):
        return list(cmd), []

    adapted = list(cmd)
    found_mtp = False
    i = 1
    while i < len(adapted):
        token = adapted[i]
        if _flag_name(token) != "--spec-type":
            i += 1
            continue
        inline = "=" in token
        if inline:
            raw = token.split("=", 1)[1]
        elif i + 1 < len(adapted):
            raw = adapted[i + 1]
        else:
            i += 1
            continue
        values = [value.strip() for value in raw.split(",") if value.strip()]
        if "draft-mtp" not in values:
            i += 1 if inline else 2
            continue
        found_mtp = True
        kept = [value for value in values if value != "draft-mtp"]
        if kept:
            replacement = ",".join(kept)
            if inline:
                adapted[i] = f"--spec-type={replacement}"
            else:
                adapted[i + 1] = replacement
            i += 1 if inline else 2
        else:
            del adapted[i : i + (1 if inline else 2)]

    if not found_mtp:
        return adapted, []

    draft_value_flags = {
        "-md",
        "--model-draft",
        "--spec-draft-model",
        "--spec-draft-ngl",
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--spec-draft-p-min",
        "--spec-draft-p-split",
    }
    cleaned: List[str] = [adapted[0]]
    i = 1
    while i < len(adapted):
        token = adapted[i]
        if _flag_name(token) in draft_value_flags:
            i += 1
            if "=" not in token and i < len(adapted):
                i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned, [
        "draft-mtp disabled for llama.cpp "
        f"b{build} (upstream NextN regression in b{_BROKEN_NEXTN_BUILD_START}-"
        f"b{_FIXED_NEXTN_BUILD - 1}; fixed in b{_FIXED_NEXTN_BUILD}+)"
    ]


def prepare_command_for_binary(cmd: List[str]) -> Tuple[List[str], List[str]]:
    """Return ``cmd`` adapted/pruned for the selected binary plus changes.

    If ``--help`` cannot be probed, unsupported optional flags are left
    unchanged so explicit wrappers and unusual forks keep working. A separately
    probeable numeric b10741-b10748 runtime still receives the narrow NextN/MTP
    safety adaptation because that known crash does not depend on help parsing.
    """
    if not cmd:
        return [], []
    flags = _probe_supported_flags(cmd[0])
    if not flags:
        return _adapt_nextn_regression_for_binary(cmd)
    adapted, mode_changes = _adapt_load_mode_for_binary(cmd)
    adapted, lazy_changes = _adapt_lazy_mode_for_binary(adapted, flags)
    adapted, spec_changes = _adapt_spec_types_for_binary(adapted)
    adapted, nextn_changes = _adapt_nextn_regression_for_binary(adapted)
    filtered, removed = _filter_command_for_supported_flags(adapted, flags)
    return (
        filtered,
        mode_changes + lazy_changes + spec_changes + nextn_changes + removed,
    )


def gemma_draft_needs_ik_fork(
    model_name: str, use_draft: bool, resolved_binary: str
) -> bool:
    """True only when a Gemma-4 + external-drafter launch must fall back to
    the legacy ik_llama.cpp fork.

    Mainline llama.cpp runs the Gemma-4 drafter natively since PR #23398
    (standalone ``gemma4-assistant`` MTP head via ``--spec-type draft-mtp``;
    a plain sibling drafter is auto-detected from ``-md``). The historic
    unconditional redirect to ik_llama.cpp therefore only remains correct
    for builds too old to advertise ``--spec-type`` (pre-b9190). We probe
    the resolved binary's ``--help``; when the probe fails (binary missing /
    timeout) we do NOT redirect — the selected fork stays authoritative,
    matching prepare_command_for_binary's philosophy.
    """
    if not use_draft:
        return False
    n = model_name.lower()
    if "gemma-4" not in n and "gemma4" not in n:
        return False
    flags = _probe_supported_flags(resolved_binary)
    return flags is not None and "--spec-type" not in flags


def match_gpu_by_token(token: Optional[str], gpus: List[GPUInfo]) -> Optional[GPUInfo]:
    """Resolve a user GPU pin token ("9070", "R9700", full driver string)
    to a detected card, robust across OS name styles.

    The pin is persisted as a name-derived token, but the SAME card is
    called "AMD Radeon AI PRO R9700" by Windows WMI, "Radeon AI PRO R9700"
    by Linux lspci/DRM and "AMD Radeon AI PRO R9700 (RADV NAVI48)" by Mesa.
    A one-directional substring test (the old force_gpu matching) silently
    dropped the pin after switching OS, and the launch then fell back to
    auto-placement — the "manual GPU selection doesn't stick" report.
    Match exact → substring (either direction) → shared model-number token
    ("r9700", "9070", "3060ti"), mirroring the priority-map matching in
    compute_config. Returns None when nothing matches (auto behaviour).
    """
    if not token or not gpus:
        return None
    needle = token.strip().lower()
    if not needle:
        return None
    for g in gpus:
        if g.name.strip().lower() == needle:
            return g
    for g in gpus:
        g_lower = g.name.strip().lower()
        if g_lower and (needle in g_lower or g_lower in needle):
            return g
    # Model-number tokens: any alnum token containing a digit that both
    # names share identifies the card across driver-string variants. The
    # needle side comes from a user pin / stored token and never contains
    # the Mesa "(RADV NAVI48)" generation suffix, so generation tokens
    # shared by two cards of the same chip cannot cause a false match.
    n_tokens = {
        t for t in re.findall(r"[a-z0-9]+", needle) if any(c.isdigit() for c in t)
    }
    if n_tokens:
        for g in gpus:
            g_tokens = {
                t
                for t in re.findall(r"[a-z0-9]+", g.name.lower())
                if any(c.isdigit() for c in t)
            }
            if g_tokens & n_tokens:
                return g
    return None


# ---------------------------------------------------------------------------
# Model-size helpers


def extract_params_billion(name: str) -> float:
    """Extract parameter count in billions from a model filename."""
    counts = [
        float(m)
        for m in re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*B(?![a-zA-Z0-9_])", name)
    ]
    counts.extend(
        float(m) * 1000.0
        for m in re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*T(?![a-zA-Z0-9_])", name)
    )
    if counts:
        return max(counts)
    m = re.search(r"E(\d+(?:\.\d+)?)B", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def kv_per_token_mb_f16(params_billion: float) -> float:
    """Approximate KV-cache memory per token at f16 quant, in MB.

    Fallback heuristic when GGUF metadata is unavailable. NOTE: this is
    calibrated for *dense* models. For MoE models (e.g. Qwen3.6-35B-A3B
    and Gemma-4-26B-A4B where only ~3B and ~4B params are active per token),
    this heuristic overestimates KV by roughly an order of magnitude — prefer
    `kv_per_token_mb_from_metadata()` whenever metadata is present.
    """
    if params_billion <= 0:
        return 0.20
    if params_billion < 1.5:
        return 0.04
    if params_billion < 4:
        return 0.10
    if params_billion < 9:
        return 0.18
    if params_billion < 16:
        return 0.30
    if params_billion < 32:
        return 0.50
    if params_billion < 70:
        return 0.85
    return 1.40


_KV_HEAD_COUNT_SUFFIXES = (
    "attention.head_count_kv",
    "attention.kv_head_count",
    "attention.num_key_value_heads",
    "kv_head_count",
    "num_key_value_heads",
)


def _metadata_arch_value(
    md: Dict[str, Any], arch: str, suffixes: Tuple[str, ...]
) -> Any:
    """Return an architecture metadata value with suffix fallbacks.

    Community converters sometimes preserve a valid value under a legacy
    architecture prefix or HF-style key. Exact current-arch keys always win;
    only then do we scan suffix-compatible alternatives.
    """
    for suffix in suffixes:
        key = f"{arch}.{suffix}" if arch else suffix
        if key in md:
            return md[key]
    lowered_suffixes = tuple(f".{suffix.lower()}" for suffix in suffixes)
    for key, value in md.items():
        if str(key).lower().endswith(lowered_suffixes):
            return value
    return None


def _metadata_arch_int(md: Dict[str, Any], arch: str, *suffixes: str) -> int:
    value = _metadata_arch_value(md, arch, tuple(suffixes))
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _kv_per_token_for_interleaved_attention(
    md: Dict[str, Any],
    arch: str,
    n_kv_heads_per_layer: List[Any],
) -> float:
    """KV per-token (MB) for models with per-layer KV-head arrays.

    The Gemma-4 family (26B-A4B, 31B, and likely future siblings)
    stores ``<arch>.attention.head_count_kv`` as a **per-layer array**
    rather than a single scalar, because each layer alternates between
    full attention and sliding-window attention (SWA) with different
    head/dim configurations. Example from gemma-4-26B-A4B:

        head_count_kv          = [8,8,8,8,8,2, 8,8,8,8,8,2, …]   # 30 layers
        sliding_window_pattern = [T,T,T,T,T,F, T,T,T,T,T,F, …]   # 30 layers
        sliding_window         = 1024

    The pattern array tells us which layers do SWA (True) and which
    do full attention (False). SWA layers cap their KV at
    ``sliding_window`` tokens — they contribute a **constant** overhead
    that does NOT scale with ctx. Full-attention layers (the False
    entries) scale linearly with ctx.

    For the AutoTuner's "per-token KV size" estimate at large ctx, we
    return only the asymptotic part: the sum over full-attention
    layers. At typical ctx >> sliding_window (e.g. 32k >> 1024) the
    constant SWA overhead is well under 100 MB and can be ignored.

    Fallback: if ``sliding_window_pattern`` is missing or shorter than
    the head array, sum every entry (treat all as full-attention).
    That overshoots, but on the safe side — better the AutoTuner
    reserves too much KV than too little.

    **Broadcast arrays (DiffusionGemma).** Some forks store scalar
    values inside a single-element list (e.g. ``head_count_kv=[2]``
    and ``sliding_window_pattern=[False]``) that apply to EVERY layer
    rather than per-layer 30-element arrays like Gemma-4. Without
    expansion the loop below sums only 1 layer's worth of KV and
    under-estimates the cache by the layer count (factor 30 on a
    30-block model), so compute_config picks huge contexts and the
    backend OOMs on KV allocation. When the head array is shorter
    than ``block_count`` we expand both it and the pattern to the full
    layer count before iterating.
    """

    sliding_pattern = _metadata_arch_value(
        md, arch, ("attention.sliding_window_pattern",)
    )
    kl = _metadata_arch_int(md, arch, "attention.key_length")
    vl = _metadata_arch_int(md, arch, "attention.value_length")
    n_heads = _metadata_arch_int(md, arch, "attention.head_count")
    embd = _metadata_arch_int(md, arch, "embedding_length")
    n_layers = _metadata_arch_int(md, arch, "block_count")

    # Expand broadcast (single-element / short) per-layer arrays to the
    # full layer count. This is the DiffusionGemma case: head_count_kv=[2]
    # and sliding_window_pattern=[False] are scalars wrapped in a list,
    # broadcast across all 30 layers. Without this the KV estimate is off
    # by the layer-count factor and compute_config over-selects context.
    if n_layers > 1 and 0 < len(n_kv_heads_per_layer) < n_layers:
        scalar_kv = n_kv_heads_per_layer[0]
        n_kv_heads_per_layer = [scalar_kv] * n_layers
    if (
        n_layers > 1
        and isinstance(sliding_pattern, list)
        and 0 < len(sliding_pattern) < n_layers
    ):
        scalar_pat = sliding_pattern[0]
        sliding_pattern = [scalar_pat] * n_layers

    # Fall back to embd/n_heads if explicit head dims absent.
    if kl <= 0 or vl <= 0:
        if n_heads > 0 and embd > 0:
            head_size = max(1, embd // n_heads)
            if kl <= 0:
                kl = head_size
            if vl <= 0:
                vl = head_size
        else:
            return 0.0

    full_kv = 0
    if isinstance(sliding_pattern, list) and sliding_pattern:
        # Sum KV-heads only for full-attention layers (pattern entry False).
        for i, h in enumerate(n_kv_heads_per_layer):
            if i >= len(sliding_pattern):
                break
            # Pattern entry True = SWA → skip (constant overhead).
            if not bool(sliding_pattern[i]):
                try:
                    full_kv += int(h)
                except (TypeError, ValueError):
                    continue
    else:
        # No pattern info — treat every layer as full attention.
        for h in n_kv_heads_per_layer:
            try:
                full_kv += int(h)
            except (TypeError, ValueError):
                continue

    if full_kv <= 0:
        return 0.0

    bytes_per_token = full_kv * (kl + vl) * 2
    return bytes_per_token / (1024.0 * 1024.0)


def _kv_per_token_total_mb_from_metadata(md: Dict[str, Any]) -> float:
    """Compute exact f16 K+V cache size per token (MB) from GGUF metadata.

    Standard transformer formula:
        bytes/token = n_attention_layers * n_kv_heads * (key_length + value_length) * 2

    Three special cases are handled before the formula:

    1. **Interleaved-attention models** (Gemma-4 family) — when
       ``head_count_kv`` is stored as a *per-layer array* instead of a
       scalar, we route to :func:`_kv_per_token_for_interleaved_attention`
       which uses the sliding-window pattern to count only the
       full-attention layers (those that actually scale with ctx).

    2. **Hybrid Mamba/Transformer models** (Nemotron-H, Jamba, …) —
       only a fraction of layers carry KV cache. We multiply by that
       fraction via :func:`metadata_attention_layer_count`. Otherwise
       we'd over-reserve VRAM by 4–5× on these architectures.

    3. **GQA** — when ``head_count_kv`` is present and a positive scalar,
       it's already smaller than ``head_count`` and the formula uses
       it directly. When the value is missing (``head_count_kv = 0``)
       we fall back to ``head_count``, which over-estimates KV for
       any modern GQA model. Such over-estimates are visible in the
       config preview; if you see them, the GGUF likely stored the
       value under a non-canonical key or as an array (case 1).

    Returns 0.0 when metadata is incomplete; the caller should then
    fall back to the params-billion heuristic.
    """
    if not md:
        return 0.0
    arch = md.get("general.architecture") or ""
    if not arch:
        return 0.0

    # ── Case 1: interleaved attention with per-layer KV-head array ─────
    n_kv_raw = _metadata_arch_value(md, arch, _KV_HEAD_COUNT_SUFFIXES)
    if isinstance(n_kv_raw, list) and n_kv_raw:
        return _kv_per_token_for_interleaved_attention(md, arch, n_kv_raw)

    # ── Standard scalar path (existing behaviour) ─────────────────────
    def _int(key: str) -> int:
        return _metadata_arch_int(md, arch, key)

    # Use the attention-bearing layer count for hybrids; for pure
    # Transformer this equals block_count and behaves as before.
    from scanner import metadata_attention_layer_count

    n_layers = metadata_attention_layer_count(md)
    if n_layers <= 0:
        # Fallback for older models / incomplete metadata: use total blocks.
        n_layers = _int("block_count")

    n_heads = _int("attention.head_count")
    n_kv_heads = _metadata_arch_int(md, arch, *_KV_HEAD_COUNT_SUFFIXES)
    embd = _int("embedding_length")
    key_length = _int("attention.key_length")
    value_length = _int("attention.value_length")

    if n_layers <= 0:
        return 0.0

    # Default head dim = embedding_length / head_count when not explicit.
    if key_length <= 0 or value_length <= 0:
        if n_heads > 0 and embd > 0:
            head_size = max(1, embd // n_heads)
            if key_length <= 0:
                key_length = head_size
            if value_length <= 0:
                value_length = head_size
        else:
            return 0.0

    # No GQA → KV heads == query heads.
    if n_kv_heads <= 0:
        n_kv_heads = n_heads if n_heads > 0 else 1

    bytes_per_token = n_layers * n_kv_heads * (key_length + value_length) * 2
    return bytes_per_token / (1024.0 * 1024.0)


def kv_per_token_parts_mb_from_metadata(
    md: Dict[str, Any],
) -> Tuple[float, float]:
    """Return exact f16 ``(K, V)`` MB/token parts when metadata permits.

    Keeping K and V separate matters for asymmetric cache quants on MLA and
    other architectures where key/value head dimensions differ.
    """
    total = _kv_per_token_total_mb_from_metadata(md)
    if total <= 0:
        return 0.0, 0.0
    arch = str(md.get("general.architecture") or "")
    key_length = _metadata_arch_int(md, arch, "attention.key_length")
    value_length = _metadata_arch_int(md, arch, "attention.value_length")
    if key_length <= 0 or value_length <= 0:
        n_heads = _metadata_arch_int(md, arch, "attention.head_count")
        embd = _metadata_arch_int(md, arch, "embedding_length")
        if n_heads > 0 and embd > 0:
            head_size = max(1, embd // n_heads)
            key_length = key_length if key_length > 0 else head_size
            value_length = value_length if value_length > 0 else head_size
    denom = key_length + value_length
    if denom <= 0:
        key_mb, value_mb = total / 2.0, total / 2.0
    else:
        key_mb = total * key_length / denom
        value_mb = total * value_length / denom

    # qwen4exp's QSA path creates a second cache over the same full-attention
    # layers. b10666 shapes it as one key head of indexer.key_length and one
    # value head of the model's normal value length. It uses the selected K/V
    # cache quants, so keep the additions split for asymmetric quant planning.
    if arch.lower() == "qwen4exp":
        from scanner import metadata_attention_layer_count

        n_attention = metadata_attention_layer_count(md)
        indexer_key_length = _metadata_arch_int(
            md, arch, "attention.indexer.key_length"
        )
        if n_attention > 0 and indexer_key_length > 0 and value_length > 0:
            key_mb += n_attention * indexer_key_length * 2 / (1024.0 * 1024.0)
            value_mb += n_attention * value_length * 2 / (1024.0 * 1024.0)

    return key_mb, value_mb


def kv_per_token_mb_from_metadata(md: Dict[str, Any]) -> float:
    """Compute f16 K+V cache size per token (MB) from GGUF metadata."""
    key_mb, value_mb = kv_per_token_parts_mb_from_metadata(md)
    return key_mb + value_mb


def recurrent_state_gb_from_metadata(
    md: Dict[str, Any],
    n_parallel: int = 1,
    snapshot_count: int = 0,
) -> float:
    """Estimate llama.cpp's fixed F32 recurrent-state buffers in GiB.

    Unlike attention KV, this state does not grow with context length. It does
    grow with recurrent layers, parallel sequences, and speculative rollback
    snapshots. The formulas mirror ``llama_hparams::n_embd_r/n_embd_s`` and
    ``llama_memory_recurrent`` in b10441.
    """
    if not md:
        return 0.0
    arch = str(md.get("general.architecture") or "").lower()
    if not arch:
        return 0.0

    from scanner import (
        _RECURRENT_ARCHS,
        metadata_attention_layer_count,
        metadata_is_hybrid_architecture,
        metadata_layer_count,
    )

    total_layers = metadata_layer_count(md)
    if total_layers <= 0 or not metadata_is_hybrid_architecture(md):
        return 0.0
    if arch in _RECURRENT_ARCHS:
        recurrent_layers = total_layers
    else:
        recurrent_layers = max(0, total_layers - metadata_attention_layer_count(md))
    if recurrent_layers <= 0:
        return 0.0

    n_embd = _metadata_arch_int(md, arch, "embedding_length")
    n_heads = _metadata_arch_int(md, arch, "attention.head_count")
    key_length = _metadata_arch_int(md, arch, "attention.key_length")
    conv = _metadata_arch_int(md, arch, "ssm.conv_kernel")
    inner = _metadata_arch_int(md, arch, "ssm.inner_size")
    state = _metadata_arch_int(md, arch, "ssm.state_size")
    groups = _metadata_arch_int(md, arch, "ssm.group_count")

    r_elements = 0
    s_elements = 0
    wkv_head = _metadata_arch_int(md, arch, "wkv.head_size")
    if wkv_head > 0 and n_embd > 0:
        token_shifts = max(1, _metadata_arch_int(md, arch, "token_shift_count"))
        r_elements = token_shifts * n_embd
        s_elements = n_embd * wkv_head
    else:
        shortconv = _metadata_arch_int(md, arch, "shortconv.l_cache")
        kda_head = _metadata_arch_int(md, arch, "kda.head_dim")
        if shortconv > 0 and n_embd > 0:
            r_elements = n_embd * max(0, shortconv - 1)
        elif kda_head > 0 and n_heads > 0:
            r_elements = 3 * max(0, conv - 1) * n_heads * kda_head
            s_elements = kda_head * kda_head * n_heads
        elif arch in {"minimax-01", "minimax_01"}:
            # b10441 maps the full-attention key head dimension to the
            # Lightning-Attention recurrent state dimension.
            if key_length <= 0 and n_heads > 0 and n_embd > 0:
                key_length = max(1, n_embd // n_heads)
            if key_length > 0 and n_heads > 0:
                s_elements = key_length * key_length * n_heads
        else:
            if conv > 0 and inner > 0:
                r_elements = max(0, conv - 1) * (
                    inner + 2 * max(0, groups) * max(0, state)
                )
            if state > 0 and inner > 0:
                s_elements = state * inner

    elements_per_layer = r_elements + s_elements
    if elements_per_layer <= 0:
        return 0.0
    rows = max(1, int(n_parallel)) * (1 + max(0, int(snapshot_count)))
    total_bytes = recurrent_layers * elements_per_layer * rows * 4  # F32
    return total_bytes / (1024.0**3)


def _resolve_kv_per_token_mb(model: ModelEntry, params_billion: float) -> float:
    """Pick the best KV-per-token estimate available."""
    md_estimate = kv_per_token_mb_from_metadata(model.metadata)
    if md_estimate > 0:
        return md_estimate
    return kv_per_token_mb_f16(params_billion)


def _resolve_kv_per_token_parts_mb(
    model: ModelEntry, params_billion: float
) -> Tuple[float, float]:
    """Return f16 K/V parts, falling back to an even heuristic split."""
    key_mb, value_mb = kv_per_token_parts_mb_from_metadata(model.metadata)
    if key_mb + value_mb > 0:
        return key_mb, value_mb
    arch = str((model.metadata or {}).get("general.architecture") or "").lower()
    from scanner import _RECURRENT_ARCHS

    if arch in _RECURRENT_ARCHS:
        # Pure recurrent architectures have no context-growing attention KV.
        # Keep a tiny numerical sentinel so context math remains finite without
        # falling back to the dense params-based heuristic.
        return 1e-6, 1e-6
    total = kv_per_token_mb_f16(params_billion)
    return total / 2.0, total / 2.0


def kv_quant_factor(quant: str) -> float:
    """Memory factor of a given KV-cache quant, relative to f16.

    Covers the upstream cache types (f16/q8/q5/q4 + iq4_nl) plus the
    TurboQuant-fork labels (turbo2/turbo3/turbo4). The turbo factors
    come from Google's TurboQuant paper + the TheTom/AtomicBot fork
    measurements (ICLR 2026 / b9082+ branches): turbo3 ≈ 4.3× vs f16,
    turbo4 ≈ 3.8×, turbo2 ≈ 6.4×.
    """
    q = quant.lower()
    if q in ("f16", "fp16", "bf16"):
        return 1.0
    if q in ("q8_0", "q8_1", "q8"):
        return 0.55
    if q in ("q5_0", "q5_1", "q5"):
        return 0.40
    if q in ("q4_0", "q4_1", "q4", "iq4_nl"):
        return 0.32
    # TurboQuant labels (TheTom/turboquant_plus, AtomicBot, spiritbuun).
    # The Google paper quotes compression ratios vs F16; we convert
    # 1 / ratio = factor. Slightly conservative (rounded up) so the
    # auto-tuner does not over-promise context length.
    if q == "turbo4":
        return 0.27  # ~3.8× → 1/3.8 = 0.263, rounded up
    if q in ("turbo3", "tq3_0"):
        return 0.24  # ~4.3× → 1/4.3 = 0.233
    if q in ("turbo3_tcq",):
        # 3-bit Viterbi-coded, ~5x at same quality as turbo3 scalar.
        return 0.20
    if q == "turbo2":
        return 0.16  # ~6.4× → 1/6.4 = 0.156
    if q in ("turbo2_tcq",):
        return 0.13  # ~7-8× in spiritbuun benchmarks
    return 0.55


# ---------------------------------------------------------------------------
# MoE detection

# Common alternate metadata keys some quantizers emit instead of the
# canonical "<arch>.expert_count". Order matters: more specific first.
_MOE_ALT_KEY_SUFFIXES = (
    ".expert_count",  # canonical (qwen3moe.expert_count, etc.)
    ".num_local_experts",  # HF-style fallback
    ".num_experts",  # plain
    ".moe.expert_count",  # some hybrid/MTP forks
)

# Filename-level MoE marker: the "A{N}B" suffix that vendors use to
# advertise the active-parameter count of an MoE model
# (e.g. Qwen3.5-30B-A3B, Gemma-4-26B-A4B, Qwen3.5-122B-A10B). This is
# a *fallback only* — when GGUF metadata declares no expert count but
# the filename clearly says "active 3B of 30B total", we trust the
# filename and route the model through the MoE placement path.
_MOE_FILENAME_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(\d+(?:\.\d+)?)B"  # total params, e.g. "30B"
    r"[-_.]?A(\d+(?:\.\d+)?)B"  # active params, e.g. "A3B"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Architectures whose llama.cpp loaders are intrinsically MoE. This is a
# safety fallback for malformed/community GGUFs that dropped expert_count;
# the sentinel count 2 selects expert-aware placement without pretending the
# exact number is known.
_KNOWN_MOE_ARCHS = frozenset(
    {
        "afmoe",
        "arctic",
        "bailingmoe",
        "bailingmoe2",
        "bailingmoe2.5",
        "bailingmoe3",
        "bailingmoe2_5",
        "bailing_hybrid",
        "cohere2moe",
        "dbrx",
        "deepseek",
        "deepseek2",
        "deepseek32",
        "deepseek4",
        "dots1",
        "ernie4_5-moe",
        "ernie4_5_moe",
        "exaone-moe",
        "glm4moe",
        "glm5next",
        "glm5-next",
        "gpt-oss",
        "granitemoe",
        "granitemoehybrid",
        "grok",
        "grovemoe",
        "hunyuan-moe",
        "hunyuan_moe",
        "hyv4",
        "hy_v4",
        "jamba",
        "kimi-k3",
        "kimi_k3",
        "kimi-linear",
        "lfm2moe",
        "llada-moe",
        "llada_moe",
        "mellum",
        "minimax-01",
        "minimax-m2",
        "minimax-m3",
        "nemotron_h_moe",
        "nemotron-h-moe",
        "nomic-bert-moe",
        "olmoe",
        "phimoe",
        "qwen2moe",
        "qwen3moe",
        "qwen3vlmoe",
        "qwen35moe",
        "qwen4exp",
        "step35",
    }
)


def _moe_expert_count(model: ModelEntry) -> int:
    """Return expert_count from GGUF metadata, or 0 if dense / unknown.

    Detection order:
      1. ``<arch>.expert_count`` — canonical GGUF key.
      2. Any ``*.expert_count`` key in metadata (older quantizers).
      3. Common alternate suffixes (``num_local_experts`` etc.) — some
         MTP and hybrid forks emit these instead of the canonical name.
      4. Filename heuristic: ``{Total}B-A{Active}B`` pattern returns 1
         (sentinel "MoE confirmed by filename, exact count unknown" —
         enough to enter the MoE placement branch). This catches GGUFs
         where the metadata writer dropped the expert count entirely.
    """
    md = model.metadata
    if md:
        arch = md.get("general.architecture")
        # Step 1+3: try every alt suffix on the model's own architecture.
        if arch:
            for suffix in _MOE_ALT_KEY_SUFFIXES:
                key = f"{arch}{suffix}"
                if key in md:
                    try:
                        n = int(md[key])
                        if n > 0:
                            return n
                    except (TypeError, ValueError):
                        pass
        # Step 2+3: scan all keys for any of the alt suffixes.
        for k, v in md.items():
            if any(k.endswith(s) for s in _MOE_ALT_KEY_SUFFIXES):
                try:
                    n = int(v)
                    if n > 0:
                        return n
                except (TypeError, ValueError):
                    continue

    # Step 4: authoritative architecture fallback. The exact count is unknown,
    # but the loader itself is MoE, so route through expert-aware placement.
    arch = str((md or {}).get("general.architecture") or "").lower()
    if arch in _KNOWN_MOE_ARCHS:
        return 2

    # Step 5: filename fallback. Sentinel 2 triggers ``is_moe`` without lying
    # about the real expert count.
    if _MOE_FILENAME_RE.search(model.name):
        return 2
    return 0


# ---------------------------------------------------------------------------
# Configuration result


@dataclass
class TunedConfig:
    ctx: int
    ngl: int
    threads: int
    batch_threads: int
    batch: int
    ubatch: int
    cache_k: str
    cache_v: str
    flash_attn: bool
    sampling: Dict[str, Any] = field(default_factory=dict)

    # Explicit llama.cpp model-loading strategy. ``auto`` leaves the binary's
    # runtime policy untouched (b10364+ avoids mmap on iGPUs). Legacy booleans
    # remain for settings/API compatibility and are normalized by
    # ``effective_load_mode``.
    load_mode: str = "auto"
    mlock: bool = False
    no_mmap: bool = False
    numa: Optional[str] = None
    tensor_split: Optional[str] = None
    main_gpu: Optional[int] = None
    # Exact backend-qualified device for the multimodal projector (for
    # example ``Vulkan1``). b10541+ can pin MTMD independently from model
    # tensors; older binaries safely lose the flag during help-based pruning.
    mmproj_device: Optional[str] = None

    n_cpu_moe: Optional[int] = None
    is_moe: bool = False
    expert_count: int = 0

    estimated_model_vram_gb: float = 0.0
    estimated_model_ram_gb: float = 0.0
    # Giant row-gather weights left mmap-backed by current --lazy-mode auto
    # (legacy --tensor-read-lazy auto).
    # ``mapped_model_ram_gb`` is the full file-backed virtual mapping;
    # ``mapped_model_resident_gb`` is the conservative active working-set
    # reservation used for physical-memory planning. They are not ordinary
    # CPU-offloaded layers and are not eligible for GPU tensor splitting.
    mapped_model_ram_gb: float = 0.0
    mapped_model_resident_gb: float = 0.0
    estimated_kv_gb: float = 0.0
    full_offload: bool = False
    # CPU and accelerator allocations share one physical pool (Apple
    # Silicon / confirmed integrated GPU). Display and preflight code must
    # present one combined capacity instead of independent RAM + VRAM totals.
    unified_memory: bool = False

    # ---- New (display fidelity) ---------------------------------------
    # VRAM that vision (mmproj) and draft model consume on the GPU.
    # The main model placement subtracts these from `free_vram_gb` to
    # decide layer placement, but until now the display only showed the
    # main-model VRAM number — so toggling vision/draft produced
    # counter-intuitive context changes the user could not explain.
    # Surfacing both here lets the GUI render the FULL GPU picture.
    vision_vram_gb: float = 0.0
    vision_ram_gb: float = 0.0
    draft_vram_gb: float = 0.0

    # ---- mmproj host-RAM offload (``--no-mmproj-offload``) -------------
    # When True the multimodal projector (mmproj) is forced to stay in
    # system RAM instead of being offloaded to the GPU. compute_config
    # then reports the projector size as ``vision_ram_gb`` (it leaves
    # ``vision_vram_gb`` at 0) so the RAM budget reflects the real host
    # footprint. build_command emits ``--no-mmproj-offload`` ONLY when an
    # mmproj is actually loaded AND this flag is set.
    no_mmproj_offload: bool = False

    # ---- Host-memory prompt cache (``--cache-ram``) --------------------
    # Size of the host-RAM prompt cache in MiB. Semantics:
    #   * positive value → reserve exactly that many MiB;
    #   * 0              → prompt cache disabled (--cache-ram 0);
    #   * -1             → legacy "unlimited"; planned conservatively as a
    #     2 GiB reservation so the RAM budget never silently explodes.
    # The computed GiB equivalent (``prompt_cache_ram_gb``) is what the
    # RAM budget subtracts and what the preview/preflight/registry show.
    prompt_cache_ram_mib: int = 2048
    prompt_cache_ram_gb: float = 0.0

    # ---- Runner-specific VRAM overhead --------------------------------
    # Extra VRAM a particular server binary reserves beyond weights + KV
    # (e.g. DiffusionGemma's diffusion-runtime buffers). compute_config
    # subtracts this from the placement / KV budget and the preview /
    # preflight / registry add it to the total GPU footprint so the
    # displayed picture matches the real on-device allocation.
    runtime_vram_overhead_gb: float = 0.0
    # Host-side architecture graph/runtime buffers beyond model mappings,
    # CPU-offloaded weights, recurrent state, and prompt cache.
    runtime_ram_overhead_gb: float = 0.0
    # Additional MoE op-offload batch workspace beyond the generic KV/FA
    # headroom. Included in GPU footprint/preflight reporting.
    batch_vram_overhead_gb: float = 0.0
    # KV split between VRAM and RAM (set by compute_config). For
    # full-offload / MoE-on-GPU the entire KV cache lives in VRAM and
    # `kv_ram_gb == 0`. For dense-hybrid placement the small RAM share
    # is shown so the user can see why context is throttled.
    kv_vram_gb: float = 0.0
    kv_ram_gb: float = 0.0
    # Fixed F32 recurrent state for Mamba/RWKV/linear-attention layers.
    # This is separate from context-growing attention KV.
    recurrent_state_vram_gb: float = 0.0
    recurrent_state_ram_gb: float = 0.0
    # KV-quant labels actually applied (may differ from cache_k/cache_v
    # when an explicit Expert override was used — kept for diagnostics).
    kv_quant_strategy: str = (
        "symmetric"  # "symmetric" | "asymmetric" | "manual" | "turbo"
    )

    no_context_shift: bool = False

    # True when the KV cache is forced into system RAM via
    # --no-kv-offload (the LOW-VRAM / ``low_vram`` perf-target lever).
    # Set by compute_config when the resolved PerformanceTarget has
    # kv_to_ram=True and there is a GPU to offload FROM. Trades
    # generation speed for context headroom drawn from system RAM.
    no_kv_offload: bool = False

    # RoPE-Scaling: aktiviert wenn ctx > native_ctx und YaRN/rope-scaling
    # verwendet werden soll (optional, nur für Modelle die es unterstützen).
    rope_scaling: bool = False
    rope_scale_factor: float = 1.0  # z.B. 4.0 für yarn mit 4x scaling

    # Expert-panel override for --spec-draft-n-max (max draft tokens per
    # speculative step). 0 = no override → the YAML profile's draft_max
    # decides (its own fallback is 2). Only takes effect when a
    # speculative path is active (external -md drafter or integrated MTP).
    draft_n_max: int = 0

    # Optional CLI extras the GUI's Expert mode injects. Examples:
    # "--jinja", "--verbose". Built-in defaults stay empty so the
    # auto-mode behaviour is unchanged.
    extra_cli_flags: List[str] = field(default_factory=list)

    # Environment variables to set when spawning llama-server.
    # Primarily used to set HIP_VISIBLE_DEVICES on Windows AMD multi-GPU
    # setups where the Windows registry GPU order differs from HIP order.
    env_overrides: Dict[str, str] = field(default_factory=dict)

    # Active performance target name ("safe" / "balanced" / "throughput").
    # Set by compute_config so display code can show what was applied.
    performance_target: str = DEFAULT_TARGET_NAME

    # --parallel N for llama-server.  Controls how many inference slots
    # the server allocates simultaneously.  Always passed explicitly so
    # llama-server's "auto" heuristic cannot over-provision KV cache.
    # Sized by the resolved PerformanceTarget (all desktop presets default to
    # one slot). The ctx calculation in compute_config divides
    # kv_budget_gb by n_parallel so each slot gets a correctly-sized KV
    # window instead of the server silently multiplying KV by N slots.
    n_parallel: int = 1

    # True when n_parallel was explicitly pinned by the GUI's Expert
    # panel (the Parallelism override).  Lets the panel render the
    # checkbox state in both Auto and Manual mode without re-deriving it
    # from the performance target's default.
    n_parallel_forced: bool = False

    # HTTP diagnostics toggles. Metrics default on because AutoTuner has
    # historically exposed GET /metrics by default. The /slots endpoint is
    # opt-in: llama.cpp treats it as an operational/debug API and some builds
    # require --slots before exposing it.
    metrics_enabled: bool = True
    slots_api_enabled: bool = False

    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers


def _decide_offload(
    model_size_gb: float,
    free_vram_gb: float,
    n_layers: int,
    has_gpu: bool,
    vram_headroom_gb: float = DEFAULT_VRAM_SAFETY_GB,
    kv_reserve_gb: float = 0.0,
    free_ram_gb: Optional[float] = None,
) -> Tuple[int, float, float, bool]:
    if not has_gpu or free_vram_gb < 1.0:
        return 0, 0.0, model_size_gb, False

    usable = max(0.0, free_vram_gb - vram_headroom_gb)

    # Reserve VRAM for the KV cache BEFORE packing weight layers, symmetric
    # with the MoE placement path (_decide_moe_offload). A dense model that
    # nearly fills VRAM used to be packed until VRAM was full, leaving ~0 GB
    # for KV → the context collapsed to a couple thousand tokens. By trimming
    # the weight budget by the KV reserve, a few layers move to CPU and the
    # freed VRAM becomes a usable GPU-resident KV cache. The reserve is
    # clamped so it can never strand the GPU entirely (keep >=60% of the
    # weight-only budget for layers) — better a smaller KV than an idle GPU.
    kv_reserve_gb = max(0.0, kv_reserve_gb)
    weight_budget = max(usable * 0.60, usable - kv_reserve_gb)

    # Full offload needs room for the weights AND the compute buffer
    # (see FULL_OFF_HEADROOM_GB). Below that threshold the model falls
    # through to PARTIAL offload — excess layers spill to CPU and the
    # server keeps running instead of being refused at launch.
    if weight_budget >= model_size_gb + FULL_OFF_HEADROOM_GB:
        return 999, model_size_gb, 0.0, True

    if weight_budget < 0.5:
        return 0, 0.0, model_size_gb, False

    if n_layers > 0:
        per_layer_gb = model_size_gb / n_layers
        residual_overhead = model_size_gb * 0.02

        # A partial dense load splits KV by layer. Packing layers from VRAM
        # alone can push so many weights into host RAM that the CPU-side KV
        # becomes the real bottleneck (Safe then paradoxically gets less
        # context than Throughput). Search the discrete layer placements and
        # choose the fastest one that preserves the requested total KV reserve
        # in *both* pools. If the target itself cannot fit, maximize the actual
        # two-pool KV capacity rather than blindly filling either pool.
        if free_ram_gb is not None and kv_reserve_gb > 0:
            host_usable = max(0.0, float(free_ram_gb))
            feasible: List[Tuple[int, float, float, bool]] = []
            fallback: List[Tuple[float, int, float, float, bool]] = []
            for layers_on_gpu in range(n_layers + 1):
                full = layers_on_gpu == n_layers
                model_vram = layers_on_gpu * per_layer_gb
                model_ram = (
                    0.0
                    if full
                    else (n_layers - layers_on_gpu) * per_layer_gb + residual_overhead
                )
                gpu_fraction = layers_on_gpu / n_layers
                cpu_fraction = 1.0 - gpu_fraction
                gpu_fixed = model_vram + (FULL_OFF_HEADROOM_GB if full else 0.0)
                gpu_remaining = max(0.0, usable - gpu_fixed)
                ram_remaining = max(0.0, host_usable - model_ram)
                gpu_capacity = (
                    math.inf if gpu_fraction <= 0.0 else gpu_remaining / gpu_fraction
                )
                ram_capacity = (
                    math.inf if cpu_fraction <= 0.0 else ram_remaining / cpu_fraction
                )
                total_kv_capacity = min(gpu_capacity, ram_capacity)
                if model_ram <= host_usable and gpu_fixed <= usable:
                    fallback.append(
                        (
                            total_kv_capacity,
                            layers_on_gpu,
                            model_vram,
                            model_ram,
                            full,
                        )
                    )
                    if total_kv_capacity >= kv_reserve_gb:
                        feasible.append((layers_on_gpu, model_vram, model_ram, full))
            if feasible:
                layers_on_gpu, model_vram, model_ram, full = max(
                    feasible, key=lambda item: item[0]
                )
                return (
                    999 if full else layers_on_gpu,
                    model_vram,
                    model_ram,
                    full,
                )
            if fallback:
                _capacity, layers_on_gpu, model_vram, model_ram, full = max(
                    fallback, key=lambda item: (item[0], item[1])
                )
                return (
                    999 if full else layers_on_gpu,
                    model_vram,
                    model_ram,
                    full,
                )

        ngl = int(weight_budget / per_layer_gb)
        ngl = max(0, min(n_layers, ngl))
        model_vram = ngl * per_layer_gb
        model_ram = (n_layers - ngl) * per_layer_gb + residual_overhead
        return ngl, model_vram, model_ram, False

    estimated_layers = 50
    fraction = weight_budget / model_size_gb
    ngl = max(0, int(fraction * estimated_layers))
    return ngl, weight_budget, max(0.0, model_size_gb - weight_budget), False


def _decide_moe_offload(
    model_size_gb: float,
    free_vram_gb: float,
    free_ram_gb: float,
    n_layers: int,
    expert_count: int,
    params_billion: float,
    target_ctx: int,
    base_kv_per_token_mb: float = 0.0,
    ram_safety_gb: float = DEFAULT_RAM_SAFETY_GB,
    moe_vram_safety_gb: float = MOE_VRAM_SAFETY_GB,
    moe_placement_ctx_target: int = MOE_PLACEMENT_CTX_TARGET,
    batch_vram_reserve_gb: float = 0.0,
    n_parallel: int = 1,
    rope_scaling: bool = False,
) -> Tuple[int, Optional[int], float, float, bool]:
    """Decide how to split an MoE model between GPU and CPU.

    Strategy:
      1. Reserve VRAM for the KV cache up front (Vulkan requires KV to
         live entirely in VRAM for MoE — RAM-resident KV crashes with
         GGML_ASSERT(addr) on the AMD/Vulkan backend).
      2. Reserve VRAM for shared (non-expert) tensors.
      3. Pack as many expert layers as possible into the leftover VRAM;
         everything else goes to CPU via `--n-cpu-moe`.

    A practical KV target of ``moe_placement_ctx_target`` is used
    instead of the profile maximum, so we don't reserve VRAM for context
    the user is unlikely to need on this run. The actual ctx in
    compute_config can still be larger if the remaining VRAM allows it.
    The target is supplied by the active PerformanceTarget — "safe"
    keeps the legacy 128k value, "throughput" shrinks it to 32k.
    """
    if base_kv_per_token_mb <= 0:
        base_kv_per_token_mb = kv_per_token_mb_f16(params_billion)

    shared_overhead_gb = model_size_gb * 0.08
    per_layer_expert_gb = max(0.001, (model_size_gb - shared_overhead_gb) / n_layers)

    # ---- KV reservation in VRAM (global Q4_0 default) -------------------
    # Cap at moe_placement_ctx_target so we don't pessimise layer placement
    # for huge profile_max values (Qwen3.6 → 262k, but most users run 32k).
    kv_reservation_ctx = max(2048, min(target_ctx, moe_placement_ctx_target))
    kv_reserve_gb = (
        kv_reservation_ctx
        * base_kv_per_token_mb
        * kv_quant_factor(DEFAULT_KV_CACHE_TYPE)
        * max(1, n_parallel)
    ) / 1024.0

    # The final context planner withholds both an absolute backend workspace
    # and a context-scaled fractional guard from the raw KV budget. Placement
    # must reserve that same raw amount *before* packing expert layers; otherwise
    # a slightly smaller quant that just fits fully on GPU paradoxically gets
    # less context than a larger quant that is forced to spill experts to RAM.
    # Solve ``usable = raw * (1 - fraction) - absolute`` for the raw budget
    # needed to leave ``kv_reserve_gb`` usable after the final guard.
    headroom_absolute_gb, headroom_fraction = _kv_headroom_reserve(
        kv_reservation_ctx,
        max(1, n_parallel),
        rope_scaling,
    )
    raw_kv_reserve_gb = (kv_reserve_gb + headroom_absolute_gb) / max(
        0.01, 1.0 - headroom_fraction
    )

    # If even the estimated non-expert/shared tensors cannot fit, --n-cpu-moe
    # cannot rescue the model: that flag moves experts only. Fall back to a
    # fully CPU-resident load instead of reporting an impossible GPU footprint.
    if free_vram_gb - moe_vram_safety_gb < shared_overhead_gb:
        return 0, n_layers, 0.0, model_size_gb, False

    # Layer placement uses VRAM left over AFTER KV, shared tensors, and the
    # large MoE op-offload batch workspace selected by the performance tier.
    usable_for_experts = (
        free_vram_gb
        - moe_vram_safety_gb
        - shared_overhead_gb
        - raw_kv_reserve_gb
        - max(0.0, batch_vram_reserve_gb)
    )

    if usable_for_experts < 0:
        # Shared tensors fit but the desired KV/batch reservation does not.
        # Keep shared tensors on GPU, all experts in RAM, and let final context
        # sizing use the actual residual capacity rather than double-counting.
        return (
            999,
            n_layers,
            shared_overhead_gb,
            model_size_gb - shared_overhead_gb,
            False,
        )

    layers_on_gpu = int(usable_for_experts / per_layer_expert_gb)
    layers_on_gpu = max(0, min(n_layers, layers_on_gpu))
    n_cpu_moe = n_layers - layers_on_gpu

    model_vram = shared_overhead_gb + layers_on_gpu * per_layer_expert_gb

    if n_cpu_moe == 0:
        # All experts on GPU.
        return 999, 0, model_size_gb, 0.0, True

    # Some experts on CPU — they live in RAM via mmap.
    model_ram = n_cpu_moe * per_layer_expert_gb
    return 999, n_cpu_moe, model_vram, model_ram, False


def _gpu_usable_cap_gb(gpu: "GPUInfo", is_primary: bool) -> float:
    """Per-card usable VRAM cap (GB) for multi-GPU placement.

    Keep a little headroom so the card never runs bone-dry, but make it
    PROPORTIONAL to the card size instead of a flat amount. A flat 3 GB
    on a 16 GB card is a ~19 % tax that strands several GB of the RX
    9070 XT on big MoE spreads (Basti's report #1), while the same 3 GB
    is trivial on a 32 GB card. We instead reserve a fraction of TOTAL
    VRAM — more on the secondary (it also drives desktop + OBS encode)
    than the primary — floored at a small minimum so a tiny card still
    keeps a safety margin. Then clamp by the card's *free* VRAM so we
    never plan to use memory other apps already hold ("inclusive was
    schon genutzt wird"), and never let the proportional reserve grow so
    large on a big card that it wastes more than the old flat value
    would have.

    Module-level (not nested in the spread section) because the MoE
    expert-placement budget must use the SAME caps: budgeting
    ``--n-cpu-moe`` against the raw combined free VRAM while the spread
    reserves per-card headroom overcommits by exactly that headroom and
    pushes the primary card to its physical limit.
    """
    total_gb = gpu.total_vram_mb / 1024.0
    frac = 0.06 if is_primary else 0.10  # 6 % primary, 10 % secondary
    floor = 1.0 if is_primary else 1.5  # GB, minimum breathing room
    cap_flat = 2.0 if is_primary else 3.0  # legacy flat reserve = ceiling
    headroom = min(cap_flat, max(floor, total_gb * frac))
    cap_by_total = total_gb - headroom
    free_gb = gpu.free_vram_mb / 1024.0
    usable = max(0.0, min(cap_by_total, free_gb))
    # A nearly-full peer should not be pulled into a new server for a token
    # amount of memory: doing so can OOM the already-running workload and makes
    # device visibility span both cards for no practical gain. The selected
    # primary is still allowed below this threshold because it may be an
    # explicit pin or the only viable device.
    if not is_primary and usable < 2.0:
        return 0.0
    return usable


def _visibility_env_for_gpus(
    gpus: List[GPUInfo], indices: List[int]
) -> Tuple[Dict[str, str], bool]:
    """Return backend-correct visibility selectors and remap status.

    ``indices`` are exact-runtime ordinals in the desired visible order. The
    boolean tells the caller whether those selectors remap the chosen devices
    to contiguous 0..N-1 indices (and therefore whether ``main_gpu`` may be
    reset to a visible-list position).
    """
    if not gpus or len(gpus) != len(indices):
        return {}, False
    backends = {
        str(g.runtime_backend or "").strip().lower() for g in gpus if g.runtime_backend
    }
    comma = ",".join(str(i) for i in indices)

    if len(backends) == 1:
        backend = next(iter(backends))
        if backend.startswith("cuda"):
            return {"CUDA_VISIBLE_DEVICES": comma}, True
        if backend.startswith("hip") or backend.startswith("rocm"):
            return {"HIP_VISIBLE_DEVICES": comma}, True
        if backend.startswith("vulkan"):
            return {"GGML_VK_VISIBLE_DEVICES": comma}, True
        if backend.startswith("sycl"):
            selector = ";".join(f"level_zero:{i}" for i in indices)
            return {"ONEAPI_DEVICE_SELECTOR": selector}, True
        # Metal/OpenVINO use different placement semantics; do not claim a
        # visibility remap that the backend does not provide.
        return {}, False

    if backends:
        # Mixed exact backends cannot share one ordinal namespace.
        return {}, False

    # Legacy/unprobeable build fallback. Emit every plausible selector for a
    # homogeneous vendor; inactive backends ignore their variable.
    vendors = {g.vendor.lower() for g in gpus}
    if vendors == {"amd"}:
        return {
            "HIP_VISIBLE_DEVICES": comma,
            "GGML_VK_VISIBLE_DEVICES": comma,
        }, True
    if vendors == {"nvidia"}:
        return {
            "CUDA_VISIBLE_DEVICES": comma,
            "GGML_VK_VISIBLE_DEVICES": comma,
        }, True
    if vendors == {"intel"}:
        return {
            "ONEAPI_DEVICE_SELECTOR": ";".join(f"level_zero:{i}" for i in indices),
            "GGML_VK_VISIBLE_DEVICES": comma,
        }, True
    return {}, False


def _split_layers_by_bytes(layer_bytes: List[float], caps: List[float]) -> List[int]:
    """Assign contiguous front-to-back layer ranges to devices so each
    card's BYTE load fills it proportionally to its usable capacity —
    and never exceeds the cap where avoidable.

    Why this exists: llama.cpp interprets ``--tensor-split`` as a
    LAYER-COUNT proportion, not a byte proportion. Layer ``il`` goes to
    the device whose normalised cumulative split bucket contains
    ``il / n_layers`` — device 0 receives the FIRST chunk of layers
    (verified against b9859 ``src/llama-model.cpp``). Meanwhile
    ``--n-cpu-moe N`` strips the expert tensors of the FIRST N layers
    to CPU (``common/arg.cpp``: ``llm_ffn_exps_block_regex(i)`` for
    i < N), so those front layers keep only tiny attention/norm tensors
    in VRAM while layers N.. carry the full expert weight.

    Per-layer GPU bytes are therefore wildly non-uniform, and emitting a
    byte-proportional FRACTION mis-places bytes badly: on Basti's
    step-3.7-Flash run the RX 9070 XT (Vulkan device 0) received only
    expert-stripped front layers and idled at ~8/16 GB while the R9700
    was pushed to its limit — exactly the stranded-VRAM problem the
    capacity-fill strategy was meant to solve.

    This helper solves the inverse problem: given the actual per-layer
    GPU byte weights (in layer order 0..L-1) and per-device capacity
    caps (in VISIBLE device order — the device that receives layer 0
    first), it returns the layer COUNT each device should get. Those
    counts are emitted directly as ``--tensor-split`` values; llama.cpp
    normalises them, so the counts reproduce themselves exactly (up to
    ±1 rounding) and the byte distribution lands as planned.

    With uniform per-layer bytes (dense, or MoE with n_cpu_moe == 0)
    this degenerates to a capacity-proportional layer split — identical
    to the previous fraction-based behaviour.

    Granularity note: layers are indivisible, so when no contiguous
    assignment satisfies every cap exactly (a single expert layer can
    weigh 1.5+ GB), the residual overshoot — strictly less than one
    layer — remains on the LAST device after the fix-up pass. The caps
    already contain per-card headroom (_gpu_usable_cap_gb), so a
    sub-layer overshoot eats reserved breathing room, never the
    physical VRAM limit.
    """
    n_dev = len(caps)
    n_lay = len(layer_bytes)
    counts = [0] * n_dev
    if n_dev == 0 or n_lay == 0:
        return counts
    total_bytes = sum(layer_bytes)
    total_cap = sum(caps)
    if total_bytes <= 0 or total_cap <= 0:
        # Degenerate input — even layer split so we never emit all-zeros.
        base = n_lay // n_dev
        counts = [base] * n_dev
        counts[-1] += n_lay - base * n_dev
        return counts

    # Per-device byte target = capacity-proportional share of the total.
    targets = [total_bytes * (c / total_cap) for c in caps]

    # Front-fill: walk the layers in order, giving each device layers
    # until its byte target (or hard cap) is reached. The boundary layer
    # is taken only if doing so lands CLOSER to the target than stopping
    # short — this keeps the split balanced when one heavy expert layer
    # straddles the boundary.
    i = 0
    for j in range(n_dev - 1):
        acc = 0.0
        while i < n_lay:
            w = layer_bytes[i]
            if acc + w > caps[j] + 1e-9:
                break
            if acc + w > targets[j] and (acc + w - targets[j]) >= (targets[j] - acc):
                break
            acc += w
            counts[j] += 1
            i += 1
    counts[-1] = n_lay - i

    # Fix-up pass: if a later device ended up over its cap (the remainder
    # after front-filling can be heavier than the last card holds when the
    # heavy expert layers all sit at the back), shift boundary layers
    # backward onto earlier devices that still have spare capacity. Only
    # adjacent moves are possible because assignments must stay contiguous.
    def _dev_bytes(j: int) -> float:
        start = sum(counts[:j])
        return sum(layer_bytes[start : start + counts[j]])

    for _ in range(n_lay):
        moved = False
        for j in range(n_dev - 1):
            if counts[j + 1] > 0 and _dev_bytes(j + 1) > caps[j + 1] + 1e-9:
                boundary = sum(counts[: j + 1])
                w = layer_bytes[boundary]
                if _dev_bytes(j) + w <= caps[j] + 1e-9:
                    counts[j] += 1
                    counts[j + 1] -= 1
                    moved = True
        if not moved:
            break

    return counts


# ---- Turbo-Quant labels --------------------------------------------------
# The TurboQuant family of forks (TheTom/turboquant_plus,
# AtomicBot/atomic-llama-cpp-turboquant, spiritbuun/buun-llama-cpp)
# adds three new -ctk / -ctv labels that pack the KV-cache much
# tighter than the stock f16 → q4_0 ladder:
#
#     turbo4   ~3.8× vs F16   (4-bit, highest accuracy, default fallback)
#     turbo3   ~4.3× vs F16   (3-bit, the "sweet spot" — recommended default)
#     turbo2   ~6.4× vs F16   (2-bit, max compression, quality drops)
#
# We map upstream quant choices to their turbo equivalent **at the
# bit-width tier the algorithm already picked** — q8_0 → turbo4 (both
# are the "high-accuracy" tier), q5_0 → turbo3 (mid), q4_0 → turbo3
# (low, but turbo3 is still measurably better than q4_0 at long ctx).
# turbo2 is intentionally not auto-selected; users who really want it
# pick it manually in the Expert panel.
_TURBO_QUANT_MAP: Dict[str, str] = {
    "f16": "turbo4",  # if someone runs f16 on a turbo fork, give them headroom
    "q8_0": "turbo4",  # 4-bit, ~3.8x, highest-accuracy turbo tier
    "q5_0": "turbo3",  # 3-bit, ~4.3x, the canonical default
    "q5_1": "turbo3",
    "q4_0": "turbo3",  # 3-bit beats q4_0 noticeably at long context
    "q4_1": "turbo3",
    "iq4_nl": "turbo3",
}


def _turbo_quant_for(label: str) -> str:
    """Map a normal KV quant label to its TurboQuant equivalent.

    Falls back to the input label when no mapping is known — that keeps
    Turbo a *safe* toggle: the worst case is "same quant as before".
    """
    return _TURBO_QUANT_MAP.get(label.lower(), label)


def _pick_kv_quant(
    profile_recommended: str,
    target_ctx: int,
    base_kv_per_token_mb: float,
    kv_budget_gb: float,
    model_max_ctx: int = 0,  # native_ctx aus GGUF-Metadata (0 = keine Begrenzung)
    *,
    turbo: bool = False,
    asymmetric: bool = True,  # Vulkan b9106+ supports asymmetric FA
    base_k_per_token_mb: Optional[float] = None,
    base_v_per_token_mb: Optional[float] = None,
) -> Tuple[str, str]:
    """Return the capacity-first automatic K/V cache pair.

    v5.3.2 intentionally standardises stock llama.cpp Auto mode on symmetric
    Q4_0 for every normal model. This avoids spending otherwise usable context
    on an automatic F16/Q8 upgrade and avoids mixed K/V FlashAttention fallback
    differences across Vulkan, HIP, and default CUDA builds. The historical
    ``profile_recommended`` and ``asymmetric`` parameters remain in the public
    signature for compatibility, but no longer raise automatic precision.

    Manual Expert pins remain untouched by :func:`compute_config`. With
    ``turbo=True`` the Q4_0 baseline maps to the fork-only TurboQuant default.
    Runner-specific paths that cannot apply cache quantisation bypass this
    helper and report their real F16 cache.
    """
    # Beschränke target_ctx auf Modell-Maximum wenn nötig.
    if model_max_ctx > 0 and target_ctx > model_max_ctx:
        target_ctx = model_max_ctx

    # Read the legacy arguments so static analyzers and third-party wrappers
    # can keep calling the old signature without implying they affect Auto.
    _ = profile_recommended, asymmetric
    pairs: List[Tuple[str, str]] = [(DEFAULT_KV_CACHE_TYPE, DEFAULT_KV_CACHE_TYPE)]

    budget_mb = kv_budget_gb * 1024 * 0.98

    # When the user enabled Turbo-KV, the quants we are about to test
    # are NOT the labels that will actually end up in the cmd line —
    # we map (q8_0, q5_0, q4_0) → (turbo4, turbo3, turbo3) just below.
    # The Turbo labels are denser than their q-counterparts, so the
    # budget check has to use the turbo factor or the AutoTuner will
    # leave a lot of context on the table (Basti's complaint: "the
    # token count doesn't change when switching to Turbo").
    def _per_token_for_pair(k_label: str, v_label: str) -> float:
        if turbo:
            k_label = _turbo_quant_for(k_label)
            v_label = _turbo_quant_for(v_label)
        k_factor = kv_quant_factor(k_label)
        v_factor = kv_quant_factor(v_label)
        if (
            base_k_per_token_mb is not None
            and base_v_per_token_mb is not None
            and base_k_per_token_mb + base_v_per_token_mb > 0
        ):
            return base_k_per_token_mb * k_factor + base_v_per_token_mb * v_factor
        return base_kv_per_token_mb * (k_factor + v_factor) / 2

    for k, v in pairs:
        per_tok = _per_token_for_pair(k, v)
        if per_tok <= 0:
            continue
        max_fit = int(budget_mb / per_tok)
        if max_fit >= target_ctx:
            chosen_k, chosen_v = k, v
            break
    else:
        # Nothing in the table fit — fall back to the most aggressive entry.
        chosen_k, chosen_v = pairs[-1]

    if turbo:
        chosen_k = _turbo_quant_for(chosen_k)
        chosen_v = _turbo_quant_for(chosen_v)

    return chosen_k, chosen_v


def _kv_headroom_reserve(
    target_ctx: int,
    n_parallel: int,
    rope_scaling: bool,
) -> Tuple[float, float]:
    """KV-budget headroom to withhold so very long context (especially
    under YaRN rope-scaling) never fills VRAM/RAM to 99.5 % and OOMs the
    compute buffer.

    Returns ``(absolute_gb, fractional)``:
      * ``absolute_gb`` — per-slot compute-buffer / flash-attention
        workspace + Vulkan/ROCm staging reserve.
      * ``fractional``  — covers cumulative per-token KV estimation drift
        and long-context scratch; grows with ``target_ctx`` and is
        amplified under YaRN.

    Root cause this fixes: the KV budget historically left only the thin
    performance-target safety band (0.15–0.30 GB) plus a 0.5 % rounding
    factor. At ~1 M tokens the GPU compute buffer alone reaches several
    GB, so ``vkAllocateMemory`` failed ("Vulkan buffer error") and the
    server crashed on the next attempt. The reserve is modest at short
    context (so we don't waste headroom) and meaningful at long context
    (so every mode — throughput / balanced / safe / low_vram — stays
    safe instead of driving the card to the brink of OOM).
    """
    slots = max(1, n_parallel)
    # ~0.6 GB/slot covers the FA workspace + Vulkan/ROCm staging buffers
    # that llama.cpp allocates on top of the KV cache.
    absolute_gb = 0.6 * slots

    # 3 % baseline, rising on a log2 scale with context
    # (3 % → ~6.6 % at 256k → ~9 % at 1 M), capped at 15 % (20 % YaRN).
    if target_ctx > 0:
        steps = max(0.0, math.log2(target_ctx / 32768.0))
        fractional = 0.03 + 0.06 * (steps / 5.0)
    else:
        fractional = 0.03
    fractional = min(max(fractional, 0.03), 0.15)
    if rope_scaling:
        # YaRN amplifies the attention/RoPE init scratch and makes the
        # per-token KV estimate slightly less certain at extreme lengths.
        fractional = min(fractional * 1.35, 0.20)

    return round(absolute_gb, 2), round(fractional, 3)


def _usable_kv_budget_after_headroom(
    raw_budget_gb: float,
    target_ctx: int,
    n_parallel: int,
    rope_scaling: bool,
) -> float:
    """Apply the same compute/scratch reserve to any candidate KV budget."""
    absolute_gb, fractional = _kv_headroom_reserve(target_ctx, n_parallel, rope_scaling)
    raw_budget_gb = max(0.0, raw_budget_gb)
    return max(0.0, raw_budget_gb - absolute_gb - fractional * raw_budget_gb)


# ---------------------------------------------------------------------------
# mlock safety


def effective_load_mode(config: "TunedConfig") -> Optional[str]:
    """Return the normalized llama.cpp load mode for a tuned configuration."""
    mode = str(getattr(config, "load_mode", "auto") or "auto").strip().lower()
    if mode in {"none", "mmap", "mlock", "mmap+mlock", "dio"}:
        return mode
    # Legacy configs/snapshots used two independent checkboxes. Preserve their
    # intended combinations while moving command generation to --load-mode.
    if bool(getattr(config, "mlock", False)):
        return "mlock" if bool(getattr(config, "no_mmap", False)) else "mmap+mlock"
    if bool(getattr(config, "no_mmap", False)):
        return "none"
    return None


def _mlock_unsafe_with_gpu(
    system: SystemInfo,
    force_mlock: bool,
    binary: Optional[str] = None,
) -> bool:
    """True when enabling a locking load mode may abort this llama.cpp build.

    The old Vulkan pinned-host-buffer crash is conservatively assumed for
    unknown/older binaries. b10151's split load-mode implementation is allowed:
    both ``mlock`` and ``mmap+mlock`` were live-tested on Windows/Vulkan.
    """
    if not system.gpus or force_mlock:
        return False
    build = _probe_binary_build_number(binary) if binary else None
    return build is None or build < _MIN_DISTINCT_MLOCK_BUILD


def veto_unsafe_mlock(
    config: "TunedConfig",
    system: SystemInfo,
    force_mlock: bool = False,
    binary: Optional[str] = None,
) -> bool:
    """Final safety net for locking modes on old/unprobeable GPU builds."""
    mode = effective_load_mode(config)
    if mode in {"mlock", "mmap+mlock"} and _mlock_unsafe_with_gpu(
        system, force_mlock, binary
    ):
        config.load_mode = "auto"
        config.mlock = False
        config.no_mmap = False
        return True
    return False


# ---------------------------------------------------------------------------
# Main entry


def compute_config(
    model: ModelEntry,
    system: SystemInfo,
    profile: ModelProfile,
    draft_model: Optional[ModelEntry] = None,
    user_ctx: Optional[int] = None,
    ram_safety_gb: Optional[float] = None,
    vram_safety_gb: Optional[float] = None,
    force_mlock: bool = False,
    perf_target: Optional[PerformanceTarget] = None,
    mode: str = "chat",
    *,
    # ---- Expert-mode (auto-cascade) overrides --------------------------
    # When any of these is set, the AutoTuner respects the user-supplied
    # value and lets the rest of the configuration cascade around it.
    # Manual mode bypasses compute_config entirely and builds a
    # TunedConfig directly from widget values, so these only apply to
    # the cascading Auto branch of the Expert panel.
    turbo_kv: bool = False,  # Map quants → TurboQuant equivalents
    force_cache_k: Optional[str] = None,  # Pin K-quant; ctx adjusts
    force_cache_v: Optional[str] = None,  # Pin V-quant; ctx adjusts
    force_ngl: Optional[int] = None,  # Pin layer offload count
    force_n_cpu_moe: Optional[int] = None,  # Pin MoE CPU-layer count
    force_n_parallel: Optional[int] = None,  # Pin --parallel slot count
    force_draft_n_max: Optional[int] = None,  # Pin speculative rollback depth
    force_rope_scale: Optional[bool] = None,  # Force YaRN on/off
    # ---- GPU priority overrides ----------------------------------------
    # Optional mapping of GPU name → user-assigned priority (≥1).
    # When provided, the GPU with the highest priority×VRAM score is
    # selected as the primary compute device (main_gpu). Priorities are
    # read from autotuner_settings.json → gpu_overrides.priority and
    # exposed through app_settings.get_gpu_priorities(). When absent or
    # None, pure VRAM size determines the primary GPU (legacy behaviour).
    gpu_priorities: Optional[Dict[str, int]] = None,
    # ---- Hard GPU pin --------------------------------------------------
    # Optional GPU *name* the user wants this server to boot on EXCLUSIVELY.
    # When set (and the named card is present) it overrides both the
    # priority×VRAM primary selection AND the free-VRAM demotion below:
    # the model is pinned to that single card and every other GPU is hidden
    # via the visibility env vars. This is the "du hast jetzt auf der GPU
    # only zu booten die ich bestimme" escape hatch — used when launching a
    # second server so the user can send it to the still-empty card instead
    # of letting it pile onto an already-full one. Matched case-insensitively
    # against GPUInfo.name; an unknown name is ignored (falls back to auto).
    force_gpu: Optional[str] = None,
    # ---- mmproj host-RAM offload (``--no-mmproj-offload``) -------------
    # When True the multimodal projector is forced to stay in system RAM
    # instead of being offloaded to the GPU. The projector size then shows
    # up as ``vision_ram_gb`` (and is subtracted from the RAM budget)
    # rather than ``vision_vram_gb``. build_command emits the flag only
    # when an mmproj is actually loaded AND this is True.
    no_mmproj_offload: bool = False,
    # ---- Host-memory prompt cache (``--cache-ram``) --------------------
    # Size of the host-RAM prompt cache in MiB. Semantics:
    #   * positive -> reserve exactly that many MiB;
    #   * 0        -> cache disabled (--cache-ram 0), no RAM reservation;
    #   * -1        -> legacy "unlimited"; planned conservatively as a
    #     PROMPT_CACHE_UNLIMITED_RESERVE_GB reservation so the RAM budget
    #     never silently explodes. build_command emits the resolved value
    #     (unless an explicit caller override is supplied).
    prompt_cache_ram_mib: int = PROMPT_CACHE_RAM_MIB_DEFAULT,
) -> TunedConfig:
    """Compute a TunedConfig that fits this model on this system.

    Priority order for VRAM allocation:
      1. Vision model (mmproj) — always placed on GPU first
      2. Draft model (speculative decoding) — always placed on GPU first
      3. Main model (weights + KV cache)

    The ``perf_target`` argument controls the safety/headroom regime
    (see ``performance_target.py``). If ``None``, it is resolved from
    ``profile.performance_target`` — falling back to "balanced" if the
    profile doesn't specify one. Callers (CLI, GUI) typically resolve
    the target themselves so a user override beats the YAML default.

    Explicit ``ram_safety_gb`` / ``vram_safety_gb`` arguments still win
    over the perf_target's values; pass ``None`` (the default) to use
    whatever the resolved target prescribes.

    Expert overrides (keyword-only)
    --------------------------------
    These are exposed primarily for the GUI's Expert panel. The plain
    CLI path keeps using the auto-tuned defaults — only set these when
    you have a specific reason to pin a value. The cascading rule:
    *whatever you pin stays; everything not pinned recomputes around it*.
    """
    # ---- Resolve performance target. Caller-supplied wins; otherwise we
    # fall back to whatever the profile recommends (or "balanced").
    if perf_target is None:
        perf_target = resolve_performance_target(
            cli_choice=None,
            profile_choice=getattr(profile, "performance_target", "") or None,
        )

    # ---- Apply the target's safety values where the caller didn't override.
    if ram_safety_gb is None:
        ram_safety_gb = perf_target.ram_safety_gb
    if vram_safety_gb is None:
        vram_safety_gb = perf_target.dense_vram_safety_gb

    # Number of parallel inference slots — always passed as --parallel N
    # to llama-server to prevent auto-detection from over-provisioning KV.
    # An Expert-panel pin (force_n_parallel) wins over the performance
    # target's default AND drives the per-slot KV-budget calculation
    # below, so context shrinks to fit N slots instead of llama-server
    # silently multiplying KV by N (the documented RAM-explosion bug).
    n_parallel_forced = force_n_parallel is not None
    n_parallel: int = (
        max(1, int(force_n_parallel))
        if force_n_parallel is not None
        else max(1, perf_target.n_parallel)
    )

    has_gpu = bool(system.gpus) and system.total_vram_gb > 1
    unified_memory = bool(getattr(system, "has_unified_memory", False))
    free_vram = max(0.0, system.free_vram_gb)
    n_layers = model.n_layers
    model_arch = str((model.metadata or {}).get("general.architecture") or "").lower()
    placement_model_size_gb = model.placement_size_gb
    read_lazy_table_gb = model.read_lazy_size_gb
    # The complete lazy tensor remains visible as a file-backed virtual map,
    # while only rows touched by inference need physical residency. Charging
    # the entire mapping against *currently free* RAM made a 26.8-GiB PLE map
    # consume more planner budget whenever unrelated applications were open,
    # collapsing an otherwise valid Qwen3.8 Flash-Next context to the 2k floor.
    # Keep both facts: full mapping for diagnostics, bounded active residency
    # for capacity planning. File-backed pages remain reclaimable on Windows
    # and POSIX; additional residency may page under pressure but is not an OOM
    # requirement at mapping time.
    mapped_model_ram_gb = read_lazy_table_gb
    mapped_model_resident_gb = (
        min(
            read_lazy_table_gb,
            max(
                _READ_LAZY_RESIDENT_MIN_GB,
                read_lazy_table_gb * _READ_LAZY_RESIDENT_FRACTION,
            ),
        )
        if read_lazy_table_gb > 0
        else 0.0
    )
    is_qwen4exp = model_arch == "qwen4exp"
    qwen4exp_ubatch = _qwen4exp_ubatch_for_target(perf_target.name)

    # ---- (0) MoE detection
    expert_count = _moe_expert_count(model)
    is_moe = expert_count > 1
    params_b = extract_params_billion(model.name)

    # ---- (0.2) Primary inference GPU selection (multi-GPU only)
    # The user's preferred main GPU is the one with the highest
    # priority×VRAM score (e.g. R9700 32 GB @ priority 2 beats RX 9070 XT
    # 16 GB @ priority 1).  Two things are computed against THIS card:
    #   • MoE expert placement (n_cpu_moe) — experts never spread onto the
    #     secondary GPU, they spill to CPU/RAM, so only the primary's free
    #     VRAM is relevant.  Using the *summed* free VRAM of all GPUs (the
    #     old behaviour) overcommits and crashes with ErrorOutOfDeviceMemory
    #     once the KV cache grows past what the primary alone can hold.
    #   • Single-GPU pinning via device-visibility env vars (section 4d).
    # Falls back gracefully to the summed value on single-GPU / CPU systems.
    _prio_map = gpu_priorities or {}

    def _gpu_priority(g: GPUInfo) -> int:
        """Priority for *g* with OS-robust name matching.

        Priorities are persisted keyed by the GPU name of the OS they were
        set on ("AMD Radeon AI PRO R9700" from Windows WMI), but the SAME
        card is called "Radeon AI PRO R9700" by Linux lspci/DRM and
        "AMD Radeon AI PRO R9700 (RADV NAVI48)" by Mesa. An exact dict.get
        therefore silently dropped every priority after switching OS — the
        Ubuntu "AUTO picks the 16 GB card" report. Match exact → case-
        insensitive → substring (either direction) → shared model-number
        token (e.g. "r9700", "9070"), so one settings file serves both
        boots. Falls back to 1 (neutral) when nothing matches.
        """
        hit = _prio_map.get(g.name)
        if hit is not None:
            return max(1, hit)
        g_lower = g.name.strip().lower()
        if not g_lower:
            return 1
        for key, prio in _prio_map.items():
            k_lower = key.strip().lower()
            if k_lower == g_lower or k_lower in g_lower or g_lower in k_lower:
                return max(1, prio)
        # Model-number token match: any alnum token containing a digit that
        # both names share ("r9700", "9070", "3060ti") identifies the card
        # across driver-string variants; generation tokens like "navi48"
        # appear on BOTH of Basti's cards and are therefore excluded by
        # requiring the token to exist in the *stored* key too (a settings
        # key never contains the Mesa "(RADV NAVI48)" suffix).
        g_tokens = {
            t for t in re.findall(r"[a-z0-9]+", g_lower) if any(c.isdigit() for c in t)
        }
        if g_tokens:
            for key, prio in _prio_map.items():
                k_tokens = {
                    t
                    for t in re.findall(r"[a-z0-9]+", key.lower())
                    if any(c.isdigit() for c in t)
                }
                if k_tokens and (g_tokens & k_tokens):
                    return max(1, prio)
        return 1

    def _gpu_score(g: GPUInfo) -> float:
        return _gpu_priority(g) * g.total_vram_mb

    # Resolve an explicit user pin (force_gpu) up front. Matched via
    # match_gpu_by_token — exact → bidirectional substring → shared
    # model-number token — so a short label ("R9700", "9070") AND a full
    # driver string from the other OS's name style both resolve to the
    # same physical card. None / unknown name → no forced pin (auto).
    forced_gpu: Optional[GPUInfo] = None
    if force_gpu and has_gpu and system.gpus:
        forced_gpu = match_gpu_by_token(force_gpu, system.gpus)

    primary_gpu: Optional[GPUInfo] = None
    primary_free_vram_gb = free_vram  # default = summed (single-GPU / CPU)
    if has_gpu and system.gpus:
        if forced_gpu is not None:
            # Hard pin: the user named the card this server must boot on.
            primary_gpu = forced_gpu
        elif len(system.gpus) > 1:
            # Auto primary = highest priority×VRAM score — BUT a card whose
            # *free* VRAM cannot even hold the model weights is no longer a
            # sane primary, regardless of its size/priority. This is the
            # second-server case: server #1 has filled the 32 GB R9700
            # (≈1 GB free), so scoring it highest and pinning there OOMs
            # while a 16 GB card sits with 9–13 GB free. We therefore pick
            # the best-scoring card *that can actually fit the weights in its
            # free VRAM*, and only fall back to the raw top score if none
            # qualifies (then the spread/CPU-spill logic downstream copes).
            ranked = sorted(system.gpus, key=_gpu_score, reverse=True)
            primary_gpu = ranked[0]
            # model_vram (the precise GPU weight footprint) is computed
            # further down, so here we use the on-disk file size as a
            # conservative proxy: for a given quant the GPU-resident weights
            # are ≈ the file size, so a card that can't fit the file in its
            # free VRAM certainly can't host the model as primary.
            need_gb = placement_model_size_gb
            if need_gb > 0:
                fit = next(
                    (g for g in ranked if (g.free_vram_mb / 1024.0) >= need_gb),
                    None,
                )
                if fit is not None:
                    primary_gpu = fit
        else:
            primary_gpu = max(system.gpus, key=_gpu_score)

        if len(system.gpus) > 1 and primary_gpu is not None:
            primary_free_vram_gb = max(0.0, primary_gpu.free_vram_mb / 1024.0)

    if forced_gpu is not None:
        # A hard pin hides every peer device, so every placement/KV decision
        # must use the selected card's capacity rather than aggregate VRAM.
        free_vram = primary_free_vram_gb

    # ---- (0.1) KV per-token: MUST be defined before any branch uses it.
    # This is the bug that caused crashes on selection of any non-Qwen
    # model in v3.x — base_kv_mb was previously only set inside the
    # rope-scaling branch, but referenced unconditionally further below.
    base_k_mb, base_v_mb = _resolve_kv_per_token_parts_mb(model, params_b)
    base_kv_mb = base_k_mb + base_v_mb

    # Context-independent recurrent state for Mamba/RWKV/hybrid layers.
    # Model-based speculative methods retain rollback snapshots (draft_max;
    # default 2), each of which duplicates state per parallel slot. b10441's
    # common_params_speculative::need_n_rs_seq explicitly excludes n-gram
    # methods, so draftless n-gram does not reserve snapshots here.
    has_state_snapshots = bool(draft_model is not None or model.has_embedded_mtp)
    resolved_draft_n_max = resolve_draft_n_max(profile, draft_model, force_draft_n_max)
    snapshot_count = resolved_draft_n_max if has_state_snapshots else 0
    recurrent_state_total_gb = recurrent_state_gb_from_metadata(
        model.metadata,
        n_parallel=n_parallel,
        snapshot_count=snapshot_count,
    )

    native_ctx = model.native_context  # GGUF metadata: model's native ctx

    # RoPE-Scaling Konfiguration aus Profil lesen
    profile_rope_scale = profile.rope_scale_enabled
    profile_rope_max = profile.rope_scale_max_ctx  # Standard: 1M
    profile_rope_factor = profile.rope_scale_factor  # Standard: 4.0

    rope_scaled_ctx = (
        0  # Wird später berechnet (braucht free_vram_after/free_ram_after)
    )
    rope_scaling_active = False  # Flag für build_command

    profile_max = profile.max_context
    if native_ctx > 0:
        profile_max = min(profile_max, native_ctx)
    target_ctx_for_placement = user_ctx if user_ctx is not None else profile_max

    # ---- DiffusionGemma runner special case ---------------------------
    # PR #24427's llama-diffusion-gemma-server parses the common llama.cpp
    # load flags but its model/context params do NOT apply cache_type_k/v,
    # tensor_buft_overrides (n_cpu_moe) or main_gpu/tensor_split the way
    # llama-server does. Treat it as a special runner:
    #   * force F16 KV (the fork ignores -ctk/-ctv anyway, so a denser
    #     quant is a lie — surface the honest F16 number and labels);
    #   * do NOT use the n_cpu_moe expert-only placement path even though
    #     the GGUF reports expert_count>1 (the fork's op-offload is
    #     layer-based like a dense model). We keep is_moe=True for display /
    #     architecture info but route placement + multi-GPU split through
    #     the dense -ngl path;
    #   * reserve DIFFUSION_GEMMA_RUNTIME_VRAM_OVERHEAD_GB of extra VRAM
    #     for the diffusion-runtime buffers the server allocates beyond
    #     weights + KV, so the placement / KV budget / footprint all harden
    #     against the Vulkan OOM that motivated this;
    #   * cap auto context at the profile's max_context (4096), because the
    #     server's huge F16 KV cache OOMs long before the generic 32k floor
    #     would ever bind.
    is_diffusion_gemma = (
        getattr(profile, "runner", "") == "llama-diffusion-gemma-server"
    )
    runtime_vram_overhead_gb = (
        DIFFUSION_GEMMA_RUNTIME_VRAM_OVERHEAD_GB if is_diffusion_gemma else 0.0
    )
    runtime_ram_overhead_gb = 0.0

    # ---- (0.5) Calculate VRAM reserved for Vision + Draft models
    # These MUST be on GPU for optimal performance — UNLESS the user asked
    # to keep the mmproj in system RAM (--no-mmproj-offload). In that case
    # the projector size is reported as ``vision_ram_gb`` (and subtracted
    # from the RAM budget below) while ``vision_vram_gb`` stays 0.
    vision_vram_gb = 0.0
    vision_ram_gb = 0.0
    draft_vram_gb = 0.0

    if model.mmproj is not None:
        # Vision model (mmproj) — estimate from file size
        try:
            mmproj_size_bytes = model.mmproj.stat().st_size
            mmproj_gb = mmproj_size_bytes / (1024**3)
        except (OSError, AttributeError):
            # Fallback: ~6 GB for typical F16 mmproj files
            mmproj_gb = 6.0
        if no_mmproj_offload:
            # Projector forced to host RAM (--no-mmproj-offload).
            vision_ram_gb = mmproj_gb
        else:
            vision_vram_gb = mmproj_gb

    if draft_model is not None:
        # Draft model — must fit in VRAM for speculative decoding to work well
        draft_vram_gb = draft_model.size_gb

    # ---- Host-memory prompt cache RAM reservation ---------------------
    # Resolve the configured prompt-cache size to a GiB figure the RAM
    # budget subtracts (and the preview/preflight/registry display). A
    # positive value is reserved exactly; 0 disables the cache (no RAM);
    # -1 (legacy "unlimited") is planned conservatively as a bounded
    # PROMPT_CACHE_UNLIMITED_RESERVE_GB so the budget never under-
    # provisions against an unbounded host cache.
    if prompt_cache_ram_mib > 0:
        prompt_cache_ram_gb = prompt_cache_ram_mib / 1024.0
    elif prompt_cache_ram_mib == -1:
        prompt_cache_ram_gb = PROMPT_CACHE_UNLIMITED_RESERVE_GB
    else:
        prompt_cache_ram_gb = 0.0

    # qwen4exp's sparse-QSA graph scales with context × ubatch and its giant
    # PLE row table is a fixed mmap host allocation. Plan against the user's
    # exact pin; Auto uses the performance tier's intended MoE context target
    # and receives a final physical RAM/VRAM clamp after KV precision is known.
    qwen4exp_plan_ctx = 0
    if is_qwen4exp:
        qwen4exp_plan_ctx = max(
            2048,
            min(
                int(target_ctx_for_placement),
                int(
                    target_ctx_for_placement
                    if user_ctx is not None
                    else perf_target.moe_placement_ctx_target
                ),
            ),
        )
        qwen_gpu_plan_gb, qwen_host_plan_gb = qwen4exp_compute_buffers_gb(
            qwen4exp_plan_ctx, qwen4exp_ubatch, n_parallel
        )
        runtime_vram_overhead_gb += qwen_gpu_plan_gb
        runtime_ram_overhead_gb = _QWEN4EXP_FIXED_HOST_RUNTIME_GB + qwen_host_plan_gb

    # Host RAM genuinely available for ordinary CPU-offloaded weights. The
    # PLE mapping and qwen4exp graph are not interchangeable with expert
    # offload and therefore come out before placement.
    effective_free_ram_for_weights = max(
        0.0,
        system.free_ram_gb
        - ram_safety_gb
        - mapped_model_resident_gb
        - runtime_ram_overhead_gb
        - vision_ram_gb
        - prompt_cache_ram_gb,
    )

    # Effective VRAM available for main model placement. Vision/draft that
    # live on the GPU AND the runner's runtime overhead are subtracted up
    # front so placement + KV sizing see the real headroom.
    shared_host_reserve_gb = (
        vision_ram_gb
        + prompt_cache_ram_gb
        + mapped_model_resident_gb
        + runtime_ram_overhead_gb
        if unified_memory
        else 0.0
    )
    effective_free_vram = (
        free_vram
        - vision_vram_gb
        - draft_vram_gb
        - runtime_vram_overhead_gb
        - shared_host_reserve_gb
    )
    if effective_free_vram < 0:
        effective_free_vram = 0.0

    # Same, but scoped to the PRIMARY GPU only — MoE expert placement must
    # use this (experts spill to CPU, never to the secondary GPU). On
    # single-GPU systems this equals effective_free_vram.
    effective_primary_free_vram = (
        primary_free_vram_gb
        - vision_vram_gb
        - draft_vram_gb
        - runtime_vram_overhead_gb
        - shared_host_reserve_gb
    )
    if effective_primary_free_vram < 0:
        effective_primary_free_vram = 0.0

    # Use a multi-GPU pool only when at least one peer has a meaningful usable
    # contribution. A nearly-full peer (<2 GiB usable) is intentionally ignored
    # by _gpu_usable_cap_gb; treating the raw device count as a pool would place
    # weights against aggregate VRAM and later pin that oversized placement to
    # the sole viable card. When no peer remains, recompute every placement
    # budget from the primary's safe cap so excess dense layers/experts spill to
    # CPU before context and device visibility are decided.
    multi_gpu_candidate = (
        has_gpu
        and len(system.gpus) > 1
        and forced_gpu is None
        and primary_gpu is not None
    )
    gpu_usable_caps: Dict[int, float] = {}
    if multi_gpu_candidate:
        gpu_usable_caps = {
            id(g): _gpu_usable_cap_gb(g, g is primary_gpu) for g in system.gpus
        }
    has_multiple_gpus = bool(
        multi_gpu_candidate
        and any(
            g is not primary_gpu and gpu_usable_caps.get(id(g), 0.0) > 0
            for g in system.gpus
        )
    )
    combined_usable_vram_gb = 0.0
    if has_multiple_gpus:
        # Combined USABLE VRAM across all active GPUs for placement — the sum
        # of the exact per-card caps section 4d enforces, never raw free VRAM.
        combined_usable_vram_gb = sum(gpu_usable_caps.values())
        effective_moe_vram = (
            combined_usable_vram_gb
            - vision_vram_gb
            - draft_vram_gb
            - runtime_vram_overhead_gb
        )
    else:
        if multi_gpu_candidate and primary_gpu is not None:
            primary_pool_cap = gpu_usable_caps.get(
                id(primary_gpu), _gpu_usable_cap_gb(primary_gpu, True)
            )
            free_vram = primary_pool_cap
            effective_free_vram = max(
                0.0,
                primary_pool_cap
                - vision_vram_gb
                - draft_vram_gb
                - runtime_vram_overhead_gb
                - shared_host_reserve_gb,
            )
            effective_primary_free_vram = effective_free_vram
        effective_moe_vram = effective_primary_free_vram
    if effective_moe_vram < 0:
        effective_moe_vram = 0.0

    # ---- (1) Model placement
    # DiffusionGemma is architecturally a MoE (expert_count>1) but its
    # dedicated server (PR #24427) ignores n_cpu_moe and uses layer-based
    # -ngl offload like a dense model, so we skip the expert-only
    # placement branch entirely. is_moe stays True for display / split
    # decisions are handled below via the disable_moe_placement flag.
    disable_moe_placement = is_diffusion_gemma
    state_vram_reserve_for_placement = (
        0.0 if perf_target.kv_to_ram else recurrent_state_total_gb
    )
    dense_placement_safety_gb = (
        max(vram_safety_gb, ram_safety_gb) if unified_memory else vram_safety_gb
    )
    moe_placement_safety_gb = (
        max(perf_target.moe_vram_safety_gb, ram_safety_gb)
        if unified_memory
        else perf_target.moe_vram_safety_gb
    )
    n_cpu_moe: Optional[int] = None
    if is_moe and has_gpu and n_layers > 0 and not disable_moe_placement:
        ngl, n_cpu_moe, model_vram, model_ram, full_off = _decide_moe_offload(
            model_size_gb=placement_model_size_gb,
            free_vram_gb=effective_moe_vram,
            free_ram_gb=effective_free_ram_for_weights,
            n_layers=n_layers,
            expert_count=expert_count,
            params_billion=params_b,
            target_ctx=target_ctx_for_placement,
            base_kv_per_token_mb=base_kv_mb,
            ram_safety_gb=ram_safety_gb,
            moe_vram_safety_gb=moe_placement_safety_gb,
            moe_placement_ctx_target=perf_target.moe_placement_ctx_target,
            batch_vram_reserve_gb=(
                perf_target.moe_batch_vram_reserve_gb + state_vram_reserve_for_placement
            ),
            n_parallel=n_parallel,
            rope_scaling=bool(
                force_rope_scale is True
                or (
                    profile_rope_scale
                    and native_ctx > 0
                    and target_ctx_for_placement > native_ctx
                )
            ),
        )

        # ---- Two-pass placement fallback ---------------------------------
        # If the first pass dumped *every* expert layer to CPU but >4 GB
        # of VRAM is still free, the KV reservation was clearly too
        # pessimistic for this model. Retry once with the placement
        # target halved (down to a 16k floor). This is a defensive net
        # for hybrid architectures we don't recognise yet, or for
        # quantisations where our heuristic mis-estimates KV footprint.
        if (
            n_cpu_moe is not None
            and n_layers > 0
            and n_cpu_moe >= n_layers
            and effective_moe_vram > 4.0
            and perf_target.moe_placement_ctx_target > 16384
        ):
            shrunk_target = max(16384, perf_target.moe_placement_ctx_target // 2)
            ngl_2, cpu_moe_2, vram_2, ram_2, full_2 = _decide_moe_offload(
                model_size_gb=placement_model_size_gb,
                free_vram_gb=effective_moe_vram,
                free_ram_gb=effective_free_ram_for_weights,
                n_layers=n_layers,
                expert_count=expert_count,
                params_billion=params_b,
                target_ctx=target_ctx_for_placement,
                base_kv_per_token_mb=base_kv_mb,
                ram_safety_gb=ram_safety_gb,
                moe_vram_safety_gb=moe_placement_safety_gb,
                moe_placement_ctx_target=shrunk_target,
                batch_vram_reserve_gb=(
                    perf_target.moe_batch_vram_reserve_gb
                    + state_vram_reserve_for_placement
                ),
                n_parallel=n_parallel,
                rope_scaling=bool(
                    force_rope_scale is True
                    or (
                        profile_rope_scale
                        and native_ctx > 0
                        and target_ctx_for_placement > native_ctx
                    )
                ),
            )
            # Only adopt the second pass if it actually placed layers on GPU.
            if cpu_moe_2 is not None and cpu_moe_2 < n_cpu_moe:
                ngl, n_cpu_moe, model_vram, model_ram, full_off = (
                    ngl_2,
                    cpu_moe_2,
                    vram_2,
                    ram_2,
                    full_2,
                )

        if ngl <= 0:
            # Fully CPU-resident fallback: --n-cpu-moe is redundant and would
            # incorrectly select the large MoE GPU-hybrid batch regime.
            n_cpu_moe = None
        elif n_cpu_moe == 0:
            n_cpu_moe = None
    else:
        # Reserve VRAM for the KV cache before placing dense weight layers,
        # sized by the tier's dense_kv_reserve_ctx (0 for low_vram, whose KV
        # goes to RAM instead). Use the same Q4_0 pair Auto will emit so layer
        # placement and final context are planned against one cache contract.
        # Capped at the model's native context so we never
        # reserve for tokens the model can't address.
        #
        # Reserve only when weights + the tier's desired KV target do not fit
        # together. A roomy GPU remains full-offload; a near-full GPU spills a
        # few weight layers to CPU so the VRAM-resident KV cache is real rather
        # than being (incorrectly) supplemented with host RAM.
        desired_dense_kv_reserve_gb = state_vram_reserve_for_placement
        if not perf_target.kv_to_ram and perf_target.dense_kv_reserve_ctx > 0:
            reserve_ctx = perf_target.dense_kv_reserve_ctx
            if native_ctx > 0:
                reserve_ctx = min(reserve_ctx, native_ctx)
            desired_dense_kv_reserve_gb += (
                reserve_ctx * base_kv_mb * kv_quant_factor(DEFAULT_KV_CACHE_TYPE)
            ) / 1024.0
        model_weights_fit_vram = (
            placement_model_size_gb
            + FULL_OFF_HEADROOM_GB
            + state_vram_reserve_for_placement
            <= effective_free_vram - dense_placement_safety_gb
        )
        model_and_kv_fit_vram = (
            placement_model_size_gb + FULL_OFF_HEADROOM_GB + desired_dense_kv_reserve_gb
            <= effective_free_vram - dense_placement_safety_gb
        )
        # Preserve established multi-GPU full-offload behavior when the model
        # weights fit across the pool. Context is then limited by the genuine
        # post-weight VRAM remainder (never by a fictitious RAM supplement).
        if has_multiple_gpus and model_weights_fit_vram:
            model_and_kv_fit_vram = True
        dense_kv_reserve_gb = (
            0.0 if model_and_kv_fit_vram else desired_dense_kv_reserve_gb
        )
        ngl, model_vram, model_ram, full_off = _decide_offload(
            model_size_gb=placement_model_size_gb,
            free_vram_gb=effective_free_vram,
            n_layers=n_layers,
            has_gpu=has_gpu,
            vram_headroom_gb=dense_placement_safety_gb,
            kv_reserve_gb=dense_kv_reserve_gb,
            free_ram_gb=effective_free_ram_for_weights,
        )

    # ---- (1.5) Expert overrides: force_ngl / force_n_cpu_moe -----------
    # Applied AFTER the automatic placement so model_vram / model_ram
    # estimates reflect the user's pinned values. The user owns the
    # consequences (over/undercommit); we only redistribute the model
    # size estimate to match the new layer split.
    if (
        force_n_cpu_moe is not None
        and is_moe
        and has_gpu
        and n_layers > 0
        and not disable_moe_placement
    ):
        new_cpu_moe = max(0, min(n_layers, int(force_n_cpu_moe)))
        # Re-derive model_vram/ram from the new split, holding shared
        # overhead constant (it scales with model size, not layer
        # placement).
        shared_overhead_gb = placement_model_size_gb * 0.08
        per_layer_expert_gb = max(
            0.001, (placement_model_size_gb - shared_overhead_gb) / n_layers
        )
        layers_on_gpu = n_layers - new_cpu_moe
        model_vram = shared_overhead_gb + layers_on_gpu * per_layer_expert_gb
        model_ram = new_cpu_moe * per_layer_expert_gb
        n_cpu_moe = new_cpu_moe if new_cpu_moe > 0 else None
        full_off = new_cpu_moe == 0
        ngl = 999

    if (
        force_ngl is not None
        and n_layers > 0
        and not (is_moe and has_gpu and not disable_moe_placement)
    ):
        new_ngl = max(0, min(n_layers, int(force_ngl)))
        per_layer_gb = placement_model_size_gb / n_layers
        ngl = new_ngl if new_ngl < n_layers else 999
        if new_ngl >= n_layers:
            model_vram = placement_model_size_gb
            model_ram = 0.0
            full_off = True
        else:
            model_vram = new_ngl * per_layer_gb
            residual_overhead = placement_model_size_gb * 0.02
            model_ram = (n_layers - new_ngl) * per_layer_gb + residual_overhead
            full_off = False

    # Fixed recurrent state follows K/Q/V offload placement but does not grow
    # with context. Account for it separately from attention KV.
    if perf_target.kv_to_ram or not has_gpu or ngl <= 0:
        recurrent_state_vram_gb = 0.0
        recurrent_state_ram_gb = recurrent_state_total_gb
    elif full_off or ngl == 999 or n_layers <= 0:
        recurrent_state_vram_gb = recurrent_state_total_gb
        recurrent_state_ram_gb = 0.0
    else:
        recurrent_gpu_fraction = min(1.0, max(0.0, ngl / n_layers))
        recurrent_state_vram_gb = recurrent_state_total_gb * recurrent_gpu_fraction
        recurrent_state_ram_gb = recurrent_state_total_gb - recurrent_state_vram_gb

    # ---- (2) Remaining KV budget — include vision/draft/state in total
    effective_vram_safety = (
        perf_target.moe_vram_safety_gb if n_cpu_moe is not None else vram_safety_gb
    )
    if unified_memory:
        effective_vram_safety = max(effective_vram_safety, ram_safety_gb)
    moe_batch_vram_reserve_gb = (
        perf_target.moe_batch_vram_reserve_gb
        if n_cpu_moe is not None and n_cpu_moe > 0 and ngl > 0
        else 0.0
    )

    # Prefer one GPU when it can deliver the requested/model-maximum context
    # with the global Q4_0 cache default. Before v5.3.2 this gate required F16,
    # so Auto spread onto an otherwise idle peer solely to upgrade KV precision.
    # Q4 is now the explicit capacity-first contract: preserve context first,
    # then keep peer GPUs free whenever the Q4 cache and fixed footprint fit.
    # A hard force_gpu pin remains authoritative even when it sacrifices context.
    planned_single_gpu = forced_gpu is not None
    gpu_budget_free_vram = free_vram
    fixed_gpu_footprint = (
        model_vram
        + vision_vram_gb
        + draft_vram_gb
        + runtime_vram_overhead_gb
        + moe_batch_vram_reserve_gb
        + recurrent_state_vram_gb
    )
    if not planned_single_gpu and has_multiple_gpus and primary_gpu is not None:
        # A spread must respect the same per-card caps used by tensor placement;
        # raw summed free VRAM includes headroom that cannot safely host KV.
        gpu_budget_free_vram = combined_usable_vram_gb
        primary_cap = _gpu_usable_cap_gb(primary_gpu, True)
        if fixed_gpu_footprint <= primary_cap:
            quality_target_ctx = max(2048, int(target_ctx_for_placement))
            quality_rope_scaling = bool(
                force_rope_scale is True
                or profile_rope_scale
                or (
                    user_ctx is not None
                    and native_ctx > 0
                    and user_ctx > native_ctx
                    and (model.supports_rope_scale or profile_rope_scale)
                )
            )
            primary_raw_kv_budget = max(
                0.0,
                primary_cap - effective_vram_safety - fixed_gpu_footprint,
            )
            primary_usable_kv_budget = _usable_kv_budget_after_headroom(
                primary_raw_kv_budget,
                quality_target_ctx,
                n_parallel,
                quality_rope_scaling,
            )
            default_kv_gb = (
                quality_target_ctx
                * base_kv_mb
                * kv_quant_factor(DEFAULT_KV_CACHE_TYPE)
                * n_parallel
            ) / 1024.0
            if default_kv_gb <= primary_usable_kv_budget * 0.98:
                planned_single_gpu = True
                gpu_budget_free_vram = primary_cap

    free_vram_after = max(
        0.0,
        gpu_budget_free_vram
        - effective_vram_safety
        - model_vram
        - vision_vram_gb
        - draft_vram_gb
        - runtime_vram_overhead_gb
        - moe_batch_vram_reserve_gb
        - recurrent_state_vram_gb,
    )
    # The RAM budget must also account for the mmproj when it is forced
    # into host RAM (--no-mmproj-offload → vision_ram_gb) and for the
    # host-memory prompt cache (prompt_cache_ram_gb). Without these the
    # KV-to-RAM budget (low_vram) silently over-commits system RAM.
    free_ram_after = max(
        0.0,
        system.free_ram_gb
        - ram_safety_gb
        - model_ram
        - mapped_model_resident_gb
        - vision_ram_gb
        - prompt_cache_ram_gb
        - runtime_ram_overhead_gb
        - recurrent_state_ram_gb,
    )

    if unified_memory:
        # CPU and accelerator allocations consume one physical pool. Count all
        # resident components once and expose the same remainder to either
        # compute placement; never add RAM and VRAM capacities together.
        shared_available = min(free_vram, max(0.0, system.free_ram_gb))
        shared_remaining = max(
            0.0,
            shared_available
            - max(effective_vram_safety, ram_safety_gb)
            - model_vram
            - model_ram
            - mapped_model_resident_gb
            - vision_vram_gb
            - vision_ram_gb
            - draft_vram_gb
            - runtime_vram_overhead_gb
            - runtime_ram_overhead_gb
            - prompt_cache_ram_gb
            - moe_batch_vram_reserve_gb
            - recurrent_state_vram_gb
            - recurrent_state_ram_gb,
        )
        free_vram_after = shared_remaining
        free_ram_after = shared_remaining

    # KV-cache placement rules:
    #   - kv_to_ram (low_vram): the ENTIRE KV cache lives in system RAM via
    #     --no-kv-offload, regardless of model type. Attention compute
    #     follows the KV onto the CPU. This is the LOW-VRAM escape hatch —
    #     it lets an 8 GB-VRAM / 64 GB-RAM box reach 90k+ context on a
    #     20 GB MoE whose KV would never fit in the leftover VRAM. The
    #     experts are already mostly on CPU (--n-cpu-moe), so the marginal
    #     cost is slower attention, paid back with huge context headroom.
    #   - MoE on GPU: KV must live in VRAM only. The Vulkan backend
    #     crashes with GGML_ASSERT(addr) when MoE KV spills to RAM.
    #   - Dense full-offload: KV in VRAM only (it's already on GPU).
    #   - Dense partial: KV split MIRRORS layer split. The VRAM portion
    #     limits the total budget; RAM portion is intentionally capped
    #     so we never bleed multi-GB KV cache into slow main memory.
    #     This was the root cause of the gemma-31B-Q3-with-draft bug:
    #     the old code added free_ram_after wholesale and produced an
    #     11 GB KV cache living in RAM, dragging inference to a crawl.
    #   - CPU-only: KV lives entirely in RAM.
    #
    # The dense-hybrid RAM-KV cap is now tier-scaled (perf_target.
    # dense_kv_ram_cap_gb) and clamped to free RAM, replacing the old flat
    # 4 GB that ignored high-RAM systems. See the dense-hybrid branch below.

    no_kv_offload = False
    if is_diffusion_gemma and has_gpu:
        # The dedicated PR #24427 server keeps its F16 KV on the GPU and does
        # not copy common_params.no_kv_offload into its context params. This
        # runner-specific contract wins even over the generic low_vram tier:
        # budgeting against host RAM would approve a context the process still
        # allocates in VRAM.
        kv_budget_gb = free_vram_after
    elif perf_target.kv_to_ram and has_gpu:
        # LOW-VRAM mode: KV cache → system RAM via --no-kv-offload.
        # Budget is the RAM headroom (capped by the tier), NOT the VRAM
        # remainder — with KV offload disabled, no KV lands in VRAM.
        ram_kv = min(free_ram_after, perf_target.kv_ram_cap_gb)
        kv_budget_gb = ram_kv
        no_kv_offload = True
    elif unified_memory and has_gpu:
        # Placement changes which processor owns the cache, not physical
        # capacity. Both views point at the same deduplicated pool.
        kv_budget_gb = min(free_vram_after, free_ram_after)
    elif is_moe and has_gpu and ngl > 0 and not disable_moe_placement:
        if has_multiple_gpus and not planned_single_gpu:
            # MoE KV lives in VRAM and follows the layer split, so it must
            # fit inside the SAME per-card caps the spread enforces. Using
            # the raw free-VRAM remainder here (free_vram_after) hands the
            # KV sizing the per-card headroom the spread deliberately
            # reserves — the resulting context then physically cannot be
            # placed without pushing a card past its cap.
            kv_budget_gb = max(
                0.0,
                combined_usable_vram_gb
                - effective_vram_safety
                - model_vram
                - vision_vram_gb
                - draft_vram_gb
                - runtime_vram_overhead_gb
                - moe_batch_vram_reserve_gb
                - recurrent_state_vram_gb,
            )
        else:
            kv_budget_gb = free_vram_after
    elif full_off:
        # With KV offload enabled, every fully-offloaded layer allocates its
        # KV buffer on the GPU device. llama.cpp does not transparently spill
        # this cache into host RAM; only --no-kv-offload changes that. Budget
        # strictly from post-weight VRAM so the promised context can allocate.
        kv_budget_gb = free_vram_after
    elif ngl > 0 and n_layers > 0:
        # Dense hybrid: llama.cpp splits the KV cache BY LAYER — the KV of
        # GPU-resident layers lives in VRAM, the KV of CPU-resident layers in
        # RAM (no --no-kv-offload here). So the total context is bounded by
        # whichever SIDE fills first, not by a single additive pool:
        #     gpu_frac * kv_total <= free_vram_after   (VRAM side)
        #     cpu_frac * kv_total <= ram_cap           (RAM side)
        # => kv_total <= min(free_vram_after / gpu_frac, ram_cap / cpu_frac)
        #
        # This physically-correct per-side binding is only usable because
        # _decide_offload now RESERVES VRAM for KV (dense_kv_reserve_gb), so
        # free_vram_after is no longer ~0 and the VRAM term no longer collapses
        # to zero the way it did before the reservation existed (which is why
        # the old code fell back to an additive pool). The RAM cap is
        # tier-scaled and clamped to what is actually free — high-RAM systems
        # get a much larger CPU-side budget than the old flat 4 GB allowed.
        gpu_layer_fraction = ngl / n_layers
        ram_cap = min(free_ram_after, perf_target.dense_kv_ram_cap_gb)
        if gpu_layer_fraction >= 0.99:
            # Effectively full offload — treat KV as VRAM-only.
            kv_budget_gb = free_vram_after
        elif gpu_layer_fraction <= 0.0:
            # No GPU layers — should not happen (ngl > 0) but stay defensive.
            kv_budget_gb = ram_cap
        else:
            cpu_layer_fraction = 1.0 - gpu_layer_fraction
            vram_side = free_vram_after / gpu_layer_fraction
            ram_side = ram_cap / cpu_layer_fraction
            kv_budget_gb = max(0.0, min(vram_side, ram_side))
    else:
        # CPU-only — KV lives entirely in RAM.
        kv_budget_gb = free_ram_after

    # ---- (2.5) RoPE-Scaling (YaRN) auto-detection
    # Aktiviere RoPE-Scaling automatisch wenn:
    # 1. Modell RoPE-Scaling unterstützt (qwen2 etc.) ODER das YAML-Profil
    #    rope_scale.enabled=true setzt (erlaubt Profil-Autoren, RoPE-Scaling
    #    für Architekturen zu aktivieren die nicht in _ROPE_SCALE_SUPPORTED_ARCHS
    #    stehen — z.B. phi3/Phi-4 mit nativem 16k-Kontext aber 128k-Kapazität)
    # 2. Genügend Speicher für Context > native_ctx vorhanden ist
    # 3. Entweder profil-configured (rope_scale.enabled=true) ODER
    #    berechneter max_fit_ctx überschreitet native_ctx
    rope_scaled_ctx = 0
    rope_scaling_active = False

    if (
        (model.supports_rope_scale or profile_rope_scale)
        and native_ctx > 0
        and native_ctx < profile_rope_max
    ):
        # KV-Speicherbedarf pro Token (globales Q4_0-Auto-Default)
        kv_per_tok_q4 = base_kv_mb * kv_quant_factor(DEFAULT_KV_CACHE_TYPE)

        # ---- RoPE-Ziel-Context ----------------------------------------
        # Wie weit wir via YaRN ausdehnen wollen. Das alte Gate prüfte
        # `desired_ctx > native_ctx` mit desired_ctx = profile.max_context
        # — da jedes Profil max_context sinnvollerweise auf den NATIVEN
        # Context setzt (z.B. qwen3_5-3_6.yaml: 262144 = nativ), war das
        # strukturell nie erfüllt: RoPE-Scaling aktivierte in KEINEM Modus,
        # nicht einmal bei rope_scale.enabled=true (der Schalter war Dead
        # Code). YaRN wurde ausschließlich über den GUI-Expert-Schalter
        # (force_rope_scale) emittiert.
        #
        # Opt-in-Reihenfolge (erster Treffer gewinnt):
        #   1. Profil rope_scale.enabled=true  → profile_rope_max (z.B. 1M)
        #   2. User pinnt user_ctx > native   → genau dieser Wert
        #   3. Sonst (Auto, enabled=false)     → keine Ausdehnung
        # Fall 3 respektiert bewusst die Profil-Default-Entscheidung
        # („Standard aus"): YaRN über native hinaus hat einen
        # Qualitätspreis und darf nicht still für jedes Qwen-Modell
        # aktiviert werden, nur weil RAM vorhanden ist.
        if profile_rope_scale:
            rope_target_ctx = user_ctx if user_ctx is not None else profile_rope_max
        elif user_ctx is not None and user_ctx > native_ctx:
            rope_target_ctx = user_ctx
        else:
            rope_target_ctx = 0

        if rope_target_ctx > native_ctx:
            # Per-Slot-Budget: llama-server legt N Slots an, jeder braucht
            # einen vollen KV-Buffer der Zielgröße. Das GESAMTE Budget durch
            # n_parallel zu teilen verhindert Überprovisionierung.
            kv_budget_per_slot_gb = kv_budget_gb / n_parallel

            # Context, den das Budget (Q4_0-Basis) pro Slot tatsächlich hält.
            # kv_budget_gb wird in Schritt (2.6) um den Compute-Buffer-Reserve
            # verkleinert, daher fällt der finale ctx (max_fit_ctx) etwas
            # kleiner aus — model_ctx_limit fängt das über min() ab.
            budget_ctx = (
                int((kv_budget_per_slot_gb * 1024) / kv_per_tok_q4)
                if kv_per_tok_q4 > 0
                else 0
            )

            # PARTIELLE Aktivierung: schon einschalten, sobald das Budget
            # MEHR als native_ctx zulässt — nicht erst, wenn die volle
            # pinned-Zielgröße × 1.1 reinpasst. Das alte All-or-Nothing-
            # Verhalten clamppte einen user_ctx > native still auf native
            # zurück, OHNE YaRN zu emittieren (der User bekam weniger
            # Context als bestellt und wusste nicht warum).
            if budget_ctx > native_ctx:
                rope_scaled_ctx = min(budget_ctx, rope_target_ctx, profile_rope_max)
                rope_scaling_active = True

    # Expert override: force_rope_scale = True turns it on unconditionally;
    # force_rope_scale = False turns it off. Either choice respects native_ctx
    # as a hard upper bound.
    if force_rope_scale is True:
        rope_scaled_ctx = min(
            (user_ctx if user_ctx is not None else profile_rope_max),
            profile_rope_max,
        )
        rope_scaling_active = True
    elif force_rope_scale is False:
        rope_scaled_ctx = 0
        rope_scaling_active = False

    # ---- (2.6) Compute-buffer reserve (long-context / YaRN OOM guard)
    # The KV budget accounts for the KV cache itself but NOT for the GPU
    # compute buffer / flash-attention scratch that llama.cpp's Vulkan/ROCm
    # backend allocates on top. That scratch grows with context length and
    # YaRN amplifies it; at ~1 M tokens it reaches several GB. With only the
    # thin safety band (0.15–0.30 GB) + a 0.5 % rounding factor reserved, the
    # card was driven to ~99.5 % utilisation and vkAllocateMemory failed
    # ("Vulkan buffer error"), crashing the server on the retry. We withhold
    # a context-scaled headroom so no mode can reach that brink.
    _reserve_target_ctx = (
        user_ctx
        if user_ctx is not None
        else (rope_scaled_ctx if rope_scaling_active else profile_max)
    )
    kv_budget_gb = _usable_kv_budget_after_headroom(
        kv_budget_gb,
        _reserve_target_ctx,
        n_parallel,
        rope_scaling_active,
    )

    # ---- (3) Context + KV quant

    # Bestimme das effektive Modell-Maximum für die KV-Quantisierung:
    # - rope_scaled_ctx: erweiterbares Maximum via YaRN (wenn aktiviert)
    # - native_ctx: natives Maximum des Modells (aus GGUF)
    model_ctx_limit = rope_scaled_ctx if rope_scaled_ctx > 0 else native_ctx
    if model_ctx_limit <= 0:
        model_ctx_limit = profile_max
    # DiffusionGemma's dedicated server has a huge F16 KV cache that OOMs
    # long before the generic 32k auto floor would ever bind, and its
    # canvas-based generation (256-token blocks) needs little context. Cap
    # the model limit at the profile's max_context (4096) so the auto path
    # honours it instead of being overridden by native_ctx (262144).
    if is_diffusion_gemma and user_ctx is None:
        model_ctx_limit = min(model_ctx_limit, profile_max)

    # Pick precision against the context Auto will actually attempt. The old
    # target used profile_max even when the final auto branch aimed at a larger
    # native/YaRN limit. Once F16 upgrades became possible this mismatch chose
    # F16 for an easy 8k profile target, then discovered it could only hold
    # 152k of a 262k+ model — sacrificing context despite denser KV fitting it.
    target_ctx = user_ctx if user_ctx is not None else model_ctx_limit

    # Expert overrides for KV-quant: when both K and V are pinned we
    # respect the user's pair as-is; when only one is pinned we still
    # let _pick_kv_quant decide the other within budget.
    #
    # NVIDIA CUDA builds default GGML_CUDA_FA_ALL_QUANTS=OFF. At b9888 the
    # CUDA FlashAttention selector correctly validates BOTH K and V cache
    # types, but without FA_ALL_QUANTS it still requires K == V. Because the
    # AutoTuner deliberately emits `-fa on`, automatic K/V asymmetry would be
    # risky on NVIDIA CUDA systems (high- and low-VRAM alike): it can disable
    # the FA kernel or abort depending on the model/backend. Keep auto KV
    # symmetric on NVIDIA; AMD's common AutoTuner builds are Vulkan/ROCm and
    # keep the asymmetric headroom win. Manual Expert pins are left untouched.
    primary_vendor = (
        primary_gpu.vendor if primary_gpu else system.primary_vendor
    ).lower()
    auto_asymmetric_kv = primary_vendor != "nvidia"

    kv_quant_strategy = "symmetric"
    if is_diffusion_gemma:
        # PR #24427's llama-diffusion-gemma-server ignores -ctk/-ctv (its
        # model/context params do not apply cache_type_k/v), so any denser
        # quant the AutoTuner would pick is a lie. Surface the honest F16
        # estimate (base_kv_mb is already metadata-F16) and F16 labels so
        # the displayed KV footprint matches what the server actually
        # allocates, hardening the placement against the Vulkan OOM.
        cache_k, cache_v = "f16", "f16"
        kv_quant_strategy = "symmetric"
    elif force_cache_k is not None and force_cache_v is not None:
        cache_k, cache_v = force_cache_k, force_cache_v
        if turbo_kv:
            cache_k = _turbo_quant_for(cache_k)
            cache_v = _turbo_quant_for(cache_v)
            kv_quant_strategy = "manual+turbo"
        else:
            kv_quant_strategy = "manual"
    else:
        cache_k, cache_v = _pick_kv_quant(
            profile.recommended_kv_quant,
            target_ctx,
            base_kv_mb,
            kv_budget_gb,
            model_ctx_limit,
            turbo=turbo_kv,
            asymmetric=auto_asymmetric_kv,
            base_k_per_token_mb=base_k_mb,
            base_v_per_token_mb=base_v_mb,
        )
        if force_cache_k is not None:
            cache_k = _turbo_quant_for(force_cache_k) if turbo_kv else force_cache_k
        if force_cache_v is not None:
            cache_v = _turbo_quant_for(force_cache_v) if turbo_kv else force_cache_v
        if cache_k != cache_v:
            kv_quant_strategy = "asymmetric"
        if turbo_kv:
            kv_quant_strategy = (
                f"{kv_quant_strategy}+turbo"
                if kv_quant_strategy != "symmetric"
                else "turbo"
            )

    actual_per_tok_mb = base_k_mb * kv_quant_factor(
        cache_k
    ) + base_v_mb * kv_quant_factor(cache_v)

    # Memory-safe ceiling — computed ONCE and used by both the auto and
    # the user-pin paths. Dividiere durch n_parallel, da llama-server N
    # Slots anlegt (jeder Slot braucht einen vollen KV-Buffer der
    # angeforderten Größe). Ohne diese Division würde llama-server "auto"
    # n_parallel auf z.B. 4 setzen und 4× das Budget belegen.
    #
    # Beispiel: 21 GB KV-Budget, n_parallel=1 →
    #   max_fit_ctx bei Q8 (0.060 MB/tok) = 356k → cap auf 262k ✓
    #   RAM-Nutzung ~3 GB statt ~60 GB (bei n_parallel=4 ohne diesen Fix).
    max_fit_ctx: int = 0  # also surfaced to the floor guard below
    pin_clamped_to_budget: Optional[int] = None
    kv_budget_per_slot_gb = kv_budget_gb / n_parallel
    if actual_per_tok_mb > 0:
        max_fit_ctx = int((kv_budget_per_slot_gb * 1024 * 0.995) / actual_per_tok_mb)
    else:
        max_fit_ctx = profile_max

    if user_ctx is not None:
        # User-specified context — honour it, but apply TWO clamps so no
        # mode can drive the card into OOM:
        #   (1) model cap  — never exceed native / rope-scaled ctx
        #   (2) budget cap — never exceed what the (reserved) KV budget
        #       can actually hold. Previously a 1 M YaRN pin sailed past
        #       the budget cap and crashed the Vulkan compute buffer.
        #       Now the tuner always delivers the maximum that works.
        ctx = user_ctx
        if model_ctx_limit > 0 and ctx > model_ctx_limit:
            ctx = model_ctx_limit
        if max_fit_ctx > 0 and ctx > max_fit_ctx:
            pin_clamped_to_budget = ctx
            ctx = max_fit_ctx
    else:
        # Auto: beschränke auf das Modell-Maximum (native oder rope-scaled)
        if model_ctx_limit > 0:
            ctx = min(max_fit_ctx, model_ctx_limit)
        else:
            ctx = min(max_fit_ctx, profile_max * 3)

    # Minimum context floor — AUTO MODE ONLY.
    # When the user explicitly sets a context (user_ctx is not None) we
    # respect that value as-is (already clamped to model limits above).
    # The 32k floor is a quality-of-life default for the auto calculation
    # so that system-prompts + tool scaffolding (e.g. zoo-code starts at
    # ~10-12k) leave meaningful room for the actual conversation.
    # Two guards prevent over-promising in auto mode:
    #   (a) model cap  — if the model's native context is below 32k, use that
    #   (b) VRAM cap   — never exceed what the KV budget can actually fit
    if user_ctx is None:
        _PREF_MIN_CTX = 32768
        effective_min = _PREF_MIN_CTX
        if model_ctx_limit > 0 and model_ctx_limit < effective_min:
            effective_min = (model_ctx_limit // 1024) * 1024  # model too small for 32k
        if max_fit_ctx > 0 and max_fit_ctx < effective_min:
            effective_min = max(2048, (max_fit_ctx // 1024) * 1024)  # budget too tight
        ctx = max(effective_min, (ctx // 1024) * 1024)
    else:
        # Explicit user pin: honour it EXACTLY (only the 2048 absolute
        # floor applies). Previously this rounded down to a 1024 boundary,
        # which silently shrank a pin like 99840 → 99328 — contradicting
        # the "user_ctx wins" contract. llama-server accepts arbitrary
        # -c values, so no quantisation is needed here.
        ctx = max(2048, ctx)

    # qwen4exp's QSA graph is not represented by ordinary KV bytes. Apply a
    # final physical two-pool ceiling now that cache precision and placement
    # are known. Coefficients are the conservative b10666 measurements above;
    # this is what prevents a 90k/ubatch-1024 graph from silently reserving
    # ~18 GiB host + ~5.6 GiB device compute buffers.
    if is_qwen4exp:
        base_runtime_vram_gb = (
            DIFFUSION_GEMMA_RUNTIME_VRAM_OVERHEAD_GB if is_diffusion_gemma else 0.0
        )
        kv_per_ctx_gb = actual_per_tok_mb * n_parallel / 1024.0
        qwen_gpu_per_ctx_gb = (
            _QWEN4EXP_GPU_COMPUTE_BYTES * qwen4exp_ubatch * n_parallel / (1024.0**3)
        )
        qwen_host_per_ctx_gb = (
            _QWEN4EXP_HOST_COMPUTE_BYTES * qwen4exp_ubatch * n_parallel / (1024.0**3)
        )

        if unified_memory:
            shared_fixed_gb = (
                model_vram
                + model_ram
                + mapped_model_resident_gb
                + vision_vram_gb
                + vision_ram_gb
                + draft_vram_gb
                + base_runtime_vram_gb
                + _QWEN4EXP_FIXED_HOST_RUNTIME_GB
                + prompt_cache_ram_gb
                + moe_batch_vram_reserve_gb
                + recurrent_state_total_gb
                + max(effective_vram_safety, ram_safety_gb)
            )
            shared_dynamic_per_ctx_gb = (
                qwen_gpu_per_ctx_gb + qwen_host_per_ctx_gb + kv_per_ctx_gb
            )
            shared_cap_gb = min(free_vram, max(0.0, system.free_ram_gb))
            qwen_physical_max_ctx = int(
                max(0.0, shared_cap_gb - shared_fixed_gb)
                / max(shared_dynamic_per_ctx_gb, 1e-12)
            )
        else:
            gpu_fixed_gb = (
                model_vram
                + vision_vram_gb
                + draft_vram_gb
                + base_runtime_vram_gb
                + moe_batch_vram_reserve_gb
                + recurrent_state_vram_gb
                + effective_vram_safety
            )
            host_fixed_gb = (
                model_ram
                + mapped_model_resident_gb
                + vision_ram_gb
                + prompt_cache_ram_gb
                + recurrent_state_ram_gb
                + _QWEN4EXP_FIXED_HOST_RUNTIME_GB
                + ram_safety_gb
            )
            gpu_dynamic_per_ctx_gb = qwen_gpu_per_ctx_gb + (
                0.0 if no_kv_offload else kv_per_ctx_gb
            )
            host_dynamic_per_ctx_gb = qwen_host_per_ctx_gb + (
                kv_per_ctx_gb if no_kv_offload else 0.0
            )
            gpu_max_ctx = int(
                max(0.0, gpu_budget_free_vram - gpu_fixed_gb)
                / max(gpu_dynamic_per_ctx_gb, 1e-12)
            )
            host_max_ctx = int(
                max(0.0, system.free_ram_gb - host_fixed_gb)
                / max(host_dynamic_per_ctx_gb, 1e-12)
            )
            qwen_physical_max_ctx = min(gpu_max_ctx, host_max_ctx)

        if qwen_physical_max_ctx < 2048:
            raise MemoryError(
                "qwen4exp fixed model/runtime reservations leave insufficient "
                "RAM or VRAM for the minimum 2,048-token context"
            )
        if ctx > qwen_physical_max_ctx:
            if user_ctx is not None and pin_clamped_to_budget is None:
                pin_clamped_to_budget = ctx
            ctx = qwen_physical_max_ctx

        qwen_gpu_actual_gb, qwen_host_actual_gb = qwen4exp_compute_buffers_gb(
            ctx, qwen4exp_ubatch, n_parallel
        )
        runtime_vram_overhead_gb = base_runtime_vram_gb + qwen_gpu_actual_gb
        runtime_ram_overhead_gb = _QWEN4EXP_FIXED_HOST_RUNTIME_GB + qwen_host_actual_gb

    # Total KV across ALL n_parallel slots — llama-server allocates one
    # full KV buffer per slot, so the real VRAM/RAM footprint is
    # n_parallel × per-slot. Previously this was per-slot only, which
    # undercounted the "Total GPU" display, the VRAM-overcommit warning,
    # the pre-launch balance check, and the MoE tensor-split byte weights
    # by a factor of n_parallel (worst in safe mode, n_parallel=4).
    estimated_kv_gb = (ctx * actual_per_tok_mb * n_parallel) / 1024

    # ---- (3b) VRAM Overcommit Warning
    warning: Optional[str] = None
    if pin_clamped_to_budget is not None:
        # The user pinned a context larger than the reserved budget can
        # hold; we clamped it to the safe maximum. Tell them explicitly so
        # a "why isn't my 1 M pin honoured?" question answers itself.
        warning = (
            f"Requested context {pin_clamped_to_budget:,} exceeds the "
            f"safe KV/compute-memory budget; clamped to {ctx:,} to avoid "
            "VRAM/RAM OOM."
        )
    if unified_memory:
        shared_total = (
            model_vram
            + model_ram
            + mapped_model_resident_gb
            + estimated_kv_gb
            + recurrent_state_total_gb
            + vision_vram_gb
            + vision_ram_gb
            + draft_vram_gb
            + runtime_vram_overhead_gb
            + runtime_ram_overhead_gb
            + prompt_cache_ram_gb
            + moe_batch_vram_reserve_gb
            + max(effective_vram_safety, ram_safety_gb)
        )
        shared_free = min(free_vram, max(0.0, system.free_ram_gb))
        if shared_total > shared_free * 0.98:
            tight = (
                f"Unified-memory budget tight: resident model "
                f"{model_vram + model_ram + mapped_model_resident_gb:.1f} "
                f"GB + KV {estimated_kv_gb:.1f} GB + overhead/reserves "
                f"{shared_total - model_vram - model_ram - mapped_model_resident_gb - estimated_kv_gb:.1f} "
                f"GB ≈ {shared_total:.1f} GB of {shared_free:.1f} GB available."
            )
            warning = f"{warning} {tight}" if warning else tight
    elif n_cpu_moe is not None or full_off:
        # When --no-kv-offload is active the KV cache lives in system RAM,
        # so it must NOT count toward the dedicated-VRAM overcommit check.
        vram_kv_component = 0.0 if no_kv_offload else estimated_kv_gb
        gpu_total = (
            model_vram
            + vram_kv_component
            + runtime_vram_overhead_gb
            + moe_batch_vram_reserve_gb
            + recurrent_state_vram_gb
            + effective_vram_safety
        )
        if gpu_total > gpu_budget_free_vram * 0.98:
            tight = (
                f"VRAM budget tight: model {model_vram:.1f} GB + KV "
                f"{vram_kv_component:.1f} GB + runtime/batch "
                f"{runtime_vram_overhead_gb + moe_batch_vram_reserve_gb:.1f} "
                f"GB + recurrent state {recurrent_state_vram_gb:.1f} GB + "
                f"safety {effective_vram_safety:.1f} GB ≈ "
                f"{gpu_total:.1f} GB of {gpu_budget_free_vram:.1f} GB free."
            )
            warning = f"{warning} {tight}" if warning else tight

    if not unified_memory:
        host_total = (
            model_ram
            + mapped_model_resident_gb
            + vision_ram_gb
            + prompt_cache_ram_gb
            + runtime_ram_overhead_gb
            + recurrent_state_ram_gb
            + (estimated_kv_gb if no_kv_offload else 0.0)
            + ram_safety_gb
        )
        if host_total > system.free_ram_gb * 0.98:
            tight = (
                f"RAM budget tight: CPU weights {model_ram:.1f} GB + lazy-map "
                f"resident budget {mapped_model_resident_gb:.1f} GB + runtime/cache/state "
                f"{host_total - model_ram - mapped_model_resident_gb - ram_safety_gb:.1f} "
                f"GB + safety {ram_safety_gb:.1f} GB ≈ {host_total:.1f} GB "
                f"of {system.free_ram_gb:.1f} GB free."
            )
            warning = f"{warning} {tight}" if warning else tight

        full_lazy_pressure_gb = (
            host_total - mapped_model_resident_gb + mapped_model_ram_gb
        )
        if (
            mapped_model_ram_gb > mapped_model_resident_gb + 0.05
            and full_lazy_pressure_gb > system.free_ram_gb * 0.98
        ):
            paging = (
                f"Lazy tensor map is {mapped_model_ram_gb:.1f} GB file-backed "
                f"({mapped_model_resident_gb:.1f} GB active-row budget); Windows/OS "
                "may reclaim or page cold rows under current RAM pressure."
            )
            warning = f"{warning} {paging}" if warning else paging

    # ---- (4) Threads — weniger Threads für bessere Performance
    # start_llama.py verwendet: cpu_count // 2 (max 8 bei <16 cores)
    physical = system.cpu_cores_physical
    logical = system.cpu_cores_logical
    optimal_threads = (logical // 2) if logical > 8 else logical

    if full_off:
        threads = min(optimal_threads, 16)
        batch_threads = min(physical, 16)
    elif n_cpu_moe is not None and n_cpu_moe > 0:
        threads = min(optimal_threads, 24)
        batch_threads = min(logical, 32)
    elif ngl > 0:
        threads = min(optimal_threads, 20)
        batch_threads = min(logical, 32)
    else:
        threads = min(optimal_threads, 32)
        batch_threads = min(logical, 64)

    # ---- (4b) Batch / ubatch sizing
    # Three regimes, picked in order:
    #
    #   1. MoE with CPU-resident experts (`--n-cpu-moe` > 0): use the
    #      perf_target's moe_hybrid_batch/ubatch. Larger batches let
    #      llama.cpp's op-offload prompt processing copy CPU-resident
    #      expert tensors to the GPU as a single batched operation,
    #      which is much faster than per-token round-trips. Reference:
    #      HuggingFace MoE-offload guide (Doctor-Shotgun, Feb 2026) and
    #      the gfx1151 ROCm/Vulkan benchmark in llama.cpp issue #21284,
    #      both showing near-linear PP scaling up to -ub 2048/4096.
    #
    #   2. Full GPU offload of a large dense model (>30 GB) OR long
    #      context (>32k): 1024/1024 — keeps the compute buffer modest
    #      so the model itself doesn't get squeezed.
    #
    #   3. Everything else (small-to-mid dense, short ctx): 2048/512 —
    #      the historical default that's optimal for pure GPU inference.
    if is_qwen4exp:
        # QSA's ctx×ubatch graph dominates both host and device memory. Use
        # 64/128/256 for Safe/Balanced/Throughput so long requested contexts
        # consume batch throughput before they consume the context window.
        batch, ubatch = 1024, qwen4exp_ubatch
    elif n_cpu_moe is not None and n_cpu_moe > 0:
        batch = perf_target.moe_hybrid_batch
        ubatch = perf_target.moe_hybrid_ubatch
        # When integrated MTP is active on a MoE model, the speculative hook
        # fires at every ubatch boundary during generation. With moe_hybrid_ubatch
        # at 2048 or 4096 the D2H transfer overhead per token grows and write speed
        # regresses below baseline. Cap ubatch at 512 for MTP MoE models so the
        # generation phase has the same granularity the community uses (b 2048 ub 512).
        # Prompt processing (PP) is unaffected because PP fills full batches anyway.
        if model.has_embedded_mtp and ubatch > 512:
            ubatch = 512
    elif placement_model_size_gb > 30 or ctx > 32768 or placement_model_size_gb > 10:
        batch, ubatch = 1024, 1024
    else:
        batch, ubatch = 2048, 512

    # ---- (4c) mlock + no_mmap (Windows Admin Check)
    ram_resident_gb = model_ram

    is_windows = platform.system() == "Windows"
    is_admin = False
    if is_windows:
        if ctypes:
            try:
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            except Exception:
                is_admin = False
    else:
        # Auf Linux/Mac prüfen wir auf Root
        try:
            # Benutze getattr, damit Pylance nicht direkt nach dem Attribut sucht
            getuid = getattr(os, "getuid", None)
            is_admin = getuid() == 0 if getuid else True
        except Exception:
            is_admin = True

    # Option A: VRAM-basierte Bedingung für full-off Modelle
    # Wenn das Modell vollständig auf der GPU ist (full_off=True), kann mlock/no-mmap
    # trotzdem sinnvoll sein, um VRAM-Paging zu verhindern.
    vram_resident_gb = model_vram
    has_enough_vram = system.total_vram_gb > 8

    if force_mlock:
        # Option B: User-Override — aktiviert mlock/no-mmap wenn System-Ressourcen reichen
        mlock = (has_enough_vram or vram_resident_gb > 0) and (
            is_windows and is_admin or not is_windows
        )
    else:
        # Automatische Logik: zwei Fälle
        if full_off:
            # Full GPU offload: prüfe VRAM statt RAM
            mlock = (
                has_enough_vram
                and vram_resident_gb > 0
                and vram_resident_gb < (system.free_vram_gb - 2)
                and (not is_windows or is_admin)
            )
        else:
            # Partial/CPU offload: prüfe RAM
            mlock = (
                system.total_ram_gb > 32
                and ram_resident_gb > 0
                and ram_resident_gb < (system.free_ram_gb - 8)
                and (not is_windows or is_admin)
            )
    # Auto-lazy row tables require mmap. Never let automatic mlock/no-mmap
    # turn a 26+ GiB PLE table into an eager resident read; an explicit
    # --force-mlock remains the user's deliberate override.
    if mlock and read_lazy_table_gb > 0 and not force_mlock:
        mlock = False

    # GPU-Gate: llama.cpp b9895 bricht mit --mlock IMMER ab, sobald das
    # Vulkan-Backend geladen ist — unabhängig von RLIMIT_MEMLOCK und auch
    # mit -ngl 0 (auf RDNA4 reproduziert). Ursache: CPU-Gewichte landen im
    # "host"-(pinned)-Buffer des GPU-Backends; ggml_vk_host_malloc kann
    # nullptr OHNE Exception liefern (der CPU-Fallback greift nur bei
    # vk::SystemError), llama-model.cpp ruft dann mlock.init(NULL) und
    # llama_mlock::grow_to stirbt an GGML_ASSERT(addr) (llama-mmap.cpp:744).
    # Bis das upstream gefixt ist: mit GPUs im System kein automatisches
    # mlock. --force-mlock übergeht das bewusst (z. B. für gefixte Builds).
    # Dieselbe Regel greift als finales Sicherheitsnetz in veto_unsafe_mlock(),
    # falls GUI-Overrides mlock nach compute_config wieder einschalten.
    if mlock and _mlock_unsafe_with_gpu(system, force_mlock):
        mlock = False
    # POSIX-Gate: mlock ist durch RLIMIT_MEMLOCK gedeckelt (Desktop-Linux
    # default: 8 MiB). Ein nicht-root Prozess kann damit kein Modell pinnen;
    # mlock daher nur aktivieren, wenn das Limit das komplette Modell
    # abdeckt (root/CAP_IPC_LOCK umgeht das Limit → is_admin reicht).
    if mlock and not is_windows and not is_admin:
        memlock_limit_gb = _memlock_limit_gb()
        model_total_gb = ram_resident_gb + vram_resident_gb
        if memlock_limit_gb is not None and memlock_limit_gb < model_total_gb + 0.25:
            mlock = False
    no_mmap = mlock

    # ---- (4d) Multi-GPU placement & device visibility.
    #
    # Runs for BOTH dense and MoE configs.  The previous version skipped
    # MoE entirely (`n_cpu_moe is None` gate), which left llama.cpp to
    # default to Vulkan0 — the 16 GB gaming GPU — and crash with
    # ErrorOutOfDeviceMemory while building the (MTP) draft context, even
    # though the 32 GB R9700 sat idle at Vulkan1.
    #
    # Fill strategy (matches the requested target: ~30/32 GB on the R9700,
    # ~13/16 GB on the RX 9070 XT, *then* system RAM):
    #
    #   1. Compute a per-card usable cap = total_vram − headroom, where the
    #      headroom keeps a card breathing for the OS/compositor/OBS.  The
    #      primary keeps ~2 GB, secondary cards keep ~3 GB (OBS encode).
    #   2. If the whole GPU footprint (weights + KV + vision + draft) fits in
    #      the PRIMARY's cap → pin everything to the primary and hide the
    #      secondary GPU completely, so it stays free for gaming/OBS.
    #   3. Otherwise → SEQUENTIALLY fill the primary up to its cap, then spill
    #      the remainder onto the secondary (and so on).  Only once every GPU
    #      cap is exhausted does llama.cpp fall back to RAM (dense: reduced
    #      ngl handled upstream; MoE: --n-cpu-moe).
    #
    # Device visibility / indices come from gpu.hip_index, resolved in
    # hardware.py by PCI-device-id (vulkaninfo --summary) → --list-devices →
    # vulkaninfo name match.  We NEVER use the Windows registry/detection
    # position as a device index (it is the opposite order on this system).
    # The exact runtime backend selects the visibility mechanism: CUDA,
    # HIP, Vulkan, or SYCL/oneAPI. Legacy unprobeable homogeneous systems
    # receive conservative vendor-appropriate selectors.
    tensor_split: Optional[str] = None
    main_gpu: Optional[int] = None
    mmproj_device: Optional[str] = None
    env_overrides: Dict[str, str] = {}

    if has_gpu and len(system.gpus) > 1 and primary_gpu is not None:
        primary_pos = system.gpus.index(primary_gpu)
        hip_known = all(g.hip_index is not None for g in system.gpus)
        # An MoE that fits entirely on the GPUs returns n_cpu_moe=None +
        # full_off=True, so "n_cpu_moe is not None" would mis-classify it as
        # dense and route it through the priority-weighted spread — leaving
        # the secondary card half-empty. Use the architectural is_moe flag
        # (combined with has_gpu) so EVERY MoE that spreads uses the
        # capacity-fill strategy, whether or not any experts spilled to CPU.
        is_moe_cfg = is_moe and has_gpu and ngl > 0 and not disable_moe_placement

        primary_cap = _gpu_usable_cap_gb(primary_gpu, True)

        # Full GPU footprint we need to place. --no-kv-offload keeps the KV
        # cache in RAM; every other full-offload/MoE path places it on GPUs.
        # A visibility pin hides every peer device, so approving this footprint
        # from aggregate VRAM would otherwise create a deterministic OOM.
        gpu_kv_footprint_gb = 0.0 if no_kv_offload else estimated_kv_gb
        fixed_primary_footprint_gb = (
            model_vram
            + vision_vram_gb
            + draft_vram_gb
            + runtime_vram_overhead_gb
            + moe_batch_vram_reserve_gb
            + recurrent_state_vram_gb
        )
        model_footprint_gb = fixed_primary_footprint_gb + gpu_kv_footprint_gb

        # Pin only when the selected cache PLUS the same long-context scratch
        # reserve used during sizing fits the primary. Testing the raw footprint
        # alone can repin a config that borrowed aggregate VRAM, discarding the
        # reserve and recreating a single-card Vulkan OOM. Conversely, an Expert
        # Q8 pin may legitimately fit one card even when Auto's preferred F16
        # needed both; this exact post-selection test keeps that card free.
        primary_raw_kv_budget = max(
            0.0,
            primary_cap - effective_vram_safety - fixed_primary_footprint_gb,
        )
        primary_usable_kv_budget = _usable_kv_budget_after_headroom(
            primary_raw_kv_budget,
            ctx,
            n_parallel,
            rope_scaling_active,
        )
        fixed_footprint_fits_primary = (
            fixed_primary_footprint_gb + effective_vram_safety <= primary_cap
        )
        selected_cache_fits_primary = (
            fixed_footprint_fits_primary
            and gpu_kv_footprint_gb <= primary_usable_kv_budget * 0.98
        )

        # A user-supplied force_gpu ALWAYS pins exclusively: the user has
        # explicitly chosen the card this server boots on, so we hide every
        # other GPU and never spread — even if the model is overcommitted.
        pin_to_primary = (forced_gpu is not None) or selected_cache_fits_primary

        if pin_to_primary:
            if hip_known:
                primary_index = int(primary_gpu.hip_index)  # type: ignore[arg-type]
                selectors, remapped = _visibility_env_for_gpus(
                    [primary_gpu], [primary_index]
                )
                env_overrides.update(selectors)
                if remapped:
                    # The selected physical device is now visible as index 0.
                    main_gpu = 0
                else:
                    # Backend has no supported visibility remap. Steer weights
                    # with a runtime-order one-hot split and keep the original
                    # backend index for --main-gpu.
                    idx_order = sorted(
                        range(len(system.gpus)),
                        key=lambda i: int(system.gpus[i].hip_index),  # type: ignore[arg-type]
                    )
                    parts = ["0.000"] * len(system.gpus)
                    parts[idx_order.index(primary_pos)] = "1.000"
                    tensor_split = ",".join(parts)
                    main_gpu = primary_index
            else:
                # Index unknown — position-based steering cannot hide peers.
                parts = ["0.000"] * len(system.gpus)
                parts[primary_pos] = "1.000"
                tensor_split = ",".join(parts)
                main_gpu = primary_pos
        else:
            # ---- Spread across GPUs -------------------------------------
            # Two distinct strategies, because dense and MoE want opposite
            # things from the second GPU:
            #
            #   • DENSE → priority-weighted. Keep the gaming GPU (low
            #     priority) as free as possible for OBS/desktop; push the
            #     bulk of the weights onto the high-priority AI card. A
            #     half-empty secondary is FINE here — the model runs fully
            #     on GPU either way, and the user wants the headroom.
            #
            #   • MoE → capacity-fill. Every expert layer that sits in VRAM
            #     instead of spilling to CPU (--n-cpu-moe) is a large speed
            #     win, so when an MoE has to use both cards we want them BOTH
            #     as full as possible. Priority-weighting here was Basti's
            #     reported bug #1: it left several GB unused on the
            #     secondary, forcing extra layers onto the CPU and slowing
            #     the MoE down. We distribute proportionally to each card's
            #     USABLE CAPACITY (not its priority) — and, crucially, we
            #     emit the split as LAYER COUNTS computed from per-layer
            #     GPU byte weights (report #2): with --n-cpu-moe the front
            #     layers are expert-stripped and ~10× lighter in VRAM, so a
            #     byte-FRACTION split (which llama.cpp maps onto layer
            #     counts) stranded the Vulkan-device-0 card at ~8/16 GB.
            ordered = sorted(
                system.gpus,
                key=_gpu_score,
                reverse=True,  # primary (highest score) first
            )
            caps = [_gpu_usable_cap_gb(g, g is primary_gpu) for g in ordered]

            # Vision, external draft, and runner/batch workspaces live wholly
            # on main_gpu; tensor-split distributes only layer-owned weights,
            # KV, and recurrent state. Dense placement previously folded these
            # primary-only allocations into a proportional footprint, making a
            # split look cap-safe while the real primary exceeded its cap once
            # the complete mmproj was added back.
            primary_only_gpu_gb = (
                vision_vram_gb
                + draft_vram_gb
                + runtime_vram_overhead_gb
                + moe_batch_vram_reserve_gb
            )
            if is_moe_cfg:
                distribution_caps = list(caps)
                distribution_footprint_gb = model_footprint_gb
            else:
                distribution_caps = [
                    max(0.0, cap - primary_only_gpu_gb) if gpu is primary_gpu else cap
                    for gpu, cap in zip(ordered, caps)
                ]
                distribution_footprint_gb = max(
                    0.0, model_footprint_gb - primary_only_gpu_gb
                )
            total_cap = sum(distribution_caps)

            # ---- MoE per-layer GPU byte weights --------------------------
            # llama.cpp splits --tensor-split by LAYER COUNT (device 0 gets
            # the first chunk of layers), while --n-cpu-moe strips the
            # expert tensors of the FIRST n_cpu_moe layers to CPU. The GPU
            # byte weight per layer is therefore non-uniform: front layers
            # carry only attention/norm tensors (+ their KV slice), back
            # layers additionally carry the full expert weight. We model
            # that here and later emit --tensor-split as LAYER COUNTS
            # computed by _split_layers_by_bytes, so the BYTES land
            # capacity-proportionally on both cards. A plain byte-fraction
            # split (the old code) handed the Vulkan-device-0 card (the
            # 9070 XT) only expert-stripped front layers — ~8/16 GB used
            # while the R9700 ran at its limit (Basti's step-3.7 report).
            #
            # KV is included per layer because MoE KV must live in VRAM on
            # Vulkan and llama.cpp allocates each layer's KV on that
            # layer's device. estimated_kv_gb / n_layers is an average —
            # hybrid SWA/full-attention archs (step35, gemma…) deviate per
            # layer, but the error is small against expert-weight deltas.
            layer_gpu_bytes: List[float] = []
            if is_moe_cfg and n_layers > 0:
                cpu_moe_layers = min(n_layers, n_cpu_moe or 0)
                shared_gb = placement_model_size_gb * 0.08
                per_layer_expert = max(
                    0.001, (placement_model_size_gb - shared_gb) / n_layers
                )
                light_gb = shared_gb / n_layers
                kv_layer_gb = (
                    0.0 if no_kv_offload else max(0.0, estimated_kv_gb) / n_layers
                )
                recurrent_layer_gb = max(0.0, recurrent_state_vram_gb) / n_layers
                layer_gpu_bytes = [
                    light_gb
                    + kv_layer_gb
                    + recurrent_layer_gb
                    + (0.0 if li < cpu_moe_layers else per_layer_expert)
                    for li in range(n_layers)
                ]

            if is_moe_cfg:
                # Capacity-proportional weights — fill both cards in step so
                # the model packs as tightly as the combined VRAM allows.
                weights = list(caps)
            else:
                # Priority×VRAM weights — bias the high-priority AI card.
                weights = [_gpu_score(g) for g in ordered]
            total_weight = sum(weights)

            # First pass: allocate the distributable footprint proportionally
            # by the chosen weighting, respecting each adjusted card cap.
            alloc: List[float] = []
            for i, cap in enumerate(distribution_caps):
                proportion = weights[i] / total_weight if total_weight > 0 else 0
                alloc.append(min(cap, distribution_footprint_gb * proportion))

            # Second pass: distribute the full remainder across every card's
            # residual capacity. The old three-round "take half" loop stopped
            # early, then normalised the undersized allocation back to 100%; the
            # resulting tensor split could exceed a card's own cap by ~0.3 GB.
            remaining = max(0.0, distribution_footprint_gb - sum(alloc))
            if remaining > 1e-9:
                spaces = [
                    max(0.0, cap - used) for cap, used in zip(distribution_caps, alloc)
                ]
                total_space = sum(spaces)
                if total_space > 0:
                    if remaining >= total_space:
                        alloc = [used + space for used, space in zip(alloc, spaces)]
                    else:
                        alloc = [
                            used + remaining * (space / total_space)
                            for used, space in zip(alloc, spaces)
                        ]

            denom = sum(alloc) if sum(alloc) > 0 else (total_cap or 1.0)

            if hip_known:
                # Order by ascending device index so the visibility env vars and
                # the tensor_split fractions line up with what llama.cpp sees.
                idx_order = sorted(
                    range(len(ordered)),
                    key=lambda i: int(ordered[i].hip_index),  # type: ignore[arg-type]
                )
                visible_gpus = [ordered[i] for i in idx_order]
                visible_indices = [
                    int(ordered[i].hip_index)  # type: ignore[arg-type]
                    for i in idx_order
                ]
                selectors, remapped = _visibility_env_for_gpus(
                    visible_gpus, visible_indices
                )
                env_overrides.update(selectors)
                if layer_gpu_bytes:
                    # MoE → byte-aware LAYER-COUNT split (see comment above).
                    # Caps in visible-device order; the primary's cap is
                    # reduced by vision/draft VRAM since mmproj + drafter
                    # load onto the main GPU, not spread by tensor-split.
                    caps_vis: List[float] = []
                    for i in idx_order:
                        c = caps[i]
                        if ordered[i] is primary_gpu:
                            c = max(
                                0.0,
                                c
                                - vision_vram_gb
                                - draft_vram_gb
                                - runtime_vram_overhead_gb
                                - moe_batch_vram_reserve_gb,
                            )
                        caps_vis.append(c)
                    counts_vis = _split_layers_by_bytes(layer_gpu_bytes, caps_vis)
                    tensor_split = ",".join(f"{c:.3f}" for c in counts_vis)
                else:
                    tensor_split = ",".join(
                        f"{alloc[i] / denom:.3f}" for i in idx_order
                    )
                primary_visible_pos = idx_order.index(ordered.index(primary_gpu))
                # Visibility selectors remap devices to 0..N-1; otherwise
                # retain the runtime's original ordinal.
                main_gpu = (
                    primary_visible_pos if remapped else int(primary_gpu.hip_index)  # type: ignore[arg-type]
                )
            else:
                # Indices unknown — position-based split in the system.gpus
                # order (may be wrong on Windows AMD; keep the llama binary /
                # vulkaninfo reachable so hip_index resolves).
                if layer_gpu_bytes:
                    # MoE → byte-aware layer counts, positional device order.
                    caps_pos: List[float] = []
                    for g in system.gpus:
                        c = _gpu_usable_cap_gb(g, g is primary_gpu)
                        if g is primary_gpu:
                            c = max(
                                0.0,
                                c
                                - vision_vram_gb
                                - draft_vram_gb
                                - runtime_vram_overhead_gb
                                - moe_batch_vram_reserve_gb,
                            )
                        caps_pos.append(c)
                    counts_pos = _split_layers_by_bytes(layer_gpu_bytes, caps_pos)
                    tensor_split = ",".join(f"{c:.3f}" for c in counts_pos)
                else:
                    pos_alloc = {id(g): a for g, a in zip(ordered, alloc)}
                    tensor_split = ",".join(
                        f"{pos_alloc.get(id(g), 0.0) / denom:.3f}" for g in system.gpus
                    )
                main_gpu = primary_pos

    # b10541 lets MTMD select a projector device independently. Before this,
    # a dual-GPU launch budgeted the whole mmproj on ``main_gpu`` but upstream
    # silently initialized the first visible GPU instead. Keep planning and
    # runtime placement identical whenever the exact binary supplied a
    # backend-qualified device map. Visibility selectors renumber devices, so
    # use the post-remap ``main_gpu`` ordinal rather than the original one.
    if model.mmproj is not None and not no_mmproj_offload and primary_gpu is not None:
        backend = str(primary_gpu.runtime_backend or "").strip()
        if main_gpu is not None and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", backend):
            mmproj_device = f"{backend}{main_gpu}"
        elif not env_overrides:
            runtime_device = str(primary_gpu.runtime_device or "").strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*\d+", runtime_device):
                mmproj_device = runtime_device

    # ---- (4d) NUMA — immer aktivieren bei genügend Kernen für bessere Performance
    numa = None
    if system.cpu_cores_physical >= 16:
        numa = "distribute"

    # ---- (4f) Sampling
    # Two YAML schema variants are supported:
    #   New (chat/coding split):   sampling: { chat: {...}, coding: {...} }
    #   Old (flat / shared):       sampling: { temperature: ..., top_k: ... }
    #
    # The flat form is detected by the ABSENCE of both "chat" and
    # "coding" sub-dicts — in that case we use the flat dict for every
    # mode. New-format profiles that define only one of the two modes
    # still fall back to the flat dict for the missing mode, so a
    # half-migrated file behaves predictably.
    raw_sampling = profile.sampling or {}
    has_chat_block = isinstance(raw_sampling.get("chat"), dict)
    has_coding_block = isinstance(raw_sampling.get("coding"), dict)
    has_split = has_chat_block or has_coding_block

    # Resolve to a concrete dict that the rest of the function can
    # call .get() on. Done in two passes (mode → fallback) so the
    # type checker can narrow sd to `dict` after the assignment.
    sd: Dict[str, Any] = {}
    if has_split:
        mode_block = raw_sampling.get(mode)
        if isinstance(mode_block, dict):
            sd = mode_block
        else:
            # Mode not defined in this profile — fall back to the other
            # mode if present.
            other = "coding" if mode == "chat" else "chat"
            other_block = raw_sampling.get(other)
            if isinstance(other_block, dict):
                sd = other_block
    else:
        # Old flat format: every mode shares the same sampling block.
        sd = {k: v for k, v in raw_sampling.items() if not isinstance(v, dict)}

    # ---- Sampling source priority ------------------------------------
    # Per field, highest priority wins:
    #   1. An explicit value in a MATCHED profile (non-empty patterns) —
    #      Basti hand-tuned these, so a real family profile always wins.
    #   2. The model author's GGUF recommendation (general.sampling.*),
    #      read via ModelEntry.recommended_sampling. This is what fixes
    #      loops / broken tool-calls on models that hit only the generic
    #      fallback profile: e.g. Qwen3.6 ships top_k 20 / temp 1.0, but
    #      _default.yaml would otherwise force top_k 40 / temp 0.7.
    #   3. The hard-coded generic default (last resort).
    #
    # The fallback profile (_default.yaml or the builtin) has EMPTY
    # patterns. Its values are treated as soft: GGUF metadata outranks
    # them. A matched profile's values are hard and outrank GGUF.
    gguf_sampling = model.recommended_sampling  # subset, may be empty
    profile_is_fallback = not profile.patterns

    _DEFAULTS = {
        "temperature": 0.7,
        "top_k": 40.0,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
        "presence_penalty": 0.0,
    }

    def _resolve_sampling(field_name: str) -> float:
        profile_val = sd.get(field_name)
        gguf_val = gguf_sampling.get(field_name)
        if profile_is_fallback:
            # Soft profile: GGUF first, then profile (≈ default), then default.
            if gguf_val is not None:
                return gguf_val
            if profile_val is not None:
                return float(profile_val)
        else:
            # Matched profile: explicit profile value first, then GGUF, then default.
            if profile_val is not None:
                return float(profile_val)
            if gguf_val is not None:
                return gguf_val
        return _DEFAULTS[field_name]

    sampling = {
        "temperature": float(_resolve_sampling("temperature")),
        "top_k": int(_resolve_sampling("top_k")),
        "top_p": float(_resolve_sampling("top_p")),
        "min_p": float(_resolve_sampling("min_p")),
        "repeat_penalty": float(_resolve_sampling("repeat_penalty")),
        "presence_penalty": float(_resolve_sampling("presence_penalty")),
    }

    # no_context_shift für bessere Performance bei grossen Kontexten aktivieren
    no_context_shift = (ctx >= 32768) or full_off

    # ---- KV split between VRAM and RAM for display fidelity -----------
    # Mirrors the budget logic in step (2): no_kv_offload puts ALL KV in
    # RAM; MoE/full_off keep KV on GPU entirely; dense-hybrid splits
    # proportionally to the layer split; CPU-only keeps it all in RAM.
    if no_kv_offload:
        kv_vram_gb = 0.0
        kv_ram_gb = estimated_kv_gb
    elif is_moe and has_gpu and ngl > 0 and not disable_moe_placement:
        kv_vram_gb = estimated_kv_gb
        kv_ram_gb = 0.0
    elif full_off:
        kv_vram_gb = estimated_kv_gb
        kv_ram_gb = 0.0
    elif ngl > 0 and n_layers > 0:
        gpu_layer_fraction = min(1.0, ngl / n_layers)
        kv_vram_gb = estimated_kv_gb * gpu_layer_fraction
        kv_ram_gb = estimated_kv_gb * (1.0 - gpu_layer_fraction)
    else:
        kv_vram_gb = 0.0
        kv_ram_gb = estimated_kv_gb

    # ---- Seed extra_cli_flags with whatever the profile declares ------
    # Until now, profile.extra_args (e.g. "--jinja" for the reasoning
    # families) were appended directly in build_cmd, never landing in
    # cfg.extra_cli_flags. Result: the Expert panel's "--jinja" checkbox
    # stayed unchecked even for models whose profile demands it. We
    # surface them here so the GUI reflects the truth, and build_cmd
    # de-dupes when it emits the final argv.
    seed_extras: List[str] = []
    if getattr(profile, "extra_args", None):
        seed_extras = [str(a) for a in profile.extra_args if a]

    return TunedConfig(
        ctx=ctx,
        ngl=ngl,
        threads=threads,
        batch_threads=batch_threads,
        batch=batch,
        ubatch=ubatch,
        cache_k=cache_k,
        cache_v=cache_v,
        flash_attn=(
            bool(profile.flash_attn)
            if getattr(profile, "flash_attn", None) is not None
            else True
        ),
        sampling=sampling,
        mlock=mlock,
        no_mmap=no_mmap,
        numa=numa,
        tensor_split=tensor_split,
        main_gpu=main_gpu,
        mmproj_device=mmproj_device,
        n_cpu_moe=n_cpu_moe,
        is_moe=is_moe,
        expert_count=expert_count,
        estimated_model_vram_gb=model_vram,
        estimated_model_ram_gb=model_ram,
        mapped_model_ram_gb=mapped_model_ram_gb,
        mapped_model_resident_gb=mapped_model_resident_gb,
        estimated_kv_gb=estimated_kv_gb,
        full_offload=full_off,
        unified_memory=unified_memory,
        vision_vram_gb=vision_vram_gb,
        vision_ram_gb=vision_ram_gb,
        draft_vram_gb=draft_vram_gb,
        no_mmproj_offload=bool(no_mmproj_offload and model.mmproj is not None),
        prompt_cache_ram_mib=prompt_cache_ram_mib,
        prompt_cache_ram_gb=prompt_cache_ram_gb,
        runtime_vram_overhead_gb=runtime_vram_overhead_gb,
        runtime_ram_overhead_gb=runtime_ram_overhead_gb,
        batch_vram_overhead_gb=moe_batch_vram_reserve_gb,
        kv_vram_gb=kv_vram_gb,
        kv_ram_gb=kv_ram_gb,
        recurrent_state_vram_gb=recurrent_state_vram_gb,
        recurrent_state_ram_gb=recurrent_state_ram_gb,
        kv_quant_strategy=kv_quant_strategy,
        no_context_shift=no_context_shift,
        no_kv_offload=no_kv_offload,
        rope_scaling=rope_scaling_active,
        # YaRN-Faktor aus dem TATSÄCHLICH erreichten finalen ctx ableiten
        # (ceil(ctx/native)), gedeckelt auf den profil-erlaubten Maximal-
        # faktor (z.B. 4.0 für Qwen3.5/3.6). Früher war der Faktor immer fix
        # profile_rope_factor (4.0) — selbst wenn nur ein 300k-Context auf
        # einem 262k-nativen Modell aktiv wurde, wurde RoPE bis 1M
        # überdehnt (Qualitätsverlust an den tatsächlich genutzten
        # Positionen). Jetzt passt sich der Faktor exakt an.
        rope_scale_factor=(
            max(
                1.0,
                min(
                    float(profile_rope_factor),
                    float(math.ceil(ctx / native_ctx))
                    if (rope_scaling_active and native_ctx > 0)
                    else float(profile_rope_factor),
                ),
            )
            if rope_scaling_active
            else 1.0
        ),
        performance_target=perf_target.name,
        n_parallel=n_parallel,
        n_parallel_forced=n_parallel_forced,
        draft_n_max=(
            max(1, int(force_draft_n_max))
            if force_draft_n_max is not None and int(force_draft_n_max) > 0
            else 0
        ),
        warning=warning,
        extra_cli_flags=seed_extras,
        env_overrides=env_overrides,
    )


def _has_integrated_mtp(model: ModelEntry) -> bool:
    """Detect models that ship an integrated MTP drafter inside the GGUF.

    Delegates to ``ModelEntry.has_embedded_mtp`` in scanner.py, which is
    the canonical source of truth for this detection.  Detection is
    metadata-first (``<arch>.nextn_predict_layers > 0`` or tensor-info
    scan) with a filename pattern (``MTP`` token) as fallback.  See that
    property for the full rationale and examples.
    """
    return model.has_embedded_mtp


# Diffusion algorithm names → llama.cpp's integer --diffusion-algorithm
# values (examples/diffusion/README.md, b9672):
#   0 ORIGIN, 1 ENTROPY_BASED, 2 MARGIN_BASED, 3 RANDOM, 4 CONFIDENCE_BASED
_DIFFUSION_ALGORITHMS = {
    "origin": 0,
    "entropy": 1,
    "entropy_based": 1,
    "margin": 2,
    "margin_based": 2,
    "random": 3,
    "confidence": 4,
    "confidence_based": 4,
}


def _diffusion_algorithm_value(raw: Any) -> Optional[int]:
    """Normalise a profile ``algorithm:`` value to the CLI integer.

    Accepts either the integer (0..4) or a friendly name
    ("confidence", "entropy", …). Returns None when unset/invalid so the
    builder simply omits the flag and lets the CLI use its own default
    (4 = confidence-based)."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):  # guard: bool is an int subclass
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw <= 4 else None
    name = str(raw).strip().lower()
    if name.isdigit():
        v = int(name)
        return v if 0 <= v <= 4 else None
    return _DIFFUSION_ALGORITHMS.get(name)


def build_diffusion_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    diffusion_binary: str = "llama-diffusion-cli",
    prompt: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build a ``llama-diffusion-cli`` command line for a diffusion LLM.

    Diffusion text models (Dream, LLaDA, LLaDA-MoE, RND1 in mainline;
    DiffusionGemma in a fork) are NOT served by ``llama-server`` — as of
    b9672 the server has no diffusion path. They run through the single-
    shot ``llama-diffusion-cli`` example binary: it takes a prompt, runs a
    fixed number of denoising steps, prints the result and exits. There is
    no /health endpoint, no OpenAI API, no port.

    Flags emitted are mainline b9700 (examples/diffusion/README.md):
      -m / -p / -c / -ngl / -b / -ub  — standard load + batch knobs
      --diffusion-steps N             — denoising steps (profile, def 256)
      --diffusion-algorithm 0..4      — token-selection algorithm
      --diffusion-eps F  XOR  --diffusion-block-length N  — schedule
      --diffusion-visual              — live visualisation (optional)
      -n / --predict N                — max tokens (profile n_predict)
    plus any verbatim ``fork_args`` from the profile's ``diffusion`` block
    for fork-only flags mainline doesn't have (e.g. --diffusion-eb,
    --diffusion-kv-cache). User ``extra_args`` (CLI passthrough) come last.

    ``prompt`` is optional: when omitted the caller is expected to add a
    ``-p`` itself or run interactively. The diffusion CLI parses the
    same backend-placement flags as llama-server (``--main-gpu``,
    ``--tensor-split``, ``-ngl``), so we forward the multi-GPU placement
    computed by ``compute_config`` — critical for big models that must
    boot on the large card and would OOM if left on Vulkan device 0.
    """
    diff = profile.diffusion or {}

    cmd: List[str] = [
        diffusion_binary,
        "-m",
        str(model.path),
        "-c",
        str(config.ctx),
        "-ngl",
        str(config.ngl),
        "-b",
        str(config.batch),
        "-ub",
        str(config.ubatch),
    ]

    # Performance timings are enabled by default in current llama.cpp.
    # Assert --perf explicitly so fork defaults cannot hide prompt/eval timing
    # and tokens/s for single-shot diffusion runs.
    cmd.append("--perf")

    # ---- multi-GPU placement (mirror build_command) ------------------
    # DiffusionGemma (25 GB) must boot on the 32 GB card. Without these
    # flags the binary defaults to Vulkan device 0 — often the smaller
    # gaming GPU — and the KV-cache allocation fails with
    # ``alloc_tensor_range: failed to allocate Vulkan0 buffer of size
    # 1073741824``. The env_overrides (HIP/VK_VISIBLE_DEVICES) are applied
    # by the launcher separately; tensor-split/main-gpu cover the
    # hip-index-unknown (pure Vulkan) case where env hiding is not emitted.
    if config.tensor_split:
        cmd += ["--tensor-split", config.tensor_split]
    if config.main_gpu is not None:
        cmd += ["--main-gpu", str(config.main_gpu)]

    # ---- max tokens to generate (-n / --predict) ----------------------
    n_predict = diff.get("n_predict")
    if n_predict is not None:
        try:
            n = int(n_predict)
            if n > 0:
                cmd += ["-n", str(n)]
        except (TypeError, ValueError):
            pass

    # ---- denoising steps ----------------------------------------------
    steps = diff.get("steps")
    if steps is not None:
        try:
            s = int(steps)
            if s > 0:
                cmd += ["--diffusion-steps", str(s)]
        except (TypeError, ValueError):
            pass

    # ---- token-selection algorithm ------------------------------------
    alg = _diffusion_algorithm_value(diff.get("algorithm"))
    if alg is not None:
        cmd += ["--diffusion-algorithm", str(alg)]

    # ---- scheduling: eps XOR block-length -----------------------------
    # The CLI takes one or the other. If a profile (mistakenly) sets both,
    # prefer block-length and warn — silently dropping one would hide a
    # config error.
    eps = diff.get("eps")
    block_length = diff.get("block_length")
    if eps is not None and block_length is not None:
        print(
            "[AutoTuner] diffusion profile sets BOTH eps and block_length; "
            "using block_length (pick one to silence this)."
        )
        eps = None
    if block_length is not None:
        try:
            bl = int(block_length)
            if bl > 0:
                cmd += ["--diffusion-block-length", str(bl)]
        except (TypeError, ValueError):
            pass
    elif eps is not None:
        try:
            cmd += ["--diffusion-eps", str(float(eps))]
        except (TypeError, ValueError):
            pass

    # ---- live visualisation (optional) --------------------------------
    if bool(diff.get("visual", False)):
        cmd += ["--diffusion-visual"]

    # ---- sampling temperature (CLI uses --temp) -----------------------
    # Diffusion-cli honours --temp / --top-k / --top-p; pull temperature
    # from the profile's chat sampling if present (kept minimal — the
    # other samplers are rarely meaningful for diffusion decoding).
    chat_sampling = profile.sampling.get("chat") or profile.sampling or {}
    if isinstance(chat_sampling, dict):
        temp = chat_sampling.get("temperature")
        if temp is not None:
            try:
                cmd += ["--temp", str(float(temp))]
            except (TypeError, ValueError):
                pass

    # ---- prompt -------------------------------------------------------
    if prompt:
        cmd += ["-p", prompt]

    # ---- fork-only flags (verbatim passthrough) -----------------------
    # Mainline b9700 has no --diffusion-eb / --diffusion-kv-cache; the
    # DiffusionGemma fork does. Keeping them in a profile list means the
    # same schema serves both — mainline profiles just leave fork_args out.
    fork_args = diff.get("fork_args") or []
    if isinstance(fork_args, list):
        cmd += [str(a) for a in fork_args]

    # ---- user CLI passthrough (highest precedence) --------------------
    if extra_args:
        cmd += [str(a) for a in extra_args]

    return cmd


def build_diffusion_server_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    server_binary: str = "llama-diffusion-gemma-server",
    host: str = "127.0.0.1",
    port: int = 8080,
    alias: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    enable_metrics: Optional[bool] = None,
    enable_slots_api: Optional[bool] = None,
) -> List[str]:
    """Build a ``llama-diffusion-gemma-server`` command line.

    PR #24427 ships a dedicated OpenAI-compatible HTTP server for
    DiffusionGemma: it exposes ``/health``, ``/v1/chat/completions``,
    ``/props``, ``/metrics`` and binds ``--host``/``--port`` (defaults
    127.0.0.1:8080). Unlike the single-shot ``llama-diffusion-cli`` this is
    a PERSISTENT, queryable server — the right choice for DiffusionGemma.

    The fork's server uses a manual arg parser (the common llama.cpp flags
    PLUS --host/--port/--api-key/--metrics/--slots) and does NOT understand
    the llama-server-only flags (``--fit``, ``--jinja``, ``--spec-type``,
    ``--cache-ram`` …). Emitting any of those aborts with "unknown
    argument", so this builder keeps the flag set minimal and verified
    against the binary's own ``--help``.
    """
    diff = profile.diffusion or {}

    cmd: List[str] = [
        server_binary,
        "-m",
        str(model.path),
        "-c",
        str(config.ctx),
        "-ngl",
        str(config.ngl),
        "-b",
        str(config.batch),
        "-ub",
        str(config.ubatch),
    ]

    # Performance timings are enabled by default in current llama.cpp.
    # Assert --perf explicitly so fork defaults cannot hide throughput details
    # (tokens/s) for DiffusionGemma's persistent server.
    cmd.append("--perf")

    # ---- multi-GPU placement (mirror build_command) ------------------
    # DiffusionGemma (25 GB) must boot on the 32 GB card; without these
    # flags the binary defaults to Vulkan device 0 (often the smaller card)
    # and OOMs on KV allocation.
    if config.tensor_split:
        cmd += ["--tensor-split", config.tensor_split]
    if config.main_gpu is not None:
        cmd += ["--main-gpu", str(config.main_gpu)]

    # ---- multimodal projector (if present) ---------------------------
    if getattr(model, "mmproj", None):
        cmd += ["--mmproj", str(model.mmproj)]

    # ---- denoising steps (profile; Doku default 48) ------------------
    steps = diff.get("steps")
    if steps is not None:
        try:
            s = int(steps)
            if s > 0:
                cmd += ["--diffusion-steps", str(s)]
        except (TypeError, ValueError):
            pass

    # ---- sampling (temperature from chat block) ----------------------
    chat_sampling = profile.sampling.get("chat") or profile.sampling or {}
    if isinstance(chat_sampling, dict):
        temp = chat_sampling.get("temperature")
        if temp is not None:
            try:
                cmd += ["--temp", str(float(temp))]
            except (TypeError, ValueError):
                pass

    # ---- HTTP binding -------------------------------------------------
    cmd += ["--host", host, "--port", str(port)]

    # ---- readable alias / optional diagnostics endpoints --------------
    if alias:
        cmd += ["-a", alias]
    metrics_on = (
        bool(getattr(config, "metrics_enabled", True))
        if enable_metrics is None
        else bool(enable_metrics)
    )
    slots_on = (
        bool(getattr(config, "slots_api_enabled", False))
        if enable_slots_api is None
        else bool(enable_slots_api)
    )
    if metrics_on:
        cmd.append("--metrics")
    if slots_on:
        cmd.append("--slots")

    # ---- user CLI passthrough (highest precedence) -------------------
    if extra_args:
        cmd += [str(a) for a in extra_args]

    return cmd


def build_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    draft_model: Optional[ModelEntry] = None,
    server_binary: str = "llama-server",
    host: str = "127.0.0.1",
    port: int = 1234,
    extra_args: Optional[List[str]] = None,
    use_thinking: bool = False,
    enable_speculative: bool = True,
    enable_ngram: bool = False,
    enable_prompt_cache: bool = True,
    prompt_cache_ram_mib: Optional[int] = None,
    enable_metrics: Optional[bool] = None,
    enable_slots_api: Optional[bool] = None,
) -> List[str]:
    """Build the llama-server command line for ``model`` and ``config``.

    Speculative decoding paths
    --------------------------
    * ``draft_model`` is set → sibling-drafter path. Adds ``-md`` plus
      ``--spec-draft-n-max`` (no ``--spec-type``; mainline auto-detects from ``-md``).
      Skipped when vision is loaded (three model graphs in VRAM simultaneously
      is too risky on 16-GB-class cards).
    * ``draft_model`` is None and the model has embedded MTP (detected
      via ``<arch>.nextn_predict_layers`` metadata or tensor-info scan,
      with filename token ``MTP`` as fallback) →
      ``--spec-type draft-mtp`` + ``--spec-draft-n-max`` only (the drafter
      rides inside the GGUF). Compatible with ``--mmproj`` / vision since
      mainline b9180 (PR #22673, merged 2026-05-16).
    * ``enable_speculative=False`` overrides both paths and emits no
      speculative flags at all — for the case where the user explicitly
      unchecked Draft on an MTP-named model.
    * ``enable_ngram=True`` adds a draftless self-speculative method (Path C),
      selected by ``profile.ngram_method`` (default ``ngram-mod``). As of
      b9334 the choices are ``ngram-mod`` / ``ngram-map-k`` / ``ngram-map-k4v``
      / ``ngram-simple`` / ``ngram-cache``. It is model-agnostic — no draft
      model required — so it can run standalone on any GGUF, or be combined with
      an *external* sibling drafter (Path A, ``-md``). ``--spec-type`` is a
      comma-separated list and llama.cpp allows mixing a draft-model path with a
      draftless one.

      INTEGRATED MTP (Path B) coexistence: only ``ngram-mod`` conflicts with
      ``draft-mtp``. The pair ``draft-mtp,ngram-mod`` triggers random
      mid-generation crashes on MTP models such as Qwen3.6-27B-MTP — see
      llama.cpp issue #23154 (open as of b9334; "issue not reproduced when
      ngram-mod is removed"). So on an MTP model with ``ngram_method ==
      ngram-mod`` the redundant ngram-mod is suppressed and ``draft-mtp`` wins.
      The ``ngram-map-*`` family is different: ggerganov's MTP clean-up
      (PR #23269) wired ``ngram-map-k4v`` into ``--spec-default`` and runs it
      together with ``draft-mtp`` by design, so setting
      ``ngram_method: ngram-map-k4v`` is the supported way to combine
      "MTP + ngram" on an MTP model.
    """
    cmd: List[str] = [
        server_binary,
        "-m",
        str(model.path),
        "-c",
        str(config.ctx),
        "-ngl",
        str(config.ngl),
        "-t",
        str(config.threads),
        "-tb",
        str(config.batch_threads),
        "-b",
        str(config.batch),
        "-ub",
        str(config.ubatch),
        "-ctk",
        config.cache_k,
        "-ctv",
        config.cache_v,
        "--host",
        host,
        "--port",
        str(port),
    ]

    # ---- AutoTuner authority over memory placement --------------------
    # Mainline llama.cpp gained an auto-fit pass (`--fit`, default 'on')
    # that silently adjusts UNSET arguments to fit device memory. The
    # AutoTuner deliberately computes ngl / ctx / n-cpu-moe / tensor-split,
    # so we turn auto-fit OFF: the values we computed and logged are the
    # ones that run. If they overcommit we want a visible, debuggable OOM
    # — not a silent ctx/ngl downscale that desyncs the running config
    # from what the launcher reported.
    # NOTE: `--fit` (env LLAMA_ARG_FIT, default 'on') is confirmed present in
    # b9334. If a server binary predates it, this will abort with
    # "unknown argument"; in that case drop the two tokens below.
    cmd += ["--fit", "off"]

    # b10653+ reads giant architecture-marked row tables on demand. Assert
    # Auto explicitly for reproducibility; compatibility adaptation translates
    # this b10700 spelling back to --tensor-read-lazy on older builds and prunes
    # the complete pair when the feature is absent. qwen4exp's 26.8 GiB PLE
    # table is the motivating case and must never be treated as GPU layer weights.
    if model.read_lazy_size_bytes > 0 or model.architecture.lower() in {
        "qwen4exp",
        "gemma4",
    }:
        cmd += ["--lazy-mode", "auto"]

    # ---- Performance timings + optional diagnostics endpoints ----------
    # b10743 defaults performance timings off. Assert --perf explicitly so
    # fork/current defaults cannot hide prompt/eval timing
    # and tokens/s. Users can still append --no-perf for a quieter server.
    # --metrics exposes GET /metrics on the SAME host:port as inference.
    # Current mainline defaults /slots ON, so emit the positive or negative
    # flag explicitly to make the Expert toggle authoritative.
    cmd.append("--perf")
    metrics_on = (
        bool(getattr(config, "metrics_enabled", True))
        if enable_metrics is None
        else bool(enable_metrics)
    )
    slots_on = (
        bool(getattr(config, "slots_api_enabled", False))
        if enable_slots_api is None
        else bool(enable_slots_api)
    )
    if metrics_on:
        cmd.append("--metrics")
    cmd.append("--slots" if slots_on else "--no-slots")

    # ---- Host-memory prompt caching (-cram / --cache-ram) -------------
    # ggerganov's PR #16391 added a host-RAM cache for computed prompt
    # prefixes. Repeated system prompts / RAG scaffolds can then skip most
    # prefill work. -1 means unlimited, 0 disables, positive values are MiB.
    #
    # Older builds cannot safely combine this cache with mtmd. Mainline b10045
    # (PR #25076) lifted the blanket mtmd state restriction, and b10058 was
    # runtime-verified with Gemma 4 + an actual image: the repeated request
    # reported cached_tokens=279 and completed ~10x faster. Keep old forks safe
    # by requiring a numeric --version build at or above b10045; an unknown
    # version conservatively retains the historic --cache-ram 0 behaviour.
    vision_active = model.mmproj is not None
    vision_cache_ok = not vision_active
    if vision_active:
        build_number = _probe_binary_build_number(server_binary)
        vision_cache_ok = bool(
            build_number is not None and build_number >= _MIN_VISION_PROMPT_CACHE_BUILD
        )

    resolved_cache_ram_mib = (
        int(config.prompt_cache_ram_mib)
        if prompt_cache_ram_mib is None
        else int(prompt_cache_ram_mib)
    )
    if enable_prompt_cache and vision_cache_ok:
        cmd += ["--cache-ram", str(resolved_cache_ram_mib)]
    else:
        cmd += ["--cache-ram", "0"]

    if vision_active and config.no_mmproj_offload:
        cmd.append("--no-mmproj-offload")

    # Speculative decoding — composable paths combined into one --spec-type:
    #   - sibling drafter passed in        → Path A (-md, auto-detected type)
    #   - integrated MTP                    → Path B (--spec-type draft-mtp)
    #   - n-gram (enable_ngram)             → Path C (--spec-type <ngram_method>)
    #   - enable_speculative=False          → suppresses Path A and B; Path C
    #                                         (ngram) is independent and still
    #                                         honours its own enable_ngram flag.
    #
    # --spec-type accepts a comma-separated list and llama.cpp allows mixing a
    # draft-model path (draft-mtp) with a draftless one, so we assemble the
    # active types and emit a single token (e.g. "draft-mtp,ngram-map-k4v").
    #
    # Vision / draft compatibility:
    #   - OLD builds (before the --spec-type speculative rework, pre-b9190)
    #     abort when -md and --mmproj are both given ("speculative decoding
    #     is not supported with multimodal") — Path A is skipped for those.
    #   - CURRENT builds load draft + mmproj side by side: verified against
    #     b9940 server-context.cpp (unverändert bis einschließlich b9963) —
    #     the draft/MTP context and mtmd_init
    #     coexist and can_speculate() ignores mctx entirely. Path A must
    #     stay ACTIVE there: every Gemma 4 GGUF ships with an mmproj, so the
    #     old unconditional "vision wins" skip silently disabled the Gemma
    #     drafter whenever the Vision checkbox was on — the "Gemma drafter
    #     doesn't work on mainline" report. VRAM for both graphs is already
    #     budgeted by compute_config (vision_vram_gb + draft_vram_gb).
    #   - Integrated MTP (Path B) embeds the drafter inside the main GGUF —
    #     no second model-load conflict on any build. Vision and embedded MTP
    #     can coexist; Qwen3.6-MTP models in fact require the mmproj.
    #   - n-gram (Path C) loads no model at all → always compatible.
    # Precedence for --spec-draft-n-max: Expert-panel override (config,
    # 0 = unset) → YAML profile draft_max → 2.
    draft_val = resolve_draft_n_max(
        profile,
        draft_model,
        int(getattr(config, "draft_n_max", 0) or 0) or None,
    )
    # Qwen3.8's profile p-min=0.75 is calibrated for its embedded MTP head.
    # DFlash2's candidate selector already performs its own lattice pruning;
    # the model author's PR #27342 command intentionally uses the upstream
    # p-min=0.0 default. Keep that path explicit so a shared target profile
    # cannot accidentally apply MTP confidence tuning to the external drafter.
    draft_p_min = (
        0.0
        if draft_model is not None and draft_model.is_dflash2_drafter
        else float(getattr(profile, "draft_p_min", 0.75) or 0.75)
    )
    vision_loaded = model.mmproj is not None
    # Path A gating with vision: allow -md alongside --mmproj only when the
    # selected binary advertises --spec-type (the new spec system, whose
    # server no longer rejects the combination). Unprobeable binaries keep
    # the conservative skip so an old build never aborts at startup.
    vision_blocks_external = vision_loaded
    if vision_loaded and enable_speculative and draft_model is not None:
        _bin_flags = _probe_supported_flags(server_binary)
        vision_blocks_external = not (
            _bin_flags is not None and "--spec-type" in _bin_flags
        )
    use_external = (
        enable_speculative and draft_model is not None and not vision_blocks_external
    )
    # An external MTP head is NOT a plain sibling draft model and therefore
    # is not auto-detected from -md alone. Gemma 4 exposes this through its
    # ``gemma4-assistant`` architecture, while Tess-4's Qwen3.5-based sidecar
    # keeps ``general.architecture=qwen35`` and declares
    # ``qwen35.nextn_predict_layers=1``. Both need the dedicated draft-mtp
    # path; loading Tess's 18-tensor head as a normal draft model crashes
    # llama-server with 0xC0000005 before startup. ``has_embedded_mtp`` is
    # safe here because draft_model is already an external draft candidate.
    # A plain sibling drafter still keeps the auto-detected -md path.
    external_is_mtp_head = use_external and bool(
        getattr(draft_model, "is_standalone_drafter", False)
        or getattr(draft_model, "has_embedded_mtp", False)
    )
    # The spec-type token an EXTERNAL drafter needs, if any. Plain sibling
    # drafters (auto-detected from -md) need no token. Standalone MTP heads
    # (Gemma 4 assistant) need "draft-mtp"; EAGLE-3 draft models need
    # "draft-eagle3"; DFlash needs "draft-dflash"; and b10164+ DSpark needs
    # "draft-dspark" because its DFlash-derived graph carries an additional
    # Markov head. EAGLE-3 reads the target's hidden states for higher
    # acceptance than a plain draft of the same size.
    external_spec_type: Optional[str] = None
    if use_external:
        external_spec_type = getattr(draft_model, "drafter_spec_type", None)
    # Path B: integrated MTP (draft-mtp) — compatible with vision (--mmproj)
    # since mainline b9180 (PR #22673, merged 2026-05-16). The MTP draft head
    # lives inside the same GGUF as the main model; llama.cpp loads it as part
    # of the same graph so there is no second-model-load conflict.
    use_integrated = (
        enable_speculative and _has_integrated_mtp(model) and draft_model is None
    )

    # ---- Draftless ("ngram") method selection (b9334) ------------------
    # As of b9334 the draftless --spec-type vocabulary is a family:
    #   ngram-mod, ngram-map-k, ngram-map-k4v, ngram-simple, ngram-cache.
    # The profile picks one (default "ngram-mod" -> unchanged behaviour). It is
    # validated at load time in settings_loader, so it is always a known token.
    ngram_method = (
        getattr(profile, "ngram_method", "ngram-mod") or "ngram-mod"
    ).lower()

    # Only ngram-mod conflicts with integrated MTP. Combining draft-mtp,ngram-mod
    # in one --spec-type list causes random mid-generation crashes on MTP models
    # (e.g. Qwen3.6-27B-MTP): CUDA/Vulkan device error, or the model stalling
    # mid-thought. That is llama.cpp issue #23154, still OPEN as of b9334 — the
    # reporter confirms "issue not reproduced when ngram-mod is removed". Both
    # speculators write into the same decode graph and corrupt each other's
    # draft state.
    #
    # The ngram-map-* family is different: ggerganov's MTP clean-up (PR #23269)
    # wired ngram-map-k4v into --spec-default and demonstrates draft-mtp +
    # ngram-map-k4v running together, so those methods ARE allowed alongside
    # integrated MTP. That is exactly how to combine "MTP + ngram" on an MTP
    # model — set ngram_method: ngram-map-k4v in the profile.
    #
    # Resolution when the conflicting pair would occur: on an MTP model with
    # ngram_method == ngram-mod, draft-mtp wins (it is the trained, model-native
    # draft head with higher acceptance than a generic hash lookup), and the
    # redundant ngram-mod is suppressed. Every other case keeps ngram active:
    # standalone (dense / MoE-without-MTP), alongside an external sibling drafter
    # (Path A, -md, separate context — no conflict), or any ngram-map-* method
    # next to integrated MTP.
    ngram_conflicts_with_mtp = ngram_method == "ngram-mod"
    use_ngram = enable_ngram and not (use_integrated and ngram_conflicts_with_mtp)

    # Assemble the --spec-type list (embedded-draft + draftless types) and emit
    # it BEFORE the per-path parameter flags. -md (Path A) is auto-detected by
    # mainline, so it contributes no type token — only its parameter flags.
    spec_types: List[str] = []
    if use_integrated:
        spec_types.append("draft-mtp")
    elif external_is_mtp_head:
        # External standalone MTP head (Gemma 4 assistant) — needs the
        # explicit draft-mtp path; it is not auto-detected from -md.
        spec_types.append("draft-mtp")
    elif external_spec_type == "eagle3":
        # EAGLE-3 draft model: a one-layer transformer that reads the target's
        # hidden states. Loaded via -md but requires the explicit token
        # (mainline does NOT auto-detect eagle3 from -md).
        spec_types.append("draft-eagle3")
    elif external_spec_type == "dflash":
        # DFlash block-diffusion draft model: emits a whole block per step.
        spec_types.append("draft-dflash")
    elif external_spec_type == "dspark":
        # DSpark extends DFlash with an anchor-first Markov proposal head.
        # prepare_command_for_binary removes this whole path on pre-b10164
        # binaries because flag-name probing cannot validate enum values.
        spec_types.append("draft-dspark")
    if use_ngram:
        spec_types.append(ngram_method)
    if spec_types:
        cmd += ["--spec-type", ",".join(spec_types)]

    if use_external:
        # Path A — sibling drafter file.
        # -md MUST come before --spec-draft-n-max (llama-server parses
        # left-to-right). For a plain sibling drafter mainline auto-detects
        # the draft path from -md, so no --spec-type token is needed. A
        # standalone MTP head (gemma4-assistant) is the exception: its
        # "draft-mtp" type was already added to --spec-type above
        # (external_is_mtp_head), because -md alone does not select that path.
        # If ngram is also enabled it was added to --spec-type too and runs as
        # the draftless path alongside -md.
        assert draft_model is not None  # guaranteed by use_external condition
        cmd += ["-md", str(draft_model.path)]
        cmd += ["--spec-draft-ngl", "99"]
        cmd += ["--spec-draft-n-max", str(draft_val)]
        cmd += ["--spec-draft-p-min", str(draft_p_min)]
    elif use_integrated:
        # Path B — integrated MTP drafter inside the main GGUF.
        # `--spec-draft-ngl 99` keeps the MTP head on GPU; without it the
        # drafter layers fall back to CPU and the speedup vanishes.
        # `--spec-draft-p-min` is emitted explicitly: as of b9334 (PR #23269
        # made the argument functional again) the mainline DEFAULT is 0.0,
        # meaning the MTP hook fires on every decode step regardless of
        # confidence, adding constant D2H-transfer overhead that can make
        # write-speed slower than baseline on Vulkan/ROCm. We pass the
        # profile's value (default 0.75) so drafting only triggers on
        # confident steps — do not rely on the upstream default here.
        cmd += ["--spec-draft-n-max", str(draft_val)]
        cmd += ["--spec-draft-ngl", "99"]
        cmd += ["--spec-draft-p-min", str(draft_p_min)]

    if use_ngram:
        # Path C — draftless self-speculative decoding (no draft model).
        # Emit per-method parameter flags. Gated by use_ngram (not enable_ngram)
        # so nothing is emitted when an ngram-mod request was suppressed next to
        # integrated MTP — see the #23154 note above.
        if ngram_method == "ngram-mod":
            # Rolling-hash lookup table built from the live context (~16 MB,
            # constant memory). Params per llama.cpp docs/speculative.md.
            ngram_match = getattr(profile, "ngram_n_match", 24) or 24
            ngram_min = getattr(profile, "ngram_n_min", 48) or 48
            ngram_max = getattr(profile, "ngram_n_max", 64) or 64
            cmd += ["--spec-ngram-mod-n-match", str(ngram_match)]
            cmd += ["--spec-ngram-mod-n-min", str(ngram_min)]
            cmd += ["--spec-ngram-mod-n-max", str(ngram_max)]
        elif ngram_method == "ngram-map-k4v":
            # Key+value n-gram map — the method ggerganov's MTP clean-up
            # (PR #23269) pairs with draft-mtp. The values below are
            # AutoTuner's own calibrated defaults (16 / 24 / 1), emitted
            # explicitly so they are authoritative regardless of upstream
            # drift. For reference, verified against b9829 source:
            #   - mainline runtime struct default (common.h,
            #     common_params_speculative_ngram_map, shared by k4v):
            #     size_n 12, size_m 48, min_hits 1
            #   - the commented --spec-default reference config (arg.cpp):
            #     size_n 8, size_m 24, min_hits 2
            # We deliberately don't track either of those moving targets —
            # the profile value wins. Override per-profile via
            # ngram_k4v_size_n / ngram_k4v_size_m / ngram_k4v_min_hits.
            size_n = getattr(profile, "ngram_k4v_size_n", 16) or 16
            size_m = getattr(profile, "ngram_k4v_size_m", 24) or 24
            min_hits = getattr(profile, "ngram_k4v_min_hits", 1) or 1
            cmd += ["--spec-ngram-map-k4v-size-n", str(size_n)]
            cmd += ["--spec-ngram-map-k4v-size-m", str(size_m)]
            cmd += ["--spec-ngram-map-k4v-min-hits", str(min_hits)]
        # ngram-map-k / ngram-simple / ngram-cache: emit only the --spec-type
        # token (added above) and let llama.cpp apply its own well-tuned
        # defaults — we deliberately don't guess sub-parameter flag names for
        # methods the AutoTuner hasn't explicitly calibrated.

    # Tri-state llama.cpp defaults to ``auto``. Emit both states explicitly so
    # an OCR/reference profile (or an unchecked Expert toggle) can genuinely
    # disable FA instead of silently inheriting auto=on from b10329.
    cmd += ["-fa", "on" if config.flash_attn else "off"]
    if config.numa:
        cmd += ["--numa", config.numa]
    load_mode = effective_load_mode(config)
    if load_mode is not None:
        cmd += ["--load-mode", load_mode]
    if config.no_context_shift:
        cmd.append("--no-context-shift")
    # LOW-VRAM lever (low_vram perf-target): keep the KV cache in system
    # RAM instead of offloading it to VRAM. Attention compute follows
    # onto the CPU, so this trades generation speed for the context
    # headroom drawn from abundant system RAM. Confirmed flag on
    # llama-server (b9334+, present in current mainline).
    if config.no_kv_offload:
        cmd.append("--no-kv-offload")

    # RoPE-Scaling (YaRN) optional aktivieren für erweiterte Context-Längen
    # Bei Qwen3.5/3.6 möglich: native 262144 → bis 1048576 mit yarn scaling
    if config.rope_scaling and config.rope_scale_factor > 1.0:
        cmd += ["--rope-scaling", "yarn"]
        cmd += ["--rope-scale", str(int(config.rope_scale_factor))]

    if config.n_cpu_moe is not None and config.n_cpu_moe > 0:
        cmd += ["--n-cpu-moe", str(config.n_cpu_moe)]
    if config.tensor_split:
        cmd += ["--tensor-split", config.tensor_split]
    if config.main_gpu is not None:
        cmd += ["--main-gpu", str(config.main_gpu)]

    # Always pass --parallel explicitly.  llama-server's "auto" mode infers
    # n_parallel from the total KV budget ÷ per-slot KV cost.  On large
    # dual-GPU systems this can produce n_parallel=4 or more, multiplying
    # the actual KV allocation by that factor and filling all available RAM
    # (confirmed on R9700 32 GB + RX 9070 XT 16 GB with Qwen3.6-27B-Q8).
    # Passing the value explicitly prevents the server from picking a
    # different N than what compute_config budgeted for.
    cmd += ["--parallel", str(max(1, config.n_parallel))]

    s = config.sampling
    cmd += [
        "--temp",
        str(s["temperature"]),
        "--top-k",
        str(s["top_k"]),
        "--top-p",
        str(s["top_p"]),
        "--min-p",
        str(s["min_p"]),
        "--repeat-penalty",
        str(s["repeat_penalty"]),
    ]
    pp = s.get("presence_penalty", 0.0)
    if pp:
        cmd += ["--presence-penalty", str(pp)]

    if model.mmproj is not None:
        cmd += ["--mmproj", str(model.mmproj)]
        if config.mmproj_device and not config.no_mmproj_offload:
            cmd += ["--mmproj-device", config.mmproj_device]

    # Thinking/Reasoning-Modus (Gemma 4, DeepSeek, etc.)
    # Thinking wird über Prompt-Tags gesteuert (<|think|>), nicht über CLI-Argumente.
    # use_thinking ist ein internes Flag - extra_args werden immer angehängt:

    # ---- Extra-flag merge (de-duplicated, order-preserving) -----------
    # Two sources feed `cmd` here:
    #   1. profile.extra_args  — declared in the YAML (e.g. "--jinja")
    #   2. cfg.extra_cli_flags — what the GUI's Expert panel emitted
    # Until v3.1 we appended both blindly, which produced duplicate
    # flags whenever compute_config (correctly) seeded extra_cli_flags
    # from profile.extra_args so the Expert checkbox would reflect it.
    # Walk both lists with a "seen" set so a flag appears at most once,
    # and the relative ordering of first occurrences is preserved.
    #
    # The seen set is pre-populated with the entire cmd built so far —
    # this also catches the case where a profile lists "--no-context-shift"
    # in extra_args *and* the tuner separately decided to emit it (line
    # 1408): without prepopulating, the same flag would land twice.
    #
    # Value-flags (those in _ARG_FLAGS_WITH_VALUES) are keyed on the FLAG
    # NAME so a duplicate ``--load-mode none`` in the free-form Extras
    # field is dropped entirely (flag + value) when the dropdown already
    # emitted ``--load-mode mlock`` — no stray value token leaks.
    def _flag_keys(seq: List[str]) -> List[str]:
        keys: List[str] = []
        j = 0
        n = len(seq)
        while j < n:
            tok = seq[j]
            flag = _flag_name(tok)
            takes_value = flag in _ARG_FLAGS_WITH_VALUES
            if "=" in tok:
                keys.append(flag)
                j += 1
            elif takes_value and j + 1 < n:
                keys.append(flag)
                j += 2
            else:
                keys.append(tok)
                j += 1
        return keys

    seen: set = set(_flag_keys(cmd))

    def _append_unique(src: Optional[List[str]]) -> None:
        if not src:
            return
        j = 0
        n = len(src)
        while j < n:
            tok = src[j]
            flag = _flag_name(tok)
            inline = "=" in tok
            takes_value = flag in _ARG_FLAGS_WITH_VALUES
            if inline:
                key = flag
                chunk = [tok]
                j += 1
            elif takes_value and j + 1 < n:
                key = flag
                chunk = [tok, src[j + 1]]
                j += 2
            else:
                key = tok
                chunk = [tok]
                j += 1
            if key in seen:
                continue
            seen.add(key)
            cmd.extend(chunk)

    _append_unique(getattr(profile, "extra_args", None))
    _append_unique(config.extra_cli_flags)
    _append_unique(extra_args)

    return cmd
