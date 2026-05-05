"""Compute an optimal llama-server configuration for a given model
on the detected hardware, then build the command line.

v3 changes vs v2:
  * MoE-aware KV budget: when --n-cpu-moe is active, attention + KV live
    on the GPU. The KV budget must therefore be VRAM-only — counting
    free RAM (as v2 did, by accident) makes the tuner pick a context
    that overflows VRAM and crashes Vulkan with `GGML_ASSERT(addr)`.
  * More conservative placement for MoE: AMD/Vulkan is sensitive when
    VRAM is packed close to the limit, and the placement-time KV
    reserve was sized against the full profile_max context (often
    256k+), which is unrealistic on a 16 GB GPU. Now: bump the safety
    margin to 2.5 GB and size KV reserve for a sensible 32k target —
    the user can still go higher with --ctx, but won't be silently
    over-committed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hardware import SystemInfo
from scanner import ModelEntry
from settings_loader import ModelProfile


# ---------------------------------------------------------------------------
# Tunables. Kept at module scope so tests / callers can override them.

DEFAULT_VRAM_SAFETY_GB = 1.5
DEFAULT_RAM_SAFETY_GB = 4.0

# MoE-specific knobs. AMD/Vulkan crashes near the VRAM ceiling, so we
# leave more slack for the allocator. The placement-time KV reserve is
# sized against this target context, NOT the profile's max_context, so
# we don't pack experts based on a 256k-token assumption that nobody
# actually uses for a 35B model on a 16 GB card.
MOE_VRAM_SAFETY_GB = 2.5
MOE_PLACEMENT_CTX_TARGET = 32768
MOE_KV_RESERVE_FRAC = 0.30


# ---------------------------------------------------------------------------
# Model-size helpers

def extract_params_billion(name: str) -> float:
    """Extract parameter count in billions from a model filename."""
    matches = re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*B(?![a-zA-Z0-9_])",
                         name)
    if matches:
        return max(float(m) for m in matches)
    m = re.search(r"E(\d+(?:\.\d+)?)B", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def kv_per_token_mb_f16(params_billion: float) -> float:
    """Approximate KV-cache memory per token at f16 quant, in MB."""
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


def kv_quant_factor(quant: str) -> float:
    """Memory factor of a given KV-cache quant, relative to f16."""
    q = quant.lower()
    if q in ("f16", "fp16", "bf16"):
        return 1.0
    if q in ("q8_0", "q8_1", "q8"):
        return 0.55
    if q in ("q5_0", "q5_1", "q5"):
        return 0.40
    if q in ("q4_0", "q4_1", "q4"):
        return 0.32
    return 0.55


# ---------------------------------------------------------------------------
# MoE detection

def _moe_expert_count(model: ModelEntry) -> int:
    """Return expert_count from GGUF metadata, or 0 if dense / unknown."""
    md = model.metadata
    if not md:
        return 0
    arch = md.get("general.architecture")
    if arch:
        key = f"{arch}.expert_count"
        if key in md:
            try:
                return int(md[key])
            except (TypeError, ValueError):
                pass
    for k, v in md.items():
        if k.endswith(".expert_count"):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
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

    mlock: bool = False
    no_mmap: bool = False
    numa: Optional[str] = None
    tensor_split: Optional[str] = None
    main_gpu: Optional[int] = None

    n_cpu_moe: Optional[int] = None
    is_moe: bool = False
    expert_count: int = 0

    estimated_model_vram_gb: float = 0.0
    estimated_model_ram_gb: float = 0.0
    estimated_kv_gb: float = 0.0
    full_offload: bool = False

    # Optional warning (e.g. "VRAM tight, consider lowering --ctx").
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers

def _decide_offload(
    model_size_gb: float,
    free_vram_gb: float,
    n_layers: int,
    has_gpu: bool,
    vram_headroom_gb: float = DEFAULT_VRAM_SAFETY_GB,
) -> Tuple[int, float, float, bool]:
    """Dense-model placement: returns (ngl, model_vram, model_ram, full)."""
    if not has_gpu or free_vram_gb < 1.0:
        return 0, 0.0, model_size_gb, False

    usable = max(0.0, free_vram_gb - vram_headroom_gb)
    if usable >= model_size_gb:
        return 999, model_size_gb, 0.0, True

    if usable < 0.5:
        return 0, 0.0, model_size_gb, False

    if n_layers > 0:
        per_layer_gb = model_size_gb / n_layers
        ngl = int(usable / per_layer_gb)
        ngl = max(0, min(n_layers, ngl))
        model_vram = ngl * per_layer_gb
        residual_overhead = model_size_gb * 0.03
        model_ram = (n_layers - ngl) * per_layer_gb + residual_overhead
        return ngl, model_vram, model_ram, False

    estimated_layers = 50
    fraction = usable / model_size_gb
    ngl = max(0, int(fraction * estimated_layers))
    return ngl, usable, max(0.0, model_size_gb - usable), False


def _decide_moe_offload(
    model_size_gb: float,
    free_vram_gb: float,
    free_ram_gb: float,
    n_layers: int,
    expert_count: int,
    params_billion: float,
    target_ctx: int,
    ram_safety_gb: float = DEFAULT_RAM_SAFETY_GB,
) -> Tuple[int, Optional[int], float, float, bool]:
    """MoE placement using --n-cpu-moe. Returns
    (ngl, n_cpu_moe, model_vram_gb, model_ram_gb, full_offload).

    KV cache also lives on GPU with this strategy. Sizing the placement
    KV reserve against `target_ctx` (which can be 256k from a permissive
    profile) packs experts way too sparsely on a 16 GB card, or — worse —
    leaves no room for the actual KV at runtime. We size the reserve
    against a realistic working context (MOE_PLACEMENT_CTX_TARGET), and
    clamp by MOE_KV_RESERVE_FRAC so a single huge model can't accidentally
    block all KV space.
    """
    base_kv_mb = kv_per_token_mb_f16(params_billion)

    placement_ctx = min(target_ctx, MOE_PLACEMENT_CTX_TARGET)
    kv_reserve_gb = (placement_ctx * base_kv_mb * 0.55) / 1024
    kv_reserve_gb = min(kv_reserve_gb, free_vram_gb * MOE_KV_RESERVE_FRAC)

    shared_overhead_gb = model_size_gb * 0.10
    per_layer_expert_gb = max(0.001,
                              (model_size_gb - shared_overhead_gb) / n_layers)

    usable_for_experts = (free_vram_gb - MOE_VRAM_SAFETY_GB
                          - kv_reserve_gb - shared_overhead_gb)

    if usable_for_experts < 0:
        if free_ram_gb - ram_safety_gb < model_size_gb - shared_overhead_gb:
            return 999, n_layers, shared_overhead_gb, model_size_gb, False
        return 999, n_layers, shared_overhead_gb, \
            model_size_gb - shared_overhead_gb, False

    layers_on_gpu = int(usable_for_experts / per_layer_expert_gb)
    layers_on_gpu = max(0, min(n_layers, layers_on_gpu))
    n_cpu_moe = n_layers - layers_on_gpu

    if n_cpu_moe == 0:
        return 999, 0, model_size_gb, 0.0, True

    model_vram = shared_overhead_gb + layers_on_gpu * per_layer_expert_gb
    model_ram = n_cpu_moe * per_layer_expert_gb
    return 999, n_cpu_moe, model_vram, model_ram, False


def _pick_kv_quant(
    profile_recommended: str,
    target_ctx: int,
    base_kv_per_token_mb: float,
    kv_budget_gb: float,
) -> Tuple[str, str]:
    """Pick the best-quality KV cache quant that fits the target context."""
    order = ["q8_0", "q5_0", "q4_0"]
    rec = profile_recommended.lower()
    if rec in order:
        order.remove(rec)
        order.insert(0, rec)

    budget_mb = kv_budget_gb * 1024 * 0.92
    for q in order:
        per_tok = base_kv_per_token_mb * kv_quant_factor(q)
        if per_tok <= 0:
            continue
        max_fit = int(budget_mb / per_tok)
        if max_fit >= target_ctx:
            return q, q
    return "q4_0", "q4_0"


# ---------------------------------------------------------------------------
# Main entry

def compute_config(
    model: ModelEntry,
    system: SystemInfo,
    profile: ModelProfile,
    user_ctx: Optional[int] = None,
    ram_safety_gb: float = DEFAULT_RAM_SAFETY_GB,
    vram_safety_gb: float = DEFAULT_VRAM_SAFETY_GB,
) -> TunedConfig:
    """Compute a TunedConfig that fits this model on this system."""
    has_gpu = bool(system.gpus) and system.total_vram_gb > 1
    free_vram = max(0.0, system.free_vram_gb)
    n_layers = model.n_layers

    # ---- (0) MoE detection
    expert_count = _moe_expert_count(model)
    is_moe = expert_count > 1
    params_b = extract_params_billion(model.name)

    native_ctx = model.native_context
    profile_max = profile.max_context
    if native_ctx > 0:
        profile_max = min(profile_max, native_ctx)
    target_ctx_for_placement = (user_ctx if user_ctx is not None
                                else profile_max)

    # ---- (1) Model placement
    n_cpu_moe: Optional[int] = None
    if is_moe and has_gpu and n_layers > 0:
        ngl, n_cpu_moe, model_vram, model_ram, full_off = _decide_moe_offload(
            model_size_gb=model.size_gb,
            free_vram_gb=free_vram,
            free_ram_gb=system.free_ram_gb,
            n_layers=n_layers,
            expert_count=expert_count,
            params_billion=params_b,
            target_ctx=target_ctx_for_placement,
            ram_safety_gb=ram_safety_gb,
        )
        if n_cpu_moe == 0:
            n_cpu_moe = None
    else:
        ngl, model_vram, model_ram, full_off = _decide_offload(
            model_size_gb=model.size_gb,
            free_vram_gb=free_vram,
            n_layers=n_layers,
            has_gpu=has_gpu,
            vram_headroom_gb=vram_safety_gb,
        )

    # ---- (2) Remaining KV budget
    # The MoE case is the subtle one: with --n-cpu-moe, attention + KV
    # cache live on the GPU. RAM does NOT contribute to the KV budget,
    # even though the offloaded experts sit there. Counting RAM here (the
    # v2 bug) made the tuner pick contexts of 100k+ tokens that produced
    # 25+ GB of would-be KV — far past what a 16 GB GPU can hold —
    # leading to GGML_ASSERT crashes during model load on Vulkan/AMD.
    effective_vram_safety = (MOE_VRAM_SAFETY_GB if n_cpu_moe is not None
                             else vram_safety_gb)
    free_vram_after = max(0.0, free_vram - effective_vram_safety - model_vram)
    free_ram_after = max(0.0, system.free_ram_gb - ram_safety_gb - model_ram)

    if n_cpu_moe is not None:
        # MoE expert offload: KV is GPU-only.
        kv_budget_gb = free_vram_after
    elif full_off:
        # Dense, fully on GPU: KV is GPU-only.
        kv_budget_gb = free_vram_after
    else:
        # Dense hybrid or CPU-only: KV is split / on CPU.
        kv_budget_gb = free_vram_after + free_ram_after

    # ---- (3) Context + KV quant
    base_kv_mb = kv_per_token_mb_f16(params_b)
    target_ctx = user_ctx if user_ctx is not None else profile_max

    cache_k, cache_v = _pick_kv_quant(
        profile.recommended_kv_quant, target_ctx, base_kv_mb, kv_budget_gb,
    )
    actual_per_tok_mb = base_kv_mb * (
        kv_quant_factor(cache_k) + kv_quant_factor(cache_v)
    ) / 2

    if user_ctx is not None:
        ctx = user_ctx
    else:
        if actual_per_tok_mb > 0:
            max_fit_ctx = int((kv_budget_gb * 1024 * 0.92) / actual_per_tok_mb)
        else:
            max_fit_ctx = profile_max
        ctx = min(profile_max, max_fit_ctx)

    ctx = max(2048, (ctx // 1024) * 1024)
    estimated_kv_gb = (ctx * actual_per_tok_mb) / 1024

    # ---- (3b) Sanity check: did user_ctx push us into VRAM overcommit?
    warning: Optional[str] = None
    if n_cpu_moe is not None or full_off:
        gpu_total = model_vram + estimated_kv_gb + effective_vram_safety
        if gpu_total > free_vram * 0.98:
            warning = (
                f"VRAM budget tight: model {model_vram:.1f} GB + KV "
                f"{estimated_kv_gb:.1f} GB + safety "
                f"{effective_vram_safety:.1f} GB ≈ {gpu_total:.1f} GB of "
                f"{free_vram:.1f} GB free. If launch fails with "
                f"GGML_ASSERT(addr), re-run with a smaller --ctx."
            )

    # ---- (4) Threads
    physical = system.cpu_cores_physical
    logical = system.cpu_cores_logical
    if full_off:
        threads = min(8, physical)
        batch_threads = min(physical, 16)
    elif n_cpu_moe is not None and n_cpu_moe > 0:
        threads = min(physical, 24)
        batch_threads = min(logical, 32)
    elif ngl > 0:
        threads = min(physical, 24)
        batch_threads = min(logical, 32)
    else:
        threads = min(physical, 32)
        batch_threads = min(logical, 64)

    # ---- (4b) Batch
    if model.size_gb > 30:
        batch, ubatch = 512, 128
    elif ctx > 32768 or model.size_gb > 10:
        batch, ubatch = 1024, 512
    else:
        batch, ubatch = 2048, 512

    # ---- (4c) mlock + no_mmap (paired)
    ram_resident_gb = model_ram
    mlock = (
        not full_off
        and system.total_ram_gb > 32
        and ram_resident_gb > 0
        and ram_resident_gb < (system.free_ram_gb - 8)
    )
    no_mmap = mlock

    # ---- (4d) Multi-GPU tensor split. Skipped for MoE.
    tensor_split: Optional[str] = None
    main_gpu: Optional[int] = None
    if has_gpu and len(system.gpus) > 1 and n_cpu_moe is None:
        sizes = [g.total_vram_mb for g in system.gpus]
        total = sum(sizes)
        if total > 0:
            tensor_split = ",".join(f"{s/total:.3f}" for s in sizes)
            main_gpu = max(range(len(system.gpus)),
                           key=lambda i: system.gpus[i].total_vram_mb)

    # ---- (4e) NUMA
    numa = None
    if not has_gpu and system.cpu_cores_physical >= 16:
        numa = "distribute"

    # ---- (4f) Sampling
    sd = profile.sampling or {}
    sampling = {
        "temperature": float(sd.get("temperature", 0.7)),
        "top_k": int(sd.get("top_k", 40)),
        "top_p": float(sd.get("top_p", 0.9)),
        "min_p": float(sd.get("min_p", 0.05)),
        "repeat_penalty": float(sd.get("repeat_penalty", 1.05)),
        "presence_penalty": float(sd.get("presence_penalty", 0.0)),
    }

    return TunedConfig(
        ctx=ctx,
        ngl=ngl,
        threads=threads,
        batch_threads=batch_threads,
        batch=batch,
        ubatch=ubatch,
        cache_k=cache_k,
        cache_v=cache_v,
        flash_attn=True,
        sampling=sampling,
        mlock=mlock,
        no_mmap=no_mmap,
        numa=numa,
        tensor_split=tensor_split,
        main_gpu=main_gpu,
        n_cpu_moe=n_cpu_moe,
        is_moe=is_moe,
        expert_count=expert_count,
        estimated_model_vram_gb=model_vram,
        estimated_model_ram_gb=model_ram,
        estimated_kv_gb=estimated_kv_gb,
        full_offload=full_off,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Build command

def build_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    server_binary: str = "llama-server",
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    cmd: List[str] = [
        server_binary,
        "-m", str(model.path),
        "-c", str(config.ctx),
        "-ngl", str(config.ngl),
        "-t", str(config.threads),
        "-tb", str(config.batch_threads),
        "-b", str(config.batch),
        "-ub", str(config.ubatch),
        "-ctk", config.cache_k,
        "-ctv", config.cache_v,
        "--host", host,
        "--port", str(port),
    ]

    if config.flash_attn:
        cmd += ["-fa", "on"]
    if config.numa:
        cmd += ["--numa", config.numa]
    if config.mlock:
        cmd.append("--mlock")
    if config.no_mmap:
        cmd.append("--no-mmap")
    if config.n_cpu_moe is not None and config.n_cpu_moe > 0:
        cmd += ["--n-cpu-moe", str(config.n_cpu_moe)]
    if config.tensor_split:
        cmd += ["--tensor-split", config.tensor_split]
    if config.main_gpu is not None:
        cmd += ["--main-gpu", str(config.main_gpu)]

    s = config.sampling
    cmd += [
        "--temp", str(s["temperature"]),
        "--top-k", str(s["top_k"]),
        "--top-p", str(s["top_p"]),
        "--min-p", str(s["min_p"]),
        "--repeat-penalty", str(s["repeat_penalty"]),
    ]
    pp = s.get("presence_penalty", 0.0)
    if pp:
        cmd += ["--presence-penalty", str(pp)]

    if model.mmproj is not None:
        cmd += ["--mmproj", str(model.mmproj)]

    if profile.extra_args:
        cmd.extend(profile.extra_args)
    if extra_args:
        cmd.extend(extra_args)

    return cmd