"""Compute an optimal llama-server configuration for a given model
on the detected hardware, then build the command line.

v2 changes:
  * MoE detection via GGUF metadata (<arch>.expert_count). Routed-MoE
    models switch from layer-based --ngl partial offload to a
    `-ngl 999 --n-cpu-moe N` strategy, which keeps attention + dense
    parts on GPU and pushes only the rarely-used routed-expert FFNs
    to CPU. For models like Qwen3.6-35B-A3B or gpt-oss-120B this is
    much faster than naive layer offload.
  * --mlock and --no-mmap are now paired: mlock alone has limited
    effect when the model is mmap'd (pages can still be evicted),
    so the tuner emits both whenever it decides to keep the
    CPU-resident portion pinned.
  * mlock decision uses model_ram (the RAM-resident slice), not the
    full model size — important for MoE where only the experts live
    on CPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hardware import SystemInfo
from scanner import ModelEntry
from settings_loader import ModelProfile


# ---------------------------------------------------------------------------
# Model-size helpers

def extract_params_billion(name: str) -> float:
    """Extract parameter count in billions from a model filename.

    Handles patterns like 9B, 0.8B, 27B, 128B, and Gemma-style E2B/E4B.
    For MoE names like '35B-A3B' returns the larger number (total params),
    since KV cache size scales with full attention dimensions, not active ones.
    """
    matches = re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*B(?![a-zA-Z0-9_])",
                         name)
    if matches:
        return max(float(m) for m in matches)

    m = re.search(r"E(\d+(?:\.\d+)?)B", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def kv_per_token_mb_f16(params_billion: float) -> float:
    """Approximate KV-cache memory per token at f16 quant, in MB.

    Coarse heuristic by model size class. Real values depend on
    n_layers * n_kv_heads * head_dim, which we may not always know.
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
    """Return expert_count from GGUF metadata, or 0 if dense / unknown.

    For MoE models llama.cpp metadata has e.g. `qwen3moe.expert_count = 128`
    or `gpt_oss.expert_count = 128`. A value > 1 means the model has
    routed experts and benefits from --n-cpu-moe instead of plain
    layer-based offload via --ngl.
    """
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
    # Fallback: look for any *.expert_count in the metadata blob
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

    # MoE-specific. When set, llama-server gets `-ngl 999 --n-cpu-moe N`,
    # which keeps attention + dense parts on GPU and pushes the routed
    # experts of N (highest-numbered) layers to CPU. None means the model
    # is dense, or fits fully on GPU, or runs CPU-only.
    n_cpu_moe: Optional[int] = None
    is_moe: bool = False
    expert_count: int = 0

    # Estimates for display
    estimated_model_vram_gb: float = 0.0
    estimated_model_ram_gb: float = 0.0
    estimated_kv_gb: float = 0.0
    full_offload: bool = False


# ---------------------------------------------------------------------------
# Internal helpers

def _decide_offload(
    model_size_gb: float,
    free_vram_gb: float,
    n_layers: int,
    has_gpu: bool,
    vram_headroom_gb: float = 1.5,
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
    vram_safety_gb: float = 1.5,
    ram_safety_gb: float = 4.0,
) -> Tuple[int, Optional[int], float, float, bool]:
    """MoE placement using --n-cpu-moe.

    Strategy: keep attention + dense parts + as many routed-expert layers
    as possible on GPU, push the rest of the routed experts to CPU RAM
    via --n-cpu-moe N. Always uses -ngl 999 conceptually so attention
    is GPU-resident even when most experts are on CPU.

    Returns (ngl, n_cpu_moe, model_vram_gb, model_ram_gb, full_offload).
      * n_cpu_moe == 0 means everything fits on GPU → full offload, the
        caller should treat n_cpu_moe as None and not emit the flag.
      * n_cpu_moe == n_layers means all experts on CPU.
      * If RAM is too tight even for the experts that need to go to CPU,
        the caller will fall back to the regular dense path.
    """
    # Reserve a chunk of VRAM for the KV cache so we don't push so much
    # MoE onto the GPU that there's no room left for context. Estimate
    # against the target context at q8_0 KV (factor 0.55), and cap at
    # 40% of free VRAM so a pathologically large context doesn't push
    # all experts onto CPU.
    base_kv_mb = kv_per_token_mb_f16(params_billion)
    kv_reserve_gb = (target_ctx * base_kv_mb * 0.55) / 1024
    kv_reserve_gb = min(kv_reserve_gb, free_vram_gb * 0.4)

    # "Always-on" weights: attention, embeddings, dense FFN, shared
    # experts. The exact ratio depends on architecture (gpt-oss-120B is
    # ~4%, Qwen3-MoE ~10%, GLM-4.7 in between). 10% is a slightly
    # conservative upper bound that errs toward more CPU offload — safer
    # than under-reserving and OOM'ing at runtime.
    shared_overhead_gb = model_size_gb * 0.10
    per_layer_expert_gb = max(0.001,
                              (model_size_gb - shared_overhead_gb) / n_layers)

    usable_for_experts = (free_vram_gb - vram_safety_gb
                          - kv_reserve_gb - shared_overhead_gb)

    if usable_for_experts < 0:
        # Even the shared parts compete with KV for the VRAM slot. Try
        # the most extreme MoE-offload anyway (all routed experts to
        # CPU) — llama.cpp will deal with whether the shared parts spill.
        if free_ram_gb - ram_safety_gb < model_size_gb - shared_overhead_gb:
            # Even RAM is too tight for the experts. Caller will fall
            # back to dense logic (which usually means CPU only).
            return 999, n_layers, shared_overhead_gb, model_size_gb, False
        return 999, n_layers, shared_overhead_gb, \
            model_size_gb - shared_overhead_gb, False

    layers_on_gpu = int(usable_for_experts / per_layer_expert_gb)
    layers_on_gpu = max(0, min(n_layers, layers_on_gpu))
    n_cpu_moe = n_layers - layers_on_gpu

    if n_cpu_moe == 0:
        # Everything fits on GPU → tell caller to use full offload, no
        # --n-cpu-moe flag needed.
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

    budget_mb = kv_budget_gb * 1024 * 0.92  # 8% safety margin
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
    ram_safety_gb: float = 4.0,
    vram_safety_gb: float = 1.5,
) -> TunedConfig:
    """Compute a TunedConfig that fits this model on this system.

    Strategy:
      0. Detect MoE via GGUF metadata.
      1. Place the model:
         * MoE on GPU? → -ngl 999 + --n-cpu-moe N (N depending on VRAM)
         * Dense, fits on GPU? → full offload
         * Dense, partial fit? → -ngl per-layer split
         * Else → CPU only
      2. Compute remaining VRAM + RAM budget for KV cache.
      3. Choose KV quant + context length to fit within that budget.
      4. Set threads / batch / mlock+no-mmap (paired) / numa / multi-GPU.
    """
    has_gpu = bool(system.gpus) and system.total_vram_gb > 1
    free_vram = max(0.0, system.free_vram_gb)
    n_layers = model.n_layers  # 0 if unknown

    # ---- (0) MoE detection
    expert_count = _moe_expert_count(model)
    is_moe = expert_count > 1

    params_b = extract_params_billion(model.name)

    # Target context for placement reservation. We don't know the final
    # ctx yet (it depends on KV budget which depends on placement) — so
    # use the profile cap as a conservative upper bound here. The actual
    # ctx is recomputed below.
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
            vram_safety_gb=vram_safety_gb,
            ram_safety_gb=ram_safety_gb,
        )
        # n_cpu_moe == 0 means everything fits on GPU; treat as full offload
        # and don't emit --n-cpu-moe.
        if n_cpu_moe == 0:
            n_cpu_moe = None
    else:
        # Dense, no GPU, or MoE without layer metadata → fall through to
        # the original dense logic.
        ngl, model_vram, model_ram, full_off = _decide_offload(
            model_size_gb=model.size_gb,
            free_vram_gb=free_vram,
            n_layers=n_layers,
            has_gpu=has_gpu,
            vram_headroom_gb=vram_safety_gb,
        )

    # ---- (2) Remaining KV budget
    free_vram_after = max(0.0, free_vram - vram_safety_gb - model_vram)
    free_ram_after = max(0.0, system.free_ram_gb - ram_safety_gb - model_ram)
    kv_budget_gb = free_vram_after + free_ram_after

    # ---- (3) Context + KV quant
    base_kv_mb = kv_per_token_mb_f16(params_b)

    if user_ctx is not None:
        target_ctx = user_ctx
    else:
        target_ctx = profile_max

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

    # ---- (4) Threads
    physical = system.cpu_cores_physical
    logical = system.cpu_cores_logical
    if full_off:
        # Mostly GPU-bound; few threads needed
        threads = min(8, physical)
        batch_threads = min(physical, 16)
    elif n_cpu_moe is not None and n_cpu_moe > 0:
        # MoE with experts on CPU: the routed experts ARE the hot path on
        # CPU side, so use plenty of threads for those FFN multiplications.
        # Logical (SMT) threads help for batch prefill on Intel hybrid CPUs.
        threads = min(physical, 24)
        batch_threads = min(logical, 32)
    elif ngl > 0:
        # Dense hybrid
        threads = min(physical, 24)
        batch_threads = min(logical, 32)
    else:
        # Pure CPU — use everything
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
    # mlock alone has limited effect when the model is mmap'd from disk:
    # the OS can still evict pages under memory pressure because the
    # backing file is the source of truth. --no-mmap forces a real RAM
    # load that mlock can then pin. So whenever we decide to lock, we
    # also bypass mmap.
    #
    # The headroom check uses model_ram (the actually CPU-resident slice)
    # rather than model.size_gb — important for MoE where only the
    # offloaded experts live on CPU, and for hybrid placement where only
    # the non-offloaded layers live on CPU.
    ram_resident_gb = model_ram
    mlock = (
        not full_off
        and system.total_ram_gb > 32
        and ram_resident_gb > 0  # nothing on CPU side → nothing to lock
        and ram_resident_gb < (system.free_ram_gb - 8)
    )
    no_mmap = mlock

    # ---- (4d) Multi-GPU tensor split
    # Skip tensor-split when we're using --n-cpu-moe: the interaction
    # between per-GPU split and per-layer expert offload is not something
    # the tuner currently models well. llama.cpp will pick a sensible
    # default (typically: dense parts split, experts on the highest-VRAM
    # GPU plus CPU). Safer to leave that alone than to fight it.
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
    # MoE expert offload — emit only when we actually decided to push
    # some routed experts to CPU. Goes together with -ngl 999 above.
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
