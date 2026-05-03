"""Compute an optimal llama-server configuration for a given model
on the detected hardware, then build the command line."""
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
    # Standard: digits + B, with non-letter boundary
    matches = re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*B(?![a-zA-Z0-9_])",
                         name)
    if matches:
        return max(float(m) for m in matches)

    # Gemma "effective" size markers (E2B, E4B, E10B, ...)
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
    """Return (ngl, model_vram_gb, model_ram_gb, full_offload)."""
    if not has_gpu or free_vram_gb < 1.0:
        return 0, 0.0, model_size_gb, False

    usable = max(0.0, free_vram_gb - vram_headroom_gb)
    if usable >= model_size_gb:
        # Plenty of room — full offload
        return 999, model_size_gb, 0.0, True

    if usable < 0.5:
        return 0, 0.0, model_size_gb, False

    # Partial offload
    if n_layers > 0:
        per_layer_gb = model_size_gb / n_layers
        ngl = int(usable / per_layer_gb)
        ngl = max(0, min(n_layers, ngl))
        model_vram = ngl * per_layer_gb
        # Embeddings + output layer live on CPU when not fully offloaded
        residual_overhead = model_size_gb * 0.03
        model_ram = (n_layers - ngl) * per_layer_gb + residual_overhead
        return ngl, model_vram, model_ram, False

    # No layer count known — assume ~50 layers, set ngl proportionally
    estimated_layers = 50
    fraction = usable / model_size_gb
    ngl = max(0, int(fraction * estimated_layers))
    return ngl, usable, max(0.0, model_size_gb - usable), False


def _pick_kv_quant(
    profile_recommended: str,
    target_ctx: int,
    base_kv_per_token_mb: float,
    kv_budget_gb: float,
) -> Tuple[str, str]:
    """Pick the best-quality KV cache quant that fits the target context."""
    # Quality order: q8 > q5 > q4
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
    # Even q4 doesn't fit — return q4 anyway
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
      1. Place the model: full GPU offload if it fits, else partial, else CPU.
      2. Compute remaining VRAM + RAM budget for KV cache.
      3. Choose KV quant + context length to fit within that budget.
      4. Set threads / batch / mlock / numa / multi-GPU split.
    """
    has_gpu = bool(system.gpus) and system.total_vram_gb > 1

    # ---- (1) Model placement
    free_vram = max(0.0, system.free_vram_gb)
    n_layers = model.n_layers  # 0 if unknown
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
    params_b = extract_params_billion(model.name)
    base_kv_mb = kv_per_token_mb_f16(params_b)

    # Cap the requested context by the model's training-time context length,
    # if known from GGUF metadata. (User can still override via --ctx.)
    native_ctx = model.native_context  # 0 if unknown
    profile_max = profile.max_context
    if native_ctx > 0:
        profile_max = min(profile_max, native_ctx)

    if user_ctx is not None:
        target_ctx = user_ctx
    else:
        target_ctx = profile_max  # try to use the full supported context

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

    # Round to multiples of 1024 for nicer numbers; minimum 2048
    ctx = max(2048, (ctx // 1024) * 1024)
    estimated_kv_gb = (ctx * actual_per_tok_mb) / 1024

    # ---- (4) Threads
    physical = system.cpu_cores_physical
    logical = system.cpu_cores_logical
    if full_off:
        # Mostly GPU-bound; few threads needed
        threads = min(8, physical)
        batch_threads = min(physical, 16)
    elif ngl > 0:
        # Hybrid
        threads = min(physical, 24)
        batch_threads = min(logical, 32)
    else:
        # Pure CPU — use everything
        threads = min(physical, 32)
        batch_threads = min(logical, 64)

    # ---- (4b) Batch
    # True giants stay conservative to avoid prefill OOM. Mid-size / long-ctx
    # gets a healthier batch (the v1 numbers were way too small for prefill
    # throughput on a 16+ GB GPU).
    if model.size_gb > 30:
        batch, ubatch = 512, 128
    elif ctx > 32768 or model.size_gb > 10:
        batch, ubatch = 1024, 512
    else:
        batch, ubatch = 2048, 512

    # ---- (4c) mlock — only worth it when model is RAM-resident and we
    # have plenty of RAM headroom
    mlock = (
        not full_off
        and system.total_ram_gb > 32
        and model.size_gb < (system.free_ram_gb - 8)
    )

    # ---- (4d) Multi-GPU tensor split (proportional to total VRAM)
    tensor_split: Optional[str] = None
    main_gpu: Optional[int] = None
    if has_gpu and len(system.gpus) > 1:
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
        no_mmap=False,
        numa=numa,
        tensor_split=tensor_split,
        main_gpu=main_gpu,
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
    # Only emit presence_penalty when it's actually non-zero — llama-server
    # defaults to 0.0 already, so emitting it everywhere just bloats the CLI.
    pp = s.get("presence_penalty", 0.0)
    if pp:
        cmd += ["--presence-penalty", str(pp)]

    if model.mmproj is not None:
        cmd += ["--mmproj", str(model.mmproj)]

    # Profile-defined extras (e.g. fork-specific flags)
    if profile.extra_args:
        cmd.extend(profile.extra_args)
    if extra_args:
        cmd.extend(extra_args)

    return cmd
