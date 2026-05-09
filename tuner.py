from __future__ import annotations

import re
import platform
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from hardware import SystemInfo
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
MOE_PLACEMENT_CTX_TARGET = PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME].moe_placement_ctx_target
MOE_KV_RESERVE_FRAC = 0.06


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
    """Approximate KV-cache memory per token at f16 quant, in MB.

    Fallback heuristic when GGUF metadata is unavailable. NOTE: this is
    calibrated for *dense* models. For MoE models (e.g. Qwen3.6-35B-A3B,
    where only ~3B params are active per token), this heuristic
    overestimates KV by roughly an order of magnitude — prefer
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


def kv_per_token_mb_from_metadata(md: Dict[str, Any]) -> float:
    """Compute exact f16 K+V cache size per token (MB) from GGUF metadata.

    Formula:
        bytes/token = n_attention_layers * n_kv_heads * (key_length + value_length) * 2

    The trailing 2 is K+V at FP16 (2 bytes each). With GQA, `n_kv_heads`
    is smaller than `n_heads`, which is why MoE models like
    Qwen3.6-35B-A3B have a small KV footprint despite a high total
    parameter count — they share KV across many query heads.

    For *hybrid* Mamba/Transformer models (Nemotron-H, Jamba, …) only
    a fraction of the layers actually carry KV cache. We multiply by
    that fraction (via ``metadata_attention_layer_count``) instead of
    the total block count, otherwise we'd over-reserve VRAM by 4–5×
    on these architectures and pessimise placement.

    Returns 0.0 when metadata is incomplete; the caller should then fall
    back to the params-billion heuristic.
    """
    if not md:
        return 0.0
    arch = md.get("general.architecture") or ""
    if not arch:
        return 0.0

    def _int(key: str) -> int:
        v = md.get(f"{arch}.{key}", 0)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    # Use the attention-bearing layer count for hybrids; for pure
    # Transformer this equals block_count and behaves as before.
    from scanner import metadata_attention_layer_count
    n_layers = metadata_attention_layer_count(md)
    if n_layers <= 0:
        # Fallback for older models / incomplete metadata: use total blocks.
        n_layers = _int("block_count")

    n_heads = _int("attention.head_count")
    n_kv_heads = _int("attention.head_count_kv")
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


def _resolve_kv_per_token_mb(
    model: ModelEntry, params_billion: float
) -> float:
    """Pick the best KV-per-token estimate available.

    Preference: exact GGUF metadata first (precise; works for MoE),
    falling back to params-based heuristic (for tests / metadata-less
    models).
    """
    md_estimate = kv_per_token_mb_from_metadata(model.metadata)
    if md_estimate > 0:
        return md_estimate
    return kv_per_token_mb_f16(params_billion)


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

    no_context_shift: bool = False

    # RoPE-Scaling: aktiviert wenn ctx > native_ctx und YaRN/rope-scaling
    # verwendet werden soll (optional, nur für Modelle die es unterstützen).
    rope_scaling: bool = False
    rope_scale_factor: float = 1.0  # z.B. 4.0 für yarn mit 4x scaling

    # Active performance target name ("safe" / "balanced" / "throughput").
    # Set by compute_config so display code can show what was applied.
    performance_target: str = DEFAULT_TARGET_NAME

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
        residual_overhead = model_size_gb * 0.02  # Reduced overhead
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
    base_kv_per_token_mb: float = 0.0,
    ram_safety_gb: float = DEFAULT_RAM_SAFETY_GB,
    moe_vram_safety_gb: float = MOE_VRAM_SAFETY_GB,
    moe_placement_ctx_target: int = MOE_PLACEMENT_CTX_TARGET,
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
    per_layer_expert_gb = max(0.001,
                              (model_size_gb - shared_overhead_gb) / n_layers)

    # ---- KV reservation in VRAM (q5_0 assumption) -----------------------
    # Cap at moe_placement_ctx_target so we don't pessimise layer placement
    # for huge profile_max values (Qwen3.6 → 262k, but most users run 32k).
    kv_reservation_ctx = max(2048, min(target_ctx,
                                       moe_placement_ctx_target))
    kv_reserve_gb = (
        kv_reservation_ctx * base_kv_per_token_mb * kv_quant_factor("q5_0")
    ) / 1024.0

    # Layer placement uses VRAM left over AFTER KV + shared overhead.
    usable_for_experts = (free_vram_gb - moe_vram_safety_gb
                          - shared_overhead_gb - kv_reserve_gb)

    if usable_for_experts < 0:
        # Not even the shared overhead + KV fits → everything via mmap/RAM.
        if free_ram_gb - ram_safety_gb < model_size_gb - shared_overhead_gb:
            return 999, n_layers, shared_overhead_gb, model_size_gb, False
        return 999, n_layers, shared_overhead_gb, \
            model_size_gb - shared_overhead_gb, False

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


def _pick_kv_quant(
    profile_recommended: str,
    target_ctx: int,
    base_kv_per_token_mb: float,
    kv_budget_gb: float,
    model_max_ctx: int = 0,  # native_ctx aus GGUF-Metadata (0 = keine Begrenzung)
) -> Tuple[str, str]:
    """Wähle die beste KV-Quantisierung die target_ctx unterstützt.
    
    Priorität: höchste Qualität (q8 > q5 > q4), die den target_ctx in das
    verfügbare Budget passt. Wenn model_max_ctx > 0 und target_ctx darüber
    hinausgeht, wird target_ctx auf model_max_ctx beschränkt (sonst muss
    rope-scaling aktiviert werden).
    """
    # Beschränke target_ctx auf Modell-Maximum wenn nötig
    if model_max_ctx > 0 and target_ctx > model_max_ctx:
        target_ctx = model_max_ctx

    # Reihenfolge: von hochwertig nach niedrig — erste die passt gewinnt
    order = ["q8_0", "q5_0", "q4_0"]
    rec = profile_recommended.lower()
    if rec in order:
        order.remove(rec)
        order.insert(0, rec)

    budget_mb = kv_budget_gb * 1024 * 0.98
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
    draft_model: Optional[ModelEntry] = None,
    user_ctx: Optional[int] = None,
    ram_safety_gb: Optional[float] = None,
    vram_safety_gb: Optional[float] = None,
    force_mlock: bool = False,
    perf_target: Optional[PerformanceTarget] = None,
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

    has_gpu = bool(system.gpus) and system.total_vram_gb > 1
    free_vram = max(0.0, system.free_vram_gb)
    n_layers = model.n_layers

    # ---- (0) MoE detection
    expert_count = _moe_expert_count(model)
    is_moe = expert_count > 1
    params_b = extract_params_billion(model.name)

    # ---- (0.1) KV per-token: MUST be defined before any branch uses it.
    # This is the bug that caused crashes on selection of any non-Qwen
    # model in v3.x — base_kv_mb was previously only set inside the
    # rope-scaling branch, but referenced unconditionally further below.
    base_kv_mb = _resolve_kv_per_token_mb(model, params_b)

    native_ctx = model.native_context  # GGUF metadata: model's native ctx
    
    # RoPE-Scaling Konfiguration aus Profil lesen
    profile_rope_scale = profile.rope_scale_enabled
    profile_rope_max = profile.rope_scale_max_ctx  # Standard: 1M
    profile_rope_factor = profile.rope_scale_factor  # Standard: 4.0
    
    rope_scaled_ctx = 0  # Wird später berechnet (braucht free_vram_after/free_ram_after)
    rope_scaling_active = False  # Flag für build_command
    
    profile_max = profile.max_context
    if native_ctx > 0:
        profile_max = min(profile_max, native_ctx)
    target_ctx_for_placement = (user_ctx if user_ctx is not None
                                else profile_max)

    # ---- (0.5) Calculate VRAM reserved for Vision + Draft models
    # These MUST be on GPU for optimal performance.
    vision_vram_gb = 0.0
    draft_vram_gb = 0.0
    
    if model.mmproj is not None:
        # Vision model (mmproj) — estimate from file size
        try:
            mmproj_size_bytes = model.mmproj.stat().st_size
            vision_vram_gb = mmproj_size_bytes / (1024 ** 3)
        except (OSError, AttributeError):
            # Fallback: ~6 GB for typical F16 mmproj files
            vision_vram_gb = 6.0
    
    if draft_model is not None:
        # Draft model — must fit in VRAM for speculative decoding to work well
        draft_vram_gb = draft_model.size_gb

    # Effective VRAM available for main model placement
    effective_free_vram = free_vram - vision_vram_gb - draft_vram_gb
    if effective_free_vram < 0:
        effective_free_vram = 0.0

    # ---- (1) Model placement
    n_cpu_moe: Optional[int] = None
    if is_moe and has_gpu and n_layers > 0:
        ngl, n_cpu_moe, model_vram, model_ram, full_off = _decide_moe_offload(
            model_size_gb=model.size_gb,
            free_vram_gb=effective_free_vram,
            free_ram_gb=system.free_ram_gb,
            n_layers=n_layers,
            expert_count=expert_count,
            params_billion=params_b,
            target_ctx=target_ctx_for_placement,
            base_kv_per_token_mb=base_kv_mb,
            ram_safety_gb=ram_safety_gb,
            moe_vram_safety_gb=perf_target.moe_vram_safety_gb,
            moe_placement_ctx_target=perf_target.moe_placement_ctx_target,
        )

        # ---- Two-pass placement fallback ---------------------------------
        # If the first pass dumped *every* expert layer to CPU but >4 GB
        # of VRAM is still free, the KV reservation was clearly too
        # pessimistic for this model. Retry once with the placement
        # target halved (down to a 16k floor). This is a defensive net
        # for hybrid architectures we don't recognise yet, or for
        # quantisations where our heuristic mis-estimates KV footprint.
        if (n_cpu_moe is not None
                and n_layers > 0
                and n_cpu_moe >= n_layers
                and effective_free_vram > 4.0
                and perf_target.moe_placement_ctx_target > 16384):
            shrunk_target = max(
                16384, perf_target.moe_placement_ctx_target // 2)
            ngl_2, cpu_moe_2, vram_2, ram_2, full_2 = _decide_moe_offload(
                model_size_gb=model.size_gb,
                free_vram_gb=effective_free_vram,
                free_ram_gb=system.free_ram_gb,
                n_layers=n_layers,
                expert_count=expert_count,
                params_billion=params_b,
                target_ctx=target_ctx_for_placement,
                base_kv_per_token_mb=base_kv_mb,
                ram_safety_gb=ram_safety_gb,
                moe_vram_safety_gb=perf_target.moe_vram_safety_gb,
                moe_placement_ctx_target=shrunk_target,
            )
            # Only adopt the second pass if it actually placed layers on GPU.
            if cpu_moe_2 is not None and cpu_moe_2 < n_cpu_moe:
                ngl, n_cpu_moe, model_vram, model_ram, full_off = (
                    ngl_2, cpu_moe_2, vram_2, ram_2, full_2)

        if n_cpu_moe == 0:
            n_cpu_moe = None
    else:
        ngl, model_vram, model_ram, full_off = _decide_offload(
            model_size_gb=model.size_gb,
            free_vram_gb=effective_free_vram,
            n_layers=n_layers,
            has_gpu=has_gpu,
            vram_headroom_gb=vram_safety_gb,
        )

    # ---- (2) Remaining KV budget — include vision/draft VRAM in total
    effective_vram_safety = (perf_target.moe_vram_safety_gb
                             if n_cpu_moe is not None
                             else vram_safety_gb)
    free_vram_after = max(0.0, free_vram - effective_vram_safety
                          - model_vram - vision_vram_gb - draft_vram_gb)
    free_ram_after = max(0.0, system.free_ram_gb - ram_safety_gb - model_ram)

    # KV-cache placement rules:
    #   - MoE on GPU: KV must live in VRAM only. The Vulkan backend
    #     crashes with GGML_ASSERT(addr) when MoE KV spills to RAM.
    #   - Dense full-offload: KV in VRAM only (it's already on GPU).
    #   - Dense partial / CPU-only: KV may use VRAM + RAM.
    if is_moe and has_gpu:
        kv_budget_gb = free_vram_after
    elif full_off:
        kv_budget_gb = free_vram_after
    else:
        kv_budget_gb = free_vram_after + free_ram_after

    # ---- (2.5) RoPE-Scaling (YaRN) auto-detection
    # Aktiviere RoPE-Scaling automatisch wenn:
    # 1. Modell RoPE-Scaling unterstützt (qwen2 etc.)
    # 2. Genügend Speicher für Context > native_ctx vorhanden ist
    # 3. Entweder profil-configured (rope_scale.enabled=true) ODER
    #    berechneter max_fit_ctx überschreitet native_ctx
    rope_scaled_ctx = 0
    rope_scaling_active = False

    if (model.supports_rope_scale
            and native_ctx > 0
            and native_ctx < profile_rope_max):
        # KV-Speicherbedarf pro Token (q5_0 als Entscheidungsgrundlage)
        kv_per_tok_q5 = base_kv_mb * kv_quant_factor("q5_0")

        # Bestimme gewünschten Context (user-specified oder profile-basiert)
        effective_profile_max = profile.max_context
        if native_ctx > 0:
            effective_profile_max = min(effective_profile_max, native_ctx)
        desired_ctx = user_ctx if user_ctx is not None else effective_profile_max

        # Wenn gewünschter Context das native Limit überschreitet
        if desired_ctx > native_ctx:
            # Prüfe ob Speicher vorhanden ist
            rope_kv_gb = (desired_ctx * kv_per_tok_q5) / 1024
            total_available = free_vram_after + free_ram_after

            # Aktiviere RoPE-Scaling wenn >= 90% des Bedarfs verfügbar
            if (profile_rope_scale or total_available >= rope_kv_gb * 1.1):
                rope_scaled_ctx = min(desired_ctx, profile_rope_max)
                rope_scaling_active = True
    
    # ---- (3) Context + KV quant
    target_ctx = user_ctx if user_ctx is not None else profile_max

    # Bestimme das effektive Modell-Maximum für die KV-Quantisierung:
    # - rope_scaled_ctx: erweiterbares Maximum via YaRN (wenn aktiviert)
    # - native_ctx: natives Maximum des Modells (aus GGUF)
    model_ctx_limit = rope_scaled_ctx if rope_scaled_ctx > 0 else native_ctx
    if model_ctx_limit <= 0:
        model_ctx_limit = profile_max

    cache_k, cache_v = _pick_kv_quant(
        profile.recommended_kv_quant, target_ctx, base_kv_mb, kv_budget_gb,
        model_ctx_limit,
    )
    actual_per_tok_mb = base_kv_mb * (
        kv_quant_factor(cache_k) + kv_quant_factor(cache_v)
    ) / 2

    if user_ctx is not None:
        # User-specified context — respect it but clamp to model limits
        ctx = user_ctx
        if model_ctx_limit > 0 and ctx > model_ctx_limit:
            ctx = model_ctx_limit
    else:
        # Berechne den maximal möglichen Kontext basierend auf dem verfügbaren
        # KV-Cache-Budget. Verwende 99,5% des Budgets um nahe an die physische
        # Grenze zu kommen — llama.cpp alloziert den KV-cache nur bei
        # tatsächlicher Nutzung.
        if actual_per_tok_mb > 0:
            max_fit_ctx = int((kv_budget_gb * 1024 * 0.995) / actual_per_tok_mb)
        else:
            max_fit_ctx = profile_max
        
        # Beschränke auf das Modell-Maximum (native oder rope-scaled)
        if model_ctx_limit > 0:
            ctx = min(max_fit_ctx, model_ctx_limit)
        else:
            ctx = min(max_fit_ctx, profile_max * 3)

    ctx = max(2048, (ctx // 1024) * 1024)
    estimated_kv_gb = (ctx * actual_per_tok_mb) / 1024

    # ---- (3b) VRAM Overcommit Warning
    warning: Optional[str] = None
    if n_cpu_moe is not None or full_off:
        gpu_total = model_vram + estimated_kv_gb + effective_vram_safety
        if gpu_total > free_vram * 0.98:
            warning = (
                f"VRAM budget tight: model {model_vram:.1f} GB + KV "
                f"{estimated_kv_gb:.1f} GB + safety "
                f"{effective_vram_safety:.1f} GB ≈ {gpu_total:.1f} GB of "
                f"{free_vram:.1f} GB free."
            )

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

    # ---- (4b) Batch — ubatch=1024 für bessere Performance bei großen Kontexten
    if model.size_gb > 30:
        batch, ubatch = 1024, 1024
    elif ctx > 32768 or model.size_gb > 10:
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
        mlock = (
            (has_enough_vram or vram_resident_gb > 0)
            and (is_windows and is_admin or not is_windows)
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

    # ---- (4d) NUMA — immer aktivieren bei genügend Kernen für bessere Performance
    numa = None
    if system.cpu_cores_physical >= 16:
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

    # no_context_shift für bessere Performance bei grossen Kontexten aktivieren
    no_context_shift = (ctx >= 32768) or full_off

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
        no_context_shift=no_context_shift,
        rope_scaling=rope_scaling_active,
        rope_scale_factor=float(profile_rope_factor) if rope_scaling_active else 1.0,
        performance_target=perf_target.name,
        warning=warning,
    )


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

    # Add draft model if provided (MTP speculative decoding)
    # NOTE: -md MUST come BEFORE --spec-type because llama-server parses
    # arguments left-to-right. If --spec-type is seen before -md, the server
    # thinks no draft model was given and aborts with:
    #   "unknown speculative decoding type without draft model"
    if draft_model is not None:
        draft_val = getattr(profile, 'draft_max', 0) or 3
        cmd += ["-md", str(draft_model.path)]
        cmd += ["--spec-type", "mtp"]  # Erforderlich für ik_llama.cpp
        cmd += ["-ngld", "99"]
        cmd += ["--draft-max", str(draft_val)] # Dieser Fork nutzt oft wieder --draft-max
        cmd += ["--draft-p-min", str(getattr(profile, 'draft_p_min', 0.0) or 0.0)]

    if config.flash_attn:
        cmd += ["-fa", "on"]
    if config.numa:
        cmd += ["--numa", config.numa]
    if config.mlock:
        cmd.append("--mlock")
    if config.no_mmap:
        cmd.append("--no-mmap")
    if config.no_context_shift:
        cmd.append("--no-context-shift")
    
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

    # Thinking/Reasoning-Modus (Gemma 4, DeepSeek, etc.)
    # Thinking wird über Prompt-Tags gesteuert (<|think|>), nicht über CLI-Argumente.
    # use_thinking ist ein internes Flag - extra_args werden immer angehängt:

    if profile.extra_args:
        cmd.extend(profile.extra_args)
    if extra_args:
        cmd.extend(extra_args)

    return cmd