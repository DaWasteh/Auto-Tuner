"""Bounded, visible Auto fallback for single-device, memory-constrained MoE.

No hardware allocations or saved-settings writes. Uses scanned expert PREFIX
sizes, not the historical 8%-shared heuristic. Multi-GPU/UMA/unknown scans keep
the existing planner. All launch consumers must use the resulting config flags.
"""

from dataclasses import replace


def eligible(model, system):
    spans = (model.metadata or {}).get("__moe_expert_bytes_by_layer__")
    return bool(
        system.os_name.lower().startswith("windows")
        and len(system.gpus) == 1
        and not system.has_unified_memory
        and str(system.gpus[0].runtime_backend or "").lower() == "vulkan"
        and system.total_vram_gb <= 12
        and system.total_ram_gb <= 32
        and model.placement_size_gb > system.free_vram_gb
        and model.read_lazy_size_bytes == 0
        and isinstance(spans, list)
        and len(spans) == model.n_layers
        and all(isinstance(v, int) and v >= 0 for v in spans)
        and 0 < sum(spans) <= model.size_bytes
        and model.architecture not in {"qwen4exp", "k2-horizon"}
    )


def fits(cfg, system, target, ram_safety=None, vram_safety=None):
    host = (
        cfg.estimated_model_ram_gb
        + cfg.mapped_model_resident_gb
        + cfg.vision_ram_gb
        + cfg.prompt_cache_ram_gb
        + cfg.kv_ram_gb
        + cfg.runtime_ram_overhead_gb
        + cfg.recurrent_state_ram_gb
    )
    device = (
        cfg.estimated_model_vram_gb
        + cfg.vision_vram_gb
        + cfg.draft_vram_gb
        + cfg.kv_vram_gb
        + cfg.recurrent_state_vram_gb
        + cfg.runtime_vram_overhead_gb
        + 0.6 * cfg.n_parallel
        + cfg.batch_vram_overhead_gb
    )
    host += target.ram_safety_gb if ram_safety is None else ram_safety
    device += (
        (
            target.moe_vram_safety_gb
            if cfg.n_cpu_moe is not None
            else target.dense_vram_safety_gb
        )
        if vram_safety is None
        else vram_safety
    )
    return host <= system.free_ram_gb and device <= system.free_vram_gb


def plan(compute, arguments):
    from performance_target import resolve_performance_target

    args = dict(arguments)
    original_model, system, profile = args["model"], args["system"], args["profile"]
    model = replace(
        original_model,
        metadata={
            **original_model.metadata,
            "__exact_moe_placement__": True,
            "__prefer_cpu_tied_output__": True,
            "__host_runtime_reserve_gb__": 0.5,
        },
    )
    args["model"] = model
    target = args["perf_target"] or resolve_performance_target(
        profile_choice=profile.performance_target or None
    )
    # On an 8-GiB Vulkan card, 1024..4096-token expert-transfer batches can
    # cause driver loss even when weight/KV arithmetic fits. Keep Auto bounded.
    small = replace(
        target,
        moe_hybrid_batch=256,
        moe_hybrid_ubatch=128,
        moe_batch_vram_reserve_gb=0.0,
    )
    args["perf_target"] = small
    candidates = []
    error = ""
    # Explore four bounded option sets and maximize usable context; ties keep
    # more requested features. No pin or persistent preference is rewritten.
    for stage in range(4):
        if stage >= 1:
            args["prompt_cache_ram_mib"] = 0
        if stage >= 2:
            args["draft_model"] = None
            args["force_draft_n_max"] = 0
        if stage >= 3:
            args["model"] = replace(model, mmproj=None)
        try:
            cfg = compute(**args)
        except MemoryError as exc:
            error = str(exc)
            continue
        if not fits(cfg, system, small, args["ram_safety_gb"], args["vram_safety_gb"]):
            error = "Fixed weights, runtime workspace and safety reserves exceed available RAM/VRAM."
            continue
        cfg.adaptive_memory = True
        cfg.memory_disable_vision = stage >= 3 and model.mmproj is not None
        cfg.memory_disable_draft = stage >= 2
        cfg.batch = min(cfg.batch, 256)
        cfg.ubatch = min(cfg.ubatch, 128)
        # Loading GPU weights through an mmap can temporarily crowd CPU
        # weights out of RAM. Stream the selected buffers on Windows instead.
        if system.os_name.lower().startswith("windows") and not args["force_mlock"]:
            cfg.load_mode, cfg.no_mmap, cfg.mlock = "none", True, False
        cfg.extra_cli_flags = list(cfg.extra_cli_flags) + ["--no-op-offload"]
        adjustments = [
            "small Vulkan batches (256/128)",
            "0.5 GiB host runtime reserve",
            "CPU experts computed on CPU (no Vulkan operation staging)",
        ]
        if model.metadata.get("__tied_output_embedding__"):
            adjustments.append("tied output on CPU (no duplicate vocabulary tensor)")
        if stage >= 1 and arguments["prompt_cache_ram_mib"] != 0:
            adjustments.append("host prompt cache disabled")
        if cfg.memory_disable_draft:
            adjustments.append("drafting disabled")
        if cfg.memory_disable_vision:
            adjustments.append("vision disabled (text-only)")
        cfg.memory_adjustments = adjustments
        detail = "Adaptive low-memory Auto: " + "; ".join(adjustments) + "."
        cfg.warning = ((cfg.warning + " ") if cfg.warning else "") + detail
        candidates.append(cfg)
    if candidates:
        return max(candidates, key=lambda cfg: cfg.ctx)
    raise MemoryError(
        "No safe Auto placement even after disabling optional vision, drafting "
        "and host prompt cache. Free more RAM/VRAM or use a smaller model. " + error
    )
