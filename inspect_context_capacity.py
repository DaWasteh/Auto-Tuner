"""Read-only capacity report; never equate model native context with free memory.

Run: python inspect_context_capacity.py --models dist/models --binary PATH
This plans text-only, single-slot configurations without draft/prompt caches.
It does not load models or modify user settings. Numbers are estimates, not a
full-context inference guarantee. OS paging is not counted as physical RAM.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

from hardware import detect_system
from performance_target import PERFORMANCE_TARGETS
from scanner import scan_models
from settings_loader import load_profiles, match_profile
from tuner import compute_config


def capacity_report(models, system, profiles, target=131072):
    rows = []
    for original in models:
        model = copy.copy(original)
        model.mmproj = None
        profile = match_profile(model.name, profiles, model.architecture)
        choices = []
        for tier in ("safe", "balanced", "throughput", "low_vram"):
            try:
                cfg = compute_config(
                    model,
                    system,
                    profile,
                    user_ctx=target,
                    perf_target=PERFORMANCE_TARGETS[tier],
                    prompt_cache_ram_mib=0,
                    force_n_parallel=1,
                    force_draft_n_max=0,
                )
            except MemoryError as exc:
                choices.append({"tier": tier, "error": str(exc)})
                continue
            host = (
                cfg.estimated_model_ram_gb
                + cfg.mapped_model_resident_gb
                + cfg.kv_ram_gb
                + cfg.recurrent_state_ram_gb
                + cfg.runtime_ram_overhead_gb
                + cfg.prompt_cache_ram_gb
                + PERFORMANCE_TARGETS[tier].ram_safety_gb
            )
            choices.append(
                {
                    "tier": tier,
                    "context": cfg.ctx,
                    "ngl": cfg.ngl,
                    "n_cpu_moe": cfg.n_cpu_moe,
                    "cache_k": cfg.cache_k,
                    "cache_v": cfg.cache_v,
                    "no_kv_offload": cfg.no_kv_offload,
                    "host_including_safety_gib": round(host, 3),
                    "host_fits": host <= system.free_ram_gb,
                    "config": asdict(cfg),
                }
            )
        eligible = [c for c in choices if c.get("host_fits")]
        best = max(eligible, key=lambda c: c["context"], default=None)
        rows.append(
            {
                "model": model.name,
                "native_context": model.native_context,
                "best_estimate": best,
                "choices": choices,
            }
        )
    return {
        "system": asdict(system),
        "requested_context": target,
        "conditions": "text only; one slot; no draft; host prompt cache disabled",
        "validation": "static estimates only; no full-context inference test",
        "models": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--target", type=int, default=131072)
    parser.add_argument("--output", type=Path, default=Path("context-capacity.json"))
    args = parser.parse_args()
    report = capacity_report(
        scan_models(args.models),
        detect_system(args.binary),
        load_profiles(Path(__file__).parent / "settings"),
        args.target,
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in report["models"]:
        best = row["best_estimate"]
        print(
            row["model"],
            (
                f"{best['context']:,} ({best['tier']})"
                if best
                else "no physically fitting automatic estimate"
            ),
        )
    print(args.output)


if __name__ == "__main__":
    main()
