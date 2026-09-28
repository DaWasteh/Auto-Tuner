"""Sequential real-runtime checks of generated Auto settings (no tuning pins).

Writes local results; never overwrites user settings or measured winners.
Validation: allocation + a small real prompt, NOT a filled context window.
"""

import argparse
import concurrent.futures
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import psutil

from hardware import detect_system
from performance_target import PERFORMANCE_TARGETS
from scanner import scan_models
from settings_loader import load_profiles, match_profile
from tuner import compute_config, build_command, prepare_command_for_binary


def verify(model, system, profile, target, binary, output):
    row = {
        "model": model.name,
        "mode": target.name,
        "system": asdict(system),
        "validation": "allocated context + ~2k-token prompt/64-token decode; not full-context validation",
    }
    start = time.monotonic()
    try:
        cfg = compute_config(model, system, profile, perf_target=target, mode="coding")
    except MemoryError as exc:
        row.update(status="refused", error=str(exc))
        return row
    row["config"] = asdict(cfg)
    # Exactly the GUI's enabled default launch options, with the same command
    # builder/pruning. The adaptive plan must override them when necessary.
    cmd = build_command(
        model,
        cfg,
        profile,
        server_binary=str(binary),
        port=1247,
        enable_speculative=True,
        enable_ngram=True,
        enable_prompt_cache=True,
        prompt_cache_ram_mib=2048,
        use_thinking=True,
    )
    cmd += ["-lv", "4"]  # diagnostic logging only; no tuning override
    cmd, notices = prepare_command_for_binary(cmd)
    row.update(command=cmd, compatibility_notices=notices)
    logfile = output / f"{model.name}-{target.name}-{time.time_ns()}.log"
    row["log"] = str(logfile)
    row["min_free_ram_gib"] = psutil.virtual_memory().available / 2**30
    proc = None
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with logfile.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(
                cmd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, **cfg.env_overrides},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            def request(path, data=None, timeout=180):
                req = urllib.request.Request(
                    "http://127.0.0.1:1247" + path,
                    data=json.dumps(data).encode() if data is not None else None,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.load(response)

            def guard():
                free = psutil.virtual_memory().available / 2**30
                row["min_free_ram_gib"] = min(row["min_free_ram_gib"], free)
                if free < 1.0:
                    raise RuntimeError("Safety stop: available RAM below 1 GiB")
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited ({proc.returncode})")

            deadline = time.monotonic() + 180
            while True:
                guard()
                if time.monotonic() > deadline:
                    raise RuntimeError("Startup timeout")
                try:
                    request("/health", timeout=0.2)
                    break
                except (OSError, ValueError):
                    time.sleep(0.1)
            props = request("/props")
            row["runtime_context"] = props.get("default_generation_settings", {}).get(
                "n_ctx"
            )
            row["startup_s"] = round(time.monotonic() - start, 2)
            prompt = (
                "Summarize the following text in one sentence.\n"
                + "A local inference server must reserve memory for weights, attention and runtime buffers. "
                * 128
            )
            future = pool.submit(
                request,
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": 64,
                    "temperature": 0,
                    "cache_prompt": False,
                    "seed": 1,
                },
            )
            deadline = time.monotonic() + 180
            while not future.done():
                guard()
                if time.monotonic() > deadline:
                    raise RuntimeError("Inference timeout")
                time.sleep(0.1)
            reply = future.result()
            if reply.get("error") or not reply.get("tokens_predicted"):
                raise RuntimeError(
                    f"No completed generation: {reply.get('error', reply)}"
                )
            row.update(
                status="passed",
                timings=reply.get("timings"),
                tokens_evaluated=reply.get("tokens_evaluated"),
                tokens_predicted=reply.get("tokens_predicted"),
            )
    except Exception as exc:
        row.update(status="failed", error=str(exc))
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        pool.shutdown(wait=True)
        row["elapsed_s"] = round(time.monotonic() - start, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("dist/models"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-filter", default="")
    parser.add_argument("--mode", choices=list(PERFORMANCE_TARGETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Never terminate or overlap a user's inference server.
    if any(
        p.info["name"] and "llama-server" in p.info["name"].lower()
        for p in psutil.process_iter(["name"])
    ):
        raise SystemExit(
            "A llama-server is already running; stop it before verification."
        )
    profiles = load_profiles(Path(__file__).parent / "settings")
    results = []
    result_file = args.output / ("auto-matrix-" + str(time.time_ns()) + ".json")
    for model in scan_models(args.models):
        if args.model_filter.lower() not in model.name.lower():
            continue
        profile = match_profile(model.name, profiles, model.architecture)
        for name, target in PERFORMANCE_TARGETS.items():
            if args.mode and args.mode != name:
                continue
            system = detect_system(str(args.binary))
            print(
                f"Testing {model.name} / {name}; free RAM {system.free_ram_gb:.2f} GiB",
                flush=True,
            )
            row = verify(model, system, profile, target, args.binary, args.output)
            results.append(row)
            result_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k
                        in (
                            "model",
                            "mode",
                            "status",
                            "error",
                            "runtime_context",
                            "tokens_evaluated",
                            "tokens_predicted",
                            "min_free_ram_gib",
                        )
                    }
                ),
                flush=True,
            )
            time.sleep(3)
    print(result_file, flush=True)


if __name__ == "__main__":
    main()
