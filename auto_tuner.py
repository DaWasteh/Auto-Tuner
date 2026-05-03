"""AutoTuner for llama.cpp — interactive launcher.

Workflow:
  1. Detect the system (CPU, RAM, GPUs across AMD / NVIDIA / Intel / Apple).
  2. Scan a folder of GGUF files, pair each main model with its mmproj.
  3. Show a numbered terminal menu so the user can pick a model.
  4. Match the model against per-family YAML profiles in settings/.
  5. Compute an optimal llama-server config that fits in free RAM/VRAM.
  6. Start llama-server with proper Ctrl+C handling.

Usage:
  python auto_tuner.py
  python auto_tuner.py --models-path D:/models --port 8080
  python auto_tuner.py --model Devstral --dry-run

Environment variables:
  AUTOTUNER_MODELS    default models path
  LLAMA_SERVER        path to the llama-server binary
  LLAMA_CPP_DIR       llama.cpp checkout (build/bin/[Release/]llama-server is auto-found)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from hardware import detect_system, format_system, SystemInfo
from launcher import launch
from scanner import scan_models, group_entries, ModelEntry
from settings_loader import load_profiles, match_profile, ModelProfile
from tuner import build_command, compute_config, TunedConfig

# ---------------------------------------------------------------------------
# Pretty-printing helpers

_BAR = "─" * 64

def _print_banner() -> None:
    print()
    print(_BAR)
    print("  AutoTuner for llama.cpp  —  interactive launcher")
    print(_BAR)

def _print_system(info: SystemInfo) -> None:
    print(format_system(info))

def _vision_marker(entry: ModelEntry) -> str:
    return " 👁" if entry.has_vision else "  "

def _print_menu(groups: dict) -> List[ModelEntry]:
    """Print grouped model menu and return a flat list in display order."""
    flat: List[ModelEntry] = []
    print("\nAvailable models:")
    print(_BAR)
    idx = 1
    for group_name in sorted(groups.keys()):
        entries = sorted(groups[group_name], key=lambda e: e.name.lower())
        if not entries:
            continue
        print(f"\n  [{group_name}]")
        for e in entries:
            size = f"{e.size_gb:>5.1f} GB"
            ctx = ""
            if e.native_context:
                ctx = f"  ({e.native_context // 1024}k native)"
            print(f"    {idx:>2}. {_vision_marker(e)} {e.name:<55} "
                  f"{size}{ctx}")
            flat.append(e)
            idx += 1
    print()
    return flat

def _print_config(model: ModelEntry, profile: ModelProfile,
                  cfg: TunedConfig, system: SystemInfo) -> None:
    print(_BAR)
    print(f"Model:    {model.name}")
    print(f"Profile:  {profile.display_name}"
          + (f"  ({profile.source_file})" if profile.source_file else ""))
    if profile.notes:
        print(f"Notes:    {profile.notes}")
    if model.mmproj is not None:
        print(f"Vision:   {model.mmproj.name}")
    print(_BAR)

    if cfg.full_offload:
        placement = f"GPU full offload (ngl=all of {model.n_layers or '?'})"
    elif cfg.ngl > 0:
        placement = (f"hybrid: {cfg.ngl} layers on GPU, rest on CPU")
    else:
        placement = "CPU only"
    print(f"  Placement       : {placement}")
    print(f"  Context         : {cfg.ctx:,} tokens")
    print(f"  KV cache quant  : K={cfg.cache_k}  V={cfg.cache_v}")
    print(f"  Threads         : {cfg.threads} (batch: {cfg.batch_threads})")
    print(f"  Batch / ubatch  : {cfg.batch} / {cfg.ubatch}")
    print(f"  Flash attention : {'on' if cfg.flash_attn else 'off'}")
    if cfg.mlock:
        print("  mlock           : on (model pinned in RAM)")
    if cfg.numa:
        print(f"  NUMA            : {cfg.numa}")
    if cfg.tensor_split:
        print(f"  Tensor split    : {cfg.tensor_split}")
    if cfg.main_gpu is not None:
        print(f"  Main GPU        : {cfg.main_gpu}")

    s = cfg.sampling
    print(f"  Sampling        : temp={s.get('temperature')} "
          f"top_k={s.get('top_k')} top_p={s.get('top_p')} "
          f"min_p={s.get('min_p')} rep={s.get('repeat_penalty')}")

    print()
    print("  Memory estimate:")
    print(f"    model on GPU  ~ {cfg.estimated_model_vram_gb:5.1f} GB"
          f"    (free VRAM:  {system.free_vram_gb:5.1f} GB)")
    print(f"    model on CPU  ~ {cfg.estimated_model_ram_gb:5.1f} GB"
          f"    (free RAM:   {system.free_ram_gb:5.1f} GB)")
    print(f"    KV cache      ~ {cfg.estimated_kv_gb:5.1f} GB")
    print(_BAR)

# ---------------------------------------------------------------------------
# Selection

def _confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        raw = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default_yes
    if not raw:
        return default_yes
    return raw in ("y", "yes", "j", "ja")

def _pick_model(flat: List[ModelEntry], cli_query: Optional[str]) -> Optional[ModelEntry]:
    if cli_query:
        # Check if the query ends with --novision
        novision = False
        if cli_query.lower().endswith("--novision"):
            cli_query = cli_query[:-len("--novision")].strip()
            novision = True

        q = cli_query.lower()
        matches = [e for e in flat if q in e.name.lower()]
        if not matches:
            print(f"[AutoTuner] No model matched --model '{cli_query}'.")
            return None
        if len(matches) > 1:
            print(f"[AutoTuner] '{cli_query}' is ambiguous — matches:")
            for e in matches:
                print(f"    - {e.name}")
            return None

        model = matches[0]
        if novision and model.mmproj is not None:
            print(f"[AutoTuner] Vision disabled per --novision (ignoring {model.mmproj.name})")
            model.mmproj = None
        return model

    while True:
        try:
            raw = input(f"Select a model [1-{len(flat)}, q to quit]: ").strip()
        except EOFError:
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None

        # Support "--novision" suffix (e.g., "18 --novision")
        novision = False
        parsed = raw
        if raw.lower().endswith("--novision"):
            parsed = raw[:raw.lower().rfind("--novision")].strip()
            novision = True

        if not parsed.isdigit():
            print("  please enter a number (optionally followed by ' --novision').")
            continue
        n = int(parsed)
        if not 1 <= n <= len(flat):
            print(f"  number must be between 1 and {len(flat)}.")
            continue
        model = flat[n - 1]
        if novision and model.mmproj is not None:
            print(f"[AutoTuner] Vision disabled per '--novision' suffix (ignoring {model.mmproj.name})")
            model.mmproj = None
        return model

# ---------------------------------------------------------------------------
# llama-server discovery

_SERVER_SUBPATHS = [
    "build/bin/Release/llama-server.exe",
    "build/bin/Debug/llama-server.exe",
    "build/bin/llama-server.exe",
    "build/bin/llama-server",
    "build/llama-server",
    "llama-server.exe",
    "llama-server",
]

def _candidate_search_roots() -> List[Path]:
    """Folders to look in for a llama.cpp / 1bllama.cpp checkout."""
    roots: List[Path] = []
    seen: set = set()

    def add(p):
        try:
            rp = Path(p).expanduser().resolve()
        except (OSError, RuntimeError):
            return
        if rp in seen or not rp.exists():
            return
        seen.add(rp)
        roots.append(rp)

    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        add(env_dir)
        parent = Path(env_dir).expanduser()
        add(parent.parent / "1bllama.cpp")
        add(parent.parent / "BitNet")

    bases = [Path(__file__).resolve().parent, Path.cwd()]
    common_subs = (
        "llama.cpp", "1bllama.cpp", "BitNet",
        "ai-local/llama.cpp", "ai-local/1bllama.cpp", "ai-local/BitNet",
        "ai/llama.cpp", "ai/1bllama.cpp",
        "ml/llama.cpp",
    )
    for base in bases:
        chain = [base, *list(base.parents)[:5]]
        for p in chain:
            for sub in common_subs:
                add(p / sub)
    return roots

def _resolve_server_binary(user_value: str) -> str:
    """Turn a user-provided server name/path into something runnable."""
    p = Path(user_value).expanduser()
    if p.is_absolute() and p.is_file():
        return str(p)

    has_sep = (os.sep in user_value
               or (os.altsep is not None and os.altsep in user_value))

    if has_sep and not p.is_absolute():
        parts = list(p.parts)
        fork_name = parts[0].lower() if parts else ""
        inner = Path(*parts[1:]) if len(parts) > 1 else None

        if inner is not None and fork_name:
            for root in _candidate_search_roots():
                if root.name.lower() == fork_name:
                    candidate = root / inner
                    if candidate.is_file():
                        return str(candidate)

        anchors: List[Path] = []
        seen: set = set()

        def add_anchor(a: Path):
            try:
                ra = a.resolve()
            except (OSError, RuntimeError):
                return
            if ra in seen:
                return
            seen.add(ra)
            anchors.append(ra)

        for base in (Path(__file__).resolve().parent, Path.cwd()):
            chain = [base, *list(base.parents)[:5]]
            for a in chain:
                add_anchor(a)

        for a in anchors:
            candidate = a / p
            if candidate.is_file():
                return str(candidate)

    if not has_sep:
        which = shutil.which(user_value)
        if which:
            return which

    name = Path(user_value).name or "llama-server"
    if name in ("llama-server", "llama-server.exe"):
        candidate_subpaths = list(_SERVER_SUBPATHS)
    else:
        candidate_subpaths = [
            f"build/bin/Release/{name}",
            f"build/bin/Debug/{name}",
            f"build/bin/{name}",
            f"build/{name}",
            name,
        ]

    for root in _candidate_search_roots():
        for sub in candidate_subpaths:
            candidate = root / sub
            if candidate.is_file():
                return str(candidate)

    return user_value

# ---------------------------------------------------------------------------
# Client settings hint

def _print_client_settings(host: str, port: int, ctx: int,
                           model: ModelEntry) -> None:
    """Print a copy-pasteable block for OpenAI-API clients."""
    base_url = f"http://{host}:{port}/v1"
    print()
    print(_BAR)
    print("  Client settings (Roo-Code, Continue, Cline, Open WebUI, …)")
    print(_BAR)
    print(f"    Base URL          : {base_url}")
    print("    API key           : sk-no-key   (any non-empty string works)")
    print(f"    Model name        : {model.name}")
    print(f"    Context window    : {ctx:,} tokens   ← set this in your client")
    print(_BAR)

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="auto_tuner",
        description="Interactive launcher for llama-server with auto-tuned "
                    "config based on free RAM/VRAM.",
    )
    p.add_argument(
        "--models-path",
        default=os.environ.get("AUTOTUNER_MODELS", "./models"),
        help="Folder to scan for *.gguf models "
             "(default: ./models, env AUTOTUNER_MODELS)",
    )
    p.add_argument(
        "--settings-path",
        default=str(Path(__file__).parent / "settings"),
        help="Folder with per-model YAML profiles (default: ./settings)",
    )
    p.add_argument(
        "--server",
        default=os.environ.get("LLAMA_SERVER", "llama-server"),
        help="Path to the llama-server binary "
             "(default: llama-server, env LLAMA_SERVER)",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Server bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080,
                   help="Server port (default: 8080)")
    p.add_argument("--ctx", type=int, default=None,
                   help="Override context length (otherwise auto-tuned)")
    p.add_argument("--model", default=None,
                   help="Pick model by substring without showing the menu")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command but don't start the server")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the launch confirmation prompt")
    p.add_argument("--novision", action="store_true",
                   help="Disable vision (mmproj) even if available")
    p.add_argument(
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        help="Extra arguments after `--` are forwarded to llama-server",
    )
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    _print_banner()
    system = detect_system()
    _print_system(system)

    models_path = Path(args.models_path).expanduser()
    if not models_path.exists():
        print(f"\n[AutoTuner] Models folder not found: {models_path}")
        print("  Pass --models-path /path/to/models or set AUTOTUNER_MODELS.")
        return 2

    print(f"\n[AutoTuner] Scanning models in: {models_path}")
    entries = scan_models(models_path)
    if not entries:
        print("[AutoTuner] No *.gguf models found.")
        return 2

    profiles = load_profiles(Path(args.settings_path))
    print(f"[AutoTuner] Loaded {len(profiles)} profile(s) from "
          f"{args.settings_path}")

    # Main loop: pick model → launch → on stop, ask whether to pick another.
    # This avoids the cmd.exe "Batchvorgang abbrechen?" prompt the user gets
    # when the script exits after Ctrl+C inside a .bat wrapper, and lets
    # them switch models without restarting the whole tool.
    first_iteration = True
    last_exit_code = 0

    while True:
        # On every iteration after the first, re-detect the system so the
        # auto-tuner sees the RAM/VRAM that was just freed by the previous
        # llama-server, and clear any one-shot CLI selectors.
        if not first_iteration:
            print()
            system = detect_system()
            _print_system(system)
            args.model = None  # force the menu to show again
            args.ctx = None    # don't keep an override meant for the prev model

        groups = group_entries(entries)
        flat = _print_menu(groups)

        try:
            model = _pick_model(flat, args.model)
        except KeyboardInterrupt:
            print("\n[AutoTuner] Aborted by user.")
            return 0

        if model is None:
            print("[AutoTuner] No model selected — exiting.")
            return last_exit_code if first_iteration else 0

        # Apply --novision flag if set
        if args.novision and model.mmproj is not None:
            print(f"[AutoTuner] Vision disabled per --novision (ignoring {model.mmproj.name})")
            model.mmproj = None

        profile = match_profile(model.name, profiles)
        cfg = compute_config(model, system, profile, user_ctx=args.ctx)
        _print_config(model, profile, cfg, system)

        raw_server = profile.server_binary or args.server
        server = _resolve_server_binary(raw_server)
        if server != raw_server:
            print(f"[AutoTuner] Found server binary: {server}")
        elif not Path(server).is_file() and not shutil.which(server):
            print(f"[AutoTuner] Warning: server binary '{server}' not found.")
            print("  Pass --server /path/to/llama-server, set LLAMA_SERVER, or")
            print("  set LLAMA_CPP_DIR to your llama.cpp checkout.")

        extra = args.passthrough or []
        cmd = build_command(
            model=model,
            config=cfg,
            profile=profile,
            server_binary=server,
            host=args.host,
            port=args.port,
            extra_args=extra,
        )

        if args.dry_run:
            print("[AutoTuner] --dry-run — not starting the server.")
            print("Command:")
            print("  " + " ".join(cmd))
            _print_client_settings(args.host, args.port, cfg.ctx, model)
            return 0

        try:
            launch_now = args.yes or _confirm("Launch llama-server now?")
        except KeyboardInterrupt:
            print("\n[AutoTuner] Aborted by user.")
            return 0

        if not launch_now:
            print("[AutoTuner] Launch skipped — back to the model menu.")
            first_iteration = False
            continue

        _print_client_settings(args.host, args.port, cfg.ctx, model)
        print(f"\n[AutoTuner] Web UI will be available at "
              f"http://{args.host}:{args.port}\n")
        last_exit_code = launch(cmd)

        # Server has stopped (Ctrl+C, crash, or normal exit). Offer to pick
        # another model instead of falling through and exiting.
        print()
        try:
            keep_going = _confirm("Server stopped. Pick another model?",
                                  default_yes=True)
        except KeyboardInterrupt:
            print("\n[AutoTuner] Goodbye.")
            return 0

        if not keep_going:
            print("[AutoTuner] Goodbye.")
            return 0

        first_iteration = False

if __name__ == "__main__":
    sys.exit(main())
