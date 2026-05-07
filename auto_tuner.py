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
  python auto_tuner.py --models-path D:/models --port 1234
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
_DEBUG_MODE = False
# Debug categories – expanded beyond just llama-cpp path searching
_DEBUG_CATEGORIES: set[str] = set()

def _debug_print(*args, **kwargs) -> None:
    """Print debug messages only if debugging mode is enabled."""
    if _DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)

def enable_debug_category(category: str) -> None:
    """Enable a specific debug category when global debug mode is off."""
    _DEBUG_CATEGORIES.add(category)

def debug_cat(category: str, *args, **kwargs) -> None:
    """Print a debug message for a specific category.
    
    Prints if global debug mode is ON OR if this category is explicitly enabled.
    """
    if _DEBUG_MODE or category in _DEBUG_CATEGORIES:
        print(f"[DEBUG:{category.upper()}]", *args, **kwargs)

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


def _ask_interactive_features(
    model: ModelEntry,
    draft_model: Optional[ModelEntry],
    settings_path: Path,
) -> tuple[bool, bool, bool, Optional[ModelEntry]]:
    """Interaktive Fragen-Kette nach Modellauswahl.

    Returns:
        (use_vision, use_draft, use_thinking, effective_draft) –
        use_vision/use_draft/use_thinking sind True wenn aktiviert,
        effective_draft ist das zu verwendende Draft-Modell (oder None)
    """
    # ── Vision ───────────────────────────────────────────────────────
    use_vision = False
    if model.mmproj is not None:
        use_vision = _confirm(
            f"Vision aktivieren? ({model.mmproj.name})",
            default_yes=True,
        )
        if not use_vision:
            model.mmproj = None

    # ── Draft Model ──────────────────────────────────────────────────
    use_draft = False
    effective_draft = draft_model
    if effective_draft is not None:
        use_draft = _confirm(
            f"Draft-Modell aktivieren? ({effective_draft.name})",
            default_yes=True,
        )
        if not use_draft:
            effective_draft = None

    # ── Thinking / Reasoning ────────────────────────────────────────
    use_thinking = False
    has_thinking_arch = (
        "gemma" in model.name.lower()
        or "deepseek" in model.name.lower()
        or "think" in model.name.lower()
    )
    if has_thinking_arch:
        use_thinking = _confirm(
            "Thinking/Reasoning aktivieren? (<|think|> / <|reserved_special_token>)",
            default_yes=True,
        )

    return use_vision, use_draft, use_thinking, effective_draft


def _pick_model(flat: List[ModelEntry], cli_query: Optional[str]) -> tuple[Optional[ModelEntry], List[str]]:
    if cli_query:
        parts = cli_query.split()
        query_parts = []
        flags = []
        for p in parts:
            if p.startswith('--') or p.lower() in ('novision', 'nodraft', 'nothinking'):
                flags.append(p.lower().lstrip('-'))
            else:
                query_parts.append(p)

        q = " ".join(query_parts).lower()
        matches = [e for e in flat if q in e.name.lower()]
        if not matches:
            print(f"[AutoTuner] No model matched --model '{cli_query}'.")
            return None, []
        if len(matches) > 1:
            print(f"[AutoTuner] '{cli_query}' is ambiguous — matches:")
            for e in matches:
                print(f"    - {e.name}")
            return None, []

        return matches[0], flags

    while True:
        try:
            raw = input(f"Select a model [1-{len(flat)}, q to quit]: ").strip()
        except EOFError:
            return None, []
        if raw.lower() in ("q", "quit", "exit"):
            return None, []

        parts = raw.split()
        model_idx_str = None
        flags = []
        for p in parts:
            if p.startswith('--') or p.lower() in ('novision', 'nodraft', 'nothinking'):
                flags.append(p.lower().lstrip('-'))
            elif model_idx_str is None and p.isdigit():
                model_idx_str = p
            else:
                flags.append(p.lower().lstrip('-'))

        if model_idx_str is None:
            print("  please enter a number (optionally followed by flags like '--novision').")
            continue
        n = int(model_idx_str)
        if not 1 <= n <= len(flat):
            print(f"  number must be between 1 and {len(flat)}.")
            continue
        return flat[n - 1], flags

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
    """Folders to look in for a llama.cpp / 1b_llama.cpp checkout."""
    roots: List[Path] = []
    seen: set = set()

    def add(p):
        try:
            rp = Path(p).expanduser().resolve()
            _debug_print(f"Checking path: {rp}")
        except (OSError, RuntimeError):
            return
        if rp in seen or not rp.exists():
            return
        seen.add(rp)
        roots.append(rp)

    env_dir = os.environ.get("LLAMA_CPP_DIR")
    _debug_print(f"LLAMA_CPP_DIR: {env_dir}")
    if env_dir:
        add(env_dir)
        parent = Path(env_dir).expanduser()
        add(parent.parent / "1b_llama.cpp")
        add(parent.parent / "ik_llama.cpp")
        add(parent.parent / "BitNet")

    bases = [Path(__file__).resolve().parent, Path.cwd()]
    common_subs = (
        "llama.cpp", "1b_llama.cpp", "ik_llama.cpp", "BitNet",
        "ai-local/llama.cpp", "ai-local/1b_llama.cpp", "ai-local/ik_llama.cpp",
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
                # Match fork name against directory name, ignoring .cpp suffix.
                # e.g. "ik_llama" matches "ik_llama.cpp", "1b_llama" matches "1b_llama.cpp"
                root_base = root.name.lower()
                if root_base.endswith(".cpp"):
                    root_base = root_base[:-4]
                if root_base.startswith(fork_name) or fork_name.startswith(root_base):
                    # First try the direct path (e.g. fork/llama-server)
                    candidate = root / inner
                    if candidate.is_file():
                        _debug_print(f"Found candidate: {candidate}")
                        return str(candidate)
                    # Then try build subpaths inside the matched fork directory.
                    for sub in _SERVER_SUBPATHS:
                        candidate = root / sub
                        if candidate.is_file():
                            _debug_print(f"Found candidate in fork subpath: {candidate}")
                            return str(candidate)
                        # Also check if inner exists within the subpath (e.g., build/bin/Release/llama-server)
                        candidate_with_inner = (root / sub) / inner
                        if candidate_with_inner.is_file():
                            _debug_print(f"Found candidate in fork subpath with inner: {candidate_with_inner}")
                            return str(candidate_with_inner)

        anchors: List[Path] = []
        seen: set = set()

        def add_anchor(a: Path):
            try:
                ra = a.resolve()
                _debug_print(f"Adding anchor: {ra}")
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
                _debug_print(f"Found candidate in anchors: {candidate}")
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
                _debug_print(f"Found candidate in subpaths: {candidate}")
                return str(candidate)

    _debug_print(f"Defaulting to user value: {user_value}")
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
    p.add_argument(
        "--llama-cpp-dir",
        default=os.environ.get("LLAMA_CPP_DIR"),
        help="Path to your llama.cpp checkout (env LLAMA_CPP_DIR). "
             "1b_llama.cpp/BitNet are searched in the same parent folder. "
             "Useful when llama.cpp lives outside the standard search paths "
             "(e.g. C:\\LAB\\ai-local\\llama.cpp).",
    )

    p.add_argument("--host", default="127.0.0.1",
                   help="Server bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=1234,
                   help="Server port (default: 1234)")
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
    p.add_argument("--nodraft", action="store_true",
                    help="Disable speculative decoding/draft model")
    p.add_argument("--nothinking", action="store_true",
                    help="Disable thinking/reasoning output")
    p.add_argument(
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        help="Extra arguments after `--` are forwarded to llama-server",
    )
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.llama_cpp_dir:
        # The existing _candidate_search_roots() already honors LLAMA_CPP_DIR
        # and looks for sibling 1b_llama.cpp / BitNet folders next to it.
        # Setting it here is process-local and does not affect the parent shell.
        os.environ["LLAMA_CPP_DIR"] = args.llama_cpp_dir

    _print_banner()

    # --- Debugging Mode Selection ---
    global _DEBUG_MODE
    print("\n" + "="*60)
    print("  DEBUG / VERBOSE MODE SELECTION")
    print("="*60)
    print("  1. Debugging OFF (standard)")
    print("  2. Debugging ON (alle Kategorien)")
    print("-" * 60)
    print("  Kategorie-Debugging (einzelne Bereiche):")
    print("  3. Hardware-Erkennung (GPU/RAM/CPU)")
    print("  4. Model-Scanning & Profil-Matching")
    print("  5. Server-Pfad-Suche (llama.cpp)")
    print("  6. Konfigurations-Berechnung (KV-Cache, Kontext)")
    print("-" * 60)

    debug_choice = input("Wahl [1-6] (default 1): ").strip()
    if debug_choice == "2":
        _DEBUG_MODE = True
        print("[AutoTuner] Globaler Debug-Modus aktiviert.")
    elif debug_choice == "3":
        enable_debug_category("hardware")
        print("[AutoTuner] Kategorie-Debugging: Hardware-Erkennung")
    elif debug_choice == "4":
        enable_debug_category("scanner")
        print("[AutoTuner] Kategorie-Debugging: Model-Scanning & Profile")
    elif debug_choice == "5":
        enable_debug_category("llama_cpp")
        print("[AutoTuner] Kategorie-Debugging: Server-Pfad-Suche")
    elif debug_choice == "6":
        enable_debug_category("config")
        print("[AutoTuner] Kategorie-Debugging: Konfigurations-Berechnung")
    else:
        print("[AutoTuner] Debugging deaktiviert.")
    print("="*60 + "\n")

    # --- Turbo-Quant Selection ---
    use_turbo = False
    print("\n" + "="*40)
    print("  QUANTIZATION MODE SELECTION")
    print("="*40)
    print("  1. Standard-Quant (llama.cpp)")
    print("  2. Turbo-Quant (tq_llama.cpp)")
    print("-" * 40)
    
    choice = input("Select mode [1/2] (default 1): ").strip()
    if choice == "2":
        use_turbo = True
        print("[AutoTuner] Turbo-Quant mode selected.")
    else:
        print("[AutoTuner] Standard-Quant mode selected.")
    print("="*40 + "\n")

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
            model, picked_flags = _pick_model(flat, args.model)
        except KeyboardInterrupt:
            print("\n[AutoTuner] Aborted by user.")
            return 0

        if model is None:
            print("[AutoTuner] No model selected — exiting.")
            return last_exit_code if first_iteration else 0

        # Apply flags picked during model selection
        for flag in picked_flags:
            if flag == 'novision':
                args.novision = True
            elif flag == 'nodraft':
                args.nodraft = True
            elif flag == 'nothinking':
                args.nothinking = True

        # Apply --novision flag if set
        if args.novision and model.mmproj is not None:
            print(f"[AutoTuner] Vision disabled per --novision (ignoring {model.mmproj.name})")
            model.mmproj = None

        profile = match_profile(model.name, profiles)
        
        # Handle Turbo-Quant binary override
        if use_turbo:
            # We force the server binary to be the one from tq_llama.cpp
            # The user provided path: C:\LAB\ai-local\tq_llama.cpp
            # Assuming it's a directory containing the binary, or the binary itself.
            # Usually, these are directories with a 'llama-server' inside.
            # If it's the binary itself, we use it directly.
            tq_path = Path(r"C:\LAB\ai-local\tq_llama.cpp")
            if tq_path.is_dir():
                # Try to find llama-server inside
                potential_binary = tq_path / "llama-server.exe"
                if potential_binary.exists():
                    server = str(potential_binary)
                else:
                    # Fallback to what we found or just use the path if it's a binary
                    server = str(tq_path)
            else:
                server = str(tq_path)

        # Try to find a draft model for this family if available.
        # Strategy: strip the quant suffix from the main model name, then look
        # for an assistant model that shares the same base prefix.
        #   e.g. "gemma-4-31B-it-Q3_K_S" -> base "gemma-4-31b-it"
        #        -> matches "gemma-4-31B-it-assistant-Q8_0"
        draft_model = None
        if profile.draft_max > 0 and not args.nodraft:
            import re as _re
            main_name_lower = model.name.lower()

            # Strip quant suffix (Q3_K_S, Q8_0, BF16, F16, IQ2_XXS, etc.)
            base_name = _re.sub(
                r"[-_]?(?:q\d+(?:_+[a-z]+)+|iq\d+_+[a-z]+|tf\d+|bf16|f16|f32)$",
                "", main_name_lower
            ).strip("-_")

            # STRICT matching: assistant must share the EXACT same architectural
            # variant (e.g., 26b-a4b only matches 26b-a4b, NOT e2b or 31b).
            # The assistant name must start with exactly the same base prefix
            # as the main model (after quant stripping), not just any common word.
            exact_drafts = [
                e for e in entries
                if "assistant" in e.name.lower()
                and e.name.lower().replace(".gguf", "").startswith(base_name + "-")
            ]
            if exact_drafts:
                draft_model = min(exact_drafts, key=lambda x: x.size_gb)
                print(f"[AutoTuner] Found draft model: {draft_model.name}")

            # NO fallback to arbitrary assistant models — speculative decoding
            # requires architecturally compatible draft models. Using an
            # incompatible draft (e.g., E2B for 26B-A4B) causes crashes.
        # Interaktive Fragen-Kette: Vision → Draft → Thinking
        (
            use_vision,
            use_draft,
            use_thinking,
            effective_draft,
        ) = _ask_interactive_features(model, draft_model, args.settings_path)

        # Wenn User "No" bei Draft drückt, setze draft_model auf None
        if not use_draft:
            effective_draft = None

        cfg = compute_config(model, system, profile, draft_model=effective_draft, user_ctx=args.ctx)
        _print_config(model, profile, cfg, system)

        raw_server = profile.server_binary or args.server
        
        # ── Spezialisierte Binary-Logik ────────────────────────────────
        # PrismML (bonsai-ternary) → 1b_llama.cpp
        # Gemma 4 mit Draft → ik_llama.cpp
        # Gemma 4 ohne Draft → Standard llama.cpp
        # Sonstige → Profil-Binary oder Standard
        
        def resolve_specialized_binary(
            profile: ModelProfile,
            use_draft_flag: bool,
            model_name: str,
        ) -> str:
            """Wähle die passende llama-server Binary basierend auf YAML-Setting + Features.

            Priorität:
            1. Spezialfall: Gemma 4 (MTP-Support)
            2. server_binary aus settings/*.yaml (wenn vorhanden)
            3. Fallback: Standard llama-server
            """
            # 1. Spezialfall: Gemma 4 (MTP-Support)
            if "gemma-4" in model_name.lower() or "gemma4" in model_name.lower():
                if use_draft_flag:
                    # Mit Draft → Entweder aus YAML oder ik_llama.cpp
                    return profile.server_binary if profile.server_binary else "ik_llama.cpp/llama-server"
                else:
                    # Ohne Draft → Immer Standard llama.cpp
                    return "llama.cpp/llama-server"

            # 2. ZUERST: server_binary aus YAML verwenden (wenn vorhanden)
            if profile.server_binary:
                return profile.server_binary

            # 3. Fallback: Standard
            return args.server
        
        if use_turbo:
            # Override server for Turbo-Quant
            tq_path = Path(r"C:\LAB\ai-local\tq_llama.cpp")
            if tq_path.is_dir():
                potential_binary = tq_path / "llama-server.exe"
                if potential_binary.exists():
                    server = str(potential_binary)
                else:
                    server = str(tq_path)
            else:
                server = str(tq_path)
            print(f"[AutoTuner] Using Turbo-Quant binary: {server}")
        else:
            effective_server = resolve_specialized_binary(profile, use_draft, model.name)
            server = _resolve_server_binary(effective_server)
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
            draft_model=effective_draft,  # ← FIX: effective_draft statt draft_model
            server_binary=server,
            host=args.host,
            port=args.port,
            extra_args=extra,
            use_thinking=use_thinking,  # ← Thinking-Flag übergeben
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
        
        # If Turbo-Quant is selected, we need to ensure the command uses the correct binary.
        # However, the 'launch' function in launcher.py handles the execution.
        # We must ensure 'cmd' contains the correct binary if use_turbo is True.
        # Since build_command already returns the command list, we check if it's correct.
        
        # Note: The user wants to use tq_llama.cpp.
        # If use_turbo is true, we should ensure the binary in 'cmd' is the one from C:\LAB\ai-local\tq_llama.cpp
        # This is handled by the fact that build_command uses 'server_binary'.
        # We will pass the correct binary to build_command if use_turbo is selected.
        
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
