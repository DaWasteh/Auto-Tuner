"""Launch llama-server as a child process with proper Ctrl+C handling.

The original `subprocess.run(cmd)` in auto_tuner.py had a bug on Windows:
when the user pressed Ctrl+C, the SIGINT was sent to the Python parent but
not propagated to llama-server, leaving the child running. This module
fixes that by:

- On Windows: launching the child in a new process group with
  CREATE_NEW_PROCESS_GROUP, then sending CTRL_BREAK_EVENT to its PGID.
- On Unix:    launching the child in a new session (start_new_session),
  then sending SIGTERM (and finally SIGKILL) to the entire process group.

That way the user pressing Ctrl+C in the terminal reliably terminates
llama-server, even on Windows.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import List

# Time to wait between SIGTERM and SIGKILL
_GRACEFUL_TIMEOUT_SECONDS = 10

def _is_windows() -> bool:
    return os.name == "nt"

def _spawn(cmd: List[str]) -> subprocess.Popen:
    """Start the child process detached enough that we can signal its group."""
    if _is_windows():
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return subprocess.Popen(cmd, creationflags=flags)
    return subprocess.Popen(cmd, start_new_session=True)

def _confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        raw = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default_yes
    if not raw:
        return default_yes
    return raw in ("y", "yes", "j", "ja")

def _terminate(proc: subprocess.Popen) -> None:
    """Politely ask the child (and its descendants) to exit."""
    if proc.poll() is not None:
        return
    try:
        if _is_windows():
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # On Unix, send SIGTERM to the process group
            os.kill(-proc.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass

def _force_kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if _is_windows():
            proc.kill()
        else:
            # On Unix, send SIGKILL to the process group
            os.kill(-proc.pid, getattr(signal, 'SIGKILL', 9))
    except (ProcessLookupError, OSError):
        pass

def _quote(arg: str) -> str:
    """Best-effort shell quoting for the command echo (display only)."""
    if not arg:
        return '""'
    if any(c in arg for c in (" ", "\t", '"', "'")):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg

def launch(cmd: List[str]) -> int:
    """Run llama-server until it exits or until the user presses Ctrl+C.

    Returns the child's exit code (or 130 if it had to be killed).
    """
    print(f"\n[AutoTuner] Starting:\n  {' '.join(_quote(c) for c in cmd)}\n",
          flush=True)
    try:
        proc = _spawn(cmd)
    except FileNotFoundError:
        print(
            f"[AutoTuner] ERROR: server binary '{cmd[0]}' not found in PATH.\n"
            "  Install llama.cpp or pass --server /path/to/llama-server",
            file=sys.stderr,
        )
        return 127

    print(f"[AutoTuner] llama-server PID: {proc.pid}")
    print("[AutoTuner] Press Ctrl+C to stop the server gracefully.\n",
          flush=True)

    try:
        return proc.wait()
    except KeyboardInterrupt:
        # Immediate exit on Ctrl+C — no confirmation prompt
        print("[AutoTuner] Stopping llama-server...", flush=True)
        _terminate(proc)

        # Give it a chance to flush logs / close ports cleanly.
        deadline = time.monotonic() + _GRACEFUL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print("[AutoTuner] Stopped cleanly.")
                return proc.returncode if proc.returncode is not None else 0
            time.sleep(0.1)

        print("[AutoTuner] Timed out — forcing kill.")
        _force_kill(proc)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        return 130
