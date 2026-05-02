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
        # CREATE_NEW_PROCESS_GROUP = 0x00000200
        # Required for sending CTRL_BREAK_EVENT to a child on Windows.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return subprocess.Popen(cmd, creationflags=flags)
    # Unix: put the child in its own session/process group.
    return subprocess.Popen(cmd, start_new_session=True)


def _terminate(proc: subprocess.Popen) -> None:
    """Politely ask the child (and its descendants) to exit."""
    if proc.poll() is not None:
        return
    try:
        if _is_windows():
            # CTRL_BREAK_EVENT can be sent to a process group on Windows.
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # Send SIGTERM to the whole group so child threads/forks die too.
            os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        # Already gone — fine.
        pass


def _force_kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if _is_windows():
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


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
        print("\n[AutoTuner] Ctrl+C received — stopping llama-server...",
              flush=True)
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


def _quote(arg: str) -> str:
    """Best-effort shell quoting for the command echo (display only)."""
    if not arg:
        return '""'
    if any(c in arg for c in (" ", "\t", '"', "'")):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg
