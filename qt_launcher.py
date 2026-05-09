"""AutoTuner Qt Launcher — standalone GUI for model selection and server control.

llama-server opens in its own terminal window (visible, full output).
The Qt log panel shows AutoTuner-level status messages only.

Run with:
  python qt_launcher.py
  python qt_launcher.py --models-path D:/models
"""
from __future__ import annotations

import copy
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar,
    QTextEdit, QToolBar, QVBoxLayout, QWidget,
)

from hardware import detect_system, SystemInfo
from scanner import scan_models, group_entries, ModelEntry
from settings_loader import load_profiles, match_profile, ModelProfile
from tuner import build_command, compute_config
from performance_target import (
    PERFORMANCE_TARGETS,
    list_target_names,
    resolve_performance_target,
    DEFAULT_TARGET_NAME,
)
import app_settings


def _get_fork_tools():
    """Lazy import — never triggers auto_tuner.main()."""
    from auto_tuner import _discover_llama_forks, _resolve_server_binary
    return _discover_llama_forks, _resolve_server_binary


def _default_settings_path() -> Path:
    return Path(__file__).resolve().parent / "settings"


def _default_models_path() -> Path:
    """Resolve default models folder.

    Preference order:
      1. Persisted choice (autotuner_settings.json)
      2. AUTOTUNER_MODELS environment variable
      3. <script_dir>/models or <script_dir>/../models if either exists
      4. <script_dir>/models (placeholder; user will be prompted)
    """
    saved = app_settings.get_models_path()
    if saved is not None:
        return saved
    env = os.environ.get("AUTOTUNER_MODELS", "")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    script_dir = Path(__file__).resolve().parent
    for c in (script_dir / "models", script_dir.parent / "models"):
        if c.exists():
            return c
    return script_dir / "models"


def _persisted_fork_path() -> Optional[Path]:
    """Persisted fork-folder choice, if still valid on disk."""
    return app_settings.get_fork_path()


# ---------------------------------------------------------------------------
# Terminal process — spawns llama-server in its own visible terminal window

class _TerminalProcess:
    """Spawn llama-server in an independent terminal (CREATE_NEW_CONSOLE on
    Windows, start_new_session on Unix). No stdout pipe — the user sees the
    full server output in the separate window; our log panel shows status only.
    """

    def __init__(self, cmd: List[str]) -> None:
        self.cmd = cmd
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
            self.proc = subprocess.Popen(self.cmd, creationflags=flags)
        else:
            self.proc = subprocess.Popen(self.cmd, start_new_session=True)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self.proc.returncode if self.proc is not None else None

    def stop(self) -> None:
        """Non-blocking signal + background wait."""
        if self.proc is None:
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.kill(-self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        # Capture in a local variable BEFORE clearing self.proc —
        # the daemon thread runs after self.proc is already None.
        _proc = self.proc
        self.proc = None

        def _wait() -> None:
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    _proc.kill()
                except (ProcessLookupError, OSError):
                    pass

        threading.Thread(target=_wait, daemon=True).start()


# ---------------------------------------------------------------------------
# Hardware detection worker with global timeout

class _HwDetectWorker(QObject):
    """Runs detect_system() in a background thread with a global timeout."""
    finished = pyqtSignal(object, str)  # SystemInfo|None, error_msg

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__()
        self._timeout = timeout

    def run(self) -> None:
        try:
            s = detect_system()
            self.finished.emit(s, "")
        except Exception as exc:
            self.finished.emit(None, str(exc))


# ---------------------------------------------------------------------------
# Background scanner

class _ScanWorker(QObject):
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root

    def run(self) -> None:
        try:
            self.finished.emit(scan_models(self._root))
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Draft-model detection helper (mirrors auto_tuner.py logic)

def _find_draft_model(
    entry: ModelEntry, all_entries: List[ModelEntry]
) -> Optional[ModelEntry]:
    name = entry.name.lower()
    base = re.sub(
        r"[-_]?(?:iq\d+(?:_+[a-z\d]+)*(?:[-_]\d+[\.\d]*bpw)?|"
        r"q\d+(?:_+[a-z\d]+)*|tf\d+|bf16|f16|f32)$",
        "", name,
    ).strip("-_")
    base = re.sub(r"[-_](?:ud|unsloth)$", "", base, flags=re.IGNORECASE).strip("-_")
    candidates = [
        e for e in all_entries
        if "assistant" in e.name.lower()
        and e.name.lower().startswith(base + "-")
    ]
    return min(candidates, key=lambda x: x.size_gb) if candidates else None


def _clean_model_name(name: str) -> str:
    """Strip quant/distributor suffixes for a clean --alias name."""
    import re as _re
    clean = _re.sub(
        r"[-_]?(?:iq\d+(?:_+[a-z\d]+)*(?:[-_]\d+[.\d]*bpw)?|"
        r"q\d+(?:_+[a-z\d]+)*|tf\d+|bf16|f16|f32)$",
        "", name, flags=_re.IGNORECASE,
    ).strip("-_")
    return _re.sub(r"[-_](?:ud|unsloth)$", "", clean, flags=_re.IGNORECASE).strip("-_")


# ---------------------------------------------------------------------------
# Main window

class MainWindow(QMainWindow):
    def __init__(self, models_path: Path, settings_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("AutoTuner Qt Launcher")
        self.resize(1320, 840)

        self.models_path   = models_path
        self.settings_path = settings_path

        self._server: Optional[_TerminalProcess] = None
        self._all_entries: List[ModelEntry]      = []
        self._system:      Optional[SystemInfo]  = None
        self._profiles:    List[ModelProfile]    = []
        self._forks:       List[Tuple[str, Path]] = []
        self._fork_path:   Optional[Path]      = None  # manueller Fork-Ordner

        # Currently selected model + its draft (set in _show_config)
        self._current_entry: Optional[ModelEntry] = None
        self._current_draft: Optional[ModelEntry] = None

        # Track whether the user has manually overridden the fork selection
        self._fork_manual_override = False

        self._scan_thread: Optional[QThread]     = None
        self._scan_worker: Optional[_ScanWorker] = None
        self._sysinfo_busy = False
        self._font_size    = 10

        self._build_ui()
        QTimer.singleShot(0, self._startup_load)

        # Server crash-detection (lightweight poll — no stdout read)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_server)
        self._poll_timer.start(500)

        # Sysinfo refresh (non-blocking — daemon thread)
        self._sysinfo_timer = QTimer(self)
        self._sysinfo_timer.timeout.connect(self._sysinfo_async)
        self._sysinfo_timer.start(6000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # ── Toolbar ────────────────────────────────────────────────────
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._path_label = QLabel()
        self._path_label.setStyleSheet("padding:0 6px;color:#aaa;")
        tb.addWidget(self._path_label)
        tb.addSeparator()

        for label, slot in (
            ("📂 Models folder", self._browse_models),
            ("🔄 Refresh",       self._start_scan),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label.startswith("🔄"):
                self._btn_refresh = btn
            tb.addWidget(btn)

        tb.addSeparator()
        tb.addWidget(QLabel(" Fork:"))
        self._fork_combo = QComboBox()
        self._fork_combo.setMinimumWidth(140)
        self._fork_combo.setToolTip("Default llama.cpp fork (auto-overridden by profile)")
        self._fork_combo.currentIndexChanged.connect(self._on_fork_changed)
        tb.addWidget(self._fork_combo)
        
        self._fork_path_lbl = QLabel()
        self._fork_path_lbl.setStyleSheet("color:#aaa;font-size:9pt;")
        self._fork_path_lbl.setMaximumWidth(120)
        self._fork_path_lbl.setText("")
        tb.addWidget(self._fork_path_lbl)
        
        self._btn_fork_folder = QPushButton("📂")
        self._btn_fork_folder.setFixedWidth(28)
        self._btn_fork_folder.setToolTip("Manuellen Fork-Ordner auswählen")
        self._btn_fork_folder.clicked.connect(self._browse_fork_folder)
        tb.addWidget(self._btn_fork_folder)

        tb.addSeparator()
        tb.addWidget(QLabel(" Performance:"))
        self._perf_combo = QComboBox()
        self._perf_combo.setMinimumWidth(120)
        # Build tooltip from registry so a future 4th tier auto-appears.
        tip_lines = ["VRAM utilisation preset:"]
        for tname in list_target_names():
            t = PERFORMANCE_TARGETS[tname]
            tip_lines.append(f"  • {tname}: {t.description}")
        self._perf_combo.setToolTip("\n".join(tip_lines))
        for tname in list_target_names():
            self._perf_combo.addItem(tname)
        # Restore persisted choice (may be None → default).
        persisted_perf = app_settings.get_performance_target()
        initial_perf = persisted_perf or DEFAULT_TARGET_NAME
        idx = self._perf_combo.findText(initial_perf)
        if idx < 0:
            idx = self._perf_combo.findText(DEFAULT_TARGET_NAME)
        self._perf_combo.setCurrentIndex(max(0, idx))
        self._perf_combo.currentIndexChanged.connect(self._on_perf_changed)
        tb.addWidget(self._perf_combo)

        tb.addSeparator()
        tb.addWidget(QLabel(" Font:"))
        for delta, label in ((-1, "A−"), (+1, "A+")):
            b = QPushButton(label)
            b.setFixedWidth(36)
            d = delta
            b.clicked.connect(lambda _, d=d: self._change_font(d))
            tb.addWidget(b)

        # ── Sysinfo bar ────────────────────────────────────────────────
        sysbar = QWidget()
        sl = QHBoxLayout(sysbar)
        sl.setContentsMargins(6, 1, 6, 1)
        self._cpu_lbl  = QLabel("CPU: —")
        self._vram_lbl = QLabel("VRAM: —")
        self._ram_lbl  = QLabel("RAM: —")
        self._gpu_lbl  = QLabel("GPU: —")
        for lbl in (self._cpu_lbl, self._vram_lbl, self._ram_lbl, self._gpu_lbl):
            lbl.setStyleSheet("color:#8be;padding:0 12px;")
            sl.addWidget(lbl)
        sl.addStretch()
        sysbar.setMaximumHeight(24)
        sysbar.setStyleSheet("background:#161625;")

        # ── Filter + model list ────────────────────────────────────────
        fr = QWidget()
        frl = QHBoxLayout(fr)
        frl.setContentsMargins(2, 2, 2, 2)
        frl.addWidget(QLabel("Filter:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("type to filter…")
        self._search.textChanged.connect(self._apply_filter)
        frl.addWidget(self._search)

        self._model_list = QListWidget()
        self._model_list.currentItemChanged.connect(self._on_selection_changed)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        ll.addWidget(fr)
        ll.addWidget(self._model_list)

        # ── Config preview ─────────────────────────────────────────────
        self._config_preview = QTextEdit()
        self._config_preview.setReadOnly(True)
        self._config_preview.setPlaceholderText("Select a model to see its config…")
        self._apply_mono_font(self._config_preview)

        # ── Launch options (checkboxes) ────────────────────────────────
        opts = QGroupBox("Launch options")
        ol = QVBoxLayout(opts)
        ol.setSpacing(4)

        self._chk_vision   = QCheckBox("Vision (mmproj)")
        self._chk_draft    = QCheckBox("Draft model (speculative decoding)")
        self._chk_thinking = QCheckBox("Thinking / Reasoning")

        for chk in (self._chk_vision, self._chk_draft, self._chk_thinking):
            chk.setEnabled(False)
            ol.addWidget(chk)

        # Checkbox toggles → refresh context / memory estimates
        self._chk_vision.toggled.connect(self._on_option_toggled)
        self._chk_draft.toggled.connect(self._on_option_toggled)
        self._chk_thinking.toggled.connect(self._on_option_toggled)

        opts.setMaximumHeight(110)

        right = QWidget()
        rl2 = QVBoxLayout(right)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.setSpacing(4)
        rl2.addWidget(self._config_preview, 1)
        rl2.addWidget(opts)

        # ── Top HSplitter ──────────────────────────────────────────────
        top_split = QSplitter(Qt.Orientation.Horizontal)
        top_split.addWidget(left)
        top_split.addWidget(right)
        top_split.setSizes([370, 650])

        # ── Log panel ──────────────────────────────────────────────────
        self._log_panel = QTextEdit()
        self._log_panel.setReadOnly(True)
        self._apply_mono_font(self._log_panel)
        self._log_panel.setPlaceholderText(
            "AutoTuner status messages appear here.\n"
            "Server output is shown in the separate terminal window."
        )

        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.addWidget(top_split)
        main_split.addWidget(self._log_panel)
        main_split.setSizes([560, 240])

        # ── Button row ─────────────────────────────────────────────────
        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(6, 4, 6, 4)

        bl.addWidget(QLabel("Host:"))
        self._host_edit = QLineEdit("127.0.0.1")
        self._host_edit.setFixedWidth(120)
        bl.addWidget(self._host_edit)

        bl.addWidget(QLabel(" Port:"))
        self._port_edit = QLineEdit("1234")
        self._port_edit.setFixedWidth(60)
        bl.addWidget(self._port_edit)

        bl.addStretch()

        self._btn_launch = QPushButton("▶  Launch")
        self._btn_launch.setFixedHeight(32)
        self._btn_launch.setEnabled(False)
        self._btn_launch.clicked.connect(self._launch_server)
        bl.addWidget(self._btn_launch)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_server)
        bl.addWidget(self._btn_stop)

        self._btn_quit = QPushButton("Quit")
        self._btn_quit.setFixedHeight(32)
        self._btn_quit.clicked.connect(self.close)
        bl.addWidget(self._btn_quit)

        # ── Root ───────────────────────────────────────────────────────
        root = QWidget()
        root_l = QVBoxLayout(root)
        root_l.setContentsMargins(4, 0, 4, 0)
        root_l.setSpacing(0)
        root_l.addWidget(sysbar)
        root_l.addWidget(main_split, 1)
        root_l.addWidget(btn_row)
        self.setCentralWidget(root)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Starting…")

    # ------------------------------------------------------------------
    def _apply_mono_font(self, w: QTextEdit) -> None:
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(self._font_size)
        w.setFont(f)

    def _change_font(self, delta: int) -> None:
        self._font_size = max(7, min(22, self._font_size + delta))
        for w in (self._config_preview, self._log_panel):
            f = w.font()
            f.setPointSize(self._font_size)
            w.setFont(f)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def _startup_load(self) -> None:
        # Load profiles and discover forks first (fast, no subprocess) —
        # then kick off hardware detection in a daemon thread so the window
        # is already fully visible before any PowerShell calls happen.
        self._profiles = load_profiles(self.settings_path)
        n = len(self._profiles)
        self._log(
            f"Loaded {n} profile(s) from {self.settings_path}"
            if n else
            f"[Warning] No profiles found in {self.settings_path}"
        )

        try:
            discover, _ = _get_fork_tools()
            self._forks = discover()
        except Exception as exc:
            self._log(f"[Warning] Fork discovery failed: {exc}")
            self._forks = []

        # Resolve manual fork path: persisted choice > LLAMA_CPP_DIR env var.
        # Persisted setting overrides env so that the user's last manual
        # selection in the GUI survives reboots.
        manual_path: Optional[Path] = None
        manual_source = ""  # "settings" | "env" | ""
        env_fork = os.environ.get("LLAMA_CPP_DIR", "")
        persisted = _persisted_fork_path()
        if persisted is not None and persisted.is_dir():
            manual_path = persisted.resolve()
            manual_source = "settings"
            self._log(f"[Fork] Loaded persisted path: {manual_path}")
        elif env_fork and Path(env_fork).is_dir():
            manual_path = Path(env_fork).resolve()
            manual_source = "env"

        # Prüfe ob der manuelle Pfad ein Container mit mehreren Forks ist
        # (z.B. C:\LAB\ai-local mit Unterverzeichnissen wie 1b_llama.cpp, atq_llama.cpp, ...)
        env_contains_forks = False
        if manual_path:
            try:
                for child in manual_path.iterdir():
                    if child.is_dir() and re.search(r"llama\.cpp", child.name, re.IGNORECASE):
                        # Prüfe ob dieser Fork ein lauffähiges Binary hat
                        has_binary = any(
                            (child / sub).is_file()
                            for sub in ["build/bin/Release/llama-server.exe",
                                        "build/bin/Debug/llama-server.exe",
                                        "build/bin/llama-server.exe",
                                        "build/bin/llama-server"]
                        )
                        if has_binary:
                            env_contains_forks = True
                            break
            except (OSError, PermissionError):
                pass

        self._fork_combo.blockSignals(True)
        self._fork_combo.clear()

        # If the persisted manual path matches one of the auto-discovered
        # forks, show it under its real name instead of as "📁 custom".
        # Avoids the cosmetic regression where every restart looked like
        # the path had been forgotten when it was actually loaded fine.
        matched_idx = -1
        if manual_path and self._forks and not env_contains_forks:
            for i, (_, p) in enumerate(self._forks):
                try:
                    if p.resolve() == manual_path:
                        matched_idx = i
                        break
                except OSError:
                    continue

        if matched_idx >= 0 and manual_path is not None:
            # Persisted path IS one of the discovered forks — restore by name.
            # The explicit `manual_path is not None` check is redundant at
            # runtime (matched_idx >= 0 implies it was matched against a
            # non-None path) but makes the implication legible to mypy /
            # Pylance, which can't infer it from the loop above.
            for name, path in self._forks:
                self._fork_combo.addItem(name, userData=path)
            self._fork_combo.setCurrentIndex(matched_idx)
            self._fork_path = self._forks[matched_idx][1]
            self._fork_path_lbl.setText(manual_path.name)
            src_label = ("persisted settings" if manual_source == "settings"
                         else "LLAMA_CPP_DIR")
            self._log(f"[Fork] Restored from {src_label}: "
                      f"{self._forks[matched_idx][0]}  →  {manual_path}")
            self._apply_fork(matched_idx)
        elif manual_path and not env_contains_forks:
            # Truly custom path outside the auto-discover scope. Label it
            # by directory name rather than the literal word "custom" so
            # the user can recognise their selection at a glance.
            label = f"📁 {manual_path.name}"
            self._fork_combo.addItem(label, userData=manual_path)
            self._fork_path = manual_path
            self._fork_combo.setCurrentIndex(0)
            self._fork_path_lbl.setText(manual_path.name)
            src_label = ("persisted settings" if manual_source == "settings"
                         else "LLAMA_CPP_DIR")
            self._log(f"[Fork] Using manual path from {src_label}: {manual_path}")
        elif self._forks:
            # Automatisch entdeckte Forks anzeigen
            for name, path in self._forks:
                self._fork_combo.addItem(name, userData=path)
            self._fork_combo.setCurrentIndex(0)
            if manual_path and env_contains_forks:
                self._fork_path = manual_path
                self._log(f"[Fork] Container '{manual_path.name}' enthält mehrere Forks:")
                for name, path in self._forks:
                    self._log(f"  - {name} → {path}")
            else:
                self._fork_path = self._forks[0][1] if self._forks else None
                self._log(f"Found {len(self._forks)} fork(s). Using: {self._forks[0][0]}")
            self._apply_fork(0)
        else:
            self._fork_combo.addItem("not found", userData=None)
            self._fork_path = None
            if manual_path and env_contains_forks:
                self._log(f"[Warning] LLAMA_CPP_DIR='{env_fork}' exists but no forks with llama-server binary found.")
            else:
                self._log("[Warning] No llama.cpp forks found. Set LLAMA_CPP_DIR.")
        self._fork_combo.blockSignals(False)

        # Hardware detection (spawns PowerShell on Windows) → background thread
        # so it never blocks the UI and never flashes a window.
        # Use signal/slot pattern instead of QTimer.singleShot from bg thread
        # to avoid potential PyQt6 deadlocks when COM is involved.
        self._log("Detecting system hardware…")
        self._hw_detect_worker = _HwDetectWorker(timeout=30.0)
        self._hw_detect_thread = QThread(self)
        self._hw_detect_worker.moveToThread(self._hw_detect_thread)
        self._hw_detect_thread.started.connect(self._hw_detect_worker.run)
        self._hw_detect_worker.finished.connect(self._hw_detect_done)
        self._hw_detect_worker.finished.connect(self._hw_detect_thread.quit)
        self._hw_detect_thread.finished.connect(self._hw_detect_thread.deleteLater)
        self._hw_detect_thread.start()

    # ------------------------------------------------------------------
    # Fork selection
    # ------------------------------------------------------------------
    def _on_fork_changed(self, index: int) -> None:
        self._fork_manual_override = True
        self._apply_fork(index)
        # Persist the user's combo-box selection too.
        path: Optional[Path] = self._fork_combo.itemData(index)
        if path is not None:
            try:
                app_settings.set_fork_path(path)
            except Exception as exc:
                self._log(f"[Warning] Could not save fork path: {exc}")

    def _apply_fork(self, index: int) -> None:
        path: Optional[Path] = self._fork_combo.itemData(index)
        if path is not None:
            os.environ["LLAMA_CPP_DIR"] = str(path)
            self._log(f"[Fork] → {path.name}")

    # ------------------------------------------------------------------
    # Performance target selection
    # ------------------------------------------------------------------
    def _on_perf_changed(self, index: int) -> None:
        """User picked a new performance target — persist + refresh view."""
        name = self._perf_combo.itemText(index).strip()
        try:
            app_settings.set_performance_target(name)
        except Exception as exc:
            self._log(f"[Warning] Could not save performance target: {exc}")
        self._log(f"[Perf] → {name}")
        # If a model is already selected, recompute the displayed config
        # so the user sees the effect immediately.
        entry = getattr(self, "_current_entry", None)
        if entry is not None and self._system is not None:
            try:
                self._show_config(entry)
            except Exception as exc:
                self._log(f"[Warning] Config refresh failed: {exc}")

    def _resolve_perf_target_for_profile(self, profile: ModelProfile):
        """Combine GUI choice with profile-level recommendation.

        GUI choice always wins; profile.performance_target is only used
        if the user hasn't picked anything (which currently never happens
        because the combo is initialised to "balanced", but stay robust).
        """
        gui_choice = self._perf_combo.currentText().strip() if hasattr(self, "_perf_combo") else None
        return resolve_performance_target(
            cli_choice=gui_choice,
            profile_choice=getattr(profile, "performance_target", "") or None,
        )

    def _hw_detect_done(
        self, s: Optional[SystemInfo], err: str = ""
    ) -> None:
        """Callback from hardware detection worker thread (via signal/slot)."""
        if s is not None:
            self._system = s
            self._update_sysinfo_labels(s)
            self._log(f"Hardware detected ({s.total_ram_gb:.0f}GB RAM, "
                      f"{s.total_vram_gb:.0f}GB VRAM, {len(s.gpus)} GPU(s)).")
        else:
            self._log(f"[Warning] Hardware detection failed: {err}")
            # Still allow model selection even without sysinfo
        self._start_scan()

    def _browse_fork_folder(self) -> None:
        """Manuellen Fork-Ordner auswählen (ähnlich wie Models folder)."""
        dialog = QFileDialog(self, "LLama.cpp Fork-Ordner auswählen")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        
        # Vorgabepfad: aktueller Fork oder Workspace
        if self._fork_path is not None:
            dialog.setDirectory(str(self._fork_path))
        elif self._forks:
            dialog.setDirectory(str(self._forks[0][1]))
        
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                new_path = Path(selected[0])
                self._set_manual_fork_path(new_path)
    
    def _set_manual_fork_path(self, path: Path) -> None:
        r"""Manuellen Fork-Pfad setzen und UI aktualisieren.

        Wenn der Pfad mehrere Forks enthält (z.B. C:\LAB\ai-local),
        werden alle Forks im Menü angezeigt statt nur "custom".
        """
        if not path.is_dir():
            QMessageBox.warning(self, "Ungültiger Ordner",
                              f"Der Ordner existiert nicht:\n{path}")
            return
        
        path = path.resolve()
        
        # Prüfe ob dieser Ordner mehrere Forks enthält
        child_forks = []
        try:
            for child in path.iterdir():
                if child.is_dir() and re.search(r"llama\.cpp", child.name, re.IGNORECASE):
                    has_binary = any(
                        (child / sub).is_file()
                        for sub in ["build/bin/Release/llama-server.exe",
                                    "build/bin/Debug/llama-server.exe",
                                    "build/bin/llama-server.exe",
                                    "build/bin/llama-server"]
                    )
                    if has_binary:
                        child_forks.append((child.name, child))
        except (OSError, PermissionError):
            pass
        
        self._fork_path = path
        self._log(f"[Fork] Pfad: {path}")
        
        self._fork_combo.blockSignals(True)
        self._fork_combo.clear()
        
        if child_forks:
            # Mehrere Forks im Ordner — alle anzeigen
            self._log(f"[Fork] '{path.name}' enthält {len(child_forks)} Fork(s):")
            for name, fork_path in child_forks:
                self._log(f"  - {name} → {fork_path}")
                self._fork_combo.addItem(name, userData=fork_path)
            self._fork_combo.setCurrentIndex(0)
            # LLAMA_CPP_DIR auf den Container setzen
            os.environ["LLAMA_CPP_DIR"] = str(path)
            self._fork_path_lbl.setText(path.name + " (📁)")
        else:
            # Single fork — label by directory name, not the literal "custom".
            # Keeps the combo readable when the user reopens the app and
            # also when they switch back from a multi-fork container.
            self._fork_combo.addItem(f"📁 {path.name}", userData=path)
            self._fork_combo.setCurrentIndex(0)
            self._fork_path_lbl.setText(path.name)
            os.environ["LLAMA_CPP_DIR"] = str(path)
        
        self._fork_combo.blockSignals(False)
        self._apply_fork(0)
        # Persist this manual choice so the next launch picks it up.
        try:
            app_settings.set_fork_path(path)
            self._log(f"[Fork] Saved as default: {path}")
        except Exception as exc:
            self._log(f"[Warning] Could not save fork path: {exc}")

    # ------------------------------------------------------------------
    # Background model scan
    # ------------------------------------------------------------------
    def _start_scan(self) -> None:
        try:
            if self._scan_thread is not None and self._scan_thread.isRunning():
                return
        except RuntimeError:
            self._scan_thread = None

        self._path_label.setText(f"Models: {self.models_path}")
        self._btn_refresh.setEnabled(False)
        self._btn_launch.setEnabled(False)
        self._model_list.clear()
        self._status.showMessage(f"Scanning {self.models_path} …")
        self._log(f"Scanning: {self.models_path}")

        if not self.models_path.exists():
            msg = (
                f"Models folder not found:\n  {self.models_path}\n\n"
                "Use '📂 Models folder' to pick the right location,\n"
                "or set the AUTOTUNER_MODELS environment variable."
            )
            self._config_preview.setPlainText(msg)
            self._status.showMessage(f"Folder not found: {self.models_path}")
            self._btn_refresh.setEnabled(True)
            return

        self._scan_worker = _ScanWorker(self.models_path)
        self._scan_thread = QThread(self)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.error.connect(self._scan_thread.quit)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._scan_thread.start()

    def _on_scan_done(self, entries: List[ModelEntry]) -> None:
        self._all_entries = entries
        self._btn_refresh.setEnabled(True)
        if not entries:
            self._config_preview.setPlainText(
                f"No *.gguf files found in:\n  {self.models_path}"
            )
            self._status.showMessage("No models found.")
            self._log("No models found.")
            return
        self._populate_list(entries)
        self._btn_launch.setEnabled(True)
        self._status.showMessage(f"{len(entries)} model(s) loaded.")
        self._log(f"Found {len(entries)} model(s).")

    def _on_scan_error(self, msg: str) -> None:
        self._btn_refresh.setEnabled(True)
        self._log(f"[Error] Scan failed: {msg}")
        self._status.showMessage(f"Scan error: {msg}")

    def _populate_list(self, entries: List[ModelEntry]) -> None:
        self._model_list.clear()
        groups = group_entries(entries)
        for group_name in sorted(groups.keys()):
            for entry in sorted(groups[group_name], key=lambda e: e.name.lower()):
                vision = " 👁" if entry.has_vision else ""
                item = QListWidgetItem(
                    f"{entry.name}{vision}  ({entry.size_gb:.1f} GB)"
                )
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self._model_list.addItem(item)

    def _apply_filter(self, text: str) -> None:
        q = text.strip().lower()
        self._populate_list(
            self._all_entries if not q
            else [e for e in self._all_entries if q in e.name.lower()]
        )

    def _browse_models(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select models folder", str(self.models_path)
        )
        if folder:
            self.models_path = Path(folder)
            try:
                app_settings.set_models_path(self.models_path)
                self._log(f"[Models] Saved as default: {self.models_path}")
            except Exception as exc:
                self._log(f"[Warning] Could not save models path: {exc}")
            self._start_scan()

    # ------------------------------------------------------------------
    # Config preview + options (single-click)
    # ------------------------------------------------------------------
    def _on_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _prev: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            return
        entry: ModelEntry = current.data(Qt.ItemDataRole.UserRole)
        if entry is None:
            return
        self._show_config(entry)

    def _show_config(self, entry: ModelEntry) -> None:
        """Called on model selection — updates checkboxes, auto-selects fork, refreshes preview.

        Gracefully handles the case when hardware detection has not yet
        completed (self._system is None).  The user will see a placeholder
        message and the config will update automatically once detection
        finishes.
        """
        if self._system is None:
            self._config_preview.setPlainText(
                "Hardware-Erkennung laeuft noch...\n\n"
                "Bitte warten Sie, bis die Systeminformationen geladen sind.\n"
                "Die Konfiguration wird automatisch aktualisiert."
            )
            return
        self._current_entry = entry
        self._current_draft = _find_draft_model(entry, self._all_entries)
        self._update_checkboxes(entry)
        profile = match_profile(entry.name, self._profiles)
        self._auto_select_fork(profile)
        self._update_config_text(entry, profile)

    def _update_checkboxes(self, entry: ModelEntry) -> None:
        """Set checkbox enabled/checked states (blockSignals prevents loop)."""
        mmproj = entry.mmproj
        has_vision = mmproj is not None
        for sig in (True, False):
            self._chk_vision.blockSignals(sig)
        self._chk_vision.blockSignals(True)
        self._chk_vision.setEnabled(has_vision)
        self._chk_vision.setChecked(has_vision)
        self._chk_vision.setText(
            f"Vision  ({mmproj.name})" if mmproj is not None else "Vision (no mmproj found)"
        )
        self._chk_vision.blockSignals(False)

        draft = self._current_draft
        has_draft = draft is not None
        self._chk_draft.blockSignals(True)
        self._chk_draft.setEnabled(has_draft)
        self._chk_draft.setChecked(has_draft)
        self._chk_draft.setText(
            f"Draft   {draft.name}  ({draft.size_gb:.1f} GB)"
            if draft is not None else "Draft (no assistant model found)"
        )
        self._chk_draft.blockSignals(False)

        # Reasoning/thinking detection — read the chat template from GGUF
        # metadata, fall back to a conservative filename heuristic when the
        # template is missing. This fixes the Qwen3-Coder false-positive:
        # the old heuristic matched any "qwen3" filename, but Qwen3-Coder
        # has no <think> tokens and llama-server logs "reasoning 0".
        has_thinking = entry.supports_thinking
        self._chk_thinking.blockSignals(True)
        self._chk_thinking.setEnabled(has_thinking)
        self._chk_thinking.setChecked(has_thinking)
        self._chk_thinking.blockSignals(False)

    def _auto_select_fork(self, profile: ModelProfile) -> None:
        """Auto-select fork from combo based on profile requirement.

        If the user has manually selected a fork (via dropdown or folder browse),
        respect that choice and do NOT override it — unless the profile requires
        a specific fork that is not available.
        """
        # Respect manual user override — only auto-switch if profile demands it
        if self._fork_manual_override:
            # Check if profile requires a specific fork
            if profile.server_binary:
                first = Path(profile.server_binary).parts[0]
                if not first.endswith(".cpp"):
                    first = first + ".cpp"
                first_l = first.lower()
                found = False
                for i in range(self._fork_combo.count()):
                    item_l = self._fork_combo.itemText(i).lower()
                    if item_l == first_l or item_l.rstrip(".cpp") in first_l:
                        found = True
                        break
                if not found:
                    self._log(
                        f"[Fork] Profile requires '{first}' but it's not available. "
                        f"Keeping manual selection: {self._fork_combo.currentText()}"
                    )
                # Keep manual selection regardless
            return

        # No manual override — apply profile-based auto-selection
        if profile.server_binary:
            first = Path(profile.server_binary).parts[0]
            if not first.endswith(".cpp"):
                first = first + ".cpp"
            first_l = first.lower()
            for i in range(self._fork_combo.count()):
                item_l = self._fork_combo.itemText(i).lower()
                if item_l == first_l or item_l.rstrip(".cpp") in first_l:
                    if self._fork_combo.currentIndex() != i:
                        self._fork_combo.blockSignals(True)
                        self._fork_combo.setCurrentIndex(i)
                        self._fork_combo.blockSignals(False)
                        self._apply_fork(i)
                        self._log(f"[Fork] Auto-selected: {self._fork_combo.itemText(i)}")
                    return
        else:
            # No specific fork required — keep current selection, don't reset
            pass

    def _on_option_toggled(self) -> None:
        """Checkbox changed → recompute context/memory with new options."""
        if self._current_entry is not None and self._system is not None:
            profile = match_profile(self._current_entry.name, self._profiles)
            self._update_config_text(self._current_entry, profile)

    def _update_config_text(self, entry: ModelEntry, profile: ModelProfile) -> None:
        """Recompute config using current checkbox states, refresh preview."""
        assert self._system is not None
        use_vision = self._chk_vision.isChecked() and self._chk_vision.isEnabled()
        use_draft  = self._chk_draft.isChecked()  and self._chk_draft.isEnabled()

        entry_for_cfg = copy.copy(entry)
        if not use_vision:
            entry_for_cfg.mmproj = None

        cfg = compute_config(
            model=entry_for_cfg, system=self._system, profile=profile,
            draft_model=self._current_draft if use_draft else None,
            user_ctx=None, force_mlock=False,
            perf_target=self._resolve_perf_target_for_profile(profile),
        )

        W = 64
        bar = "─" * W
        lines = [bar]
        lines.append(f"Model   : {entry.name}")
        lines.append(
            f"Profile : {profile.display_name}"
            + (f"  ({profile.source_file})" if profile.source_file else "")
        )
        if profile.notes:
            for i in range(0, len(profile.notes.strip()), W - 10):
                prefix = "Notes   : " if i == 0 else "          "
                lines.append(f"{prefix}{profile.notes.strip()[i:i+W-10]}")
        if entry.mmproj:
            vis = "✓" if use_vision else "✗"
            lines.append(f"Vision  : {entry.mmproj.name}  [{vis}]")
        if self._current_draft:
            drf = "✓" if use_draft else "✗"
            lines.append(f"Draft   : {self._current_draft.name}  [{drf}]")
        if profile.server_binary:
            lines.append(f"Requires: {profile.server_binary}")
        lines.append(bar)

        if cfg.full_offload:
            placement = f"GPU full offload  ({entry.n_layers or '?'} layers)"
        elif cfg.ngl > 0:
            placement = f"Hybrid — {cfg.ngl} layers GPU + CPU"
        else:
            placement = "CPU only"

        lines += [
            f"Placement       : {placement}",
            f"Perf target     : {cfg.performance_target}",
            f"Context         : {cfg.ctx:,} tokens",
            f"KV cache quant  : K={cfg.cache_k}  V={cfg.cache_v}",
            f"Threads         : {cfg.threads}  (batch: {cfg.batch_threads})",
            f"Batch / ubatch  : {cfg.batch} / {cfg.ubatch}",
            f"Flash attention : {'on' if cfg.flash_attn else 'off'}",
        ]
        if cfg.mlock:
            lines.append("mlock           : on")
        s = cfg.sampling
        lines.append(
            f"Sampling        : temp={s.get('temperature')}  "
            f"top_k={s.get('top_k')}  top_p={s.get('top_p')}  "
            f"min_p={s.get('min_p')}  rep={s.get('repeat_penalty')}"
        )
        lines += [
            bar,
            "Memory estimate (with current options):",
            f"  Model GPU : ~{cfg.estimated_model_vram_gb:5.1f} GB"
            f"   (free VRAM: {self._system.free_vram_gb:.1f} GB)",
            f"  Model CPU : ~{cfg.estimated_model_ram_gb:5.1f} GB"
            f"   (free RAM:  {self._system.free_ram_gb:.1f} GB)",
            f"  KV cache  : ~{cfg.estimated_kv_gb:5.1f} GB",
            bar,
        ]
        self._config_preview.setPlainText("\n".join(lines))
    # ------------------------------------------------------------------
    # System info — non-blocking (daemon thread → signal/slot)
    # ------------------------------------------------------------------
    def _sysinfo_async(self) -> None:
        if self._sysinfo_busy:
            return
        self._sysinfo_busy = True
        threading.Thread(target=self._sysinfo_bg, daemon=True).start()

    def _sysinfo_bg(self) -> None:
        """Background thread for hardware detection (runs every 6 seconds)."""
        import time
        try:
            start = time.monotonic()
            s = detect_system()
            elapsed = time.monotonic() - start
            # Qt widgets are thread-safe for updates from any thread in PyQt6
            # (they auto-marshal to the GUI thread internally)
            self._update_sysinfo_labels(s)
            self._log(f"[SysInfo] Refreshed ({elapsed:.1f}s)")
        except Exception as exc:
            self._log(f"[Warning] Sysinfo detection failed: {exc}")
        finally:
            self._sysinfo_busy = False

    def _update_sysinfo_labels(self, s: SystemInfo) -> None:
        """Update system info labels in the UI bar.

        Always updates self._system to ensure model selection and config
        preview work even if hardware detection happened after startup.
        """
        self._system = s
        
        # VRAM-Anzeige
        if s.total_vram_gb > 0:
            self._vram_lbl.setText(
                f"VRAM: {s.free_vram_gb:.1f} / {s.total_vram_gb:.1f} GB free"
            )
        else:
            self._vram_lbl.setText("VRAM: keine GPU")
        
        # RAM-Anzeige
        self._ram_lbl.setText(
            f"RAM: {s.free_ram_gb:.1f} / {s.total_ram_gb:.1f} GB free"
        )
        
        # CPU-Anzeige
        if s.cpu_name:
            self._cpu_lbl.setText(f"CPU: {s.cpu_name}")
        
        # GPU-Anzeige mit Utilization
        if s.gpus:
            gpu_parts = []
            for g in s.gpus:
                util = f"{g.gpu_util_percent:.0f}%" if g.gpu_util_percent > 0 else "—"
                gpu_parts.append(f"{g.name} ({util})")
            self._gpu_lbl.setText("GPU: " + ", ".join(gpu_parts))
        else:
            self._gpu_lbl.setText("GPU: keine")
        
        self._log(f"[SysInfo] CPU={s.cpu_name}, VRAM={s.free_vram_gb:.1f}/{s.total_vram_gb:.1f}GB, RAM={s.free_ram_gb:.1f}/{s.total_ram_gb:.1f}GB, GPU={[g.name for g in s.gpus]}")

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------
    def _resolve_binary(self, profile: ModelProfile, use_draft: bool,
                         model_name: str) -> str:
        try:
            _, resolve = _get_fork_tools()
        except Exception:
            return "llama-server"
        if ("gemma-4" in model_name.lower() or "gemma4" in model_name.lower()) and use_draft:
            spec = profile.server_binary or "ik_llama.cpp/llama-server"
        elif profile.server_binary:
            spec = profile.server_binary
        else:
            spec = "llama-server"
        resolved = resolve(spec)
        self._log(f"[Binary] {spec!r} → {resolved}")
        return resolved

    # ------------------------------------------------------------------
    # Server control
    # ------------------------------------------------------------------
    def _launch_server(self) -> None:
        if self._server is not None and self._server.is_running():
            QMessageBox.information(
                self, "Already running", "Stop the running server first."
            )
            return

        if self._current_entry is None:
            QMessageBox.warning(
                self, "No model selected", "Click a model in the list first."
            )
            return

        if self._system is None:
            QMessageBox.warning(
                self, "System info unavailable",
                "Hardware detection has not completed yet. Please wait a moment and try again."
            )
            return

        use_vision   = self._chk_vision.isChecked()   and self._chk_vision.isEnabled()
        use_draft    = self._chk_draft.isChecked()    and self._chk_draft.isEnabled()
        use_thinking = self._chk_thinking.isChecked() and self._chk_thinking.isEnabled()

        # Build a copy of entry so we can control mmproj inclusion
        entry = copy.copy(self._current_entry)
        if not use_vision:
            entry.mmproj = None

        profile = match_profile(entry.name, self._profiles)
        cfg = compute_config(
            model=entry, system=self._system, profile=profile,
            draft_model=self._current_draft if use_draft else None,
            user_ctx=None, force_mlock=False,
            perf_target=self._resolve_perf_target_for_profile(profile),
        )

        host = self._host_edit.text().strip() or "127.0.0.1"
        try:
            port = int(self._port_edit.text().strip())
        except ValueError:
            port = 1234

        server_binary = self._resolve_binary(profile, use_draft, entry.name)
        # Clean alias so RooCode/clients show a readable name, not the file path
        alias = _clean_model_name(entry.name)
        cmd = build_command(
            model=entry, config=cfg, profile=profile,
            draft_model=self._current_draft if use_draft else None,
            server_binary=server_binary, host=host, port=port,
            extra_args=["-a", alias], use_thinking=use_thinking,
        )

        self._log("\n" + "─" * 60)
        self._log(f"Starting: {' '.join(cmd)}")
        self._log(f"Options : vision={use_vision} draft={use_draft} thinking={use_thinking}")

        self._server = _TerminalProcess(cmd)
        try:
            self._server.start()
        except FileNotFoundError:
            self._log(f"[Error] Binary not found: {cmd[0]}")
            self._log("  → Check fork selection or set LLAMA_CPP_DIR / LLAMA_SERVER")
            self._server = None
            return

        pid = self._server.proc.pid if self._server.proc else "?"
        self._log(f"[AutoTuner] Server started — PID: {pid}")
        self._log("[AutoTuner] Server output → separate terminal window")
        self._log(f"[AutoTuner] Web UI → http://{host}:{port}")

        self._btn_launch.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status.showMessage(f"Running — PID {pid} — http://{host}:{port}")

    def _stop_server(self) -> None:
        if self._server is None:
            return
        self._log("[AutoTuner] Stopping server…")
        srv = self._server
        self._server = None
        srv.stop()   # sends signal + waits in daemon thread
        self._btn_launch.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.showMessage("Server stopped.")
        self._log("[AutoTuner] Stop signal sent.")

    # ------------------------------------------------------------------
    # Server crash detection
    # ------------------------------------------------------------------
    def _poll_server(self) -> None:
        if self._server is None:
            return
        if not self._server.is_running():
            code = self._server.returncode()
            self._server = None
            self._btn_launch.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._log(f"[AutoTuner] Server exited (code {code}).")
            self._status.showMessage(f"Server exited (code {code}).")

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self._log_panel.append(msg.rstrip("\n"))
        sb = self._log_panel.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        # Guard against already-deleted QThread (deleteLater race)
        try:
            if self._scan_thread is not None and self._scan_thread.isRunning():
                self._scan_thread.quit()
                self._scan_thread.wait(2000)
        except RuntimeError:
            pass
        self._scan_thread = None

        if self._server is not None and self._server.is_running():
            reply = QMessageBox.question(
                self, "Server still running",
                "Stop the server and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                if event is not None:
                    event.ignore()
                return
            self._stop_server()

        if event is not None:
            event.accept()


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="qt_launcher",
                                description="AutoTuner Qt GUI launcher")
    p.add_argument("--models-path",   default=str(_default_models_path()))
    p.add_argument("--settings-path", default=str(_default_settings_path()))
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # Hide the parent console on Windows when launched via python.exe
    if os.name == "nt":
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("AutoTuner")
    window = MainWindow(
        models_path=Path(args.models_path),
        settings_path=Path(args.settings_path),
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()