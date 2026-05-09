@echo off
REM Startet den AutoTuner Qt-Launcher ohne sichtbares Terminal-Fenster.
REM Verwendet pythonw.exe (kein Konsolenfenster) statt python.exe.

cd /d "%~dp0"

REM ── Pfade anpassen (gleiche Einstellungen wie AutoTuner.bat) ──────────────
set "AUTOTUNER_MODELS=C:\LAB\ai-local\models"
set "LLAMA_CPP_DIR=C:\LAB\ai-local\llama.cpp"

REM start "" öffnet den Prozess entkoppelt vom aktuellen cmd-Fenster.
REM pythonw.exe = Python ohne Konsolfenster (wie .pyw-Skripte).
start "" pythonw "%~dp0qt_launcher.py" %*