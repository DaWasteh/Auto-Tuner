@echo off
REM Wechselt ins Verzeichnis dieser .bat-Datei (egal von wo aus gestartet)
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM Lokale Pfade (nur in diesem Prozess gesetzt, nicht systemweit).
REM Wenn du das Repo bewegst oder die Modelle/llama.cpp woanders liegen,
REM hier anpassen.
REM ---------------------------------------------------------------------------
set "AUTOTUNER_MODELS=C:\LAB\ai-local\models"
set "LLAMA_CPP_DIR=C:\LAB\ai-local\llama.cpp"

REM Auto-Tuner starten und alle übergebenen Argumente weiterreichen.
REM Statt der ENV-Vars oben kannst du auch CLI-Flags benutzen:
REM   python auto_tuner.py --models-path "C:\LAB\ai-local\models" ^
REM                        --llama-cpp-dir "C:\LAB\ai-local\llama.cpp" %*
python auto_tuner.py %*

REM Fenster offen halten, falls ein Fehler kommt — sonst klappt es zu
pause