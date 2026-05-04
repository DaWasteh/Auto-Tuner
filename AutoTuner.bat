@echo off
REM Wechselt ins Verzeichnis dieser .bat-Datei (egal von wo aus gestartet)
cd /d "%~dp0"

REM Optional: Pfad zu deinen Modellen fest hinterlegen, damit du --models-path
REM nie wieder angeben musst. Zeile auskommentieren wenn nicht gewünscht.
set "AUTOTUNER_MODELS=C:\LAB\ai-local\models"
set "LLAMA_CPP_DIR=C:\LAB\ai-local"

REM Auto-Tuner starten und alle übergebenen Argumente weiterreichen
python auto_tuner.py %*

REM Fenster offen halten, falls ein Fehler kommt — sonst klappt es zu
pause