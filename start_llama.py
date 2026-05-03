import subprocess
import sys
import os
from pathlib import Path
from threading import Thread
from copy import deepcopy

# --- Pfade definieren ---
HERE = Path(__file__).resolve().parent
BIN_DIR = Path(r"C:\LAB\ai-local\llama.cpp\build\bin\Release")
MODELS_DIR = HERE / "models"
SERVER_EXE = BIN_DIR / "llama-server.exe"
LOG_FILE = HERE / "llama_server.log"         # Hier landet dein Log!

def handle_output(process, log_file):
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8', errors='replace')
        sys.stdout.write(decoded_line)
        sys.stdout.flush()
        log_file.write(decoded_line)
        log_file.flush()

def apply_preserve_thinking_mode(messages):
    if not messages:
        return messages

    messages = deepcopy(messages)

    preserve_thinking_system_prompt = (
        "Arbeite komplexe Aufgaben sorgfältig und schrittweise intern aus."
        "Gib standardmäßig nur die finale Antwort aus."
        "Wenn der Nutzer ausdrücklich nach deinem Denkprozess fragt, antworte entsprechend."
    )

    first_message = messages[0]
    if isinstance(first_message, dict) and first_message.get("role") == "system":
        existing = first_message.get("content", "")
        if preserve_thinking_system_prompt not in existing:
            first_message["content"] = (
                existing.rstrip()
                + "\n\n"
                + preserve_thinking_system_prompt
            ).strip()
        messages[0] = first_message
    else:
        messages.insert(0, {"role": "system", "content": preserve_thinking_system_prompt})

    return messages

def start_text_vision_engine():
    cpu_count = os.cpu_count() or 8                    # Fallback auf 8, falls None
    threads = cpu_count // 2 if cpu_count > 8 else 8
    print(f"[llama.cpp] Starte LLM auf Port 1234...")
    print(f"[llama.cpp] Logs werden parallel gespeichert in: {LOG_FILE}")
    print("-" * 50)
    
    cmd = [
        str(SERVER_EXE),
        "-m", str(MODELS_DIR / "Qwen3.6-27B-UD-Q3_K_XL.gguf"), 
        "--mmproj", str(MODELS_DIR / "mmproj-Qwen3.6-27B-BF16.gguf"),
        
        # --- Hardware & Kontext ---
        "-c", "193536",
        # 1Mio = 1048576
        # Qwen3.5 = 262144 Max (32768, 65536, 131072)
        # Gemma 4 = 131072 Max (E2B & E4B), 262144 Max (26B-A4B & 31B)
#        "--rope-scaling", "yarn",     # YaRN-Technik für die Erweiterung
#        "--rope-scale", "4",          # Skalierungsfaktor (262k * 4 = 1.04M)
        "-ngl", "99",                  # Layer auf GPU
        "-t", str(threads),            # Optimierte Threads für Intel Core Ultra 200 Serie
        "-b", "1024",                  # Batch-Größe
        "--ubatch-size", "512",        # (Optional) Physischer Batch, oft performanter auf 512 oder 1024
        "-fa", "on",                   # Flash-Attention aktiviert
#        "--jinja",                    # Modelinternes JINJA-Template nutzen
#        "--verbose",                  # DEBUGGING
        # "chat_template_kwargs": {"preserve_thinking": True}, # Erlaubt es dem Modell, "thinking..." im Chat zu behalten (nützlich für längere Antworten oder Code-Generierung)
        # Der Denkmodus gehört in die Client-Seite bei messages, nicht in cmd.
        # Für llama.cpp wird das über eine Hilfsfunktion an die Messages angepasst.

        # --- GEMMA 4 VISION SETTINGS (Für Roo Code Web-Screenshots) ---
#        "--image-min-tokens", "1120", # Zwingend erforderlich für Gemma 4! Verhindert Server-Abstürze beim Bilder-Upload
#        "--image-max-tokens", "1120", # 1120 ist ideal für detaillierte Code-Screenshots, UI-Strukturen und OCR
        # Gemma 4 unterstützt ausschließlich die exakten Budgets 70, 140, 280, 560 oder 1120

        # --- K/V-Cache Quantisierung & Speicheroptimierung ---
        "-ctk", "q8_0",           # Quantisiert den K-Cache (Key) [q4_0 sollte kaum Qualiverlust haben]
        "-ctv", "q8_0",           # Quantisiert den V-Cache (Value) [q4_0 sollte kaum Qualiverlust haben]
#        "--no-mmap",             # Keine Auslagerung auf SSD alles in VRAM/RAM
        "--mlock",                # Versucht das Modell im RAM/VRAM festzupinnen (verhindert Paging)
        "--numa", "distribute",   # Optimiert Memory-Access (besser als nur "1" bei Hybrid-CPUs)

        # --- Host & Port ---
        "--port", "1234",
        "--host", "0.0.0.0",

        # --- NEUE GENERIERUNGS-PARAMETER ---
        "--temp", "0.7",              # Temperatur: 0.0 = strikt/logisch, 1.0 = sehr kreativ (Roo Code überschreibt die Temperatur für striktes Coding ohnehin über die API)
        "--top-k", "20",              # Top-K: Begrenzt die Auswahl auf die K wahrscheinlichsten Tokens (für 9B 20-30)
        "--top-p", "0.8",            # Top-P: Begrenzt auf Tokens, deren kumulierte Wahrscheinlichkeit P ergibt
        "--min-p", "0.0",             # Min-P: Schneidet Tokens ab, die im Vergleich zum besten Token zu unwahrscheinlich sind (für 9B 0.1)
        "--repeat-penalty", "1.05",    # Wiederholungsstrafe: 1.0 = deaktiviert, >1.0 = bestraft Wiederholungen
        "--presence-penalty", "0.0",  # Präsenzstrafe: Bestraft die Generierung von Tokens, die bereits im Kontext vorhanden sind (0.0 = deaktiviert, >0.0 = bestraft)
    ]

    print(f"Starte Server mit {threads} Threads und 65536 Kontext...")
    
    log_file = open(LOG_FILE, "a", encoding="utf-8")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_thread = Thread(target=handle_output, args=(process, log_file), daemon=True)
    output_thread.start()
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[llama.cpp] Beende den Server sicher...")
        process.terminate()
        log_file.close()

if __name__ == "__main__":
    start_text_vision_engine()