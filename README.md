# AutoTuner for llama.cpp

Interactive launcher for `llama-server` that **detects your hardware**,
**scans your local GGUF collection**, and **auto-tunes** context length,
KV-cache quantization, GPU offload, threading, and batch size to fit in
the RAM/VRAM you actually have free — without manual edits.

```
────────────────────────────────────────────────────────────────
  AutoTuner for llama.cpp  —  interactive launcher
────────────────────────────────────────────────────────────────

============================================================
  DEBUG / VERBOSE MODE SELECTION
============================================================
  1. Debugging OFF (standard)
  2. Debugging ON (alle Kategorien)
------------------------------------------------------------
  Kategorie-Debugging (einzelne Bereiche):
  3. Hardware-Erkennung (GPU/RAM/CPU)
  4. Model-Scanning & Profil-Matching
  5. Server-Pfad-Suche (llama.cpp)
  6. Konfigurations-Berechnung (KV-Cache, Kontext)
------------------------------------------------------------
Wahl [1-6] (default 1):
[AutoTuner] Debugging deaktiviert.
============================================================


========================================
  QUANTIZATION MODE SELECTION
========================================
  1. Standard-Quant (llama.cpp)
  2. Turbo-Quant (tq_llama.cpp)
----------------------------------------
Select mode [1/2] (default 1):
[AutoTuner] Standard-Quant mode selected.
========================================

OS:   Windows 11
CPU:  Intel(R) Core(TM) Ultra 9 285K (24C/24T)
RAM:  47.4 GB total, 18.1 GB free
GPU1: [amd] AMD Radeon RX 9070 XT (15.9 GB total, 15.1 GB free)
      (ignored: [intel] Intel(R) Graphics, 2.0 GB — too small or auxiliary)

[AutoTuner] Scanning models in: C:\LAB\ai-local\models
[AutoTuner] Loaded 17 profile(s) from C:\GitHub\Auto Tuner\settings

Available models:
────────────────────────────────────────────────────────────────
  ...

  [Alibaba/Qwen3.6]
    7.  👁 Qwen3.6-27B-UD-Q3_K_XL                                   13.5 GB  (256k native)
    8.  👁 Qwen3.6-35B-A3B-UD-IQ3_S                                 12.7 GB  (256k native)

  [Frankenmerger]
    9.    Archon-14B.Q6_K                                          11.3 GB  (40k native)
  ...

  [Google]
  ...
    16.  👁 gemma-4-E2B-it-assistant-Q8_0                             0.1 GB  (128k native)
    17.  👁 gemma-4-E2B-it-BF16                                       8.7 GB  (128k native)
  ...

  [IBM]
    21.    granite-4.1-30b-IQ4_XS                                   14.4 GB  (128k native)
    22.    granite-4.1-3b-UD-Q8_K_XL                                 4.0 GB  (128k native)
  ...

  [Mistral AI/Mistral-Medium]
    36.  👁 Mistral-Medium-3.5-128B-UD-IQ3_XXS                       45.9 GB  (256k native)

  [NVIDIA]
    37.  👁 NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ4_XS  18.2 GB  (1024k native)

  [PrismML]
    38.    Bonsai-8B                                                 1.1 GB  (64k native)
  ...

Select a model [1-40, q to quit]: 15
[AutoTuner] Found draft model: gemma-4-26B-A4B-it-assistant-Q8_0
Vision aktivieren? (mmproj-gemma-4-26B-A4B-it-BF16.gguf) [Y/n] y
Draft-Modell aktivieren? (gemma-4-26B-A4B-it-assistant-Q8_0) [Y/n] y
Thinking/Reasoning aktivieren? (<|think|> / <|reserved_special_token>) [Y/n] y
────────────────────────────────────────────────────────────────
Model:    gemma-4-26B-A4B-it-UD-IQ4_XS
Profile:  Gemma 4 (Google)  (gemma-4.yaml)
Notes:    Gemma ist empfindlich gegenüber repeat_penalty > 1.0. E2B/E4B = multimodal (Text+Bild+Audio), 26B-A4B + 31B = Text+Bild. Thinking-Modus aktivierbar durch <|think|> am Anfang des System-Prompts. Tipp: Manche Community-Tests zeigen, dass Gemma 4 für Coding sogar mit temp=1.5 besser performt - bei Bedarf mit `-- --temp 1.5` überschreiben.
Vision:   mmproj-gemma-4-26B-A4B-it-BF16.gguf
────────────────────────────────────────────────────────────────
  Placement       : GPU full offload (ngl=all of 60)
  Context         : 111,616 tokens
  KV cache quant  : K=q4_0  V=q4_0
  Threads         : 8 (batch: 16)
  Batch / ubatch  : 1024 / 512
  Flash attention : on
  Sampling        : temp=1.0 top_k=64 top_p=0.95 min_p=0.0 rep=1.0

  Memory estimate:
    model on GPU  ~  12.3 GB    (free VRAM:   15.1 GB)
    model on CPU  ~   0.0 GB    (free RAM:    18.1 GB)
    KV cache      ~  17.4 GB
────────────────────────────────────────────────────────────────
[AutoTuner] Found server binary: C:\LAB\ai-local\ik_llama.cpp\build\bin\Release\llama-server.exe
Launch llama-server now? [Y/n]
```

## Features

- **Interactive terminal menu** — pick from whatever GGUFs are in your
  models folder, no editing required.
- **Hardware auto-detection** — works on **AMD (ROCm)**, **NVIDIA**,
  **Intel**, and **Apple Silicon** (unified memory). Multi-GPU is
  supported via automatic `--tensor-split`.
- **Free-memory aware** — context length and KV quant are picked to
  use the RAM/VRAM that's actually free *right now*, not a hard-coded
  cap. The original v1 cap of 16k context is gone.
- **Per-family YAML profiles** in `settings/` — override sampling,
  max context, chat template, and llama-server flags per model family.
  Easy for contributors to extend without touching Python.
- **mmproj auto-pairing** — if your model has a sibling
  `mmproj-*.gguf`, vision is enabled automatically (longest-prefix
  match picks the most specific one).
- **Reads GGUF metadata** — pulls `n_layers` and `context_length`
  straight from the file so partial GPU offload (`-ngl`) is exact.
  
### Vision control

You can disable vision (mmproj) support in two ways:

1. **Command-line flag**:

   ```bash
   python auto_tuner.py --model "Qwen3.6" --novision
   ```

## Installation

```bash
git clone https://github.com/<you>/llama-cpp-auto-tuner
cd llama-cpp-auto-tuner
pip install -r requirements.txt
```

You also need a working `llama-server` binary. The tuner automatically discovers binaries in common local setups (like `C:\LAB\ai-local\`), or you can specify one via `--server`.

## Usage

Point it at a folder of `*.gguf` models — it will recurse:

```bash
python auto_tuner.py --models-path /path/to/models
```

Or set the environment variable once:

```bash
export AUTOTUNER_MODELS=/path/to/models     # Linux / macOS
setx  AUTOTUNER_MODELS  D:\models           # Windows
python auto_tuner.py
```

Pick a model from the menu. Once it's running, point your client at:

```
http://127.0.0.1:1234
```

Works with the built-in **llama.cpp Web UI**, **VS Code** extensions
like Continue / Cline, **Open WebUI**, or any OpenAI-API client.

### Useful flags

| Flag | Description |
|---|---|
| `--models-path PATH` | Folder to scan (default `./models`, env `AUTOTUNER_MODELS`) |
| `--settings-path PATH` | Folder with YAML profiles (default `./settings`) |
| `--server PATH` | Path to `llama-server` (default looks on `$PATH`, env `LLAMA_SERVER`) |
| `--host HOST` | Bind address (default `127.0.0.1`) |
| `--port N` | Server port (default `1234`) |
| `--ctx N` | Override the auto-tuned context length |
| `--model SUBSTR` | Skip the menu, pick a model by name substring |
| `--dry-run` | Print the command, don't start the server |
| `--yes / -y` | Skip the launch confirmation prompt |
| `--force-mlock` | Force `--mlock` / `--no-mmap` (prevents VRAM/RAM paging) |
| `-- <args...>` | Anything after `--` is forwarded to `llama-server` |

### Memory locking (`--mlock` / `--no-mmap`)

The auto-tuner automatically decides whether to enable `--mlock` and `--no-mmap`
based on available system resources. These flags pin model data in physical
memory (RAM/VRAM) and prevent the OS from paging it to disk, which is critical
for stable inference performance.

**Automatic behavior:**

| Scenario | Condition | Result |
|---|---|---|
| **Full GPU offload** | `total_vram > 8 GB` AND `free_vram > model_size + 2 GB` | `--mlock --no-mmap` enabled |
| **Partial / CPU offload** | `total_ram > 32 GB` AND `free_ram > model_ram_on_cpu + 8 GB` | `--mlock --no-mmap` enabled |
| **Insufficient memory** | Safety reserve not met | Disabled (fallback to default mmap) |

**Force memory locking:**

Use `--force-mlock` to override the automatic decision and always enable
memory locking when the OS permits it:

```bash
python auto_tuner.py --force-mlock
```

This is useful when you know your system has enough memory but the tuner's
conservative thresholds would otherwise skip it.

**Debug output:**

The tuner prints the mlock decision before every launch:

```
  [mlock] decision: model=Qwen3.6-35B-A3B-UD-Q6_K
         full_offload=True  vram=18.5GB  ram=0.0GB
         sys: total_vram=24.0GB  free_vram=5.2GB  total_ram=32.0GB  free_ram=12.1GB
         force_mlock=False  -> mlock=True  no_mmap=True
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AUTOTUNER_MODELS` | `./models` | Where to scan for `*.gguf` files |
| `LLAMA_SERVER` | `llama-server` | Path or name of the server binary |
| `LLAMA_CPP_DIR` | (auto-detected) | Your llama.cpp checkout. If set, the auto-tuner will look for `build/bin/[Release/]llama-server[.exe]` inside it. |

### Server binary auto-discovery

The tuner automatically searches for binaries in common local layouts.
If you have a workspace like this, it "Just Works" without any flags:

```
C:\GitHub\
└── Auto Tuner\         ← clone of this repo
C:\LAB\
└── ai-local\
    ├── llama.cpp\      ← standard build
    ├── tq_llama.cpp\   ← Turbo-Quant build
    ├── ik_llama.cpp\   ← Gemma 4 (MTP) build
    ├── 1b_llama.cpp\   ← BitNet fork (Ternary-Bonsai)
    └── models\         ← your models
```

It looks for `llama-server` inside these directories (including `build/bin/...` subpaths).

#### Quantization Modes

When you start the tuner, you can choose between:

1.  **Standard-Quant**: Uses standard `llama.cpp` binaries.
2.  **Turbo-Quant**: Uses the `tq_llama.cpp` binary for faster inference.

#### Specialized Binary Logic

The tuner intelligently selects the best binary based on your model and settings:
- **Gemma 4 (with Draft)** $\rightarrow$ uses `ik_llama.cpp` (MTP support).
- **Gemma 4 (without Draft)** $\rightarrow$ uses standard `llama.cpp`.
- **Ternary-Bonsai** $\rightarrow$ uses `1b_llama.cpp`.
- **Turbo-Quant Mode** $\rightarrow$ uses `tq_llama.cpp`.

Example — run Devstral, override context, and pass an extra flag:

```bash
python auto_tuner.py --model Devstral --ctx 131072 -y -- --metrics
```

## Adding profiles for new models

Drop a new YAML file into `settings/`. The filename doesn't matter;
the `patterns:` list does. The longest pattern that appears as a
substring of the model filename wins.

```yaml
# settings/my-model.yaml
display_name: "My Model"
patterns:
  - my-model
  - my-model-base

max_context: 131072
recommended_kv_quant: q8_0

sampling:
  temperature: 0.7
  top_k: 40
  top_p: 0.9
  min_p: 0.05
  repeat_penalty: 1.05

# Optional:
chat_template: chatml
extra_args:
  - --no-context-shift
notes: >
  Anything you want to remind yourself about this model.
```

Profiles with empty `patterns:` become the fallback when nothing else
matches. See `settings/_default.yaml`.

## How the auto-tuning works

1. **Detect**: total / free RAM, every GPU's total / free VRAM, total
   CPU cores.
2. **Place the model**: full GPU offload if it fits, else partial
   offload using the GGUF's exact `n_layers`, else CPU only.
3. **Compute the KV budget**: free VRAM (after the model) plus free
   RAM (minus a safety reserve).
4. **Pick KV quant + context**: try q8 → q5 → q4, pick the highest
   quality that fits the profile's `max_context`. Round context down
   to a multiple of 1024.
5. **Threads / batch**: scale with placement (full GPU offload needs
   fewer CPU threads than CPU-only inference; long context wants
   smaller batches to keep prompt-prefill memory bounded).
6. **Multi-GPU**: split tensors proportionally to free VRAM and pick
   the GPU with the most free memory as `--main-gpu`.

## Project layout

```
auto_tuner/
├── auto_tuner.py        # main entry: menu + glue
├── hardware.py          # CPU + multi-vendor GPU detection
├── scanner.py           # GGUF scanner + mmproj pairing + metadata reader
├── settings_loader.py   # YAML profile loader and matcher
├── tuner.py             # config calculation + llama-server command builder
├── launcher.py          # subprocess + Ctrl+C handling (Windows + Unix)
...
├── settings/
│   ├── _default.yaml
│   ...
│   ├── ministral.yaml
│   ├── bonsai.yaml
│   ...
├── requirements.txt
└── README.md
```

## Building llama.cpp and forks

# Recommended build settings for this system:
# - Ninja generator
# - native CPU optimizations
# - static build
# - Release mode
# - 20 parallel build jobs

```
## Building llama.cpp and forks - Example

# Example-System:
# - Intel Core Ultra 9 285K
# - AMD Radeon RX 9070 XT 16GB
# - G.Skill Trident Z 48GB DDR5-8400MHz (2x24GB)


# Main-Fork
cd C:\LAB\ai-local
git clone https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp
if (Test-Path build) { Remove-Item -Recurse -Force build }
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_COMPILER="cl.exe" -DGGML_CCACHE=OFF
cmake --build build --config Release --parallel 20

# PrismML-Fork
cd C:\LAB\ai-local
git clone https://github.com/PrismML-Eng/llama.cpp 1b_llama.cpp
cd 1b_llama.cpp
git checkout prism
if (Test-Path build) { Remove-Item -Recurse -Force build }
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_COMPILER="cl.exe" -DGGML_CCACHE=OFF
cmake --build build --config Release --parallel 20

# Ikawrakow MTP-Fork
cd C:\LAB\ai-local
git clone https://github.com/ikawrakow/ik_llama.cpp ik_llama.cpp
cd ik_llama.cpp
git fetch origin
git checkout ik/gemma4
if (Test-Path build) { Remove-Item -Recurse -Force build }
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_COMPILER="cl.exe" -DGGML_CCACHE=OFF
cmake --build build --config Release --parallel 20

# TurboQuant-Fork
cd C:\LAB\ai-local
git clone https://github.com/TheTom/llama-cpp-turboquant tq_llama.cpp
cd tq_llama.cpp
git checkout thetom
if (Test-Path build) { Remove-Item -Recurse -Force build }
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_COMPILER="cl.exe" -DGGML_CCACHE=OFF
cmake --build build --config Release --parallel 20

# Atomic MTP-Fork
cd C:\LAB\ai-local
git clone https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant atq_llama.cpp
cd atq_llama.cpp
git checkout Atomic
if (Test-Path build) { Remove-Item -Recurse -Force build }
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_COMPILER="cl.exe" -DGGML_CCACHE=OFF
cmake --build build --config Release --parallel 20
```

## License

MIT.
