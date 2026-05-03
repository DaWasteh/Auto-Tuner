# AutoTuner for llama.cpp

Interactive launcher for `llama-server` that **detects your hardware**,
**scans your local GGUF collection**, and **auto-tunes** context length,
KV-cache quantization, GPU offload, threading, and batch size to fit in
the RAM/VRAM you actually have free — without manual edits.

```
────────────────────────────────────────────────────────────────
  AutoTuner for llama.cpp  —  interactive launcher
────────────────────────────────────────────────────────────────

  CPU     : Intel Core Ultra 9 285K  (24 cores / 8 P-Cores, 16 E-Cores)
  RAM     : 47,4 GB total, 37.4 GB free
  GPU(s)  : 1 × AMD Radeon RX 9070 XT (16.0 GB, 13.1 GB free)

Available models:
────────────────────────────────────────────────────────────────

  [Alibaba/Qwen3.6]
     1.  👁 Qwen3.6-27B-UD-Q3_K_XL                          12.5 GB  (128k native)
     2.  👁 Qwen3.6-35B-A3B-UD-IQ3_S                        14.8 GB  (128k native)

  [Mistral AI]
     3.  👁 Devstral-Small-2-24B-Instruct-2512-IQ4_XS       12.1 GB  (256k native)
     ...

Select a model [1-12, q to quit]:
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

## Installation

```bash
git clone https://github.com/<you>/llama-cpp-auto-tuner
cd llama-cpp-auto-tuner
pip install -r requirements.txt
```

You also need a working `llama-server` binary from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) somewhere
on your `PATH` (or pass `--server /path/to/llama-server`).

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
http://127.0.0.1:8080
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
| `--port N` | Server port (default `8080`) |
| `--ctx N` | Override the auto-tuned context length |
| `--model SUBSTR` | Skip the menu, pick a model by name substring |
| `--dry-run` | Print the command, don't start the server |
| `--yes / -y` | Skip the launch confirmation prompt |
| `-- <args...>` | Anything after `--` is forwarded to `llama-server` |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AUTOTUNER_MODELS` | `./models` | Where to scan for `*.gguf` files |
| `LLAMA_SERVER` | `llama-server` | Path or name of the server binary |
| `LLAMA_CPP_DIR` | (auto-detected) | Your llama.cpp checkout. If set, the auto-tuner will look for `build/bin/[Release/]llama-server[.exe]` inside it. |

### Server binary auto-discovery

If `--server` isn't passed and the binary isn't on `PATH`, the auto-tuner
walks upward from the script directory and `cwd`, looking for common
layouts:

```
<root>/llama.cpp/build/bin/Release/llama-server.exe   (Windows MSVC)
<root>/llama.cpp/build/bin/llama-server               (Linux / macOS)
<root>/ai-local/llama.cpp/build/bin/...               (common dev-workspace layout)
<root>/1bllama.cpp/build/bin/...                      (BitNet fork, used by Ternary-Bonsai)
```

So if your tree looks like this:

```
C:\LAB\
├── Auto Tuner\         ← clone of this repo
└── ai-local\
    ├── llama.cpp\      ← your build
    ├── models\         ← your models (with mmproj if available)
    └── 1bllama.cpp\    ← BitNet fork (only needed for Ternary-Bonsai)
```

…then `python auto_tuner.py` Just Works without any flags. No models or
binaries need to be copied into the repo.

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
├── settings/
│   ├── _default.yaml
│   ├── qwen3.yaml
│   ├── gemma-4.yaml
│   ├── devstral.yaml
│   ├── ministral.yaml
│   ├── mistral-medium.yaml
│   ├── bonsai.yaml
│   └── frankenmerger.yaml
├── requirements.txt
└── README.md
```

## License

MIT.
