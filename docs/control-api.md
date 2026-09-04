# Local external-control API

AutoTuner exposes one stable, OpenAI-compatible loopback endpoint for Pi, the
Supercalc benchmark, and other trusted local clients. It is **off by default**.

Open **⋯ → Settings → External control API**, enable it, keep or change port
`1233`, and click **OK**. The listener binds only to `127.0.0.1`; wildcard and
LAN addresses are rejected. Every endpoint except `/health` requires the random
bearer token shown (masked) in Settings.

The control plane and generated llama-server processes are separate:

1. A client requests an AutoTuner model's stable ID, optionally naming one of
   the discovered llama-server builds (`runtime_id`).
2. AutoTuner serializes the request on the Qt GUI thread.
3. It stops the previous **API-managed** llama-server, without touching servers
   the user launched manually.
4. It selects the requested build in the toolbar (exactly like a click), selects
   the scanned model, and reuses AutoTuner's saved performance target, Expert
   profile, mmproj, draft/MTP, thinking, n-gram, prompt-cache, GPU, and mode
   controls. The API never invents its own tuning.
5. It starts llama-server and waits for `/health` to return 200.
6. Only then does the original client request continue to the backend. Streaming
   SSE bytes are flushed without buffering.

A failed stop, launch, health check, unknown runtime, or unsupported single-shot
model returns a structured HTTP error instead of routing to the wrong model.

## Client discovery file (v5.3.9)

AutoTuner writes a small JSON document next to its settings whenever the
gateway starts, stops, or the token is regenerated:

```text
~/.autotuner/control_api.json        (or <AUTOTUNER_DATA_DIR>/control_api.json)
```

```json
{
  "schema": 1,
  "enabled": true,
  "base_url": "http://127.0.0.1:1233",
  "port": 1233,
  "version": "5.3.9",
  "pid": 12345,
  "started_at": "2026-09-03T18:00:00Z",
  "token": "…"
}
```

- `token` is present **only** while `enabled` is `true`. Disabling the API,
  stopping AutoTuner, or a start failure rewrites the file with
  `"enabled": false` and no token.
- The file is written atomically and, on POSIX, with mode `0600`. On Windows it
  lives in the user profile like the settings file that already stores the
  token.
- Clients should read this file instead of `autotuner_settings.json`, which can
  hold tens of megabytes of benchmark evidence.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Minimal unauthenticated gateway liveness |
| `GET` | `/v1/models` | OpenAI-format list of runnable chat/server models |
| `GET` | `/api/v1/models` | Extended catalogue: paths, `runnable`/`unavailable_reason`, `size_bytes`, `quant`, `params_b`, `family`, `architecture`, `default_runtime_id` |
| `GET` | `/api/v1/runtimes` | Every llama-server build in the toolbar dropdown, with backend and probed build identity |
| `GET` | `/api/v1/status` | Idle/loading/ready state plus the full launch snapshot of the active model |
| `POST` | `/api/v1/switch` | Atomically activate `{"model_id":"…"}` on an optional `runtime_id` and wait until ready |
| `POST` | `/api/v1/stop` | Stop only the API-managed model server |
| any supported OpenAI request | `/v1/*` | Switch from the JSON `model` field, rewrite it to the backend alias, and proxy to llama-server |

### `GET /api/v1/runtimes`

```json
{
  "runtimes": [
    {
      "id": "b10786-vulkan-llama-cpp",
      "label": "b10786_vulkan_llama.cpp",
      "server_binary": "L:\\LAB\\ai-local\\b10786_vulkan_llama.cpp\\build\\bin\\Release\\llama-server.exe",
      "backend": "vulkan",
      "build": "b10786",
      "build_number": 10786,
      "build_info": "b10786-de8656bd9",
      "is_default": true,
      "available": true,
      "unavailable_reason": ""
    }
  ],
  "default_runtime_id": "b10786-vulkan-llama-cpp",
  "active_runtime": null
}
```

- `id` is a stable slug of the toolbar label. Resolve it from this list; never
  guess it.
- `backend` is one of `vulkan`, `hip`, `cuda`, `sycl`, `metal`, `opencl`, `cpu`,
  or `unknown` (ROCm builds report `hip`).
- `build`/`build_info` come from the cached `llama-server --version` probe and
  are `null` until the binary has been probed successfully.
- `available` is `false` when a build folder contains no runnable
  `llama-server`; selecting it returns HTTP 409 `runtime_unavailable`.
- The list is refreshed whenever the toolbar build list changes.

### `POST /api/v1/switch`

Request body:

```json
{"model_id": "qwen3.8-27b-ud-q3_k_xl--1a2b3c4d5e", "runtime_id": "b10786-hip-llama-cpp", "timeout_s": 900}
```

- `model_id` (required): a catalogue ID from `/v1/models`.
- `runtime_id` (optional): a build from `/api/v1/runtimes`. Omitted or empty
  keeps the build currently selected in the toolbar. Selecting a runtime changes
  the toolbar selection and is persisted like a user click.
- `timeout_s` (optional): how long the request waits for `/health`
  (1–86400 s, default 900). Non-numeric values return HTTP 400.
- Idempotent: an already-ready model on the same (or unspecified) runtime
  returns immediately without a restart; a different runtime forces a real
  stop/start transition.
- The response is the `/api/v1/status` document below with `"status":"ready"`.

### `GET /api/v1/status`

```json
{
  "status": "ready",
  "active_model": "qwen3.8-27b-ud-q3_k_xl--1a2b3c4d5e",
  "loading_model": null,
  "active_since": 1756922400.0,
  "inflight_requests": 0,
  "endpoint": "http://127.0.0.1:1233",
  "ready": true,
  "backend_url": "http://127.0.0.1:1234",
  "alias": "Qwen3.8-27B-UD-Q3_K_XL",
  "backend_api_key": null,
  "active_runtime": "b10786-hip-llama-cpp",
  "default_runtime_id": "b10786-hip-llama-cpp",
  "pid": 4242,
  "log_path": null,
  "runtime": {"id": "b10786-hip-llama-cpp", "label": "b10786_hip_llama.cpp", "server_binary": "…\\llama-server.exe", "backend": "hip", "build": "b10786", "build_number": 10786, "build_info": "b10786-de8656bd9"},
  "model": {"id": "…", "name": "Qwen3.8-27B-UD-Q3_K_XL", "path": "I:\\models\\…gguf", "quant": "Q3_K_XL", "ftype": "Q3_K_XL", "params_b": 27.0, "size_bytes": 12345678901, "architecture": "qwen35", "draft_model_path": null, "mmproj_path": null, "profile": "Qwen3.8 (Alibaba)", "profile_file": "qwen3_8.yaml"},
  "launch": {"ctx_size": 262144, "gpu_layers": 999, "threads": 12, "batch_threads": 16, "batch": 2048, "ubatch": 512, "kv_type_k": "q4_0", "kv_type_v": "q4_0", "flash_attention": "on", "spec_type": "ngram-map-k4v", "draft_n_max": null, "main_gpu": 0, "tensor_split": null, "parallel": 1, "thinking": true, "profile": "Qwen3.8 (Alibaba)", "performance_target": "throughput", "mode": "chat", "load_mode": "mmap", "n_cpu_moe": null, "port": 1234},
  "devices": [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vendor": "amd", "backend": "hip", "device": "ROCm0", "vram_mb": 32768, "free_vram_mb": 30000}],
  "env": {"HIP_VISIBLE_DEVICES": "0"},
  "command_line": ["…\\llama-server.exe", "-m", "…", "--api-key", "<redacted>"]
}
```

- `endpoint` is the gateway itself; `backend_url` is the **direct** llama-server
  URL. Benchmark clients may call llama-server's `/props`, `/tokenize`,
  `/metrics`, `/health`, and streaming endpoints there directly, bypassing the
  proxy. `alias` is the value llama-server expects in the `model` field, and
  `backend_api_key` is the separate `--api-key` of that server when one exists.
- Every field below `ready` is `null` while no API-managed model is ready.
  `log_path` is only set on Linux/macOS, where llama-server output is captured
  to a per-launch file; Windows launches use a separate console.
- `command_line` is the full argv with credentials redacted. `env` contains only
  the backend visibility variables AutoTuner sets; secrets are never included.
- `launch` values come from the tuned configuration that started the server.

### Error codes

`{"error":{"message":"…","type":"autotuner_control_error","code":"…"}}`

| HTTP | `code` | Meaning |
|---|---|---|
| 400 | `invalid_request`, `invalid_json`, `invalid_action` | Malformed body or parameters |
| 401 | `unauthorised` | Missing/invalid bearer token |
| 404 | `model_not_found` | Unknown or vanished catalogue ID |
| 409 | `model_not_runnable` | Standalone drafter, embedding profile, or single-shot diffusion runner |
| 409 | `runtime_unavailable` | Unknown `runtime_id`, a build without a runnable server, or a toolbar selection failure |
| 409 | `model_busy` | The active model has in-flight proxied requests; retry later |
| 409 | `no_active_model` | Proxy request without a `model` and no active model |
| 409 | `autotuner_busy` | An exclusive benchmark or OCR workflow owns the GUI |
| 409 | `launch_failed` | AutoTuner refused the launch (compatibility, VRAM, missing binary) |
| 500 | `launch_exception`, `invalid_backend`, `internal_error` | Unexpected failure |
| 502 | `backend_unavailable`, `backend_exited` | llama-server unreachable or exited while loading |
| 503 | `hardware_pending`, `shutting_down` | Retry after hardware detection, or AutoTuner is closing |
| 504 | `switch_timeout`, `stop_timeout` | The model did not become ready / the old server did not exit in time |

### Example (PowerShell)

```powershell
$base = "http://127.0.0.1:1233"
$discovery = Get-Content "$HOME\.autotuner\control_api.json" | ConvertFrom-Json
$headers = @{ Authorization = "Bearer $($discovery.token)" }

$runtimes = Invoke-RestMethod "$base/api/v1/runtimes" -Headers $headers
$models = Invoke-RestMethod "$base/v1/models" -Headers $headers

$body = @{ model_id = $models.data[0].id; runtime_id = $runtimes.runtimes[0].id; timeout_s = 900 } | ConvertTo-Json
$status = Invoke-RestMethod "$base/api/v1/switch" -Method Post -Headers $headers -ContentType "application/json" -Body $body
Invoke-RestMethod "$($status.backend_url)/props"   # talk to llama-server directly
```

## Pi integration

The dynamic extension is [`integrations/pi/autotuner.ts`](../integrations/pi/autotuner.ts).
It reads `control_api.json` first, falls back to a bounded regex scan of the
legacy settings file, fetches `/v1/models` before registering the `autotuner`
provider, and implements Pi's `refreshModels` callback. It does not put a fixed
model list in `models.json`.

Copy the file to Pi's global extension folder:

```text
~/.pi/agent/extensions/autotuner.ts
```

Then enable the API in AutoTuner, wait for the model scan to finish, and run
`/reload` in Pi. Select a model under provider **AutoTuner** with `/model`.
See [`integrations/pi/README.md`](../integrations/pi/README.md) for environment
and troubleshooting details.

## Benchmark campaigns (Supercalc)

A campaign client iterates models × runtimes: read `/api/v1/runtimes` and
`/api/v1/models`, call `/api/v1/switch` with each pair, take `backend_url` and
`alias` from the response, run its workload against llama-server directly, and
call `/api/v1/stop` when finished. `409 autotuner_busy`/`model_busy` and
`503 hardware_pending` are transient and should be retried with backoff;
`launch_failed`, `runtime_unavailable`, and `switch_timeout` mark that campaign
point as failed.

## Credential and network safety

- The gateway cannot bind beyond loopback and only proxies to a loopback
  llama-server URL returned by AutoTuner.
- Tokens are generated with at least 256 bits of randomness. Regenerating one
  invalidates prior clients after Settings is accepted and immediately rewrites
  the discovery file.
- `AUTOTUNER_CONTROL_API_ENABLED`, `AUTOTUNER_CONTROL_API_PORT`, and
  `AUTOTUNER_CONTROL_API_KEY` may override persisted gateway settings.
- The Pi extension also accepts `AUTOTUNER_API_URL` and `AUTOTUNER_API_KEY`.
- AutoTuner strips the gateway credential before forwarding. If the underlying
  llama-server has its own `--api-key`, only that separate key is sent upstream,
  and `/api/v1/status` reveals it only to clients that already hold the gateway
  token.
- API keys and key-file values are redacted from launch logs and from
  `command_line`. Request payloads, query strings, and bearer tokens are not
  logged.
- No permissive browser CORS header is emitted.
