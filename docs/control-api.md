# Local external-control API

AutoTuner v5.3.4 can expose one stable, OpenAI-compatible loopback endpoint for
Pi and other trusted local clients. It is **off by default**.

Open **⋯ → Settings → External control API**, enable it, keep or change port
`1233`, and click **OK**. The listener binds only to `127.0.0.1`; wildcard and
LAN addresses are rejected. Every endpoint except `/health` requires the random
bearer token shown (masked) in Settings.

The control plane and generated llama-server processes are separate:

1. A client requests an AutoTuner model's stable ID.
2. AutoTuner serializes the request on the Qt GUI thread.
3. It stops the previous **API-managed** llama-server, without touching servers
   the user launched manually.
4. It selects the scanned model and reuses AutoTuner's saved performance target,
   Expert profile, mmproj, draft/MTP, thinking, n-gram, prompt-cache, GPU, and
   mode controls.
5. It starts llama-server and waits for `/health` to return 200.
6. Only then does the original client request continue to the backend. Streaming
   SSE bytes are flushed without buffering.

A failed stop, launch, health check, or unsupported single-shot model returns a
structured HTTP error instead of routing to the wrong model.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Minimal unauthenticated gateway liveness |
| `GET` | `/v1/models` | OpenAI-format list of runnable chat/server models |
| `GET` | `/api/v1/models` | Extended catalogue, including paths and reasons that non-server models are unavailable |
| `GET` | `/api/v1/status` | Idle/loading/ready state and active model ID |
| `POST` | `/api/v1/switch` | Atomically activate `{"model_id":"…"}` and wait until ready |
| `POST` | `/api/v1/stop` | Stop only the API-managed model server |
| any supported OpenAI request | `/v1/*` | Switch from the JSON `model` field, rewrite it to the backend alias, and proxy to llama-server |

Example (PowerShell):

```powershell
$base = "http://127.0.0.1:1233"
$key = "paste-the-key-from-AutoTuner-Settings"
$headers = @{ Authorization = "Bearer $key" }

$models = Invoke-RestMethod "$base/v1/models" -Headers $headers
$model = $models.data[0].id

$body = @{
  model = $model
  messages = @(@{ role = "user"; content = "Hello" })
  stream = $false
} | ConvertTo-Json -Depth 8

Invoke-RestMethod "$base/v1/chat/completions" `
  -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

## Pi integration

The dynamic extension is [`integrations/pi/autotuner.ts`](../integrations/pi/autotuner.ts).
It reads the same per-user AutoTuner settings file, fetches `/v1/models` before
registering the `autotuner` provider, and implements Pi's `refreshModels`
callback. It does not put a fixed model list in `models.json`.

Copy the file to Pi's global extension folder:

```text
~/.pi/agent/extensions/autotuner.ts
```

Then enable the API in AutoTuner, wait for the model scan to finish, and run
`/reload` in Pi. Select a model under provider **AutoTuner** with `/model`.
See [`integrations/pi/README.md`](../integrations/pi/README.md) for environment
and troubleshooting details.

## Credential and network safety

- The gateway cannot bind beyond loopback and only proxies to a loopback
  llama-server URL returned by AutoTuner.
- Tokens are generated with at least 256 bits of randomness. Regenerating one
  invalidates prior clients after Settings is accepted.
- `AUTOTUNER_CONTROL_API_ENABLED`, `AUTOTUNER_CONTROL_API_PORT`, and
  `AUTOTUNER_CONTROL_API_KEY` may override persisted gateway settings.
- The Pi extension also accepts `AUTOTUNER_API_URL` and `AUTOTUNER_API_KEY`.
- AutoTuner strips the gateway credential before forwarding. If the underlying
  llama-server has its own `--api-key`, only that separate key is sent upstream.
- API keys and key-file values are redacted from launch logs. Request payloads,
  query strings, and bearer tokens are not logged.
- No permissive browser CORS header is emitted.
