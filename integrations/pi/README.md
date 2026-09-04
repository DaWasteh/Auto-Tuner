# AutoTuner provider for Pi

`autotuner.ts` is a dependency-free Pi extension that discovers AutoTuner's
current model catalogue at runtime. A static `~/.pi/agent/models.json` cannot do
that: it is read as configuration and does not fetch newly scanned GGUF files.

## Install

1. In AutoTuner, open **⋯ → Settings**, enable **External control API**, and
   accept the dialog.
2. Wait for AutoTuner's model scan to finish.
3. Copy `autotuner.ts` to:

   ```text
   ~/.pi/agent/extensions/autotuner.ts
   ```

4. Start Pi or run `/reload`, then choose provider **AutoTuner** under `/model`.

Pi's extension loader runs the async factory before normal startup. The factory
locates the gateway credentials, authenticates to `/v1/models`, and calls
`pi.registerProvider("autotuner", …)`. Pi's model refresh invokes the same live
catalogue endpoint through `refreshModels`.

## Credential discovery

The extension resolves the gateway in this order; the first hit wins:

1. `AUTOTUNER_API_URL` + `AUTOTUNER_API_KEY` (the Settings dialog's
   **Copy Pi setup** button copies exactly these two assignments).
2. `AUTOTUNER_CONTROL_API_PORT` + `AUTOTUNER_CONTROL_API_KEY`, the shared
   gateway overrides also honoured by AutoTuner itself.
3. `~/.autotuner/control_api.json`, the small discovery document AutoTuner
   v5.3.9+ rewrites whenever the gateway starts, stops, or the token is
   regenerated. It carries the token only while `enabled` is `true`, so a
   disabled gateway is reported as disabled instead of as a missing key.
4. `~/.autotuner/autotuner_settings.json` for older AutoTuner versions. The
   file can hold tens of megabytes of benchmark evidence, so only a bounded
   window is scanned with a targeted regex and it is never `JSON.parse`d.
5. `http://127.0.0.1:1233` without a key: the provider registers empty and
   prints a credential-free warning.

`AUTOTUNER_DATA_DIR` relocates the folder used in steps 3 and 4 (portable or
test installations). `AUTOTUNER_API_URL` has the highest endpoint precedence.

## Behaviour

- Provider ID: `autotuner`
- API type: `openai-completions`
- Base URL: `http://127.0.0.1:1233/v1` by default
- Catalogue: dynamic, from AutoTuner's scanned models
- Switching: the request's `model` ID triggers AutoTuner's serialized model
  transition and waits for a successful llama-server health check. Clients that
  want a progress indicator can call `POST /api/v1/switch` first; it is
  idempotent and returns the full `/api/v1/status` snapshot.
- Costs: zero, because inference is local
- Pi-level `reasoning_effort`: disabled for broad llama.cpp compatibility;
  AutoTuner's saved thinking/reasoning launch setting remains authoritative

If Pi starts while AutoTuner is closed or still scanning, the provider may first
appear empty and prints a credential-free warning. Start/finish the AutoTuner
scan, then refresh models or `/reload` Pi.

For endpoint contracts and security details, see
[`docs/control-api.md`](../../docs/control-api.md).
