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
reads `~/.autotuner/autotuner_settings.json`, authenticates to `/v1/models`, and
calls `pi.registerProvider("autotuner", …)`. Pi's model refresh invokes the same
live catalogue endpoint through `refreshModels`.

If you do not want the extension to read AutoTuner's settings file, set both:

```text
AUTOTUNER_API_URL=http://127.0.0.1:1233
AUTOTUNER_API_KEY=<key copied from AutoTuner Settings>
```

The Settings dialog's **Copy Pi setup** button copies exactly those two
newline-separated assignments. `AUTOTUNER_DATA_DIR` is respected for portable
or test installations. `AUTOTUNER_CONTROL_API_PORT` and
`AUTOTUNER_CONTROL_API_KEY` are also accepted as shared gateway overrides;
`AUTOTUNER_API_URL` has the highest endpoint precedence.

## Behaviour

- Provider ID: `autotuner`
- API type: `openai-completions`
- Base URL: `http://127.0.0.1:1233/v1` by default
- Catalogue: dynamic, from AutoTuner's scanned models
- Switching: the request's `model` ID triggers AutoTuner's serialized model
  transition and waits for a successful llama-server health check
- Costs: zero, because inference is local
- Pi-level `reasoning_effort`: disabled for broad llama.cpp compatibility;
  AutoTuner's saved thinking/reasoning launch setting remains authoritative

If Pi starts while AutoTuner is closed or still scanning, the provider may first
appear empty and prints a credential-free warning. Start/finish the AutoTuner
scan, then refresh models or `/reload` Pi.

For endpoint contracts and security details, see
[`docs/control-api.md`](../../docs/control-api.md).
