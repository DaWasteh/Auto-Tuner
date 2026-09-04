# llama.cpp b10797 compatibility audit

Date: 2026-09-04

AutoTuner release: v5.4.0

Audited upstream range: b10786 (`de8656bd94f1163188125542534e4bcbc9f9fb1f`)
through b10797 (`832fd6f1741ad9f66fb2a785002396700666595a`)

## Authority and scope

The audited source and binary are the exact local upstream build:

```text
L:\LAB\ai-local\b10797_vulkan_llama.cpp   (MSVC 19.51, build/bin/Release)
version: 0.3.0-dev (build 10797, commit 832fd6f17)
```

The source tree resolves to tag `b10797`. The audit covered the complete
eleven-commit range, the exact `llama-server --help`, every option emitted by
shipped YAML profiles, and the loader/hparams diff. The earlier
[`llama-b10786-audit.md`](llama-b10786-audit.md) remains the v5.3.9 record.

## CLI and AutoTuner feature compatibility

The b10786 and b10797 server help each expose 331 distinct long-option names.
The set difference is empty in both directions, so no flag was added, removed,
or renamed. The profile-flag regression test now reads the newest captured
help text (`.pi/b10797-llama-server-help.txt`).

## Upstream changes

| Commit | Change | AutoTuner impact |
|---|---|---|
| `9a4843cf2` | `llama_hparams::n_expert_used_max()`; loader and buffer-type probing use the per-layer maximum | none: AutoTuner reads only the scalar `expert_count`; Puzzle profile gate stays b10786 |
| `c5a5535e6` | GBNF grammar for empty object schemas | none: response-side JSON schema handling |
| `d230ddd76` | CMake no longer rebuilds all sources per commit (`llama-version.h.in`) | none: build recipes unchanged |
| `6703d7894`, `f9f09f02c` | SYCL fused residual chains, MKL FA flag refactor | transparent |
| `95ef7fc16` | OpenCL quantized lm_head / GEMV / medium-batch GEMM for speculative decoding | transparent |
| `8c1a25166` | CUDA MMVQ/MMQ crossover tuning for SM87 | transparent |
| `832fd6f17` | s390x q5_1 fix | transparent |
| `42f0225fe`, `d30500b83`, `e107984bc` | server test parallelism, Snapdragon CI, docs | none |

No architecture was added to `src/llama-arch.cpp`; the classifier comparison
from the b10786 audit remains valid.

## MTP sidecar loading contract (verified on b10786)

The v5.3.9 Quick suite for Qwen3.8 Flash Next failed in every drafter lane.
The captured llama-server output shows the real cause:

```text
common_speculative_init_result: loading draft model 'mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf'
llama_model_load: error loading model: check_tensor_dims: tensor 'output_hc_norm.weight' not found
common_speculative_init_result: loading draft model 'mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf'
llama_model_load: error loading model: check_tensor_dims: tensor 'token_embd.weight' not found
```

Both sidecars declare `general.architecture = qwen4exp` with
`block_count = 49` and `nextn_predict_layers = 1`, and contain only
`blk.48.*` plus (for the non-shared file) `token_embd.weight` and
`output.weight`. Mainline has no head-only loader: a `-md` model is loaded by
the architecture's normal `load_tensors()` path, which requires the family's
root tensors. Qwen3.8 Flash Next additionally needs `output_hc_norm.weight`
(hyper-connection head norm). Working sidecars on the same machine confirm the
contract: the Qwen3.6 MTP heads carry `token_embd`, `output_norm`, and
`output`, and the Gemma 4 assistant uses its own `gemma4-assistant`
architecture.

AutoTuner therefore adds a preflight in `check_draft_model_build()`: for a
same-architecture MTP sidecar it collects the root tensors of every target
shard and of the sidecar (header reads only), and refuses the launch with the
missing names when a required root tensor is absent. `output.weight` and
`rope_freqs.weight` are exempt because the loader treats them as optional.
The Quick suite lists such lanes under *Failed/skipped* and proceeds with the
remaining lanes.

## Conclusion

AutoTuner's command/runtime feature layer is compatible with b10797 without
changes. The v5.4.0 changes are the sidecar preflight, the scrollable
performance summary dialog, documentation, and the rebuilt Windows artifact.
Release evidence is in [`v5.4.0-validation.md`](v5.4.0-validation.md).
