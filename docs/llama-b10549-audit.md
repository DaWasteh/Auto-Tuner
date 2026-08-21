# llama.cpp b10549 integration audit

AutoTuner v5.1.9 was audited against the exact upstream range below.

| Tag | Commit | Date |
|---|---|---|
| `b10441` | `0177dcc7300bad8914bb838baabce87899812491` | 2026-08-15 |
| `b10549` | `b2e5e9b28b2484fbf94b543432ece638996a8b97` | 2026-08-21 |

The range contains **108 upstream commits**. Evidence was checked from a
full-history upstream checkout, the exact source diff, generated server option
tables, and a locally built current Windows/Vulkan `llama-server`.

## Command-line compatibility

No AutoTuner-emitted `llama-server` option was removed or renamed in this
range. The only new ordinary model/server option is:

- **b10541 / PR #23255:** `-mmdev, --mmproj-device DEVICE` selects one exact
  backend-qualified device (`Vulkan1`, `CUDA0`, etc.) for the multimodal
  projector. `none` disables projector offload.

This option matters on AutoTuner's multi-GPU path. Before b10541, upstream MTMD
initialized the first visible GPU when no projector device was supplied, while
AutoTuner budgeted the complete projector on its selected `main_gpu`. A spread
across Vulkan0 + Vulkan1 could therefore budget the projector on the R9700 but
allocate it on the RX 9070 XT. v5.1.9 now derives the post-visibility-remap
device and emits `--mmproj-device` automatically whenever an exact runtime
device map is known. Help-based compatibility pruning removes the complete
flag/value pair on b10540 and older forks.

The second new argument-like setting, `dedup-cache-models` (b10505 / PR
#27346), is deliberately **preset-only** for router mode. It hides a cached
model when a preset resolves to the same file; it is not a normal
`llama-server` command-line switch and therefore does not belong in the
AutoTuner launch UI. The existing `--models-dir`, `--models-preset`, and
`--models-max` advanced router surface remains available through explicit
Extra CLI flags.

## New model and speculative support

- **Kimi-K3 text model (b10448 / PR #26185):** llama.cpp now has a dedicated
  `kimi-k3` converter and runtime for the 2.8T-A104B hybrid KDA/MLA MoE. The
  converter explicitly skips `vision_tower.*` and `mm_projector.*`, so this is
  text-path support, not yet a claim of local K3 vision support. AutoTuner's
  profile now requires b10448+, classifies K3 as hybrid/MoE, and retains its 1M
  context and official sampling.
- **BailingMoE3 / Ling 3.0 (b10460 / PR #26608):** conversion and inference
  landed for Ling-3.0 Flash (124B-A5.1B) and Tiny (7.9B-A1.3B), including
  integrated and separate NEXTN/MTP heads. AutoTuner now claims the exact
  `bailingmoe3` architecture, requires b10460+, uses the official
  `temperature=0.6`, `top_p=0.95`, `top_k=20`, and counts the published seven
  full-attention layers instead of a generic hybrid estimate.
- **DSpark formats (b10467 / PR #26275):** SpecForge/speculators-format
  checkpoints, reduced draft vocabularies, and bonus-anchor layouts are now
  supported automatically by AutoTuner's existing `draft-dspark` path.
- **Granite SWA/MoE SWA (b10514 / PR #25505):** the new `granite_swa`
  architecture adds per-layer SWA, attention sinks, and per-layer RoPE/NoPE.
  AutoTuner already reads generic sliding-window pattern metadata and expert
  counts, so no special launch flag is required.
- **LFM2 DSpark (b10540 / PR #27383):** DSpark rollback snapshots now work for
  LFM2 targets. Existing architecture-aware draft detection benefits without a
  new AutoTuner control.
- **LFM2/LFM2MoE tensor split (b10549 / PR #26993):** tensor-parallel placement
  now supports these architectures. AutoTuner's existing device/split inputs
  benefit automatically; no new command-line option was added.

## MTMD, OCR, server, and API changes

- `--models-dir` can load MTP assistant models (b10444 / PR #24431); this is a
  router feature and does not replace AutoTuner's local target/drafter pairing.
- Processed MTMD chunks are stored as prompt-cache placeholders (b10476 / PR
  #27278), improving repeated multimodal prompts transparently.
- DeepSeek-OCR's SAM convolution keeps im2col in F32 (b10497 / PR #26727), and
  LFM2 tiling/thumbnail fixes landed. These are rebuild benefits; the existing
  OCR request and `--cache-ram` gates remain valid.
- Authenticated router model endpoints are now private (PR #26347). AutoTuner
  binds localhost and does not rely on unauthenticated router administration.
- Sleep handling was refactored and `/metrics` remains available while a model
  sleeps (PR #27376). Existing health/metrics behavior remains compatible.
- `--docker-repo` no longer incorrectly enables router mode, and startup models
  are loaded lazily after router initialization. No AutoTuner argument changed.

## Backend and performance changes

Transparent rebuild benefits include:

- HIP no longer receives CUDA's UMA override (PR #27083).
- Vulkan gains tiled transpose, safer queue cleanup, one-time Q8 KV
  dequantization for coopmat1 (b10517), and FP32 Q quantization calculations in
  Flash Attention (b10539).
- CUDA gains dense decode tuning and static cuBLAS workspace handling; Metal,
  SYCL, OpenCL, WebGPU, and OpenVINO receive correctness/performance fixes.
- GGUF conversion can evict processed layer weights, reducing peak conversion
  RAM for very large models.

None changes AutoTuner's KV type vocabulary, context semantics, tensor split,
or emitted Flash Attention state.

## Why master reported b10548 while the release page showed b10545

This was not a llama.cpp binary bug. `cmake/build-info.cmake` embeds:

```text
git rev-list --count HEAD
```

A normal clone checks out **master**, not the newest release tag. At the moment
in question, master was three commits past exact tag b10545, so the binary
correctly reported build 10548 while GitHub's latest published Release was
still b10545. The old AutoTuner recipe then used `git describe --abbrev=0` as a
fallback and incorrectly named that development checkout `b10545_llama.cpp`.

The corrected recipe now:

1. keeps full Git history and computes the same commit count as CMake;
2. accepts a b-tag only when it points at **exactly HEAD**;
3. names exact releases `bNNNN_llama.cpp`;
4. names untagged master checkouts `bNNNN_dev_<commit>_llama.cpp`;
5. verifies after compilation that `llama-server --version` reports exactly the
   expected numeric build.

Therefore no waiting or hard-coded offset is needed. A GitHub Release may still
appear shortly after its exact tag because upstream pushes the tag before the
release assets finish publishing, but the source/build identity remains
unambiguous.

## Primary upstream references

- Release: <https://github.com/ggml-org/llama.cpp/releases/tag/b10549>
- Exact comparison: <https://github.com/ggml-org/llama.cpp/compare/b10441...b10549>
- Server options: <https://github.com/ggml-org/llama.cpp/blob/b10549/tools/server/README.md>
- Build-number source: <https://github.com/ggml-org/llama.cpp/blob/b10549/cmake/build-info.cmake>
- mmproj device: <https://github.com/ggml-org/llama.cpp/pull/23255>
- Kimi-K3: <https://github.com/ggml-org/llama.cpp/pull/26185>
- BailingMoE3: <https://github.com/ggml-org/llama.cpp/pull/26608>
- Granite SWA: <https://github.com/ggml-org/llama.cpp/pull/25505>
- DSpark formats: <https://github.com/ggml-org/llama.cpp/pull/26275>
- LFM2 DSpark: <https://github.com/ggml-org/llama.cpp/pull/27383>
- LFM2 tensor split: <https://github.com/ggml-org/llama.cpp/pull/26993>
