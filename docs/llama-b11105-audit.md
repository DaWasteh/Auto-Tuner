# llama.cpp b11105 audit — AutoTuner v5.5.5

Checked 2026-09-22 against **b11105**, `348f853b7` (`0.4.1-dev`), on the
user-built Windows HIP and Vulkan trees. Stable remains **v0.4.1 / b10964**.
The [b11063…b11105 comparison](https://github.com/ggml-org/llama.cpp/compare/b11063...b11105)
contains **42 commits / 130 changed files**. The trees are clean at the same
commit; this audit does not rebuild or change the user's llama runtimes.
See the [CLI/binary manifest](llama-b11105-server-flags.json) and
[release validation](v5.5.5-validation.md) for execution evidence and limits.

## CLI and server contracts

Both binaries still advertise **415 option names / 328 long options**,
identical to b11063. No emitted flag was removed. `--rpc` is absent because
these builds use `GGML_RPC=OFF`, not because upstream removed RPC. The help
hash changed on both backends for two reasons beyond the stripped,
timestamped startup line introduced in b11063:

- **b11078**, [sampling environment variables](https://github.com/ggml-org/llama.cpp/commit/e0dff5847):
  temperature, top-p, min-p, repeat, presence and frequency penalties now
  accept `LLAMA_ARG_*` defaults. Explicit CLI values take precedence.
  AutoTuner already emitted temperature/top-k/top-p/min-p/repeat, but omitted
  **presence penalty zero**. v5.5.5 always emits `--presence-penalty`, including
  zero, so an inherited `LLAMA_ARG_PRESENCE_PENALTY` cannot override the
  profile/Expert-panel choice. Frequency penalty has no AutoTuner UI control;
  it remains usable via Extra CLI flags or llama.cpp's environment defaults.
- **b11104**, [PR #28690](https://github.com/ggml-org/llama.cpp/pull/28690):
  `--host` now accepts comma-separated bind addresses/socket paths. The
  existing single-loopback default and AutoTuner's `/health` readiness checks
  remain valid. No extra listen address or network exposure is enabled by
  this release. Multi-bind is optional through Extra CLI flags; the separate
  authenticated AutoTuner control API stays single-loopback-only.
- Router child processes no longer inherit `--api-key-file` (the router
  consumes it). AutoTuner launches single-model servers, not router children.
  cpp-httplib 0.57.0/0.57.1 updates and Jinja const-correctness require no
  AutoTuner API migration. No change to the command's fit-off, parallel,
  load-mode, device-order, KV, lazy-PLE or prompt-cache contracts is needed.

## Models and parsers

### MiMo-V2.6 Flash / Pro RL — new profile

[PR #29257](https://github.com/ggml-org/llama.cpp/pull/29257), first tagged
**b11102**, adds MXFP4 MoE expert conversion, shares the existing Kimi-K3
lazy repacker, skips audio decoder tensors and prevents the MiMo template
from being misrouted to the Qwen3-Coder parser. It uses the existing
**`mimo2`** inference architecture, not a new GGUF/ggml type.

`settings/mimo-v2_6.yaml` uses the author's **temperature 1.0 / top_p 0.95**
for both modes, disables unrequested top-k/min-p truncation, enables Jinja
and requires b11102 for the correct parser. Its **1,048,576** context is a
ceiling, still clamped by GGUF metadata and memory. Filename matching does
not capture V2-Flash/V2.5 merely because they share `mimo2`.

The Flash converter writes 48 trunk + 3 NextN blocks, a per-layer KV-head
array (4 heads for 9 global layers, 8 for SWA), K=192/V=128 and a 128-token
sliding window. AutoTuner's existing interleaved-KV estimator handles this;
a new regression checks the full-attention coefficient exactly. The
model-card five-layer DFlash-style decoder must not be confused with the
converter's NextN tensors: MTP is enabled only when the actual GGUF scan
finds a head. `ngram-map-k4v` is the compatible draftless option.

**Limits:** no local V2.6 weights; this is source/profile/command/KV coverage,
not a local inference or throughput claim. Conversion includes the matching
vision projector but excludes the audio decoder: no speech-output promise.
Sources: [Flash card/config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL),
[Pro card/config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL).

### Muse Glimmer — fixed first-token tool calls

[PR #29242](https://github.com/ggml-org/llama.cpp/pull/29242), **b11100**,
accepts a tool-call prefix at the very start of generated text, not only
after `<|start|>assistant`. The old parser could put such calls in `content`
instead of `tool_calls`. `muse-glimmer.yaml` already passes `--jinja` and
`--reasoning-preserve`; those flags select the fix without a template
replacement. Its loader minimum stays **b10353**; the note and all nine
language packs recommend **b11100+ for tools**.

### Inventory gaps, not new upstream architectures

The user's refreshed `models_metadata.md` is included unchanged. A fresh
scan of the actual GGUFs found these gaps in the bundled settings:

- **FastContext 1.0 4B SFT/RL** (`qwen3`) previously fell back to generic
  8k settings. `fastcontext-1_0-4b.yaml` follows both local GGUFs' explicit
  `general.sampling` values (**0.7 / 0.8 / top_k 20**) and **262,144** native
  context, with Jinja. This is a repository explorer; the client supplies
  search tools, AutoTuner does not execute them. No generic `qwen3` fallback
  and no claim for the 30B variant. The Microsoft HF pages returned HTTP
  401 during direct retrieval, so these defaults are attributed to the
  inspected artifacts, not to an independently fetched generation config.
- **Xing 4.0 29B-A4B**: local IQ4_NL uses **`xing4_0`**, absent from b11105's
  architecture registry and loaders. `xing-4_0.yaml` recognizes it but
  blocks launch; an architecture gate also catches renamed files/custom
  profiles. The [author's settings](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B)
  (256k, chat 1.0 / coding 0.8, top_p 0.95, repeat 1.05) are recorded only
  for a future validated MLA/mHC/MTP runtime. Do not relabel it DeepSeek.
- **VoxCPM2 BaseLM** (`minicpm4`) is a Voice Lab TTS component, not a
  standalone chat/speech server. `voxcpm2.yaml` recognizes and blocks the
  ordinary AutoTuner chat path; the name guard also protects custom
  profiles without blocking unrelated `minicpm4` models. The already
  committed, previously unpublished
  [Voice Lab PS1](../building%20llama.cpp/voicelab_voxcpm2_vulkan_build.ps1)
  ships with this release: isolated `tc-mb/llama.cpp-omni` directory,
  `build-voicelab`, targets `voxcpm2-cli` / `llama-tts-server`. It is not
  added to normal llama build orchestration. Voice Lab's Qwen3-Omni audio
  pipeline likewise remains separate; a qwen3vlmoe text-profile match is
  not proof of full speech support.

HunyuanOCR gains DFlash layer-input capture and corrected draft vocabulary
conversion in [PR #28890](https://github.com/ggml-org/llama.cpp/pull/28890),
**b11103**. No new CLI switch is needed: the existing `dflash` architecture
selects `draft-dflash`. This does **not** fix Qwen's vision/DFlash2 position
holes. No HunyuanOCR/drafter weights are installed; no local runtime claim.

## Older issues: closure is not proof of a fix

GitHub issue/PR state and discussion were checked on 2026-09-22, separately
from whether code exists in the pinned b11105 tree:

| Concern | Current evidence | AutoTuner action |
|---|---|---|
| Qwen3.5/3.8 image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | Open; the compact recurrent draft cannot accept holes left by image positions | Keep the b10896+ gate; recheck on both b11105 backends |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open draft, updated Sep 22; conversion only, no inference implementation in b11105 | Keep block, update wording to b11105 |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Open; types 142/143 and Hadamard loader absent from mainline | Keep fork marker gate and existing recipe pin |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | Open/unmerged CPU PR; types 100..111 absent from mainline | Keep tensor-type/fork gate |
| RDNA4 MMQ fallback, [ROCmFPX #26](https://github.com/ROCmFPX/ROCmFPX/issues/26) | Open, no replies; no published upstream resolution | Keep local recipe patch |
| PQ2_0 CPU repack, [Prism #180](https://github.com/PrismML-Eng/llama.cpp/issues/180) | Open; device-type-based candidate patch has successful CPU/Vulkan/HIP reports in comments, not a verified replacement of our pinned Windows build | Do not remove workarounds or silently update the pin |
| MTP + ngram-mod, [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) | **Stale-closed July 31**, not a confirmed fix | Correct outdated “still open” comments/docs; retain suppression, prefer ngram-map-k4v |
| HIP multi-GPU garbage, [#16424](https://github.com/ggml-org/llama.cpp/issues/16424) | Closed; discussion points to platform PCIe/IOMMU/P2P faults and the no-peer-copy workaround, not a general runtime fix | Keep `GGML_CUDA_NO_PEER_COPY=ON`; check actual generated text |
| b10741–b10748 NextN regression | Already fixed by b10749 (#28173/#28183), no relevant regression in this range | Keep the bounded old-build gate, not a blanket MTP disable |
| Maple ternary experts on HIP | No TQ1_0/TQ2_0 CUDA/HIP kernels added in this range | CPU expert fallback remains; no advertised HIP ternary speedup |

## Remaining upstream changes

The other changes do not require new AutoTuner controls: CUDA/HIP contiguous
conversion and conv2d implicit GEMM, CUDA FA swizzle/Gemma 4 tuning and Volta
MMVQ crossover, `ggml_permute` 64-bit stride/dimension truncation fix,
JSON enum/grammar robustness, graph-input diagnostics, save/load and
recurrent-state test improvements, CPU ARM Q1_0 repack, Metal FA mask bounds
and DSV4 hyper-connections, SYCL context/DSV4/FA fixes, OpenCL Q4_0 kernels,
WebGPU gated-delta fusion, Hexagon DMA/64-bit/GDN rework, CI/CMake/vendor/UI
updates. Metal/SYCL/OpenCL/WebGPU/Hexagon runtime performance is not locally
validated by this Windows dual-AMD audit. AutoTuner's cross-platform source
and frozen-artifact CI remains a separate release gate.
