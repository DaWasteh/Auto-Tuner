# llama.cpp b11302 audit — AutoTuner v5.6.0

**Follow-up:** [b11319](llama-b11319-audit.md) now qualifies image+DFlash2;
the protection remains for the older affected builds documented below.
The Vulkan slowdown persists on stock b11319; PR #29182 was isolated as its
cause in a separate diagnostic build.

Checked 2026-10-01 against the locally built Windows **b11302** HIP and
Vulkan servers, commit `05af0d2b1398394cfa67e1918fee7feabccaa9bc`.
The [b11249…b11302 comparison](https://github.com/ggml-org/llama.cpp/compare/b11249...b11302)
contains **53 commits / 174 changed files**. Stable remains **v0.5.0**.
See the [binary/CLI manifest](llama-b11302-server-flags.json) and
[validation record](v5.6.0-validation.md).

## CLI, loading and planning contracts

Both backends expose exactly the same **416 option names / 329 long options**
as b11249; even the normalized help SHA-256 is unchanged. All **178
profile/mode commands per backend** parse with the exact server, without
loading weights. There are no unsupported profile Extra CLI flags and no new
flag to add to AutoTuner. RPC is still advertised by non-RPC builds but fails
when used; AutoTuner does not plan RPC offload.

The CPU/GPU CMake option sets match the previous builds: AVX2/AVX-VNNI/BMI2,
AVX-512 off; HIP gfx1201 with graphs, no VMM, no peer copy, and FA quants.
The compiler banners are unchanged (Vulkan MSVC 19.51; HIP Clang 21).
Device ordering remains HIP ROCm0 = R9700 / ROCm1 = RX 9070 XT; Vulkan0 =
RX 9070 XT / Vulkan1 = R9700 / Vulkan2 = Intel iGPU. AutoTuner still owns
context/offload placement and sends `--fit off`.

## Backend and model changes relevant to this PC

- **HIP packed subtraction:** [#29478](https://github.com/ggml-org/llama.cpp/pull/29478)
  replaces saturating packed-byte subtraction with the non-saturating
  operation required by these quantized dot products. No new launch flag.
- **Vulkan MoE dispatch:** [#29182](https://github.com/ggml-org/llama.cpp/pull/29182)
  selects matmul tiles by rows per expert instead of the total token batch.
  Relevant to small per-expert prompt batches; no AutoTuner setting.
- **Vulkan GDN:** [#29476](https://github.com/ggml-org/llama.cpp/pull/29476)
  retunes kernel selection, including Intel handling. Applies to hybrid
  Qwen-family models; it is not a reason to change KV precision.
- **Lazy row prefetch:** [#29599](https://github.com/ggml-org/llama.cpp/pull/29599)
  adds the Windows prefetch path for lazy per-layer embeddings in Gemma 4
  and Qwen3.8 Flash-Next. AutoTuner already accounts for the lazy table and
  selects the load mode. Real Flash-Next requests and mapped-table residency
  checks passed on both backends without loading the full table into RAM.
  This does not mean the whole model is small: ready RSS was 34.6/33.8 GiB
  (Vulkan/HIP), private allocation about 44.5/44.0 GiB, comparable to the
  previous run. Only the mapped lazy table's resident pages were ~0.25 MiB.
- **GLM5-Next:** the audited head fixes sparse-indexer scatter rows
  ([#29745](https://github.com/ggml-org/llama.cpp/pull/29745)). No local
  GLM5-Next inference or memory-plan qualification is claimed.

These are backend/loader changes, not new tuning controls. The build recipes,
Q8-first KV policy, GPU selection and HIP no-peer-copy safeguard remain.
The A/B check nevertheless found a Vulkan MoE pp512 slowdown of 14.3%;
independent confirmation also found ~32% at pp128. Increasing the microbatch
did not restore throughput. The [validation record](v5.6.0-validation.md)
reports all cases explicitly. This audit is not a blanket performance
endorsement and publication remains gated.

## Verified AutoTuner fixes

### MTP filename cannot override an authoritative tensor scan

`metadata_has_embedded_mtp()` already rejects stale `nextn_predict_layers`
when the complete scan says `__mtp_scan__="absent"`. The `ModelEntry`
filename fallback could then turn MTP back on solely because the trunk's
name contained `MTP`. The same negative evidence now vetoes that fallback.
A separately attached head still remains usable. Positive tensor evidence,
legacy headers without a scan and inconclusive split scans retain their
previous behavior. Regression tests cover each boundary.

### Canonical OCR task prompts

GUI and TUI use the same model presets. Defaults now follow the model
publishers rather than a generic `OCR`/`OCR markdown:` task:

| Family | Default task |
|---|---|
| GLM-OCR | `Text Recognition:` |
| PaddleOCR-VL | `OCR:` |
| dots.ocr | `Extract the text content from this image.` |

Sources: [GLM-OCR prompt limitations](https://huggingface.co/zai-org/GLM-OCR),
[PaddleOCR-VL-1.6 tasks](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6),
[dots.ocr task definitions](https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/prompts.py).
The previous tasks also recognized the simple test page; this is alignment
with the supported task contract, not a claim that all previous OCR failed.
Custom prompt overrides, image-first ordering, deterministic sampling,
source preservation and atomic outputs are unchanged. Each canonical task
was exercised through the actual shared document workflow on HIP and
Vulkan, with the expected text and successful manifests.

## Older blockers rechecked, not silently lifted

| Concern | Current evidence | Action |
|---|---|---|
| Qwen3.5/3.8 image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | HTTP 500 reproduced on b11302 HIP and Vulkan; text DFlash2 and image-only requests pass | Keep b10896+ combination gate; message records the b11302 recheck |
| Xing 4.0 `xing4_0` | Both real b11302 loaders still reject the architecture | Keep recognition-only block |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open/unmerged; no validated inference implementation | Keep block; no local V4.1 inference claim |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Four stock-runtime loads fail; AutoTuner rejects them before launch | Keep fork marker gate; Prism pin unchanged |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | Open/unmerged; four Agnes stock-runtime loads fail as expected | Keep tensor-type/fork gate |
| ROCmFPX Windows build / RDNA4 MMQ, [fork #26](https://github.com/ROCmFPX/ROCmFPX/issues/26) | Fork main remains `721db4193`; no replies on #26; same source as the previously failed Windows trial | Keep validated `aed0d5fd9` pin and patch; no redundant rebuild or new compilation-success claim |
| MTP + ngram-mod, [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) | Stale-closed, not proof of a fix | Keep suppression; no unsafe crash retest |
| HIP multi-GPU P2P, [#16424](https://github.com/ggml-org/llama.cpp/issues/16424) | Dual-GPU text/draft and vision-only plans pass with no peer copy | Keep build safeguard |

## Newer upstream tag: source-only boundary

During the audit **b11303**, `60e9cf7a7`, appeared. Its single commit,
[#29601](https://github.com/ggml-org/llama.cpp/pull/29601), migrates remaining
examples/tools to `llama_batch_ext`, removes legacy common batch helpers and
changes `common_batch` materialization. The comparison changes **34 files**,
including common speculative helpers and llama-bench. There is no new
AutoTuner CLI control. This is **not** a local b11303 binary/inference
qualification and not evidence to remove the image+DFlash2 gate.

Metal, SYCL, OpenCL, Hexagon, WebGPU and RPC runtimes are not locally
validated by this Windows dual-AMD audit. Synthetic OCR pages do not replace
full document-accuracy benchmarks; representative inference does not qualify
every model/checkpoint/quantization or context size.
