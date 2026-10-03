# llama.cpp b11371 audit — AutoTuner v5.6.1

Checked 2026-10-03 against the locally built Windows **b11371** HIP and Vulkan
servers, commit `99b95488c` (`0.5.0-dev`), on Core Ultra 9 285K, AI PRO R9700
32 GB + RX 9070 XT 16 GB. The
[b11319…b11371 comparison](https://github.com/ggml-org/llama.cpp/compare/b11319...b11371)
contains **52 commits / 168 changed files**. See the
[binary/CLI manifest](llama-b11371-server-flags.json) and the
[validation record](v5.6.1-validation.md).

## CLI and planning contracts

Both backends expose **417 option names / 330 long options**: the b11319 set
plus `--spec-draft-sampling {greedy,probabilistic}`
([#27694](https://github.com/ggml-org/llama.cpp/pull/27694), b11368). All
profile Extra CLI flags are still advertised. Device order is unchanged (HIP
ROCm0 = R9700 / ROCm1 = RX 9070 XT; Vulkan0 = RX 9070 XT / Vulkan1 = R9700 /
Vulkan2 = Intel iGPU). AutoTuner still owns context/offload placement and
sends `--fit off`.

`--spec-draft-sampling` is now a known value flag: it is accepted as an Extra
CLI flag, de-duplicated, pruned together with its value on older builds and
removed with the rest of the draft path where a draft is withdrawn. AutoTuner
does **not** set it by default. With AutoTuner's own MTP plans (p-min 0.75,
profile sampling at temperature 1.0, four prompts, mirrored order):

| Backend | Plan | greedy tok/s (accept) | probabilistic tok/s (accept) |
|---|---|---:|---:|
| Vulkan | Qwen3.8-27B UD-Q4_K_XL, embedded MTP | 42.97 (0.852) | 46.68 (0.854) |
| Vulkan | Gemma 4 12B + MTP assistant | 59.20 (0.759) | 59.32 (0.776) |
| HIP | Qwen3.8-27B UD-Q4_K_XL, embedded MTP | 40.78 (0.871) | 41.78 (0.872) |
| HIP | Gemma 4 12B + MTP assistant | 50.82 (0.780) | 51.50 (0.789) |

The first Vulkan greedy run was the cold start; per prompt the modes differ by
−2…+4%. That is inside run variation, so the upstream default stays.

## Upstream changes that needed AutoTuner work

### Non-causal image budget is now capped to `--ubatch-size`

[#29773](https://github.com/ggml-org/llama.cpp/pull/29773) (b11327) caps a
non-causal projector's `image_max_tokens` to the physical batch instead of
failing the request. Gemma 4 projectors (`gemma4v` except E2B/E4B, `gemma4uv`)
allow up to **1120** image tokens, while AutoTuner planned ubatch 512 or 1024
for these models. Real Gemma 4 12B request with a detailed 1792×1344 image,
same result on HIP and Vulkan:

| `-ub` | Server log | Prompt tokens | Cell `R03C2` (expected 133) |
|---:|---|---:|---|
| 512 (old plan, ≤32k context) | `cap image_max_tokens (original=1120) to n_ubatch (512)` | 536 | wrong (185) |
| 1120 (v5.6.1) | no cap | 1078 | correct |

`compute_config` now raises batch/ubatch to the projector's budget whenever a
non-causal Gemma 4 mmproj is attached. Causal projectors (Qwen, E2B/E4B) and
Qwen3.8 Flash Next's context×ubatch graph are untouched.

### Decision models and `/v1/systemone`

[#29818](https://github.com/ggml-org/llama.cpp/pull/29818) (b11361),
[#29844](https://github.com/ggml-org/llama.cpp/pull/29844) (b11364) and
[#29831](https://github.com/ggml-org/llama.cpp/pull/29831) (b11371) add typed
decision models (OpenJev, lev, Laya, Julia-1, Kev, Nimble, Clef). They matched
chat profiles or the fallback: Kev-4B planned a 262,144-token window with chat
sampling, Laya/Julia-1 got ubatch 512 although the encoder and the joint Clef
head must see the whole prompt in one physical batch. New profiles
`decision-systemone.yaml` (b11364+) and `clef.yaml` (b11371+), plus two
planner rules keyed on GGUF metadata: `<arch>.decision.type` caps the auto
context at the profile maximum, and non-causal encoders
(`<arch>.attention.causal = false`, which also covers BERT-style embedding
models) and Clef get batch = ubatch = min(context, 8192).

Real `/v1/systemone` requests (choice + noul + score) pass on HIP and Vulkan
for Julia-1, Laya, Kev-4B and Clef-Flash, including a 2,012…2,565-token state
that the previous 512-token ubatch could not hold. Chat completions on these
servers answer HTTP 500/empty by design.

### LLM-jp-4.1

[#29681](https://github.com/ggml-org/llama.cpp/pull/29681) (b11320) adds the
`llm-jp-harmony-v1` chat handler. New profile `llm-jp-4_1.yaml` (b11320+,
`--jinja`, NII cookbook sampling temp 0.7 / top_p 0.9, 65,536 context).
`llm-jp-4.1-8b-thinking` Q4_K_M: reasoning separated from content, Japanese
answer correct, tool calls with `auto` and `required` parsed, on both backends.

### Qwen3.8 Flash Next MTP (`qwen4exp`)

[#29761](https://github.com/ggml-org/llama.cpp/pull/29761) (b11330) adds the
NextN block and its draft graph. Verified locally:

- Older builds register no qwen4exp NextN tensor; a GGUF that embeds the block
  cannot load there (`wrong number of tensors`). `check_model_build` now
  requires b11330+ for such files.
- b11371 loads both a separate `mtp-*.gguf` head (through the new MTP-only
  loader mode) and the same block merged into the target, reserves the MTP
  graph and then aborts with `ggml-backend.cpp:345: GGML_ASSERT(buffer)` while
  the draft context is initialised. Reproduced on HIP and Vulkan, with two
  GPUs and with one, with `--ctx-checkpoints 0`, F16 KV,
  `--no-spec-draft-backend-sampling` and a CPU draft.
- The merged 75.2 GiB GGUF itself is valid: without speculation it answers
  normally (Vulkan 23 tok/s, HIP 29 tok/s at 16k context).

AutoTuner therefore keeps `draft-mtp` off for qwen4exp: a separate head is
refused with the real reason (the previous message claimed missing root
tensors), an embedded head loads with a logged note and n-gram speculation
stays available. `merge_mtp_head.py` (new) writes the embedded form from a
target and its head so the file is ready once upstream fixes the draft
context; a scan-proven embedded head no longer defaults to a sibling `mtp-*`
file. This is an upstream defect, not a tuning limit; it has not been
reported upstream by this audit.

## Other relevant upstream changes (no AutoTuner change needed)

- **qwen4exp attention path** ([#29751](https://github.com/ggml-org/llama.cpp/pull/29751),
  k-pool clamp [#29805](https://github.com/ggml-org/llama.cpp/pull/29805),
  mask construction [#29824](https://github.com/ggml-org/llama.cpp/pull/29824)):
  real Flash-Next requests pass on both backends; the lazy PLE table stays
  mapped (0.00 GiB resident of 26.82 GiB).
- **Recurrent memory assert** ([#29799](https://github.com/ggml-org/llama.cpp/pull/29799)),
  **HIP FA selection for CDNA only** ([#29572](https://github.com/ggml-org/llama.cpp/pull/29572)),
  **Vulkan pipeline-compile logging** ([#29794](https://github.com/ggml-org/llama.cpp/pull/29794)),
  **Samsung matmul tiles** ([#28531](https://github.com/ggml-org/llama.cpp/pull/28531)):
  no launch-flag impact on RDNA4.
- **Direct-I/O tensor reads** ([#29749](https://github.com/ggml-org/llama.cpp/pull/29749))
  is a Linux `-lm dio` fix. **Gemma DFlash embedding scale**
  ([#29802](https://github.com/ggml-org/llama.cpp/pull/29802)) is converter
  only; existing Gemma DFlash drafts need a re-conversion to benefit.

## Throughput against b11249 (R9700, mirrored order, three repeats)

| Backend / model | Test | b11249 | Stock b11371 | `rdna4moe_b11371` |
|---|---|---:|---:|---:|
| Vulkan Qwen3.6-35B-A3B UD-IQ3_XXS | pp128 | 1886.93 | 1301.15 (**−31.0%**) | 1874.07 (−0.7%) |
| | pp512 | 3417.11 | 2926.08 (**−14.4%**) | 3413.09 (−0.1%) |
| | pp1024 | 3409.78 | 2918.72 (**−14.4%**) | 3402.38 (−0.2%) |
| | tg128 | 148.38 | 148.09 (−0.2%) | 147.66 (−0.5%) |
| Vulkan Qwen3.8-27B UD-Q4_K_XL | pp512 | 872.21 | 869.94 (−0.3%) | 869.80 (−0.3%) |
| | tg128 | 32.13 | 32.11 (0.0%) | 32.09 (−0.1%) |
| HIP Qwen3.6-35B-A3B UD-IQ3_XXS | pp512 | 2627.84 | 2635.02 (+0.3%) | — |
| | tg128 | 120.69 | 121.97 (+1.1%) | — |
| HIP Qwen3.8-27B UD-Q4_K_XL | pp512 | 1186.70 | 1190.05 (+0.3%) | — |
| | tg128 | 30.59 | 30.67 (+0.3%) | — |

The Vulkan MoE prompt regression from
[#29182](https://github.com/ggml-org/llama.cpp/pull/29182) persists on stock
b11371 and is now tracked upstream as
[#29892](https://github.com/ggml-org/llama.cpp/issues/29892) (open). The
opt-in [RDNA4 workaround](rdna4-moe-workaround.md) was re-qualified for the
exact b11371 source (byte-identical dispatch hunk, own source digest) and
restores b11249 throughput; stock builds are not modified. A first HIP run
overlapped with a 75 GiB file merge on the same machine and was discarded;
the table shows the undisturbed repeat.

## Older blockers rechecked

| Concern | Evidence on b11371 | Action |
|---|---|---|
| Qwen3.5/3.8 image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | Issue still open upstream, but real image requests pass on HIP and Vulkan with **152/168** accepted drafts | Gate stays bounded to b10896 ≤ build < b11319 |
| Xing 4.0 `xing4_0` | Both loaders reject the architecture | Block kept, wording names b11371 |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open/unmerged | Block kept, wording names b11371 |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Four stock loads fail; fork tag `prism-b10754` adds SYCL/Metal/CUDA/WebGPU work only | Marker gate and `prism-b10743` pin kept |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | Open/unmerged; four Agnes stock loads fail | Tensor-type gate kept |
| ROCmFPX Windows build | Fork main moved to `7966b1db7`; its Windows build fix is still an open PR ([fork #31](https://github.com/ROCmFPX/ROCmFPX/pull/31)), RDNA4 MMQ issue [#26](https://github.com/ROCmFPX/ROCmFPX/issues/26) unanswered | `aed0d5fd9` pin and patch kept |
| MTP + ngram-mod, [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) | Stale-closed, no fix | Suppression kept |
| Separate Qwen3.8 Flash Next MTP head | See above: loads, then aborts | Block kept with the real reason |

## Local models without a dedicated profile

A scan of all local GGUFs found GLM-OCR and dots.ocr on `_default.yaml`
(temp 0.7, repetition penalty 1.05, 8k target, no `--jinja`) whenever they
were started as a server outside the OCR workflow. New `glm-ocr.yaml`
(deterministic, as published) and `dots-ocr.yaml` (reference-parser sampling,
32k window). The OCR workflow's own per-request sampling is unchanged.

## Memory estimate audit

Loading every local model from its plan showed that llama.cpp's compute
buffers (about two attention masks on one device, about ten on a split
plan) were missing from the estimate, that mmap kept fully offloaded models
in the Windows working set, and that MLA and gpt-oss KV were estimated twice
too large. All three are corrected; measurements and limits are in the
[validation record](v5.6.1-validation.md#memory-estimate-audit).

Metal, SYCL, OpenCL, Hexagon, WebGPU, CUDA and RPC runtimes are not locally
validated by this Windows dual-AMD audit. Representative inference does not
qualify every model, checkpoint, quantization or context size.
