# llama.cpp b11319 follow-up — AutoTuner v5.6.0 preparation

Checked 2026-10-01 on R9700 + RX 9070 XT. Locally built stock HIP/Vulkan
**b11319**, commit `3ec4df42d9c1d4de896c886ebc65fad2e6e29fa4`; older builds
and user settings were preserved. [CLI/binary manifest](llama-b11319-server-flags.json).

## Source and CLI

[b11302…b11319](https://github.com/ggml-org/llama.cpp/compare/b11302...b11319)
contains **17 commits**, with no changed Vulkan backend files. The common
batch migration, speculative-layer-input batch-order correction (#29019),
Qwen LoRA V-head conversion correction and restored Qwen4Exp tensor-split
support are relevant changes. No exact commit is claimed as the first
working image+DFlash2 version; only b11319 is newly runtime-qualified here.
Both binaries expose the same **416 flags / 329 long flags** as b11302.
Profile Extra CLI flags remain supported.

## Image + DFlash2 now qualified

Stock HIP **and** Vulkan completed real Qwen3.8-27B Q8 + DFlash2 Q4 image
requests, with the appropriate projector and an actual dual-GPU plan:

- Red 448×448 image: correct color and integers 1…40; **152/168** draft
  tokens accepted, on each backend.
- Blue 672×448 and green 448×672 follow-up requests on the same server:
  correct color and all 40 integers; **133/133** accepted drafts each.
- Confirmation was repeated after removing the test-only gate bypass,
  exercising the real updated AutoTuner command builder.

The production gate is now bounded to **b10896 ≤ build < b11319**. b11319
is the first locally qualified build, not a claim that all intermediate
versions failed. The b11249/b11302 protection and text-only behavior remain.
These synthetic images do not qualify every finetune, resolution, image
count or OCR document. CLI acceptance and absence of HTTP500 alone were not
used as evidence: actual perception, generated tokens and accepted drafts
were checked. Spark/MiniCPM text/tools, Gemma assistant MTP, text DFlash2,
vision-only and existing Agnes/Prism negative-load probes also pass on the
stock b11319 HIP/Vulkan runtimes.

## Vulkan MoE regression persists; introducing change isolated

R9700 single-GPU ABBA, two runs/build with three internal repeats:

| Backend/model | Test | b11249 tok/s | b11319 tok/s | Change |
|---|---|---:|---:|---:|
| Vulkan Qwen3.8-27B UD-Q4_K_XL | pp512 | 884.28 | 882.68 | −0.2% |
| Vulkan Qwen3.8-27B UD-Q4_K_XL | tg128 | 32.16 | 32.16 | 0.0% |
| Vulkan Qwen3.6-35B-A3B UD-IQ3_XXS | pp512 | 3417.68 | 2927.80 | **−14.3%** |
| Vulkan Qwen3.6-35B-A3B UD-IQ3_XXS | tg128 | 148.62 | 148.33 | −0.2% |
| HIP Qwen3.8-27B UD-Q4_K_XL | pp512 | 1187.65 | 1190.31 | +0.2% |
| HIP Qwen3.8-27B UD-Q4_K_XL | tg128 | 30.71 | 30.80 | +0.3% |
| HIP Qwen3.6-35B-A3B UD-IQ3_XXS | pp512 | 2634.18 | 2653.20 | +0.7% |
| HIP Qwen3.6-35B-A3B UD-IQ3_XXS | tg128 | 123.12 | 124.98 | +1.5% |

A **separate diagnostic checkout/build** first reproduced the stock result.
Only the expert-tile selection change in
[PR #29182](https://github.com/ggml-org/llama.cpp/pull/29182), commit
`94a0ae3e7`, was reversed; GDN changes and all subsequent b11319 changes were
retained. Identical compiler/options, ABBA with five internal repeats:

| Device/quant | Test | Stock tok/s | Diagnostic revert tok/s |
|---|---|---:|---:|
| R9700 IQ3_XXS | pp128 | 1354.64 | 2009.94 |
| R9700 IQ3_XXS | pp512 | 2964.69 | 3462.20 |
| R9700 IQ3_XXS | pp1024 | 2949.41 | 3446.51 |
| R9700 MXFP4 | pp512 | 3865.94 | 4472.40 |
| R9700 Q6_K | pp512 | 3292.20 | 3778.63 |
| RX 9070 XT IQ3_XXS | pp512 | 2780.65 | 3165.75 |

Thus the regression is not an AutoTuner launch-code effect or merely a GDN
hypothesis: the single tile-selection change causes the measured slowdown.
The diagnostic revert is **not** an installed stock fix, universal GPU
optimization, or full inference/accuracy qualification. No original
llama.cpp checkout was patched and no benchmark-only binary was promoted
to an inference server. Following explicit approval, the separately named
[maintained RDNA4 option](rdna4-moe-workaround.md) was built with a hardware
predicate and tested through actual AutoTuner inference, independently of
the diagnostic binary. Fresh b11249 comparisons on both cards show only
−0.4…+1.2% variation across prompt lengths and decode; stock still regresses.

## Release status

AutoTuner's MTP/OCR corrections remain local. Stock b11319 is not a
regression-free Vulkan baseline, despite its working image+DFlash2 path.
The opt-in scoped runtime supplies the qualified local basis; frozen and
CI/release gates still apply. No v5.6.0 tag/push/publication is claimed here. See [validation and remaining gates](v5.6.0-validation.md).
