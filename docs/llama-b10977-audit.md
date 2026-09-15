# llama.cpp b10977 / v0.4.1 / build-recipe audit — AutoTuner v5.5.0

Checked 2026-09-15. Exact runtime: **b10977**, commit
`0ecb159c9e93056a4742afe4195d05a2912b1746`; the **v0.4.1 stable** release
(`b29c606e2`) is **b10964** and lies inside the audited range.
[Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/b10948...b10977):
29 commits, 66 files. Both Windows pre-release trees were **freshly cloned
and built** with the unchanged repository recipes and then executed; the
two `0.4.1` stable trees were verified by the stable recipes. See
[validation](v5.5.0-validation.md) for the test boundaries and
[binary/help hashes](llama-b10977-server-flags.json).

## CLI surface: byte-identical `--help`

No commit in the range touches `common/arg.cpp` or the server option table.
The local Vulkan and HIP `llama-server --help` outputs have the **same
SHA-256 as b10948** (both backends share one help text), the parsed option set
is **415 names / 328 long options** as on b10930 and b10948, nothing was added
or removed, and `--rpc` is still absent only because the recipes keep
`GGML_RPC=OFF`. Every bundled profile's `extra_args` token exists in the
manifest and all **164 standard profile × chat/coding commands per backend**
(one more profile than b10948: `maple.yaml`) parse on the real server. No
migration of `--load-mode`, `--lazy-mode`, KV-type, speculative or multimodal
flags is needed.

## Behaviour changes checked against AutoTuner

- **CPU work-buffer sizing / heap corruption**
  ([PR #28882](https://github.com/ggml-org/llama.cpp/pull/28882)): the
  `CACHE_LINE_SIZE` macro diverged between C and C++ translation units when
  precompiled headers changed the include order and undersized the CPU work
  buffer. Upstream disabled PCH for the CPU backend there and then
  ([PR #28892](https://github.com/ggml-org/llama.cpp/pull/28892)) removed every
  `target_precompile_headers` again (llama, common, server, mtmd); the
  `src/models` unity build stays. AutoTuner's CPU paths (`--n-cpu-moe`
  expert offload, host-mapped PLE tables, the HIP ternary fallback below)
  run on the fixed sizing. Both recipes needed no change; the HIP tree now
  has **504 Ninja targets** (511 on b10948, the PCH targets are gone).
- **Sliding-window pattern loaders require the array key**
  ([PR #28868](https://github.com/ggml-org/llama.cpp/pull/28868)): `gemma4`,
  `gemma4-assistant`, `step35` and `mimo2` read
  `*.attention.sliding_window_pattern` with `get_arr` instead of
  `get_key_or_arr`, so a GGUF carrying a scalar pattern would now fail to
  load. All nine local Gemma 4 GGUFs (Unsloth UD Q8/Q5, Google QAT q4_0 and
  the three `mtp-gemma-4-*` assistant drafters) carry per-layer arrays, and
  Gemma 4 12B Q8 + its `gemma4-assistant` MTP drafter was **run live** on
  b10977 Vulkan and HIP (`is_swa_any = 1`, `n_swa = 1024`, drafter
  accepted, deterministic answer).
  `scanner.py` already reads the array form for its global-layer KV count.
- **qwen4exp hyper-connection norms** ([PR #28896](https://github.com/ggml-org/llama.cpp/pull/28896)):
  the `hc_*_norm` and `ple_norm_*` gammas load as `[n_embd, hc]` with
  `TENSOR_ALLOW_RESHAPE` so the grouped RMS norm fuses `rms_norm + mul`;
  existing 1-D GGUF tensors are reshaped at load time. The Qwen3.8 Flash-Next
  lazy-PLE check (`lazy_test.py`) ran unchanged on both backends.
- **Recurrent-state context reuse** ([PR #28749](https://github.com/ggml-org/llama.cpp/pull/28749)):
  `common_context_can_seq_rm` now returns the bounded `RS` removal type
  before the memory-clear/decode probe, so hybrid and recurrent models
  (Qwen3.5/3.8 with the DFlash2 sidecar, BailingMoE3, Kimi Linear, Nemotron-H)
  no longer run the throw-away decode at start-up. AutoTuner does not read
  that probe; the DFlash2 draft runs below cover the path.
- **CUDA/HIP BF16 fallback and I16/I32 DUP** ([PR #28846](https://github.com/ggml-org/llama.cpp/pull/28846),
  [PR #28897](https://github.com/ggml-org/llama.cpp/pull/28897)): BF16 matmuls
  drop to F32 only on devices without fast BF16 (RDNA3+/CDNA keep BF16, so
  gfx1201 is unaffected); `GGML_OP_DUP` now accepts I16/I32 on CUDA/HIP.
- **Grammar / SYCL / s390x / CI**: the grammar `find + insert` coalescing,
  the SYCL oneDNN scratchpad and radix top-k, the MiMo2 SWA pattern load fix,
  the API/ABI compatibility script, the ubuntu-cuda release builds, the
  gfx1103 ROCm release target and the CUDA 13.4.1 Windows CI bump have no
  Windows HIP/Vulkan effect on AutoTuner.

## New architecture: Maple (DeepGrove Maple-Preview)

[PR #27000](https://github.com/ggml-org/llama.cpp/pull/27000) (merged
2026-09-14, first tag **b10964** = v0.4.1) adds `maple`: 24 layers, 256 experts
with 8 active, 3:1 sliding-window-512/global attention, QK norm, partial RoPE
0.5, SwiGLU clamp 7.0 on the experts, 131,072 native context, Qwen2 tokenizer
with a thinking-by-default ChatML template. The official
`deepgrove/maple-preview-GGUF` files are TQ1_0/TQ2_0 with a Q4_K or F16 head
(4.64–5.91 GiB). AutoTuner ships `settings/maple.yaml` (patterns
`maple-preview`/`deepgrove`, `arch_fallback: maple`, `min_llama_build: 10964`,
`max_context: 131072`, `--jinja --reasoning-preserve`, thinking-model
sampling temp 0.6 / top_p 0.95 / top_k 20 because DeepGrove publishes no
`generation_config`), and all nine language packs carry its note.

Backend facts from the b10977 sources: `ggml-vulkan` has TQ1_0/TQ2_0 dequant,
`mul_mat_vec`, `mul_mat_vec_id` and MMQ pipelines, so the whole model runs on
the GPU under Vulkan. `ggml-cuda` (and therefore the HIP build) has no TQ
kernels at all; llama.cpp's weight placement then keeps the TQ2_0 expert
tensors in host memory and computes them on the CPU. Live runs of
`maple-preview-TQ2_0-head-Q4_K.gguf` (5.91 GiB, `general.architecture =
maple`, 291 tensors, per-layer `sliding_window_pattern` and
`swiglu_clamp_exp` arrays) through AutoTuner's planner on b10977, 32k
context, Q8 K/V, `--jinja --reasoning-preserve`, deterministic
"17 * 23" with thinking on:

| Backend | Placement | Decode | Answer |
|---|---|---|---|
| Vulkan (RX 9070 XT) | 25/25 layers offloaded, **5,029 MiB on Vulkan0**, 594 MiB host-mapped | **295 t/s** (prompt 91 t/s) | `391` after a 71-character reasoning block |
| HIP (R9700) | 25/25 layers "offloaded", but **5,456 MiB CPU-mapped** and only 215 MiB on ROCm0 (no TQ kernels) | 27 t/s (prompt 62 t/s) | `391`, same reasoning |

Both runs confirm the profile note: use the Vulkan runtime for Maple until
CUDA/HIP gain ternary kernels; the HIP build works but is CPU-bound.

## Vision + DFlash2 on Qwen3.5/3.8 (PR #28587)

No commit in the range touches `tools/server/server-context.cpp`, the
speculative helpers or the recurrent memory module, so the failure documented
for b10896–b10948 was expected to persist. It was **re-executed**:

| Build | Actual image + DFlash2 request |
|---|---|
| b10901 / b10903 / b10930 / b10948 | HTTP 500 (v5.4.6–v5.4.8 audits) |
| **b10977 Vulkan** | **HTTP 500 reproduced** (`failed to process speculative batch`) |
| **b10977 HIP** | **HTTP 500 reproduced** (same log lines) |

The gate `tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE = 10896` stays open-ended
and its message names b10977 as the last verified build. The two documented
alternatives (Draft off for images, Vision off for text-only DFlash2) were
re-run on both backends.

## Build recipes on this workstation

Both mainline pre-release recipes were executed unchanged
(`llama_prerelease_{vulkan,hip}_build.ps1 -Tag b10977`), followed by the two
stable recipes, which resolve `latest` to **v0.4.1** and verify the installed
`0.4.1` trees. Toolchain actually used: Visual Studio 2026 (MSVC
14.51.36231 / compiler 19.51.36257), Vulkan SDK 1.4.341.1 (`glslc` with
coopmat, coopmat2, bfloat16, integer-dot and e4m3 support), ROCm HIP SDK 7.2
(clang 21) with the workspace-local LLVM PR #201563 `<cmath>` fix.

- **Precompiled headers are gone upstream** (PR #28892): the recipes never
  passed a PCH option, so nothing changed; upstream's clang
  `-Xclang -fno-pch-timestamp` (PR #28816) is still emitted but idle.
- Vulkan tree facts from the actual run: MSVC 19.51.36257 x64 Release, all
  shader capabilities detected, `MSB8027` is the known benign `src/llama.cpp`
  vs `src/models/llama.cpp` basename clash (one `llama.obj`), three Vulkan
  devices listed (RX 9070 XT, AI PRO R9700 and the Intel iGPU).
- HIP tree facts from the actual run: ROCm clang 21, `GPU_TARGETS=gfx1201`,
  **504 Ninja targets**, **11,124 clang warnings / 0 errors** (upstream
  `-Wsign-compare` and `-Wnested-anon-types`), ROCm DLLs bundled, kernel
  directories junctioned, `ROCm0` = R9700 and `ROCm1` = RX 9070 XT, and the
  deterministic two-GPU decoded-text check `HIP MULTI GPU OK`.
- The stable recipes found the `0.4.1` (b10964) trees built earlier the same
  day, re-ran `--version`, `--list-devices` and the HIP semantic check on
  them and reported success without rebuilding.

## Still open upstream (no AutoTuner change)

- **DeepSeek-V4.1-Flash:** [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
  is still open (last updated 2026-09-13) and conversion-only; the profile
  block and all nine language packs now name b10977.
- Maple has CPU and Vulkan kernels only; CUDA/HIP/Metal TQ kernels are
  announced as follow-ups in PR #27000.
