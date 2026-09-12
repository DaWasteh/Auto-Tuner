# llama.cpp b10930 / build-recipe audit — AutoTuner v5.4.7

Checked 2026-09-12. Exact runtime: **b10930**, commit
`56381e407c0ccfb3a6f71e668a27a901001d22ce`.
[Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/b10901...b10930):
29 commits, 123 files. Both Windows trees were **freshly cloned and built** with
the unchanged repository recipes and then executed; see
[validation](v5.4.7-validation.md) for the test boundaries and
[binary/help hashes](llama-b10930-server-flags.json).

## CLI surface: byte-identical to b10901

No commit in the range touches `common/arg.cpp`. The local Vulkan and HIP
`llama-server --help` outputs are byte-identical to the b10901 captures:
**415 names / 328 long options**, one shared help SHA-256 for both backends.
`--rpc` is still absent only because the local recipes keep `GGML_RPC=OFF`.
Every bundled profile's `extra_args` token exists in the manifest, and all
**162 standard profile × chat/coding commands per backend** parse on the real
server. No migration of `--load-mode`, `--lazy-mode`, KV-type, speculative or
multimodal flags is needed.

## Vision + DFlash2 is still broken on b10930 (PR #28587, not fixed by #28715)

v5.4.6 reproduced HTTP 500 on b10901 for Qwen3.5/3.8 (`qwen35`) + DFlash2 +
vision. The range was re-examined from source and re-executed:

| Build | Change | Actual image + DFlash2 request |
|---|---|---|
| **b10896** | [PR #28587](https://github.com/ggml-org/llama.cpp/pull/28587) skips pinned M-RoPE image batches instead of injecting them into the draft context | not installed; source cause |
| b10901 | (v5.4.6 audit) | HTTP 500, both backends |
| b10903 | installed Vulkan tree, AutoTuner gate bypassed for the probe | **HTTP 500 reproduced again** |
| b10906 | [PR #28715](https://github.com/ggml-org/llama.cpp/pull/28715) passes `slot.prompt.tokens.pos_next()` instead of the token count as the drafter's `pos0` | position handed over changes, gap remains |
| **b10930** | fresh HIP and Vulkan builds | **HTTP 500 reproduced on both backends** |

The b10930 logs show the identical failure: the DFlash2 sidecar is loaded as
`llama_memory_recurrent` (grouped convolution state), the target decodes the
196-token image batch, and the draft context then rejects
`X = 3 ... Y = 18 ... it is required that the sequence positions remain
consecutive`, followed by `failed to process speculative batch`. Recurrent
memory cannot skip the 14 pinned image positions that PR #28587 no longer
feeds into the draft context, so PR #28715's corrected start position does not
help this drafter. The PR #28715 author's DFlash2 measurement (RTX 5090,
acceptance 0.02 → 0.35) evidently used a setup where the request survived;
here it never did. No upstream issue for the recurrent-draft case was found
by search on 2026-09-12.

AutoTuner therefore replaces the single-build (`== 10901`) gate with an
**open-ended gate from b10896** (`tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE`).
The message still names the two working alternatives (disable Draft for
images, disable Vision for text-only DFlash2), which were both re-run
successfully on b10930 HIP and Vulkan. Builds before b10896 keep the previous
injecting behaviour and are not gated; other drafter types are untouched. The
gate must be lowered only after an actual image + DFlash2 request succeeds on
a newer build, not on the strength of a merged PR title.

## MTP draft-context KV allocation (PR #28630, b10907)

`llama_model::create_memory` previously filtered NextN layers only for
`STEP35`, `HY_V3`, `GLM_DSA`, `MIMO2` and `DEEPSEEK32`. It now applies the
filter to every architecture with `n_layer_nextn > 0`, a non-empty trunk and
no router layer, which fixes 3× over-allocation of the **MTP draft context**
on `deepseek2`, `glm4moe` and `cohere2moe` (issue #28626). AutoTuner never
budgeted that draft context separately: integrated MTP is planned inside the
target's own KV/graph reserve, and the b10930 behaviour only lowers actual
usage. The plan is therefore conservative on b10930 and still safe for
b10906-and-older runtimes; no automatic enlargement is derived from it.
b10903 also adds Kimi-Linear to the `n_seqs == 1` graph-reserve diagnostic
case (`[TAG_RESERVE_DIAG_DECAY]`); AutoTuner's graph reserve is unchanged.

## Build recipes on this workstation

Both mainline pre-release recipes and both stable recipes were executed
unchanged (`llama_prerelease_{vulkan,hip}_build.ps1 -Tag b10930`, then the
stable pair, which re-verifies the installed `0.4.0` trees). Toolchain actually
used: Visual Studio 2026 (MSVC 14.51.36231), CMake 4.4.3, Vulkan SDK
1.4.341.1 (`glslc` with coopmat, coopmat2 and bfloat16 support enabled), ROCm
HIP SDK 7.2 with the workspace-local LLVM PR #201563 `<cmath>` fix.

- **PR #28091 (b10911) / PR #28763 (b10917):** upstream now precompiles
  `common.h`, `models.h` and `ggml-impl.h` and compiles the model sources as a
  unity build (`UNITY_BUILD_BATCH_SIZE 16`). Both generators accept this
  without recipe changes. In the Visual Studio tree MSBuild reports
  **MSB8027** because `src/llama.cpp` and `src/models/llama.cpp` share a
  basename; it is benign here: `src/models/llama.cpp` is folded into
  `unity_4_cxx.cxx` and exactly one `llama.obj` exists. `LLAMA_SERVER_PCH`
  is only disabled for MSVC *shared* builds; the recipes use
  `BUILD_SHARED_LIBS=OFF`.
- **PR #28102 (b10905), CUDA/HIP flash-attention tuning for gfx1201:** the
  tuned path is the native RDNA4 WMMA MMA kernel (`amd_wmma_available`), so no
  `GGML_HIP_ROCWMMA_FATTN` option is required or added. `GGML_CUDA_FA_QUANTS=all`
  is still selected by source inspection, `GGML_CUDA_NO_PEER_COPY=ON` remains
  mandatory and is re-verified by the HIP recipe's two-GPU decoded-text check.
- **PR #28492 (b10925)** only changes the macOS RDMA link option of `ggml-rpc`;
  the local `GGML_RPC=OFF` trees are unaffected.
- Vulkan-specific: PR #28705 (b10903) fixes an argsort data race / OOB access;
  no AutoTuner change. Warnings remain visible (63 MSVC diagnostics, 0 errors;
  `D9025` is upstream's own `/w` override for third-party sources).
- HIP tree facts from the actual run: ROCm clang 21 with the workspace-local
  resource dir, `GPU_TARGETS=gfx1201`, `GGML_CUDA_FA_QUANTS=all`,
  `GGML_CUDA_NO_PEER_COPY=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`,
  PCH objects for `ggml-cpu`, `llama` and `llama-common`, ten unity sources,
  **11,125 clang warnings / 0 errors** (mostly upstream `-Wsign-compare` and
  `-Wnested-anon-types`), ROCm DLLs bundled, kernel directories junctioned,
  `ROCm0` = R9700 and `ROCm1` = RX 9070 XT, and the deterministic two-GPU
  decoded-text check `HIP MULTI GPU OK`.
- No recipe parameter needed changing: ROCm path, `gfx1201`-only target,
  parallelism, SPIRV-Headers refresh and the HIP runtime DLL/junction step all
  ran as written. Old build folders were preserved. The stable recipes found
  the existing `0.4.0` (b10809) trees, re-ran `--version`, `--list-devices`
  and the HIP semantic check on them and reported success without rebuilding.

## Still open upstream (no AutoTuner change)

- **DeepSeek-V4.1-Flash:** [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
  is still open and conversion-only; the profile block now names b10930.
- Server router/subprocess refactor (PR #28555), download-at-limit fix
  (PR #28530), child-state framing (PR #28747) and cpp-httplib 0.56.0 only
  affect `--models-dir` router mode and its tests, which AutoTuner does not use.
- PR #28739 fixes an out-of-bounds read when always offloading zero-sized
  expert `ids` tensors (`GGML_OP_OFFLOAD_MIN_BATCH=0` with qwen4exp). AutoTuner
  does not set that variable; the Qwen3.8-Flash-Next lazy test still passes.
- Metal, OpenCL, SYCL, WebGPU, Hexagon and s390x commits have no Windows
  HIP/Vulkan impact.
