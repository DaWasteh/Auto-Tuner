# llama.cpp b10948 / build-recipe audit — AutoTuner v5.4.8

Checked 2026-09-13. Exact runtime: **b10948**, commit
`5f436dddb440a288ee5611d7d1eca564a6aca9f4`.
[Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/b10930...b10948):
18 commits, 70 files. Both Windows trees were **freshly cloned and built** with
the unchanged repository recipes and then executed; see
[validation](v5.4.8-validation.md) for the test boundaries and
[binary/help hashes](llama-b10948-server-flags.json).

## CLI surface: same option set as b10930, two help sentences reworded

One commit in the range touches `common/arg.cpp`:
[PR #28736](https://github.com/ggml-org/llama.cpp/pull/28736) (`common_schema`)
rewrites the `-j, --json-schema` and `-jf, --json-schema-file` help sentences
(`{"type": "object"}` instead of `{}` as the "any object" example; the
reference to the deleted `examples/json_schema_to_grammar.py` is gone) and
makes `--log-jsonl` a global logger switch. The local Vulkan and HIP
`llama-server --help` outputs therefore share a **new** help SHA-256, but the
parsed option set is **identical to b10930: 415 names / 328 long options**,
nothing added or removed. `--rpc` is still absent only because the recipes
keep `GGML_RPC=OFF`. Every bundled profile's `extra_args` token exists in the
manifest, no profile uses `--json-schema`, `--grammar` or the removed Python
grammar helpers, and all **162 standard profile × chat/coding commands per
backend** parse on the real server. No migration of `--load-mode`,
`--lazy-mode`, KV-type, speculative or multimodal flags is needed.

## Server behaviour changes checked against AutoTuner

- **Empty JSON schema means "any object"** (`tools/server/server-common.cpp`,
  `server-schema.cpp`): an absent or `{}` `json_schema` / `response_format`
  schema is now completed with `"type": "object"` before grammar conversion.
  AutoTuner's control API, benchmark and OCR paths never send a JSON schema,
  so nothing changes for the launcher; API clients that passed `{}` get the
  documented "any object" behaviour back.
- **Structured log records** ([PR #28586](https://github.com/ggml-org/llama.cpp/pull/28586)):
  a new `LOG_JSON` macro emits typed JSON objects, and `common/fit.cpp` now
  also writes the memory-breakdown table as a `fit_memory_breakdown` record.
  Both only appear with `--log-jsonl`. AutoTuner reads the plain text log,
  emits `--fit off` and does not enable JSONL logging, so the text output it
  already understands is unchanged.
- **Chat parsing** ([PR #28742](https://github.com/ggml-org/llama.cpp/pull/28742)
  qwen3-coder complex tool-argument types, [PR #28817](https://github.com/ggml-org/llama.cpp/pull/28817)
  Jinja dot-property integer literals, and the shared PEG-parser cleanup
  across `common/parsers/*`): the structured `report_cache` tool-call check
  with Q8 K/V still passes on Spark-X2.5 and MiniCPM5 on both backends, so
  the parser rework did not break the two families the live suite exercises.
- **Nemotron-H expert FFN guard** ([PR #28779](https://github.com/ggml-org/llama.cpp/pull/28779)):
  a layer without both `expert_feed_forward_length` and `expert_used_count`
  now fails with a clear error instead of dividing by zero. Correct GGUFs are
  unaffected; the Nemotron profiles and the hybrid-KV heuristic in
  `scanner.py` are unchanged.

## Vision + DFlash2 is still broken on b10948 (PR #28587)

No commit in the range touches `tools/server/server-context.cpp`, the
speculative helpers or the recurrent memory module, so the failure documented
for b10896–b10930 was expected to persist. It was **re-executed** rather than
inferred:

| Build | Actual image + DFlash2 request |
|---|---|
| b10901 / b10903 / b10930 | HTTP 500 (v5.4.6 / v5.4.7 audits) |
| **b10948 Vulkan** | **HTTP 500 reproduced** (`X = 3`, `Y = 18`, `failed to process speculative batch`) |
| **b10948 HIP** | **HTTP 500 reproduced** (same log lines) |

The gate `tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE = 10896` therefore stays
open-ended; its message now names b10948 as the last verified build. The two
documented alternatives (Draft off for images, Vision off for text-only
DFlash2) were both re-run successfully on b10948 HIP and Vulkan. The gate must
be lowered only after an actual image + DFlash2 request succeeds.

## Build recipes on this workstation

Both mainline pre-release recipes were executed unchanged
(`llama_prerelease_{vulkan,hip}_build.ps1 -Tag b10948`), followed by the two
stable recipes, which re-verify the installed `0.4.0` trees. Toolchain actually
used: Visual Studio 2026 (MSVC 14.51.36231 / compiler 19.51.36257), CMake
4.4.3, Vulkan SDK 1.4.341.1 (`glslc` with coopmat, coopmat2, bfloat16,
integer-dot and e4m3 support), ROCm HIP SDK 7.2 (clang 21) with the
workspace-local LLVM PR #201563 `<cmath>` fix.

- **PR #28816, `-fno-pch-timestamp` for clang:** upstream now adds
  `-Xclang -fno-pch-timestamp` to every clang C/C++ compile so the
  precompiled headers introduced in b10911 no longer embed the header
  modification time. The HIP recipe compiles with ROCm clang, so this is the
  first local build where the flag is active: it is present in the generated
  `build.ninja` for both the C and C++ rules, the `ggml-cpu`, `llama` and
  `llama-common` PCH objects were produced, and the ten unity sources compiled
  without a PCH rejection. No recipe change was needed. MSVC (Vulkan tree) is
  not affected.
- **PR #28830, Vulkan NVIDIA queue-submit workaround:** a device-wide submit
  mutex is only installed when `vendor_id` is NVIDIA. The two AMD cards and
  the Intel iGPU keep the previous path; the Vulkan live suite shows no
  change.
- **`scripts/ui-assets.cmake`:** the embedded web UI now writes an embed
  fingerprint and skips re-downloading `dist.tar.gz` when the local archive
  already matches the checksum. The recipes build the UI from `tools/ui` with
  the bundled Vite toolchain, which still works; the pre-existing
  `vswhere.exe`-not-found message printed by that npm step is unchanged from
  b10930 and harmless (the recipe locates Visual Studio through its own
  absolute `vswhere` path).
- Vulkan tree facts from the actual run: Visual Studio 18 2026 generator,
  `BUILD_SHARED_LIBS=OFF`, `GGML_RPC=OFF`, AVX2/AVX-VNNI/BMI2 on and AVX-512
  off, **50 MSVC diagnostics / 0 errors** (`D9025` is upstream's own `/w`
  override for third-party sources; `MSB8027` is the known benign
  `src/llama.cpp` vs `src/models/llama.cpp` basename clash, one `llama.obj`).
- HIP tree facts from the actual run: ROCm clang 21 with the workspace-local
  resource dir, `GPU_TARGETS=gfx1201`, `GGML_CUDA_FA_QUANTS=all`,
  `GGML_CUDA_NO_PEER_COPY=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`,
  **511 Ninja targets** (510 on b10930; `common/json-schema.cpp` is new),
  **11,125 clang warnings / 0 errors** (upstream `-Wsign-compare` and
  `-Wnested-anon-types`), ROCm DLLs bundled, kernel directories junctioned,
  `ROCm0` = R9700 and `ROCm1` = RX 9070 XT, and the deterministic two-GPU
  decoded-text check `HIP MULTI GPU OK`.
- No recipe parameter needed changing. Old build folders were preserved. The
  stable recipes found the existing `0.4.0` (b10809) trees, re-ran
  `--version`, `--list-devices` and the HIP semantic check on them and
  reported success without rebuilding.

## Still open upstream (no AutoTuner change)

- **DeepSeek-V4.1-Flash:** [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
  is still open (last updated 2026-09-13) and conversion-only; the profile
  block and all nine language packs now name b10948.
- The `pi` assistant (`PI_MODEL_NAME`, PR #28853), CI job caps, reduced
  flash-attention test sizes, SYCL memory fix, OpenCL row alignment and the
  s390x repack guard have no Windows HIP/Vulkan impact.
