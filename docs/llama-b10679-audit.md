# llama.cpp b10679 integration and backend-build audit

AutoTuner v5.3.0 was reviewed against the exact upstream range below.

| Tag | Commit | Date |
|---|---|---|
| `b10666` | `4e97ac86ebe2c4cb8212d98d2641ad6768810896` | 2026-08-28 |
| `b10679` | `50f068ffffc3e0e4c9c2e4139281c6075224f429` | 2026-08-28 |

The range contains **13 commits**. The exact source and server-parser diffs,
upstream CMake contracts, official Windows HIP SDK documentation, locally
compiled backend outputs, and AutoTuner tests form the evidence for this audit.

## AutoTuner v5.3.1 correctness correction

The v5.3.0 audit's speed-only HIP checks missed silent multi-GPU output
corruption. On Windows, direct HIP peer copies between the R9700 and RX 9070 XT
produced invalid logits with every locally maintained HIP family: the server
stayed healthy and reported timings, but decoded isolated symbols, numbers,
Chinese characters, repetition, or immediate EOS. Vulkan and HIP pinned to the
R9700 remained coherent.

The defect reproduced with a 4B Qwen3 Q8_0 oracle and the original
Qwen3.8-27B IQ4_XS SuperCalc request. Rebuilding the same source commits with
`GGML_CUDA_NO_PEER_COPY=ON` fixed both. AutoTuner v5.3.1 therefore makes that
CMake option mandatory for Windows HIP and rejects a build unless a bounded
real two-GPU layer split decodes exactly `HIP MULTI GPU OK`. Throughput-only
`llama-bench` output is no longer accepted as evidence of decode correctness.

## AutoTuner compatibility

No emitted `llama-server` option was added, renamed, or removed between b10666
and b10679, so AutoTuner needs no new command-line gate or capability-pruning
rule. The normal, Vision/OCR, reasoning/tool, placement/KV, and speculative
command contracts from the b10666 audit remain valid.

The relevant runtime changes are transparent rebuild benefits:

- **b10675 / `90c26fcd4`:** Vulkan hoists MoE row IDs and per-expert counts
  instead of repeatedly scanning the routing table in each matrix shader.
- **b10677 / `b387ddfd8`:** Vulkan graph optimization now respects dependencies
  between aliased views, preventing unsafe operation reordering.
- **b10678 / `6fe749801`:** Qwen3.8 Flash Next (`qwen4exp`) builds its PLE input
  once and keeps it in the embedding graph split, reducing graph splits. Its
  architecture metadata, PLE table, QSA KV shape, and CLI are unchanged. This
  is a scheduler/graph-structure optimization, not a new allocation contract;
  the conservative measured memory coefficients need no code change.
- **b10679 / `50f068fff`:** the public C API enum/field
  `llama_tensor_read_lazy` / `tensor_read_lazy` is internally renamed to
  `llama_lazy_mode` / `lazy_mode`. The CLI remains exactly
  `--tensor-read-lazy {on,auto,off}` and now becomes available in
  `llama-bench` too. AutoTuner uses the CLI rather than linking the C API, so
  its existing argument and complete flag/value pruning remain correct.

The other commits affect Metal, SYCL, OpenVINO, conversion, CPU
`conv_transpose_2d`, or disable non-fused GDN/LID paths; none requires an
AutoTuner argument change.

## Backend-qualified Windows build contract

Every Windows Vulkan recipe now has a source-identical HIP sibling:

| Family | Vulkan output | HIP output |
|---|---|---|
| Stable | `X.Y.Z_vulkan_llama.cpp` | `X.Y.Z_hip_llama.cpp` |
| Mainline/pre-release | `bNNNN_vulkan_llama.cpp` | `bNNNN_hip_llama.cpp` |
| DiffusionGemma | `d_bNNNN_vulkan_llama.cpp` | `d_bNNNN_hip_llama.cpp` |
| DeepSeek-OCR v1 | `ocr_b17400_vulkan_llama.cpp` | `ocr_b17400_hip_llama.cpp` |
| PrismML Ternary | `2b_bNNNN_vulkan_llama.cpp` | `2b_bNNNN_hip_llama.cpp` |
| TurboQuant | `tq_bNNNN_vulkan_llama.cpp` | `tq_bNNNN_hip_llama.cpp` |

Mainline tags are resolved once by `build_all_windows_llama.ps1`; fork/PR heads
are pinned to exact reviewed commits. Existing known-good destinations are
never deleted or replaced; a destination that fails the build-safety gate is
reconfigured and rebuilt in its existing clean source tree. AutoTuner strips
only the terminal `_vulkan`/`_hip` token
for family matching, keeps both full names in the picker, preserves the active
backend when a profile switches family, and honors an explicit backend hint.

## Pandaking optimization choices

The target workstation has a Core Ultra 9 285K (AVX2, AVX-VNNI and BMI2; no
AVX-512), an RX 9070 XT 16 GB and an AI PRO R9700 32 GB. Both AMD GPUs report
`gfx1201`.

- **Vulkan:** Visual Studio 18 2026, static llama/ggml libraries, AVX2 +
  AVX-VNNI + BMI2, AVX-512 off, checks/debug/validation off. Current `glslc`
  capability probes select integer-dot, BF16 and cooperative-matrix shaders;
  no guessed GPU target flag is used.
- **HIP (audited local configuration):** AMD HIP SDK 7.2.3, Ninja, ROCm `clang`/`clang++`,
  `GPU_TARGETS=gfx1201`, HIP graphs on, VMM/RCCL off for Windows, direct peer
  copies disabled with `GGML_CUDA_NO_PEER_COPY=ON`, and all Flash-Attention KV
  quant instantiations enabled. The SDK's clang-21 wrapper
  still lacks merged LLVM PR #201563 and collides with MSVC 14.51 `<cmath>`;
  the recipe applies that exact include reorder in a workspace-local clang
  resource copy, leaving Program Files untouched. Matching ROCm runtime DLLs
  are bundled beside every HIP executable so Windows cannot load a mismatched
  System32/older-SDK runtime first; local junctions point rocBLAS/hipBLASLt at
  the exact SDK 7.2 kernel trees without duplicating ~1 GiB per build. Vulkan
  and HIP never share a CMake tree.
- **Device order is backend-specific:** the built binary's `--list-devices`
  table remains authoritative. AutoTuner does not infer HIP ordinals from the
  Vulkan order.

## Verification record

All six source families built successfully for both backends. Every tree was
checked for exact source commit, optimized CMake cache, `llama-server --version`,
backend-exclusive `--list-devices`, and both intended AMD GPUs. For v5.3.1 all
six existing HIP trees were reconfigured with no-peer-copy enabled, fully
relinked, and independently passed the deterministic dual-GPU text check. The
mainline b10679 tree additionally passed the exact 262,144-context
Qwen3.8-27B IQ4_XS + DFlash2 + ngram + mmproj SuperCalc reproduction with
coherent reasoning. The mainline pair reports b10679/`50f068fff`; stable
reports 0.3.0/b10621. Pinned fork pairs use DiffusionGemma `dd0cf0445`, OCR
`95cc56658`, PrismML `e311ed38f`, and TurboQuant `df7f54729` identically across
Vulkan/HIP.

Focused R9700 benchmarks (same model/settings, three repetitions) confirm a
backend-dependent tradeoff rather than one universal winner:

| Model | Backend | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|
| Qwen3 4B Q8_0 | Vulkan | 6,264.40 | 121.22 |
| Qwen3 4B Q8_0 | HIP | 7,634.01 | 102.86 |
| Qwen3.8 27B Ridge 3.7bpw | Vulkan | 790.20 | 35.95 |
| Qwen3.8 27B Ridge 3.7bpw | HIP | 1,038.92 | 27.36 |

HIP is materially faster for prompt processing on these workloads; Vulkan is
faster for token-by-token decode. TurboQuant's `turbo3` KV mode completed on
both backends, both DiffusionGemma servers loaded the real 25.03 GiB Q8 model
at 4,096 context and served an HTTP completion, and both OCR MTMD CLIs passed
help/startup smoke checks. The superseded backend-neutral/b10678 trees were
removed only after these replacements passed.

## Primary references

- b10679 release: <https://github.com/ggml-org/llama.cpp/releases/tag/b10679>
- Exact comparison: <https://github.com/ggml-org/llama.cpp/compare/b10666...b10679>
- b10678 qwen4exp graph change: <https://github.com/ggml-org/llama.cpp/commit/6fe74980162af0ed5e559870d5deccafaa034e7c>
- b10679 lazy-mode/bench change: <https://github.com/ggml-org/llama.cpp/commit/50f068ffffc3e0e4c9c2e4139281c6075224f429>
- LLVM HIP/MSVC `<cmath>` fix: <https://github.com/llvm/llvm-project/pull/201563>
- Windows HIP build instructions: <https://github.com/ggml-org/llama.cpp/blob/b10679/docs/build.md#hip>
- ROCm multi-GPU garbage-output report and no-peer-copy workaround: <https://github.com/ggml-org/llama.cpp/issues/16424>
- HIP SDK 7.2 release notes: <https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/about/releasenotes.html>
- HIP SDK 7.2 system requirements: <https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/reference/system-requirements.html>
