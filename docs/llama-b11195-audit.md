# llama.cpp b11195 audit — AutoTuner v5.5.7

Checked 2026-09-26 against **b11195**, `d834d44e6` (`0.5.0-dev`), on the
user-built Windows HIP and Vulkan trees (built with the unchanged repository
recipes). The [b11160…b11195 comparison](https://github.com/ggml-org/llama.cpp/compare/b11160...b11195)
contains **35 commits / 128 changed files**. Stable is still **v0.5.0 =
b11146**. See the [CLI/binary manifest](llama-b11195-server-flags.json) and
[release validation](v5.5.7-validation.md) for execution evidence and limits.

## CLI and server contracts

Both binaries still advertise **415 option names / 328 long options**, and
after stripping the timestamped `llama_server: initializing ...` startup line
the `--help` text is **byte-identical to b11160** on both backends. Device
enumeration is unchanged (HIP: ROCm0 = R9700, ROCm1 = RX 9070 XT; Vulkan:
Vulkan0 = RX 9070 XT, Vulkan1 = R9700, Vulkan2 = Intel iGPU).

Server-side changes without an AutoTuner control: cpp-httplib 0.58.0
(#29407); shared UTF-8 path/wide-string helpers for the MCP stdio launcher,
router child environment and HF cache path (#29415, #29432); a grammar
`<[token-id]>` parser that no longer truncates ids above `UINT32_MAX`
(#29382). The new `llama_batch_ext` C API (#24669) is not yet used by the
server. New environment variables are backend knobs, not server options:
`GGML_CPU_TILED_MM` (CPU tiled matmul master switch, on by default),
`GGML_CPU_TILED_MM_FORCE` (test/bench only) and `GGML_CUDA_MMQ_PREC`
(NVFP4/MXFP4 activation precision, effective on Blackwell only). AutoTuner
sets none of them.

## Models

### MiMo-V2 — separate MTP head export and a b11195 floor

[PR #29294](https://github.com/ggml-org/llama.cpp/pull/29294) fixes the
`-sm tensor` split of fused QKV weights with uneven K/V head sizes (MiMo:
d_k 192, d_v 128) and, together with it, teaches the converter the
`--mtp` / `--no-nextn` export for `mimo2` (previously refused with "not
supported") and the loader an **MTP-only** mimo2 GGUF: when `blk.0` is
absent, trunk tensors become optional. Such a head is used as `-md` with
`--spec-type draft-mtp` — AutoTuner's existing same-architecture sidecar path
(`has_embedded_mtp`, root-tensor preflight). Older builds require every trunk
block in each mimo2 file and abort while loading the head, so
`check_draft_model_build` now refuses a mimo2 MTP-style sidecar
(`mimo2.nextn_predict_layers > 0`, tensor scan not "absent") below **b11195**
with a clear message. Trunk-only exports (no nextn key), stale keys over
scan-proven trunk files and other architectures are unaffected; the
`mimo-v2_6.yaml` floor (b11102) is unchanged and its note (all nine packs)
names the head requirement. AutoTuner does not set `--split-mode`, so the
tensor-split fix only matters for manual `-sm tensor` extra arguments.

**Limits:** no MiMo weights are installed locally; unit coverage only.

### Per-tensor activation precision (NVFP4 W4A16)

[PR #24364](https://github.com/ggml-org/llama.cpp/pull/24364) adds the GGUF
keys `general.tensor_extra.name` / `general.tensor_extra.prec_a4` and a
`llama_prec_policy` that keeps W4A16-NVFP4 layers on 8-bit activations. The
native W4A4 path exists only on NVIDIA Blackwell; on HIP (RDNA4) every
NVFP4/MXFP4 MMQ already uses Q8_1 activations, so nothing changes on this
machine. AutoTuner's header scan ignores the new keys (no planner input).

### LFM2 audio

[PR #29403](https://github.com/ggml-org/llama.cpp/pull/29403) corrects the
`lfm2a` mel preprocessor (log floor, symmetric Hann window, epsilon on the
standard deviation) for LFM2.5-Audio speech input through `mtmd`. No
AutoTuner setting is involved; the `lfm2.yaml` profile note already says the
audio pipeline runs through llama.cpp's dedicated audio paths.

## CPU, Vulkan and HIP backend changes for this machine

- **[PR #27851](https://github.com/ggml-org/llama.cpp/pull/27851), b11195 —
  tiled CPU `mul_mat`:** a 256×256 int8 tile path with AVX2 / AVX-512-VNNI
  microkernels for Q2_K…Q6_K, IQ4_XS and the grid IQ types (IQ1/IQ2/IQ3); it
  replaces the earlier `iqp` IQ panel path (#27402). It is used only for
  batches of at least 64 rows and only for weights that are not in a CPU
  repack buffer (on AVX2, Q4_K, Q5_K, Q6_K and Q2_K are repacked already).
  In AutoTuner's hybrid GPU plans, prompt batches on CPU-resident experts are
  op-offloaded to the GPU from 32 tokens on, and decode stays below 64 rows,
  so the path matters for CPU-only runs. The 285K uses the AVX2 kernel (the
  recipes' `GGML_AVX2=ON`, no AVX-512). **Measured CPU-only regression for
  IQ4_XS on this CPU:** Qwen3.8-27B-UD-IQ4_XS, `llama-bench -dev none -ngl 0`,
  pp512 at 8 / 16 / 24 threads drops from 30.1 / 40.3 / 50.1 (b11160) to
  22.1 / 33.9 / 43.2 tok/s (b11195), −27 / −16 / −14 %; tg32 is unchanged
  (5.2 tok/s). `GGML_CPU_TILED_MM=0` does not help (13.6 tok/s at 16 threads):
  the faster b11160 path was the removed `iqp` panel code, not the plain
  `vec_dot` fallback. The hybrid Ling-3.0 plan (`--n-cpu-moe 13`) shows no
  change (HIP 30.8 → 31.3, Vulkan 24.2 → 22.9 tok/s decode, single runs).
  AutoTuner sets no CPU-matmul variable; this is an upstream trade-off for
  AVX2 CPUs without AVX-512 that only affects CPU-only prompt processing.
- **[PR #29409](https://github.com/ggml-org/llama.cpp/pull/29409):** Vulkan
  builds with a `glslc` lacking `GL_KHR_cooperative_matrix` compile again
  (issue #29373). The local Vulkan SDK 1.4.341.1 supports cooperative
  matrix, coopmat2, integer dot, BF16 and E4M3; the b11195 tree detected the
  same five `GGML_VULKAN_*_GLSLC_SUPPORT` capabilities as b11160.
- **HIP:** FP8 (`__hip_fp8_e4m3`) now requires HIP ≥ 6.3 (#29231; the
  recipes use ROCm HIP SDK 7.2); the CUDA/HIP backend fuses `RMS_NORM` +
  `SCALE` (#29393) and accepts F16 `CONV_2D_DW` (#29064, vision encoders).
- mtmd/llama: `n_pos_per_embd()` now uses `GGML_MROPE_SECTIONS` (same value).

## Build recipes

**Mainline:** the recipes remain optimal for this Core Ultra 9 285K (AVX2,
AVX-VNNI, BMI2, no AVX-512) with RX 9070 XT + AI PRO R9700 (both gfx1201).
b11160…b11195 adds no CMake option (only the ggml 0.25.3 version bump and the
new tiled CPU sources, picked up by upstream's lists). The user-built b11195
trees carry exactly the b11160 option set in `CMakeCache.txt` (Vulkan:
AVX2/AVX-VNNI/BMI2, AVX-512 off; HIP additionally `GPU_TARGETS=gfx1201`,
`GGML_HIP_GRAPHS`, `GGML_HIP_NO_VMM`, `GGML_CUDA_NO_PEER_COPY=ON`,
`GGML_CUDA_FA_QUANTS=all`, FMA/F16C). Stable remains v0.5.0 (b11146),
built and verified in the v5.5.6 audit.

**ROCmFPX fork:** no commit after the pinned `aed0d5fd9` (2026-09-06);
[ROCmFPX #26](https://github.com/ROCmFPX/ROCmFPX/issues/26) is still open
without replies, so the pin and the local RDNA4 MMQ patch stay.

**PrismML Ternary/Bonsai fork — pin moved to prism-b10743:** the fork
published prism-b10709, -b10735 and **prism-b10743-adfffbe** (2026-09-25,
56 commits after the old pin 5d80cff). Relevant for this machine: a Vulkan
PQ2_0 dequant path and a PTQ1_0 integer-dot mat-vec (b10687 had no Vulkan
PQ2_0 code at all), native PQ2_0 unpacking in the HIP MMQ tile loader,
vectorized HIP PTQ1_0 `vec_dot`, AVX2/AVX-VNNI PQ2_0 CPU kernels, Hadamard
rotation tensors kept out of `CPU_REPACK` (Prism #245, which fixes Prism
issue #180), the Hadamard inverse for in-file MTP token embeddings, and
DFlash2 support. No new ggml type ids (PQ2_0 = 142, PTQ1_0 = 143, `GGML_TYPE_COUNT`
144). Both recipes now pin `adfffbe41b2cabcd51fff326ab045662265062bb` with
`-FixedIdentity "b10743"`; the new `2b_b10743_{vulkan,hip}_llama.cpp` trees
built cleanly next to the untouched b10687 trees (`--version`: build 10743,
HIP two-GPU semantic check `HIP MULTI GPU OK`). `llama-bench` on the R9700,
`-ngl 99 -fa on -r 3`, ABBA order:

| Backend | Packing | Test | prism-b10687 | prism-b10743 | Δ |
|---|---|---|---:|---:|---:|
| HIP | PQ2_0 | pp512 | 933.5 | 1277.8 | **+36.9 %** |
| HIP | PQ2_0 | tg128 | 57.5 | 57.5 | ±0 |
| HIP | PTQ1_0 | pp512 | 984.8 | 984.1 | ±0 |
| HIP | PTQ1_0 | tg128 | 34.8 | 61.3 | **+76.2 %** |
| Vulkan | PTQ1_0 | pp512 | 433.0 | 430.1 | −0.7 % |
| Vulkan | PTQ1_0 | tg128 | 10.8 | 61.9 | **+473 %** |

Vulkan PQ2_0 was CPU-mapped on b10687 (3.0 tok/s through AutoTuner's plan)
and now runs on the GPU: 6.4 GiB `Vulkan0` weights, 57.6 tok/s decode.
AutoTuner's real planner ran both packings on both backends with the BF16
projector (text `391`, image `Red`, 1…80 list) — see the validation
document. The profile floor stays `min_llama_build: 10687`; notes in all
nine packs name the new pin and the Vulkan PQ2_0 path.

**Fork selection fix:** the recipes never delete an older pinned tree, and
AutoTuner resolved a backend-neutral profile hint such as
`2b_llama/llama-server` to the lexicographically first — i.e. oldest — tree
(`2b_b10687_…` next to `2b_b10743_…`), both in the CLI resolver and in the
GUI's automatic fork switch. Builds of one fork family are now ordered by
backend preference, then newest build number (`b`-builds, then semantic
versions); an explicitly selected fork (`LLAMA_CPP_DIR` / manual GUI choice)
still wins. The fork list's display order is unchanged.

## Older issues: closure is not proof of a fix

Checked on 2026-09-26:

| Concern | Current evidence | AutoTuner action |
|---|---|---|
| Qwen3.5/3.8 image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | Open (last update Aug 30); HTTP 500 reproduced on b11195 HIP and Vulkan | Keep the b10896+ gate; wording names b11195 |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open, not merged (updated Sep 22); no V4.1 loader or converter in b11195 | Keep block, wording names b11195 |
| Xing 4.0 `xing4_0` | No loader in b11195; negative load still `unknown model architecture` | Keep recognition-only block |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Open (active discussion Sep 26); `GGML_TYPE_COUNT` still 43 | Keep fork marker gate |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | Open/unmerged since Aug 3 | Keep tensor-type/fork gate |
| RDNA4 MMQ fallback, [ROCmFPX #26](https://github.com/ROCmFPX/ROCmFPX/issues/26) | Open, no replies, no new fork commits | Keep local recipe patch |
| PQ2_0 CPU repack, [Prism #180](https://github.com/PrismML-Eng/llama.cpp/issues/180) | Still formally open, but fixed by Prism #245 (Hadamard rotation tensors kept out of `CPU_REPACK`) in prism-b10735; two reporters confirm the fix on Sep 24/25 | Never seen on this AVX2 machine; the new pin prism-b10743 contains the fix; no `--no-repack` |
| MTP + ngram-mod, [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) | Stale-closed July 31, not fixed | Keep suppression |
| HIP multi-GPU garbage, [#16424](https://github.com/ggml-org/llama.cpp/issues/16424) | Closed; platform P2P faults | Keep `GGML_CUDA_NO_PEER_COPY=ON` |

## Remaining upstream changes

No AutoTuner control is needed for: Metal FWHT kernels, per-dtype FA
libraries, sparse FA and graph-capture fixes; OpenCL A8 Q8_0/Q5_K dp4a
kernels; Hexagon Q5_K, DMA concat, I32 copies and quantizer work; SYCL
sparse FA; MUSA PH1 fixes; RPC allocation-size cache keys and directory
creation; gguf-py tokenizer special-token handling (#29417, #29422); Web UI
SVG preview; CI/test workers. Metal/SYCL/OpenCL/Hexagon/MUSA/RPC runtime
paths are not locally validated by this Windows dual-AMD audit.
