# llama.cpp b11249 audit — AutoTuner v5.5.9

Checked 2026-09-29 against **b11249**, `6d78fb072` (`0.5.0-dev`), on the
user-built Windows HIP and Vulkan trees (built with the unchanged repository
recipes). The [b11195…b11249 comparison](https://github.com/ggml-org/llama.cpp/compare/b11195...b11249)
contains **54 commits / 135 changed files**. Stable is still **v0.5.0 =
b11146**. See the [CLI/binary manifest](llama-b11249-server-flags.json) and
[release validation](v5.5.9-validation.md) for execution evidence and limits.

## CLI and server contracts

Both binaries now advertise **416 option names / 329 long options**. The only
difference from b11195 is **`--rpc`**: [PR #29537](https://github.com/ggml-org/llama.cpp/pull/29537)
registers it unconditionally and checks `llama_supports_rpc()` only when the
flag is used, so a build without `GGML_RPC` (such as these recipe builds) now
lists `--rpc` and rejects it at parse time with "RPC not supported in this
build". AutoTuner never plans RPC offload. Its compatibility filter used to
drop an unadvertised `--rpc host:port` from user Extra CLI flags; on b11249 such
an extra is passed through and the launch fails with llama.cpp's own message
instead of silently running without RPC. The same PR prints the server's
`initializing ...` log line only after argument parsing, so `--help` no longer
emits the timestamped startup line the b11063…b11195 manifests had to strip
(0 stripped lines now). AutoTuner detects readiness through `/health` and
parses no startup log lines. Device enumeration is unchanged (HIP: ROCm0 =
R9700, ROCm1 = RX 9070 XT; Vulkan: Vulkan0 = RX 9070 XT, Vulkan1 = R9700,
Vulkan2 = Intel iGPU).

Other server changes without an AutoTuner control: `/v1/embeddings` accepts
typed image/audio/video content for multimodal embedding models
([#29556](https://github.com/ggml-org/llama.cpp/pull/29556)); RANK pooling may
split batches for causal rerankers such as Qwen3/Qwen3-VL
([#28876](https://github.com/ggml-org/llama.cpp/pull/28876)); the Windows
`wake_fd` warning is gone ([#29479](https://github.com/ggml-org/llama.cpp/pull/29479));
Windows Hugging Face cache paths are UTF-8/`fs::path` based
([#29475](https://github.com/ggml-org/llama.cpp/pull/29475),
[#29595](https://github.com/ggml-org/llama.cpp/pull/29595); AutoTuner scans its
own model folders, not the HF cache); a grammar without llguidance throws
instead of aborting ([#29516](https://github.com/ggml-org/llama.cpp/pull/29516));
`string_split<T>` rejects invalid numbers
([#29518](https://github.com/ggml-org/llama.cpp/pull/29518), used only by
batched-bench options); Jinja gains `sameas`, `dict` and argument tests on
non-call statements (#29443, #29448, #29477). The auto-fit context change
of PR #28849 was reverted ([#29437](https://github.com/ggml-org/llama.cpp/pull/29437));
AutoTuner always passes `--fit off` and sizes the context itself.

**Speculative decoding / mtmd / server batches** moved to the new
`llama_batch_ext` API ([#29385](https://github.com/ggml-org/llama.cpp/pull/29385)).
For a classic `-md` draft whose input width differs from the target's, image
rows are now replaced by zero rows (warning logged once) so draft positions stay
contiguous. The DFlash path is unchanged: it still skips pinned M-RoPE image
batches, so the Qwen3.5/3.8 vision + DFlash2 failure persists (below). Text
DFlash2, Gemma 4 MTP assistants, Qwen3.8 Flash-Next and Ling 3.0 were re-run
through AutoTuner's planner on both backends.

`llama-bench` (not used by AutoTuner) gains `--repack <0|1>`.

## Models

### Muse Glimmer — structured output needs b11249

[PR #29615](https://github.com/ggml-org/llama.cpp/pull/29615) makes the
dedicated Muse Glimmer chat parser (selected by `--jinja`) build a constrained
grammar for `response_format` `json_schema`; before, the schema was ignored
unless tools were offered. The `muse-glimmer.yaml` loader floor (b10353) and
the b11100 tool-call note stay; the note in all nine language packs now says
that structured output needs b11249+. Measured on the local
Muse-Glimmer-30B UD-Q5_K_XL through AutoTuner's plan — see the validation
document.

### Nemotron 3 Puzzle — HIP Mamba-2 scan with state 96

Puzzle shrinks the Mamba-2 state from 128 to 96 channels. [PR #28717](https://github.com/ggml-org/llama.cpp/pull/28717)
adds a 96-state `SSM_SCAN` kernel to the CUDA/HIP backend and reports it as
supported. Vulkan's `SSM_SCAN` still accepts only 128/256 states, so on Vulkan
(and on HIP before b11249) the scheduler runs that operation on the CPU in
every Mamba layer. The `nemotron-3-puzzle.yaml` floor (b10786) is unchanged;
the note in all nine packs recommends HIP b11249+. **Limit:** no Puzzle
weights are installed locally; the note follows the backends' `supports_op`
tables, not a measurement.

### Other model-side changes

- **Qwen3.8 Flash-Next (`qwen4exp`):** Vulkan fuses the hyper-connection gate
  chain SCALE → SIGMOID → SCALE → `hc_post` into one dispatch
  ([#29520](https://github.com/ggml-org/llama.cpp/pull/29520), upstream +3.8 %
  decode); the non-causal block path no longer depends on `causal_attn`
  ([#28751](https://github.com/ggml-org/llama.cpp/pull/28751)). No setting.
- **DFlash2 conv:** taps past the block size are skipped and padding uses
  `ggml_pad_ext` (#29567) — same output, fewer ops.
- **HRM:** `z_l_init` placement fix (#29512); PLaMo-3 YaRN export in the
  converter (#29528). No AutoTuner profile involved.
- No new architecture, tensor type or chat parser besides the Muse Glimmer fix
  (`GGML_TYPE_COUNT` stays 43).

## CPU, Vulkan and HIP backend changes for this machine

- **Vulkan:** descriptor sets are reused when a graph is re-recorded with
  identical bindings ([#29280](https://github.com/ggml-org/llama.cpp/pull/29280));
  `mul_mat`/`mul_mat_id` read the batch stride of an in-place strided view
  (e.g. the first rows of a cache) from `nb[2]` instead of assuming packed
  batches ([#28956](https://github.com/ggml-org/llama.cpp/pull/28956); found
  with a SheetSage2 decoder, upstream also names a Qwen3 `--no-fa` path).
  AutoTuner keeps flash attention on except for the DeepSeek/Unlimited OCR
  profiles and Grok; nothing to change, the fix only removes wrong results;
  Adreno argsort selection and a missing header (#29469, #29597). The A/B
  below measures the net effect on the R9700.
- **HIP:** `SSM_SCAN` state 96 (above); FWHT with F16 input (#29096, the
  Hadamard mat-mul hint); FA MMA for head sizes > 256 and its template fix
  (#28907, #29559) apply to CDNA/MFMA only; the fp16 tile FA retune (#26289)
  changes NVIDIA tables only. Nothing changes the gfx1201 kernel choice.
- **CPU:** tiled flash attention for head sizes that are not a multiple of the
  SIMD width on x86-64 ([#29423](https://github.com/ggml-org/llama.cpp/pull/29423),
  masked AVX-512 tail; on this AVX2 CPU only where the tiled kernel already
  applied). The b11195 tiled `mul_mat` observation is unchanged.
- State save/restore: failed K/V or recurrent restores are cleaned up
  ([#27530](https://github.com/ggml-org/llama.cpp/pull/27530)); relevant to
  the host prompt cache (`--cache-ram`) of hybrid models, no setting.

**Measured on the R9700** (`llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3`,
Vulkan1 / ROCm0, ABBA order, quiet system, same user-built trees):

| Backend | Model | Test | b11195 | b11249 | Δ |
|---|---|---|---:|---:|---:|
| Vulkan | Qwen3.8-27B UD-Q4_K_XL (dense) | pp512 | 887.4 | 885.1 | −0.3 % |
| Vulkan | Qwen3.8-27B UD-Q4_K_XL (dense) | tg128 | 32.16 | 32.14 | −0.1 % |
| Vulkan | Qwen3.6-35B-A3B UD-IQ3_XXS (MoE) | pp512 | 3448.4 | 3446.9 | ±0 |
| Vulkan | Qwen3.6-35B-A3B UD-IQ3_XXS (MoE) | tg128 | 149.05 | 147.62 | −1.0 % |
| HIP | Qwen3.8-27B UD-Q4_K_XL (dense) | pp512 | 1197.7 | 1196.3 | −0.1 % |
| HIP | Qwen3.8-27B UD-Q4_K_XL (dense) | tg128 | 30.54 | 30.49 | −0.2 % |
| HIP | Qwen3.6-35B-A3B UD-IQ3_XXS (MoE) | pp512 | 2649.5 | 2637.0 | −0.5 % |
| HIP | Qwen3.6-35B-A3B UD-IQ3_XXS (MoE) | tg128 | 122.79 | 123.06 | +0.2 % |

No regression and no measurable gain on this discrete RDNA4 card: the
descriptor-set reuse targets drivers where `vkUpdateDescriptorSets` is
expensive (translation layers), not RADV-class AMD drivers. The Vulkan MoE
decode delta (both b11249 runs 0.6–1.4 % below both b11195 runs, per-run σ
≈ 1–2 tok/s) is at the edge of noise.

## Build recipes

**Mainline:** unchanged and still optimal for this Core Ultra 9 285K (AVX2,
AVX-VNNI, BMI2, no AVX-512) with RX 9070 XT + AI PRO R9700 (both gfx1201).
b11195…b11249 changes CMake only for Windows ARM64/ARM64EC with MSVC (#28362);
no new x64 option. The user-built b11249 trees carry exactly the b11195
option set in `CMakeCache.txt` (Vulkan and HIP: AVX2/AVX-VNNI/BMI2, AVX-512
off, `GGML_RPC=OFF`, static libraries; HIP additionally `GPU_TARGETS=gfx1201`,
`GGML_HIP_GRAPHS`, `GGML_HIP_NO_VMM`, `GGML_CUDA_NO_PEER_COPY=ON`,
`GGML_CUDA_FA_QUANTS=all`, FMA/F16C). Stable remains v0.5.0 (b11146).

**ROCmFPX fork — pin kept at `aed0d5fd9` (fork build 11544):** the fork merged
mainline through `e613ef2c8` (= b11056) plus Flash-Next work into main
`721db4193` (fork build 11898, PRs #27 and #30, qualified on gfx1151 only). A
trial build of the unchanged recipe with that commit **fails on Windows**:
`src/models/qwen4exp.cpp` calls `llama_lazy_reader::prefetch`, which exists
only in the POSIX branch of `llama-lazy-reader.h` (`#ifdef _WIN32` has no
such member) — MSVC error C2039. The RDNA4 MMQ fallback is still missing in
`mmq-config-rdna4.cuh` and [ROCmFPX #26](https://github.com/ROCmFPX/ROCmFPX/issues/26)
is unanswered, so the validated pin and the local patch stay; the trial
folders were removed again.

**PrismML Ternary/Bonsai fork:** no release after prism-b10743-adfffbe
(2026-09-25); pin unchanged.

## Older issues: closure is not proof of a fix

Checked on 2026-09-29:

| Concern | Current evidence | AutoTuner action |
|---|---|---|
| Qwen3.5/3.8 image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | Open; a user confirmed the crash on master `1c4729414` (Sep 28); HTTP 500 reproduced on b11249 HIP and Vulkan | Keep the b10896+ gate; wording names b11249 |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open, not merged (review comments Sep 28); no V4.1 loader in b11249 | Keep block, wording names b11249 |
| Xing 4.0 `xing4_0` | No loader in b11249; negative load still `unknown model architecture` | Keep recognition-only block |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Open; `GGML_TYPE_COUNT` still 43, no `prism.hadamard` loader (only the FWHT hint, #29096) | Keep fork marker gate |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | Open/unmerged since Aug 3 | Keep tensor-type/fork gate |
| RDNA4 MMQ fallback, [ROCmFPX #26](https://github.com/ROCmFPX/ROCmFPX/issues/26) | Open, no replies; not fixed in fork main `721db4193` | Keep local recipe patch |
| PQ2_0 CPU repack, [Prism #180](https://github.com/PrismML-Eng/llama.cpp/issues/180) | Formally open, fixed by Prism #245 (in the prism-b10743 pin) | No action |
| MTP + ngram-mod, [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) | Stale-closed July 31, not fixed | Keep suppression |
| HIP multi-GPU garbage, [#16424](https://github.com/ggml-org/llama.cpp/issues/16424) | Closed; platform P2P faults | Keep `GGML_CUDA_NO_PEER_COPY=ON` |

## Remaining upstream changes

No AutoTuner control is needed for: Hexagon backend sampler and tiled
GET_ROWS; Metal FWHT and padding; SYCL FWHT widths; OpenCL kernel loading;
WebGPU unaligned writes; OpenVINO batch-stride views; MUSA/oneAPI Docker and
CI images; RPC RDMA completion channels; mtmd GCC 15 fix and left padding via
`ggml_pad_ext`; `llama-bench` README and hf_file OOB fix; test refactors.
Metal/SYCL/OpenCL/Hexagon/MUSA/WebGPU/OpenVINO/RPC runtime paths are not
locally validated by this Windows dual-AMD audit.
