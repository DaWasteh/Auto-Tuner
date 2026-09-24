# llama.cpp b11160 audit — AutoTuner v5.5.6

Checked 2026-09-24 against **b11160**, `70c4e1582` (`0.5.0-dev`), on the
user-built Windows HIP and Vulkan trees (both built today with the unchanged
repository recipes). The [b11105…b11160 comparison](https://github.com/ggml-org/llama.cpp/compare/b11105...b11160)
contains **55 commits / 95 changed files**. Stable is now **v0.5.0 = b11146**
(`7fe450e19`, published 2026-09-23); the local stable trees are still v0.4.1.
See the [CLI/binary manifest](llama-b11160-server-flags.json) and
[release validation](v5.5.6-validation.md) for execution evidence and limits.

## CLI and server contracts

Both binaries still advertise **415 option names / 328 long options**, and
after stripping the timestamped `llama_server: initializing ...` startup line
(b11063+) the `--help` text is **byte-identical to b11105** on both backends.
Only `--version` changed: `0.5.0-dev (build 11160, commit 70c4e1582)` after
[PR #29333](https://github.com/ggml-org/llama.cpp/pull/29333) (b11146).
AutoTuner parses the numeric build, never the semantic version, so the 0.5.0
bump needs no code change; the Windows recipes read the semantic version from
`CMakeLists.txt` and therefore expect `0.5.0-dev` / `0.5.0` correctly.

Server changes without an AutoTuner control: router child processes no longer
inherit the log file and presets may set one (#29212, #29334); router eviction
races (#29217); token counting no longer crashes while the server sleeps
(#29309); Responses API images inside `function_call_output` (#22575);
OpenAI `video_url` parts and `data:` video URIs (#27921, b11136, relevant for
Ling-3.0-flash-VL video input); deduplicated `-hfd` draft downloads (#27934)
and UTF-8-safe HF cache paths (#29320). AutoTuner starts single-model servers
without router mode, `--sleep-idle-seconds`, `-hf` or `-hfd`.

## Models

### Ling-3.0-flash-VL — new profile and M-RoPE gate

[PR #29151](https://github.com/ggml-org/llama.cpp/pull/29151), first tagged
**b11156**, adds the `ling3vl` projector (27-block qwen3-MoE-ViT family,
norm-only 2×2 merger, two-layer projector) and M-RoPE with sections
`[8, 12, 12]` for `bailingmoe3`. The text backbone is unchanged; text-only
Ling 3.0 GGUFs keep NORM rope.

Risk found during the audit: the filename pattern `ling-3.0-flash` in
`ling-3.yaml` (min b10749) also matched `Ling-3.0-flash-VL`. A VL GGUF on an
older build loads the text weights with plain rope instead of M-RoPE and
answers incorrectly without any load error. v5.5.6 therefore adds:

- `settings/ling-3_0-vl.yaml` (longer patterns win), `min_llama_build: 11156`,
  `--jinja --reasoning-preserve`, chat sampling from the published
  `generation_config.json` (**1.0 / top_p 0.95 / top_k 20**), coding the model
  card's evaluation default (**0.6**). The 262,144 ceiling needs YaRN (native
  131,072) and is clamped to GGUF metadata as usual.
- A metadata gate in `check_model_build`: any `bailingmoe3` GGUF with non-zero
  `rope.dimension_sections` is refused on builds below b11156, which also
  covers renamed files and custom profiles. Text GGUFs are unaffected.

**Limits:** no VL weights are installed locally; source/profile/command/unit
coverage only. Sources: [model card](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL).

### Gemma 4 DSpark drafts — build gate

[PR #29226](https://github.com/ggml-org/llama.cpp/pull/29226), **b11132**,
teaches the shared `dflash` graph Gemma 4 backbones (GELU activation,
embedding scale, post-attention/FFN norms, layer output scale, shared K/V
without `attn_v`, tied output). The GGUF keeps `general.architecture=dflash`,
so AutoTuner's existing scanner already selects `draft-dspark` (Markov head)
or `draft-dflash`. Older builds require `attn_v` and reject the extra tensors,
so `check_draft_model_build` now refuses a Gemma-backbone DFlash/DSpark sidecar
(`dflash.hidden_activation` = `gelu*` or `dflash.embedding_scale`) below
b11132 with a clear message instead of a loader abort. Qwen-style DFlash,
DFlash2 and DSpark keep their old floors (b10164 / b10658). The follow-up
converter change [#29339](https://github.com/ggml-org/llama.cpp/pull/29339)
(b11151) only allows vision targets during conversion; it does **not** fix
the Qwen vision + DFlash2 position holes (#27408). No DSpark weights are
installed locally: unit coverage only.

## Vulkan/HIP backend changes for this machine

- **[PR #27952](https://github.com/ggml-org/llama.cpp/pull/27952), b11160:**
  int8 cooperative-matrix MMQ for AMD RDNA3/RDNA4. It needs the
  `VK_KHR_cooperative_matrix` int8 shape from the driver and the
  `GL_KHR_cooperative_matrix` glslc capability the recipe already detects
  (`GGML_VULKAN_COOPMAT_GLSLC_SUPPORT`, Vulkan SDK 1.4.341.1). No new CMake
  option, environment variable or AutoTuner setting is involved; it is on by
  default on both Navi48 cards. On RDNA4, dense Q4_K/Q5_K/Q4_1/Q5_1/NVFP4
  deliberately stay on the FP16 path; Q4_0, Q5_0, Q8_0, IQ4_NL, IQ4_XS,
  MXFP4, Q3_K and Q6_K (and all MoE `mul_mat_id` types except NVFP4) use int8.
  Measured locally on the R9700 (b11159 vs b11160, Qwen3.8-27B): **Q8_0
  prompt processing +13–15 %**, IQ4_XS and UD-Q4_K_XL within ±2 %, token
  generation unchanged; details in the validation document.
- Vulkan IQ4_XS MMQ/MMV kernels (#28415, b11126), hidden internal symbols
  (#29139), conv2d/conv3d misalignment (#29365), Intel Xe FA kernels and
  Adreno tuning (not used here).
- HIP: IQ2/IQ3 SWAR dequantization (#27962, b11117; relevant to the local
  Ling IQ3_M and Flash-Next Q2_K_XL quants), fused top-k MoE always enabled
  (#28432), conv3d implicit GEMM (#29137).
- The Vulkan build lacks `GL_EXT_float_e2m1` glslc support (SDK 1.4.341.1),
  but the AMD driver 26.5.2 does not expose
  `VK_EXT_shader_ocp_microscaling_types` either, so nothing is lost;
  `VK_EXT_shader_float8` is exposed and compiled in (E4M3 support).

## Build recipes

The recipes remain optimal for this Core Ultra 9 285K (AVX2, AVX-VNNI, BMI2,
no AVX-512) with RX 9070 XT + AI PRO R9700 (both gfx1201): b11105…b11160 adds
no CMake option (only version bumps, hidden Vulkan symbols and the ling3vl
source file). The HIP option set is unchanged (`GGML_HIP_GRAPHS`,
`GGML_HIP_NO_VMM`, `GGML_CUDA_NO_PEER_COPY`, `GGML_CUDA_FA_QUANTS=all`;
`GGML_HIP_MMQ_MFMA` affects CDNA only; there is no rocWMMA FA option anymore).
The stable recipes resolve `latest` to **v0.5.0**, verify it against the
b11146 pre-release tag and expect runtime version `0.5.0`; both stable
recipes were run successfully for v0.5.0 during this audit (new
`0.5.0_{vulkan,hip}_llama.cpp` trees, HIP two-GPU semantic check passed).

## Older issues: closure is not proof of a fix

Checked on 2026-09-24:

| Concern | Current evidence | AutoTuner action |
|---|---|---|
| Qwen3.5/3.8 image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | Open (last update Aug 30; a fork patch is confirmed there, not upstream) | Keep the b10896+ gate; recheck on both b11160 backends |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open, not merged (updated Sep 22); no V4.1 loader in b11160 | Keep block, wording names b11160 |
| Xing 4.0 `xing4_0` | No loader in b11160 | Keep recognition-only block |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Open; `GGML_TYPE_COUNT` still 43 | Keep fork marker gate and recipe pin |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | Open/unmerged since Aug 3 | Keep tensor-type/fork gate |
| RDNA4 MMQ fallback, [ROCmFPX #26](https://github.com/ROCmFPX/ROCmFPX/issues/26) | Open, no replies | Keep local recipe patch |
| PQ2_0 CPU repack, [Prism #180](https://github.com/PrismML-Eng/llama.cpp/issues/180) | Open; candidate patch only | Keep workarounds and pin |
| MTP + ngram-mod, [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) | Stale-closed July 31, not fixed | Keep suppression |
| HIP multi-GPU garbage, [#16424](https://github.com/ggml-org/llama.cpp/issues/16424) | Closed; platform P2P faults | Keep `GGML_CUDA_NO_PEER_COPY=ON` |

## Remaining upstream changes

No AutoTuner control is needed for: CUDA conv3d/sparse-FA DSV4/top-k MoE,
Metal bf16 mul_mv, FA table keys and multi-buffer views, SYCL fusions and
new ops, OpenCL Q4_K/Q6_K dp4a kernels, Hexagon DMA cache and MUL_MAT_ID
guard, sampler probe size, Jinja unary operators, mtmd bounds checks
(#29276: out-of-range feature layers and resize targets now throw instead
of reading out of bounds), test/CI/release-script and Web UI (WEBM upload)
updates. Metal/SYCL/OpenCL/Hexagon runtime paths are not locally validated
by this Windows dual-AMD audit.
