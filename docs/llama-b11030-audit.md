# llama.cpp b11030 / build-recipe audit — AutoTuner v5.5.1

Checked 2026-09-18. Exact runtime: **b11030**, commit
`bdcbaaf6e7520b68c8c60ff724c67409970d70e1` (`0.4.1-dev`); the newest stable
release is still **v0.4.1** (b10964), inside the range audited in
[b10977](llama-b10977-audit.md).
[Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/b10977...b11030):
53 commits, 168 files (11,459 insertions / 10,174 deletions in the
source directories that matter here, most of it the Vulkan source split).
Both Windows pre-release trees were built the same morning with the unchanged
repository recipes (`llama_prerelease_{vulkan,hip}_build.ps1 -Tag b11030`),
then every recipe was re-run in verify mode against them and against the
`0.4.1` stable trees. See [validation](v5.5.1-validation.md) for the test
boundaries and [binary/help hashes](llama-b11030-server-flags.json).

## CLI surface: byte-identical `--help`

No commit in the range touches `common/arg.cpp`, `tools/server/` or the
speculative helpers. The local Vulkan and HIP `llama-server --help` outputs
have the **same SHA-256 as b10948 and b10977** (both backends share one help
text), the parsed option set is **415 names / 328 long options**, nothing was
added or removed, and `--rpc` is still absent only because the recipes keep
`GGML_RPC=OFF`. Every bundled profile's `extra_args` token exists in the
manifest and all **166 standard profile × chat/coding commands per backend**
(one more profile than b10977: `dfm-mimir.yaml`) parse on the real server.
No migration of `--load-mode`, `--lazy-mode`, KV-type, speculative or
multimodal flags is needed.

## Behaviour changes checked against AutoTuner

- **New architecture `hrm_text`** ([PR #27625](https://github.com/ggml-org/llama.cpp/pull/27625),
  first tagged b11003): DFM Mimir 1B, see the profile section below.
- **Nemotron MTP with latent projections**
  ([PR #29018](https://github.com/ggml-org/llama.cpp/pull/29018), b11025):
  the `nemotron_h_moe` MTP graph now computes the shared expert before the
  optional `ffn_latent_down` / `ffn_latent_up` projections and registers
  those two tensors for the MTP block. Nemotron 3 Super (latent MoE,
  `moe_latent_size` 1024) GGUFs that carry their MTP block failed the
  loader's tensor count on every older build ("wrong number of tensors;
  expected 781, got 779", upstream's own reproduction), whether speculation
  was enabled or not, because skipped MTP tensors still count. AutoTuner
  now refuses such GGUFs on builds below b11025 before launch
  (`_nemotron_latent_mtp_block_reason`, metadata-driven: architecture,
  `nextn_predict_layers`/tensor scan and `moe_latent_size`). Lightning /
  Nano (PR #26725 MTP without the latent step) and GGUFs with the MTP block
  stripped are untouched; the acceptance rate upstream reports for Lightning
  is unchanged. No Super GGUF is on this workstation, so this gate rests on
  the upstream report, not on a local run.
- **Nemotron-H `layer_norm_epsilon` only** ([PR #28989](https://github.com/ggml-org/llama.cpp/pull/28989),
  b11005): `layer_norm_rms_epsilon` became optional and falls back to
  `layer_norm_epsilon`. The local Nemotron 3 Nano Omni 30B-A3B UD_Q6_K
  carries both keys (1e-5) and was **run live** through the
  `nemotron-3-nano-omni.yaml` profile on both backends (results below).
- **Vulkan sparse flash attention** ([PR #28105](https://github.com/ggml-org/llama.cpp/pull/28105),
  b10982) for the compact-index attention that `qwen4exp` (Qwen3.8
  Flash-Next QSA) and `minimax-m3` use, plus the **qwen4exp hyper-connection
  ops** ([PR #28901](https://github.com/ggml-org/llama.cpp/pull/28901) CPU/CUDA,
  [PR #28988](https://github.com/ggml-org/llama.cpp/pull/28988) Vulkan) and the
  **`mul_mat_id` row-id hoisting limit raised from 256 to 1024 experts**
  ([PR #28501](https://github.com/ggml-org/llama.cpp/pull/28501), b11029):
  Qwen3.8 Flash-Next's 512 experts leave the slow rescanning path on Vulkan.
  The read-lazy PLE check (`lazy_test.py`) ran the user's current Flash-Next
  quant (UD-Q2_K_XL, three shards, 73.5 GiB, PLE table 26.8 GiB IQ4_NL) on
  both backends.
- **CUDA/HIP graphs for MTP draft graphs** ([PR #28549](https://github.com/ggml-org/llama.cpp/pull/28549),
  b11007): `llama_context` keeps two previous graph results (with/without
  outputs) so the draft and target graphs can both be reused. Gemma 4 12B +
  `gemma4-assistant` MTP drafter was run on HIP and Vulkan.
- **Fused-QKV tensor-parallel split state** ([PR #28965](https://github.com/ggml-org/llama.cpp/pull/28965),
  b11009) and **HIP AllReduce** ([PR #27825](https://github.com/ggml-org/llama.cpp/pull/27825),
  b10989) only affect `--split-mode tensor`; AutoTuner plans layer splits
  (`--tensor-split`/`--main-gpu`), so nothing changes.
- **`--fit` auto-context with unified KV** ([PR #28849](https://github.com/ggml-org/llama.cpp/pull/28849),
  b10999): `n_ctx_max` is now `n_ctx_train × n_seq_max` even when the KV
  cache is unified. AutoTuner always emits `-c` and `--fit off`, so its plan
  is unaffected.
- **GGUF data-section alignment** ([PR #28993](https://github.com/ggml-org/llama.cpp/pull/28993),
  b11019) pads relative to the GGUF start for GGUFs embedded in a larger
  file (`llama_model_load_from_file_ptr`); plain files start at offset 0, so
  `scanner.py`'s `data_start` computation stays correct and the tensor-info
  scan (`__mtp_scan__`) is unchanged.
- **`qwen35moe` fused `gate_up_exps` skip** ([PR #29014](https://github.com/ggml-org/llama.cpp/pull/29014),
  b11026): loading MoE Qwen3.5 GGUFs with fused MTP tensors while MTP is off
  no longer creates the expert tensors twice. AutoTuner's MTP-off path for
  those targets (`prepare_command_for_binary`) is unchanged.
- **Chat parsers**: the DeepSeek V3.2/V4 parser now declares message
  delimiters so the server can place context checkpoints at user turns
  ([PR #29008](https://github.com/ggml-org/llama.cpp/pull/29008), b11020), and
  Qwen3-Coder forces `\n</think>` when the reasoning budget ends
  ([PR #28869](https://github.com/ggml-org/llama.cpp/pull/28869), b10996).
  Both are server-internal; `--reasoning-budget` and the V4 profile are
  unchanged.
- **Vulkan source split** ([PR #28732](https://github.com/ggml-org/llama.cpp/pull/28732)):
  `ggml-vulkan.cpp` lost 5,871 lines to `ggml-vulkan-buffers.cpp`,
  `ggml-vulkan-debug.cpp` and three shared headers; upstream's CMake lists
  pick them up, the recipe needed no change. NV `argsort_large` workaround,
  im2col alignment, coopmat1 MoE skip and the cm2 `mul_mm` tail are Vulkan
  kernel fixes without a launcher-side effect.
- **Everything else** (OpenVINO 2026.4, SYCL, hexagon K-quants, OpenCL
  ssm_scan/warnings, spacemit, Metal MiniCPM3 FA, RPC hash cache / ACCEL
  skip, `llama-bench --version`, webui reasoning menu, CI) has no Windows
  HIP/Vulkan effect on AutoTuner.

## New architecture: HRM-Text (DFM Mimir 1B)

[PR #27625](https://github.com/ggml-org/llama.cpp/pull/27625) (merged
2026-09-16, first tag **b11003**) adds `hrm_text` for
`danish-foundation-models/DFM-Mimir`: two 16-layer transformer stacks (low
and high) run in alternating cycles (`H_cycles` 2 × `L_cycles` 3) over the
same token stream, so the GGUF `block_count` is the expanded cache-slot count
16 × 2 × (3 + 1) = **128** while the file holds 32 physical blocks; hidden
size 1,536, 12 attention heads without GQA (head 128), sigmoid-gated
attention, SwiGLU FFN 4,096, parameterless RMS norms, learned embedding scale
39.19, GPT-2/Gemma-style 262,144 vocabulary, `max_position_embeddings`
**4,096**, Apache 2.0. Upstream implements causal attention only (the
`prefix_lm` key round-trips unused), reserves a 128-slot looped graph
(`n_tokens × 80` nodes) and mirrors all tensors for the meta split.
Community GGUFs: `noctrex/DFM-Mimir` (BF16/F16 3.6 GB, Q8_0 1.9 GB).

AutoTuner ships `settings/dfm-mimir.yaml` (patterns `dfm-mimir`, `hrm-mimir`,
`mimir-1b` and spellings, `arch_fallback: hrm_text`, `min_llama_build:
11003`, `max_context: 4096`, `--jinja` for the Gemma-4-style template whose
thinking is opt-in through `enable_thinking`, generic sampling because DFM
publishes no `generation_config`) and all nine language packs carry its note.
The KV estimate follows the GGUF `block_count`: 128 × 12 × (128 + 128) × 2
bytes = 786,432 bytes per token, i.e. **3,072 MiB at 4,096 tokens in F16**,
exactly the figure PR #27625 documents (halved with Q8 K/V), so the planner
does not under-provision the 4× cache. Decode runs all 128 block passes per
token (roughly four times a dense model of equal width). No Mimir GGUF is on
this workstation; the profile is covered by unit tests (name/arch matching,
build gate, context cap, command flags, KV arithmetic, pack notes), not by a
live run.

## Vision + DFlash2 on Qwen3.5/3.8 (PR #28587)

No commit in the range touches `tools/server/`, the speculative helpers,
`llama-batch` or the recurrent memory module, so the failure documented for
b10896–b10977 was expected to persist. It was **re-executed**:

| Build | Actual image + DFlash2 request |
|---|---|
| b10901 / b10903 / b10930 / b10948 / b10977 | HTTP 500 (v5.4.6–v5.5.0 audits) |
| **b11030 Vulkan** | **HTTP 500 reproduced** (`failed to process speculative batch`) |
| **b11030 HIP** | **HTTP 500 reproduced** (same log lines) |

The gate `tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE = 10896` stays open-ended
and its message names b11030 as the last verified build. The two documented
alternatives (Draft off for images, Vision off for text-only DFlash2) were
re-run on both backends.

## Live runs (b11030, both backends)

| Check | Vulkan | HIP |
|---|---|---|
| `command_matrix.py`: every non-blocked profile × chat/coding command parsed by the real server | 166 / 166 | 166 / 166 |
| `runtime_test.py`: Spark-X2.5-4B and MiniCPM5-2B-F16, 16k, Q8 K/V, deterministic reply, structured tool call | PASS / PASS | PASS / PASS |
| `draft_test.py`: Qwen3.8-27B Q8 + DFlash2 drafter, two-GPU split, drafter pinned to the emptier card | PASS | PASS |
| `vision_only_test.py`: Qwen3.8-27B + mmproj, DFlash2 present but Draft off, image request | PASS | PASS |
| `vision_draft_regress.py`: gate bypassed, image + DFlash2 | HTTP 500 reproduced | HTTP 500 reproduced |
| `lazy_test.py`: Qwen3.8 Flash-Next UD-Q2_K_XL (qwen4exp, sparse FA + hc ops), read-lazy PLE residency | see validation | see validation |
| `gemma4_test.py`: Gemma 4 12B Q8 + `mtp-gemma-4-12b` drafter, thinking off | PASS | PASS |
| `maple_test.py`: Maple-Preview TQ2_0, 32k, thinking on | PASS | PASS |
| `nemotron_test.py`: Nemotron 3 Nano Omni 30B-A3B UD_Q6_K, text-only, 16k | PASS | PASS |

Exact placements, throughputs and residency figures are in
[v5.5.1-validation.md](v5.5.1-validation.md).

## Build recipes on this workstation

Both mainline pre-release trees (`b11030_{vulkan,hip}_llama.cpp`) were built
at 11:05 on 2026-09-18 with the unchanged recipes; their `CMakeCache.txt`
carries exactly the recipe options (Visual Studio 18 2026 x64 with the
SPIRV-Headers prefix, `LLAMA_BUILD_IS_DEV=ON`, `GGML_RPC=OFF`; Ninja + ROCm
7.2 clang 21 with the workspace-local LLVM PR #201563 resource dir,
`GPU_TARGETS=gfx1201`, `GGML_CUDA_NO_PEER_COPY=ON`, `GGML_CUDA_FA_QUANTS=all`,
bundled ROCm 7 DLLs and junctioned rocBLAS/hipBLASLt kernel directories). All
four recipes were then re-run for this audit:
`llama_prerelease_{vulkan,hip}_build.ps1 -Tag b11030` verified the existing
trees (`--version` = `0.4.1-dev (build 11030, commit bdcbaaf6e)`, MSVC
19.51.36257 / Clang 21.0.0, `--list-devices` with the three Vulkan devices and
`ROCm0` = R9700 / `ROCm1` = RX 9070 XT, the deterministic two-GPU decoded-text
check `HIP MULTI GPU OK`) and reported `Success (existing)`; the stable
recipes resolved `latest` to **v0.4.1 / b10964** and verified the `0.4.1`
trees the same way. The Vulkan source split (PR #28732) and the new
`flash_attn_sparse_compact.comp` shader are compiled by upstream's lists; no
recipe line changed.

## Still open upstream (no AutoTuner change)

- **DeepSeek-V4.1-Flash:** [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
  is still an open draft (last updated 2026-09-13) and conversion-only; the
  profile block and all nine language packs now name b11030.
- Maple still has CPU and Vulkan TQ kernels only; CUDA/HIP keep the ternary
  expert tensors on the CPU (re-confirmed by the HIP Maple run).
- HRM-Text prefix-LM prefill is not implemented upstream (causal only).
