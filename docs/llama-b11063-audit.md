# llama.cpp b11063 audit — AutoTuner v5.5.4

Checked 2026-09-20. Mainline runtime: **b11063**, commit `3d82ef62d`
(`0.4.1-dev`, the newest tag that day); the newest stable release is still
**v0.4.1** (b10964). [Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/b11042...b11063):
21 commits. Both Windows pre-release trees (`b11063_{vulkan,hip}_llama.cpp`)
were built by the user with the unchanged repository recipes before this
audit; the b11042 trees no longer exist locally. The fork trees from the
previous releases (`2b_b10687_*`, `fpx_b11544_*`) were not rebuilt. See
[validation](v5.5.4-validation.md) for the test boundaries and
[binary/help hashes](llama-b11063-server-flags.json).

## Mainline b11042 → b11063: same option set, one new log line

None of the 21 commits touches `common/arg.cpp`, the speculative helpers,
`llama-batch` or the recurrent memory. `tools/server/server.cpp` changes only
its startup wording (PR #29125, below). The parsed option set is unchanged:
**415 names / 328 long options**, nothing added or removed, `--rpc` still
absent only because the recipes keep `GGML_RPC=OFF`. The `--help` output
itself is no longer byte-identical: PR #29125 logs a timestamped
`0.00.000.4xx I srv  llama_server: initializing ...` line on stderr before
argument parsing, so it appears even for `--help` and differs per run and
per backend. `make_manifest.py` now drops that single line before hashing
(recorded as `notes.<backend>.stripped_startup_log_lines = 1` in the
manifest); the remaining help bytes have the **same SHA-256 as b10948,
b10977, b11030 and b11042** on both backends. AutoTuner detects readiness
through `GET /health`, not by parsing startup logs, so the reworded security
and port notices have no effect. Every bundled profile's `extra_args` token
exists in the manifest and all **170 standard profile × chat/coding commands
per backend** parse on the real server.

Commits in the range, checked against AutoTuner:

- `chat : add dedicated Ling 3.0 (Bailing V3) parser`
  ([PR #28682](https://github.com/ggml-org/llama.cpp/pull/28682)): see the
  Ling section below. This is the only behavioural change for a bundled
  profile.
- `server : improve startup log messages`
  ([PR #29125](https://github.com/ggml-org/llama.cpp/pull/29125)): the
  `initializing ...` line above, one-line security/feature warnings instead
  of the boxed multi-line ones, the port-change notice on one line, and a
  `[source]` tag per model in the router listing. Cosmetic for AutoTuner.
- `chat : fix gemma4 required tool grammar`
  ([PR #29115](https://github.com/ggml-org/llama.cpp/pull/29115)): the
  Gemma 4 template's `tool_choice: required` grammar. The Gemma 4 MTP run in
  the live chain uses a plain request; AutoTuner's tool-call smoke test
  (Spark-X2.5 / MiniCPM5 with `tool_choice: required`) passes as before.
- `CUDA: enable sparse fa for qwen4exp`
  ([PR #28770](https://github.com/ggml-org/llama.cpp/pull/28770)): the HIP
  build gains the sparse flash-attention path for the Qwen3.8 Flash-Next
  indexer (Vulkan had it since b11030). The Flash-Next lazy-PLE run on HIP
  passes with the same plan; no profile change.
- `common/peg : handle invalid utf-8 sequences in the AST`
  ([PR #29161](https://github.com/ggml-org/llama.cpp/pull/29161)),
  `json-schema : accept escaped hyphen in regex patterns`
  ([PR #29127](https://github.com/ggml-org/llama.cpp/pull/29127)): parser
  robustness for tool-call grammars; no planner impact.
- `fix(mamba) : make time-step projection input contiguous`
  ([PR #28832](https://github.com/ggml-org/llama.cpp/pull/28832)): the
  classic Mamba graph; the hybrid SSM models AutoTuner runs (Nemotron-H,
  Qwen3.5/3.8, Ling 3.0) use their own graphs and behaved as before.
- `cuda : fix CUB argsort corruption caused by in-place keys`
  ([PR #28389](https://github.com/ggml-org/llama.cpp/pull/28389)): HIP
  argsort (MoE routing top-k) now sorts a copy of the keys. The HIP MoE runs
  (Nemotron 3 Nano Omni, Maple, Flash-Next, Ling 3.0) answer correctly.
- `ui: Fix mobile breakpoint + content overflow issues`
  ([PR #29108](https://github.com/ggml-org/llama.cpp/pull/29108)): web UI
  only; AutoTuner runs with `--no-webui` in the audits.
- Metal (`F16 input to the FWHT`, `MoE and SSM_CONV fusion`, `FA support
  checks`, `qwen4exp hc ops`), Hexagon (`I32 GET_ROWS`, `GEGLU_QUICK`,
  `TOP_K`, `ROLL`, `im2col`, HMX FA head_dim padding), OpenCL FA kernel and
  `test-llama-archs` dummy vocab: other backends / test tooling, no Windows
  Vulkan/HIP effect.

The standard live chain (command matrix, runtime + tool call, DFlash2
two-GPU draft, vision, vision + DFlash2 regression, Flash-Next lazy PLE,
Gemma 4 MTP, Maple, Nemotron 3 Nano Omni) was repeated on the b11063 trees on
both backends; results are in the validation document.

## Ling 3.0 (BailingMoE3): dedicated chat parser

Ling 3.0 Flash's Jinja template writes `<role>ASSISTANT</role><think>` into
the generation prompt, so the model never emits an opening `<think>` and a
tool call can arrive before any `</think>`. Up to b11062 the generated
autoparser ended reasoning only at the close tag, which classified such a
call entirely as `reasoning_content` (clients got `content=""`, no
`tool_calls`, agent loops died as reasoning-only turns). PR #28682 adds
`common/parsers/ling3.cpp`: `common_chat_try_specialized_template` picks it
whenever the template contains `<role>ASSISTANT</role>` and `<arg_key>`, and
it ends reasoning at `</think>` **or** at a `<tool_call>` start, like the
hand-written Qwen3-Coder / Kimi K3 parsers and the reference vLLM/SGLang
Ling3 parser.

What this means for AutoTuner:

- The parser is selected by `llama-server` from the template; the only
  requirement is `--jinja`, which `ling-3.yaml` has carried since v5.3.x
  together with `--reasoning-preserve` (the template supports
  `preserve_reasoning`). AutoTuner adds no `--reasoning-format` override, so
  nothing bypasses the new parser. The local
  `Ling-3.0-flash-AD-IQ3_M-00001-of-00002.gguf` (bailingmoe3, 42 blocks, 512
  experts, one NextN/MTP block, 131,072 context, 62.2 GB over two shards)
  carries both markers in `tokenizer.chat_template`.
- `min_llama_build` stays **10749** (the corrected no-scan SSM tensor
  contract): older builds load and run the model; only a tool call emitted
  inside the pre-opened think block is mis-classified there. The profile
  note and all nine language packs now say that b11063+ is recommended for
  tool use and why. AutoTuner does not gate on it because text chat,
  thinking-off tool calls with `tool_choice: required` (the runtime
  contract) and the MTP head work on every build since b10749.
- Live on both b11063 trees (`ling3_test.py`, results in the validation
  document): the server log shows `Using specialized template: Ling 3.0
  (Bailing V3)`; with thinking **on** and tools offered, the call to
  `report_cache` surfaces as `tool_calls` with the parsed arguments and the
  reasoning stays in `reasoning_content`; `tool_choice: required`,
  thinking-off text and a plain thinking answer behave as before.
- The local AD-IQ3_M quant is what llama.cpp's loader calls *trunk-only*:
  `nextn_predict_layers = 1` in the header, block 41 present, but no
  `blk.41.nextn.*` head tensors. `--spec-type draft-mtp` on such a file
  aborts llama-server (`GGML_ASSERT(layer.nextn.eh_proj)`,
  `bailingmoe3.cpp:422`). AutoTuner's folder scan reads every shard of a
  split GGUF before trusting the key, marks the scan "absent" and plans
  without speculation; only the first version of the audit script, which
  read shard 1 alone, hit the assert. No AutoTuner change was needed.

## Still fork-only on mainline b11063

- `ggml/include/ggml.h` still ends at `GGML_TYPE_COUNT = 43`. The Prism
  types PQ2_0/PTQ1_0 (142/143, Ternary-Bonsai 2) and the ROCmFPX range
  (100..111, Agnes-3.0-Flash MTP-ROCmFP4) are absent; neither
  `prism.hadamard` nor `ROCmFP` appears in the mainline sources.
  [ggml-org/llama.cpp#29058](https://github.com/ggml-org/llama.cpp/issues/29058)
  (Prism types) gained a layout write-up, a PrismML correction on the
  `LLAMA_FTYPE` values and a maintainer pointer to discussion #22019 but no
  implementation; [PR #24185](https://github.com/ggml-org/llama.cpp/pull/24185)
  (ROCmFP4 CPU quantization) is open and untouched since 2026-08-03. Both
  metadata gates (`_prism_hadamard_block_reason`, `_rocmfpx_block_reason`)
  therefore keep refusing mainline; `bonsai_negative.py` and
  `agnes_negative.py` re-ran the raw loads on the b11063 trees (validation
  document).
- PrismML published `prism-b10709-9a9394a` (2026-09-18, 22 commits after the
  pinned `prism-b10687-5d80cff`): DFly drafter support, the DSpark corrected
  draft runtime and release-workflow fixes, entirely in `common/speculative`,
  `src/llama-*` and `src/models/{dflash,dspark}.cpp`. No ggml or Vulkan
  change, so Vulkan still has no PQ2_0 kernels and the Bonsai 2 planner
  contract is unchanged; the recipes keep the b10687 pin.
- ROCmFPX fork issue [#26](https://github.com/ROCmFPX/ROCmFPX/issues/26)
  (RDNA4 MMQ fallback, filed 2026-09-19 with the local patch) has no reply;
  `patches/rocmfpx-rdna4-mmq-fallback.patch` stays in the recipes.
- Qwen3.5/3.8 vision + DFlash2: `vision_draft_regress.py` reproduced the
  HTTP 500 (`failed to process speculative batch`, position gap after the
  pinned M-RoPE image batch) on **b11063 HIP and Vulkan**; upstream tracks it
  as [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) (open,
  zero-fill patch crash-eliminating but without speculative benefit). The
  gate `QWEN35_VISION_DFLASH2_BROKEN_SINCE = 10896` stays and its message now
  says "verified through b11063".
- DeepSeek-V4.1-Flash: [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
  is still a draft (updated 2026-09-19), conversion only; the block message,
  `settings/deepseek-v4_1.yaml` and the nine language packs now name b11063.
