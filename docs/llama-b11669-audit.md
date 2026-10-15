# llama.cpp b11669 (v0.7.0) audit — AutoTuner v5.6.3

Checked 2026-10-15/16 against the locally built Windows **b11669** HIP and
Vulkan servers, commit `620d2225c` (`0.7.0-dev`), on Core Ultra 9 285K, AI
PRO R9700 32 GB + RX 9070 XT 16 GB. The
[b11538…b11669 comparison](https://github.com/ggml-org/llama.cpp/compare/b11538...b11669)
contains **131 commits**. b11669 is also the commit of the stable release
**v0.7.0** (v0.6.0 was b11429). See the
[binary/CLI manifest](llama-b11669-server-flags.json) and the
[validation record](v5.6.3-validation.md).

## CLI and planning contracts

Both backends expose **420 option names / 333 long options**: the b11538 set
plus `--spec-draft-n-ubatch N` ([#30459](https://github.com/ggml-org/llama.cpp/pull/30459),
b11629). (`--n-ubatch` in the manifest is only quoted in that option's help
text; it is not an option.) No option was removed. All profile Extra CLI
flags are still advertised and the 190-command profile/mode matrix parses on
both servers. Device order is unchanged. AutoTuner still owns context/offload
placement and sends `--fit off`.

- **`--spec-draft-n-ubatch`** sets the physical batch of the draft context
  (default 0 = follow `-ub`). AutoTuner knows it as a value flag (pruned with
  its value on older builds, de-duplicated, removed together with a dropped
  draft) and now sends `--spec-draft-n-ubatch 64` for external DFlash-family
  drafters (DFlash, DFlash2, LiLiCorr) and standalone MTP heads — measured
  below. Plain sibling drafters keep the target's `-ub` because they process
  the whole prompt themselves; EAGLE-3 and DSpark are unmeasured and
  unchanged.

- **`--cache-ram`:** decision models now default to no prompt cache unless
  `--cache-ram` is given ([#30498](https://github.com/ggml-org/llama.cpp/pull/30498),
  b11642). AutoTuner always sends `--cache-ram`, which counts as that opt-in,
  so it now applies the default itself: decision models get `--cache-ram 0`
  and no host-RAM reservation.
- **`--slot-save-path`** got the environment variable
  `LLAMA_ARG_SLOT_SAVE_PATH` ([#30272](https://github.com/ggml-org/llama.cpp/pull/30272));
  AutoTuner does not use slot files.

### `--spec-draft-n-ubatch` measured

AutoTuner's own plans on b11669, R9700 + RX 9070 XT, 32K context, mirrored
order (0, 64, 16, 16, 64, 0), a 7.6k-token prompt for prefill and a
deterministic 256-token answer for decode. GPU memory is the server's
dedicated usage from the Windows counters after loading.

| Plan | Draft `-ub` | Vulkan GPU MiB | Vulkan prefill / decode tok/s | HIP GPU MiB | HIP prefill / decode tok/s |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q8_0 + DFlash2 | target (1024) | 32,359 | 865 / 35.7 | 23,099 | 274 / 10.4 |
| | 64 | **31,070** (−1,289) | 890 / 35.6 | 23,150 | 267 / 10.5 |
| | 16 | 31,020 | 869 / 35.5 | 23,098 | 261 / 10.5 |
| Gemma 4 31B Q5_K_XL + MTP head | target | 24,349 | 262 / 39.0 | 24,824 | 736 / 37.8 |
| | 64 | **24,125** (−224) | 262 / 39.1 | **24,600** (−224) | 736 / 37.7 |
| | 16 | 24,114 | 262 / 39.1 | 24,589 | 737 / 37.5 |

Accepted drafts are identical in every row (DFlash2 178/533 on Vulkan,
180/514 on HIP; Gemma 140/175 and 137/190). The first Vulkan DFlash2 run
(21.5 tok/s decode) was a cold start and is left out; its mirrored repeat
is in the table. 16 saves almost nothing more than 64, so AutoTuner uses
64.

## Upstream changes that needed AutoTuner work

### Prism Bonsai 2 on mainline — and a gate that would have let the wrong files through

[#29600](https://github.com/ggml-org/llama.cpp/pull/29600) (b11556) adds the
`prism.hadamard.*` runtime to mainline, but only for the mainline tensor
types Q1_0/Q2_0 (41/42). PrismML's first GGUFs use the private types PQ2_0
(142) and PTQ1_0 (143), which mainline still rejects
(`GGML_TYPE_COUNT` is still 43; issue [#29058](https://github.com/ggml-org/llama.cpp/issues/29058)).

AutoTuner's gate looked only for the `prism.hadamard` loader string in the
selected runtime. b11669 contains it, so the local
`Ternary-Bonsai-2-27B-{PTQ1_0,PQ2_0}.gguf` would have been accepted and then
failed while loading (`tensor 'output.weight' has invalid ggml type 142.
should be in [0, 43)`, measured on b11669 HIP and Vulkan). The gate now also
requires the fork's own file-type name (`PTQ1_0 - 1.75 bpw` /
`PQ2_0 - 2.13 bpw`) for every private type a GGUF uses (from the tensor-type
scan, or from `general.file_type` 141/142/143 for a header-only read).
Checked against the real binaries: prism-b10743 accepts both files,
b11538 and b11669 refuse them with the new message.

Mainline conversions (PrismML now publishes `Q2_0` for Bonsai 2 2B/4B/9B and
`F16` for 27B, all with `prism.hadamard.*`) load on b11556+ and on the fork.
Before b11556 mainline loads such a file without the transform and decodes
garbage (PR #29600 measured perplexity 1,258,506 instead of 10.23); those
runtimes lack the loader string and stay refused. New profile `bonsai-2.yaml`
for 2B/4B/9B (they matched the first-generation Bonsai profile with 64K
context before); `bonsai-2-27b.yaml` keeps the fork pin for the 27B packings.

### Nemotron 3.5 Super VL and Nemotron 3.5 Bolt

[#30520](https://github.com/ggml-org/llama.cpp/pull/30520) (b11668) adds the
`nemotron_omni` vision projector, Nemotron 3.5 Super VL (120B-A12B,
`nemotron_h_moe`, separate MTP head) and Bolt (74B-A9B with per-layer expert
counts — older builds create the expert tensors with the wrong shape). Both
fell back to `_default.yaml` (8K context, no `--jinja`): the Super VL name
does not contain the Super profile's `super-120b-a12b`. New profiles
`nemotron-3_5-super-vl.yaml` (1M context, `--jinja --reasoning-preserve`,
MTP with 3 draft tokens as NVIDIA serves it, text 1.0/0.95) and
`nemotron-3_5-bolt.yaml` (256K, `--jinja`, 6 draft tokens for the LiLiCorr
drafter as in NVIDIA's llama.cpp recipe), both with `min_llama_build: 11668`.
Sampling follows the NVIDIA model cards.

### LiLiCorr drafters

[#30506](https://github.com/ggml-org/llama.cpp/pull/30506) (b11641) adds
LiLiCorr, a `dflash`-architecture drafter recognised by
`lilicorr.candidate_topk` and run as `draft-dflash`; b11668 fixes its
lattice walk. `lilicorr-<target>-….gguf` did not pair with its target (the
word was not a known draft marker). The scanner now treats `lilicorr` like
`dflash`/`dspark`, `check_draft_model_build` requires b11668+, and the draft
runs with `--spec-draft-p-min 0` like DFlash2 (the upstream default the PR
uses).

### MiniCPM-V 4.7

[#29416](https://github.com/ggml-org/llama.cpp/pull/29416) (b11551) adds the
`minicpmv4_7` projector for MiniCPM-V 4.7 1B / 35B-A3B (`qwen35` /
`qwen35moe` text). The files fell back to the Qwen3.5/3.6 profile. New
profile `minicpm-v-4_7.yaml` with OpenBMB's instruct sampling (0.7 / 0.95 /
top-k 100 / repetition 1.03).

### Projector build floors

A new table `_MMPROJ_PROJECTOR_MIN_BUILD` refuses an attached mmproj whose
`clip.projector_type` the selected build predates (`minicpmv4_7` b11551,
`nemotron_omni` b11668); the start would otherwise abort while reading the
projector. Text-only launches are not affected.

### vLLM decision models

`vllm-decision2` ([#30378](https://github.com/ggml-org/llama.cpp/pull/30378),
b11571, own architecture) and `vllm-d3`
([#30383](https://github.com/ggml-org/llama.cpp/pull/30383), b11572, `qwen35`;
multi-sequence fix [#30408](https://github.com/ggml-org/llama.cpp/pull/30408),
b11577) join `_DECISION_TYPE_MIN_BUILD`, the decision profile matches their
file names (`vllm-d3-nano-Q8_0.gguf` used the Qwen chat profile with 262K
context), and Decision 2 reads its prompt in one batch like Laya/Clef. The
vllm-sr `Decision-2.0-*` GGUFs on Hugging Face use a fork format (decision
type `decision2`, `decision2.*` tensors); mainline answers "unsupported
decision model type", so AutoTuner names the fork they need.

### Laguna XS 2.1

[#30331](https://github.com/ggml-org/llama.cpp/pull/30331) (b11598) reads
`laguna.rope.scaling.yarn_beta_fast`; Laguna XS 2.1 stores 64 (older builds
used 32). XS had no profile (`_default.yaml`); new `laguna-xs-2_1.yaml`
(256K, Poolside sampling 1.0 / top-k 20 / top-p 1.0, DFlash with 7 tokens,
b11598+). The local Laguna S 2.1 stores the defaults and is unaffected.

### Local models found on the wrong profile

A scan of all 82 local GGUFs (isolated metadata cache) found:
`Kolibri-1-Q4_K_M` (`kolibri1`, no mainline loader, issue #29922) on
`_default.yaml` — new `kolibri-1.yaml` refuses it on runtimes without the
architecture (the hob-llama.cpp patch has it); `OrcaSAQ-2-27B` (a Qwen3.8-27B
finetune) on the Qwen3.5/3.6 profile — now `qwen3_8.yaml`. Every decision,
vision and MTP model was on its correct profile.

## Other relevant upstream changes (no AutoTuner change needed)

- **MTP with images:** [#30257](https://github.com/ggml-org/llama.cpp/pull/30257)
  (b11548) lets MTP heads decode mtmd input; for Qwen3.8 Flash Next this
  caused an abort on every image request (issue #30307) until
  [#30397](https://github.com/ggml-org/llama.cpp/pull/30397) (b11588).
  AutoTuner keeps Flash Next `draft-mtp` off anyway: its separate head still
  aborts with `GGML_ASSERT(buffer)` on b11669 HIP and Vulkan.
- **Server slots / prompt cache:** empty slots are preferred and child tasks
  save what they overwrite ([#30567](https://github.com/ggml-org/llama.cpp/pull/30567)),
  only restorable states are cached ([#30498](https://github.com/ggml-org/llama.cpp/pull/30498)),
  the SWA reuse threshold is fixed ([#30580](https://github.com/ggml-org/llama.cpp/pull/30580)),
  draft state is saved in slot files ([#30574](https://github.com/ggml-org/llama.cpp/pull/30574))
  and plain KV caches honour `PARTIAL_ONLY`, so an MTP / `-md` draft no longer
  copies its whole KV into every checkpoint ([#30514](https://github.com/ggml-org/llama.cpp/pull/30514)).
- **RDNA4:** the HIP GDN chunked kernel ([#29353](https://github.com/ggml-org/llama.cpp/pull/29353))
  is enabled only for RDNA3.5; the 64-row MMQ tiles (#28195) are RDNA3.5 too;
  the AMD Vulkan strided-KV copy ([#27703](https://github.com/ggml-org/llama.cpp/pull/27703))
  explicitly excludes RDNA4. Vulkan BF16 activations in matmul (#29069) and
  the L2-friendly cm2 scheduling (#30215, NVIDIA) need nothing from AutoTuner.
- **CPU:** AVX2 CPUs now use F16C for FP32→FP16 (#30270, #30343) and Q2_0 has
  an AVX2 dot product (#30447) — relevant for CPU-offloaded layers only.
- **Mellum** reads MoE gating from the GGUF ([#30578](https://github.com/ggml-org/llama.cpp/pull/30578));
  defaults unchanged. **NFC normalisation** for Unigram tokenizers (#30390)
  is read from the GGUF.
- **Layer split without free-memory reports** ([#29677](https://github.com/ggml-org/llama.cpp/pull/29677))
  no longer aborts; AutoTuner always passes an explicit split.

## Build recipes

No CMake option was added, renamed or removed and no default changed; the
user-built b11669 `CMakeCache.txt` files equal b11538's on both backends.
Two recipe defects were found and fixed:

- **HIP builds had no HTTPS.** ROCm `clang` uses the GNU driver, so CMake's
  FindOpenSSL (`MSVC=FALSE`) never searched `C:\Program Files\OpenSSL-Win64`
  (`OPENSSL_INCLUDE_DIR-NOTFOUND` in the b11538 and b11669 HIP caches; the
  HIP `llama-server.exe` imports no `libssl`). The recipes now pass
  `OPENSSL_ROOT_DIR`, `OPENSSL_INCLUDE_DIR` and the `SSL_EAY`/`LIB_EAY`
  libraries that this FindOpenSSL branch reads. The HIP output check treats a
  tree without OpenSSL as incomplete when OpenSSL-Win64 is installed, so
  re-running the recipe on an existing b11669 HIP tree rebuilds it with
  HTTPS (checked against the local tree); without OpenSSL it only warns. Verified with an out-of-tree ROCm-clang build
  of `llama-server` from the b11669 source: OpenSSL 4.0.3 found, binary
  imports `libssl`/`libcrypto`. Only `-hf` downloads and https MCP servers were affected;
  AutoTuner itself starts local files.
- **The stable version check accepted a dev build.** `version:\s*0\.7\.0\b`
  also matched `0.7.0-dev`; it now requires the version to end at whitespace
  or `(`.

F16C and FMA are active on both backends (MSVC: `/arch:AVX2` defines
`__F16C__`/`__FMA__` through ggml; HIP clang: `-mf16c -mfma`). ROCm 7.2 is
still the newest installed HIP SDK and still needs the LLVM PR #201563
include-order workaround. The stable recipe resolves `latest` to v0.7.0 and
would build `0.7.0_{vulkan,hip}_llama.cpp` from the same commit as b11669
(only the version string differs).

## Throughput against b11538

`llama-bench` on the R9700 alone (Vulkan1 / ROCm0), `-ngl 99 -fa 1 -r 3`,
mirrored order, no foreign GPU load.

| R9700 | b11538 | b11669 |
|---|---:|---:|
| Vulkan Qwen3.6-35B-A3B UD-IQ3_XXS pp128 / pp512 / pp1024 | 1868 / 3359 / 3389 | 1906 / 3431 / 3424 (+2.0 / +2.2 / +1.0 %) |
| tg128 | 150.6 | 151.5 (+0.6 %) |
| Vulkan Qwen3.8-27B UD-Q4_K_XL pp128 / pp512 / pp1024 | 781 / 877 / 873 | 774 / 875 / 872 (−0.9 / −0.2 / −0.1 %) |
| tg128 | 31.87 | 31.93 (+0.2 %) |
| HIP Qwen3.6-35B-A3B UD-IQ3_XXS pp128 / pp512 / pp1024 | 1630 / 2779 / 2789 | 1656 / 2783 / 2772 (+1.6 / +0.1 / −0.6 %) |
| tg128 | 123.8 | 123.5 (−0.3 %) |
| HIP Qwen3.8-27B UD-Q4_K_XL pp128 / pp512 / pp1024 | 942 / 1279 / 1277 | 935 / 1276 / 1273 (−0.8 / −0.2 / −0.3 %) |
| tg128 | 30.34 | 30.30 (−0.1 %) |

Everything is within ±2 %; nothing in the range targets RDNA4 throughput.

**Open finding (not a b11669 regression):** split over both cards, the
recurrent hybrid Qwen3.8-27B Q8_0 decodes 4.4 tok/s on HIP but 19.7 tok/s on
Vulkan; on the R9700 alone both backends reach about 20 tok/s. b11538 is
identical. AutoTuner chooses that split when a DFlash2 draft is budgeted
(10.5 tok/s with the draft on HIP, 35.8 on Vulkan). Details and the
follow-up idea are in the [validation record](v5.6.3-validation.md).

## Older blockers rechecked

| Concern | Evidence on b11669 | Action |
|---|---|---|
| Qwen3.8 Flash Next `draft-mtp` | Separate head: `GGML_ASSERT(buffer)` on HIP and Vulkan | Gate kept, verified build b11669 |
| Xing 4.0 `xing4_0` | Both loaders reject the architecture | Block kept, wording names b11669 |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open/unmerged | Block kept, wording names b11669 |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Four stock loads fail (invalid ggml type 142/143) | Gate tightened (see above) |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | `GGML_TYPE_COUNT` still 43; four Agnes stock loads fail | Tensor-type gate kept |
| Image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | Real image requests pass on HIP and Vulkan (152/168 accepted drafts, as on b11538) | Gate stays bounded to b10896 ≤ build < b11319 |
| Ling-3.0-flash AD-IQ3_M on HIP | Five probes byte-identical to b11538 (including the known early end and the Chinese numeral); Vulkan clean | Not a regression |

Metal, SYCL, OpenCL, Hexagon, WebGPU, CUDA and RPC runtimes are not locally
validated by this Windows dual-AMD audit. Representative inference does not
qualify every model, checkpoint, quantization or context size.
