# llama.cpp b11538 audit — AutoTuner v5.6.2

Checked 2026-10-09 against the locally built Windows **b11538** HIP and Vulkan
servers, commit `79e2e74eb` (`0.6.0-dev`), on Core Ultra 9 285K, AI PRO R9700
32 GB + RX 9070 XT 16 GB. The
[b11371…b11538 comparison](https://github.com/ggml-org/llama.cpp/compare/b11371...b11538)
contains **167 commits / 664 changed files**. See the
[binary/CLI manifest](llama-b11538-server-flags.json) and the
[validation record](v5.6.2-validation.md).

## CLI and planning contracts

Both backends expose **418 option names / 331 long options**: the b11371 set
plus `--moe-cache-mib N`
([#29887](https://github.com/ggml-org/llama.cpp/pull/29887), b11480;
multi-GPU [#30112](https://github.com/ggml-org/llama.cpp/pull/30112), b11507).
No option was removed. All profile Extra CLI flags are still advertised and
the 190-command profile/mode matrix parses on both servers. Device order is
unchanged (HIP ROCm0 = R9700 / ROCm1 = RX 9070 XT; Vulkan0 = RX 9070 XT /
Vulkan1 = R9700 / Vulkan2 = Intel iGPU). AutoTuner still owns context/offload
placement and sends `--fit off`.

- **Default port 8080 → 9931**
  ([#30159](https://github.com/ggml-org/llama.cpp/pull/30159), b11521).
  AutoTuner always passes `--port` (default 1234), so nothing changes.
- **Log format:** [#29895](https://github.com/ggml-org/llama.cpp/pull/29895)
  only moves the colour reset before the newline and splits router child
  output; the `srv`/`slot` lines, timings and speculative statistics that
  AutoTuner reads are unchanged.
- **Release assets:** Windows asset names are unchanged; tags are now pushed
  before the release is published.

### `--moe-cache-mib`

A GPU cache for MoE experts that stay in host memory: `MUL_MAT_ID` on host
experts runs on the GPU from an LRU cache, only misses are uploaded, for
batches of at most 32 tokens. The budget is split over the GPUs that hold
host-expert layers (by `--tensor-split`), only layers whose experts are all
in host memory are cached, and pipeline parallelism is disabled.

AutoTuner knows the flag (value flag: de-duplicated, pruned with its value on
older builds, accepted as an Extra CLI flag). It does **not** set it. Measured
on the R9700 alone (the RX 9070 XT was busy with a game), llama-server
b11538, mirrored order, two runs per variant, 16k context, three 256-token
answers for decode and a 2.7k-token prompt for prefill:

| Qwen3.6-35B-A3B UD-Q8_K_XL (39 GB), partial hybrid | Vulkan decode / prefill | HIP decode / prefill |
|---|---:|---:|
| 13 expert layers in host RAM, no cache | 44.0 / 940 | 43.0 / 2416 |
| 13 layers + 1 GiB cache | 16.5 / 900 | 44.2 / 2096 |
| 15 layers + 2 GiB cache | 19.6 / 839 | — |
| 17 layers + 4 GiB cache | 27.5 / 814 | 49.9 / 1642 |
| 21 layers + 8 GiB cache | — | aborts (`ROCm error: unspecified launch failure`) |

| Qwen3.6-35B-A3B UD-IQ3_XXS, all 40 expert layers in host RAM | Vulkan decode / prefill | HIP decode / prefill |
|---|---:|---:|
| no cache | 30.2 / 580 | 26.3 / 1824 |
| 2 GiB cache | 15.7 / 518 | 40.0 / 1172 |
| 4 GiB cache | 28.1 / 547 | 43.3 / 1021 |
| 8 GiB cache | context creation fails (`failed to allocate the MoE cache buffers`) | 45.3 / 1216 |

On Vulkan (RDNA4) every cache size decodes slower, and 8 GiB cannot be
allocated. On HIP decoding gains +3…16 % in an AutoTuner-like partial plan
and up to +72 % when all experts are in host RAM, but prompt processing loses
13…44 %, and an 8 GiB cache in the partial plan aborted. That is a workload
trade-off rather than an improvement, so Auto keeps the cache off; AutoTuner
reserves no VRAM for a cache added by hand.

## Upstream changes that needed AutoTuner work

### K2 Horizon is mainline

[#29535](https://github.com/ggml-org/llama.cpp/pull/29535) (b11454) adds the
`k2-horizon` architecture (dense 1B–32B and MoVA-36B-A4B) with a dedicated
chat parser. The profile's capability check looks for the architecture string
in the runtime's llama library, so mainline b11454+ is now accepted without a
code change; the IFM fork keeps working. The notes no longer claim a fork is
required. Locally: K2-Horizon-1B Q8_0 (new download) and MoVA-36B-A4B Q8_0 —
thinking answer, tool calls with `auto` and `required` — pass on HIP and
Vulkan; MoVA-36B-A4B decodes 65 tok/s on Vulkan and 45 tok/s on HIP across
both cards (1B: 322 / 220 tok/s). `--n-cpu-moe` does not match MoVA's
`attn_v_exps`, which confirms the profile's whole-layer offload.

### Decision models on existing architectures

pplx-decider ([#30044](https://github.com/ggml-org/llama.cpp/pull/30044),
b11447, `qwen35`), LiquidAI d1-3B
([#30110](https://github.com/ggml-org/llama.cpp/pull/30110), b11483, `lfm2`)
and d1-omni-600M ([#30114](https://github.com/ggml-org/llama.cpp/pull/30114),
b11488, non-causal `lfm2`, no memory) reuse existing architectures. Their file
names matched the Qwen and LFM2 chat profiles. The decision profile now
matches them, and `check_model_build` refuses every known
`<arch>.decision.type` on a build older than its upstream PR (an older build
loads the backbone and then fails on the unknown type). d1-3B and
d1-omni-600M with their mmproj answer text and image questions through
`/v1/systemone` on both backends (red/green/blue test images: all correct).

### Clef image input

[#29969](https://github.com/ggml-org/llama.cpp/pull/29969) (b11418) adds image
input for Clef. The Clef-Flash GGUF downloaded for v5.6.1 answers every image
request on b11538 with HTTP 500 (`unexpected layout of the decision prompt`):
its `tokenizer.chat_template.systemone` has no image slot. ggml-org
re-converted the GGUFs on 2026-10-05; with that file red and blue test
images are decided correctly, and text questions (including a 2,854-token
state) behave the same with or without the projector. b11371 loads the
mmproj but ignores the image (a red image was answered "blue"). AutoTuner
now warns before launch when a Clef mmproj is attached to a GGUF without the
image slot, or to a build older than b11418.

### EmbeddingGemma 2

[#30054](https://github.com/ggml-org/llama.cpp/pull/30054) (b11452) adds the
non-causal `gemma-embedding2` architecture. It fell back to `_default.yaml`
(chat sampling, no `--embeddings`, 262,144 context from the GGUF). New profile
`embeddinggemma-2.yaml`: embedding endpoint, mean pooling from the GGUF, 8,192
tokens (model card) read in one physical batch. The Auto context of every
single-batch encoder is now capped at its profile limit, as for decision
models: a window larger than the physical batch cannot be used.
`/v1/embeddings` returns 768-dimensional vectors on HIP and Vulkan; a search
query scores 0.87 against the matching document and 0.62 against an unrelated
one, and a 6,828-token document is encoded in one request.

### DFlash drafts that ship embeddings but no head

[#30111](https://github.com/ggml-org/llama.cpp/pull/30111) (b11512) removes a
fallback from [#29226](https://github.com/ggml-org/llama.cpp/pull/29226)
(b11132): a DFlash/DSpark draft with its own `token_embd` and no
`output.weight` used its embeddings as LM head and accepted almost nothing
(upstream measured 0 % → 40.65 % for a Nemotron-3.5-Lightning DFlash draft).
AutoTuner now warns for such drafts on b11132–b11511. Gemma backbones tie the
head to the embeddings and are not affected; the local Qwen3.8 DFlash2 and
Muse Glimmer drafts carry no embeddings.

### PLaMo-3 tokenizer

[#30045](https://github.com/ggml-org/llama.cpp/pull/30045) (b11455) adds the
`plamo3` tokenizer model. GGUFs that declare it are refused on older builds.

## EschaLabs W2 GGUFs (AutoTuner issue #7)

[Issue #7](https://github.com/DaWasteh/Auto-Tuner/issues/7) asked to allow the
MTP head of `Escha-Qwen3.8-27B-W2` on the Escha fork
([Ajay9o9/llama.cpp-escha](https://github.com/Ajay9o9/llama.cpp-escha),
`escha-w2-dense` at `c22479837`, mainline base b10366). Findings with the
original Hugging Face files (SHA-256 identical to `aj9o9/Qwen3.8-27B-Escha-W2-GGUF`):

- The target keeps the native 2/3-bit Escha code (`escha_code` I16 tensors,
  `escha_lut`, `escha_dep_k2/k3`, `qwen35.escha.version = 1`). Stock b11538
  aborts on HIP and Vulkan with `check_tensor_dims: tensor
  'blk.0.ssm_out.weight' not found`, but AutoTuner did not refuse the model.
  It now does, unless the runtime's llama library contains the Escha loader
  (`escha_lut`) — the same marker check as Prism and ROCmFPX, not a folder
  name.
- The head (`blk.64` only, `nextn_predict_layers = 1`) carries `token_embd`
  and `output` but no `output_norm` and no codec tables. Mainline needs
  `output_norm`; the fork's qwen35 loader makes the trunk tensors optional for
  an MTP-only file and loads the codec tables only for a file with its own
  `escha.version`. On an Escha runtime exactly these are exempt; anything else
  a head lacks is still refused, and the fork's qwen35moe loader still needs
  `output_norm`.
- The head's original name has no `mtp-` prefix and is 2.9 GB, so the file
  name rules listed it as a separate model with embedded MTP. The scanner now
  records the lowest block index and treats any complete single-file GGUF
  whose blocks all lie in the NextN range as a draft head, whatever its name;
  it is paired with its target and launched with `-md` + `--spec-type
  draft-mtp`. The five local `mtp-*` heads keep their pairing.

The fork itself was not built here (its GPU kernels are CUDA); the issue
reporter ran it on an RTX 5060 Ti.

## Other relevant upstream changes (no AutoTuner change needed)

- **Vulkan RDNA4:** [#29936](https://github.com/ggml-org/llama.cpp/pull/29936)
  (b11423) reverts the `mul_mat_id` tile selection from #29182 that slowed
  Vulkan MoE prompt processing; [#29934](https://github.com/ggml-org/llama.cpp/pull/29934)
  (b11389) extends the mat-vec tuning to RDNA4. Stock b11538 now matches the
  opt-in RDNA4 build, so the [workaround](rdna4-moe-workaround.md) is no
  longer needed from b11423 on (table below). Upstream issue #29892 is still
  open.
- **MTP/NextN refactors** ([#30017](https://github.com/ggml-org/llama.cpp/pull/30017),
  [#30020](https://github.com/ggml-org/llama.cpp/pull/30020),
  [#30097](https://github.com/ggml-org/llama.cpp/pull/30097)): Qwen3.8 MTP,
  Gemma 4 assistant and image + DFlash2 (152/168 accepted drafts, as on
  b11371) pass unchanged. Qwen3.8 Flash Next `draft-mtp` still aborts with
  `GGML_ASSERT(buffer)` while the draft context is initialised (separate
  head, HIP and Vulkan); without speculation the model answers normally.
- **qwen4exp indexer** ([#29825](https://github.com/ggml-org/llama.cpp/pull/29825))
  halves the indexer score buffers; k-pool fixes
  ([#29958](https://github.com/ggml-org/llama.cpp/pull/29958),
  [#29994](https://github.com/ggml-org/llama.cpp/pull/29994)). AutoTuner's
  Flash Next estimate stays on the safe side.
- **Sampling/speculation:** n-gram drafts at temperature > 0
  ([#29924](https://github.com/ggml-org/llama.cpp/pull/29924)) and greedy
  temperature-zero chains ([#29797](https://github.com/ggml-org/llama.cpp/pull/29797)).
- **Chat refactor** ([#30210](https://github.com/ggml-org/llama.cpp/pull/30210)):
  reasoning, tool calls and `json_schema` pass for Ling 3.0, Muse Glimmer,
  FastContext, K2 Horizon and the other live models.
- **Ling-3.0-flash AD-IQ3_M on HIP:** sporadic deterministic glitches at
  temperature 0 on both b11371 and b11538 (early end token, a repetition loop,
  a Chinese numeral in a number list) at different prompts; Vulkan b11538 is
  clean on the same five probes. Not a b11538 regression (details in the
  validation record).
- **Session files** ([#28498](https://github.com/ggml-org/llama.cpp/pull/28498))
  move to a new version; AutoTuner keeps prompt caches in memory only.
- **CUDA/HIP fixes** ([#29941](https://github.com/ggml-org/llama.cpp/pull/29941),
  [#29953](https://github.com/ggml-org/llama.cpp/pull/29953)) and GDN/SSM
  speed-ups ([#30087](https://github.com/ggml-org/llama.cpp/pull/30087),
  [#29807](https://github.com/ggml-org/llama.cpp/pull/29807)) also compile into
  the HIP build.

## Throughput against b11371

The RX 9070 XT ran a game during these runs; the R9700 was idle apart from
the benchmark, and the mirrored order keeps background load symmetric.

| R9700, mirrored order, `-r 3` | b11371 | rdna4moe_b11371 | b11538 |
|---|---:|---:|---:|
| Vulkan Qwen3.6-35B-A3B UD-IQ3_XXS pp128 | 1246.92 | 1782.51 (+43.0%) | 1779.14 (**+42.7%**) |
| pp512 | 2763.41 | 3218.68 (+16.5%) | 3209.70 (**+16.1%**) |
| pp1024 | 2817.51 | 3271.94 (+16.1%) | 3244.45 (**+15.2%**) |
| tg128 | 143.50 | 143.18 (−0.2%) | 144.45 (+0.7%) |
| Vulkan Qwen3.8-27B UD-Q4_K_XL pp512 | 836.83 | 838.54 (+0.2%) | 848.96 (+1.5%) |
| tg128 | 31.79 | 31.87 (+0.3%) | 31.47 (−1.0%) |
| HIP Qwen3.6-35B-A3B UD-IQ3_XXS pp512 | 2610.29 | — | 2744.04 (+5.1%) |
| tg128 | 121.32 | — | 117.65 (−3.0%) |
| HIP Qwen3.8-27B UD-Q4_K_XL pp512 | 1184.26 | — | 1268.00 (+7.1%) |
| tg128 | 30.51 | — | 30.16 (−1.1%) |

On HIP prompt processing gains 3.6…7.2 % (Qwen3.8 dense +7.1 %), generation
stays within −1…−3 %.

## Older blockers rechecked

| Concern | Evidence on b11538 | Action |
|---|---|---|
| Qwen3.8 Flash Next `draft-mtp` | Separate head: `GGML_ASSERT(buffer)` on HIP and Vulkan, as on b11371 | Gate kept, verified build b11538 |
| Xing 4.0 `xing4_0` | Both loaders reject the architecture | Block kept, wording names b11538 |
| DeepSeek V4.1, [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) | Open/unmerged | Block kept, wording names b11538 |
| Prism PQ2_0/PTQ1_0, [#29058](https://github.com/ggml-org/llama.cpp/issues/29058) | Open; four stock loads fail | Marker gate and `prism-b10743` pin kept |
| ROCmFPX, [#24185](https://github.com/ggml-org/llama.cpp/pull/24185) | Open; `GGML_TYPE_COUNT` still 43; four Agnes stock loads fail | Tensor-type gate kept |
| ROCmFPX Windows build | Fork main moved to `e289c1092`; Windows fix ([fork #31](https://github.com/ROCmFPX/ROCmFPX/pull/31)) and RDNA4 MMQ issue [#26](https://github.com/ROCmFPX/ROCmFPX/issues/26) still open | `aed0d5fd9` pin and patch kept |
| Image + DFlash2, [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) | Issue open; real image requests pass on HIP and Vulkan | Gate stays bounded to b10896 ≤ build < b11319 |

Metal, SYCL, OpenCL, Hexagon, WebGPU, CUDA and RPC runtimes are not locally
validated by this Windows dual-AMD audit. Representative inference does not
qualify every model, checkpoint, quantization or context size.
