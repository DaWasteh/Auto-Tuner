# llama.cpp b11042 / PrismML prism-b10687 audit — AutoTuner v5.5.2

Checked 2026-09-18. Mainline runtime: **b11042**, commit `ec9281505`
(`0.4.1-dev`); the newest stable release is still **v0.4.1** (b10964).
[Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/b11030...b11042):
12 commits. Both Windows pre-release trees (`b11042_{vulkan,hip}_llama.cpp`)
were built by the user with the unchanged repository recipes before this
audit. The second runtime under audit is the **PrismML fork** of llama.cpp
(branch `prism`) at its release tag
[`prism-b10687-5d80cff`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10687-5d80cff)
(2026-09-17, mainline base b10615 + fork commits; `llama-server --version`
prints `0.2.0-dev (build 10687, commit 5d80cff0b)`), rebuilt for Vulkan and
HIP with the re-pinned `ternary_bonsai_{vulkan,hip}_llama_build.ps1` recipes
(`2b_b10687_{vulkan,hip}_llama.cpp`). See [validation](v5.5.2-validation.md)
for the test boundaries and [binary/help hashes](llama-b11042-server-flags.json).

## Mainline b11030 → b11042: byte-identical `--help`

None of the twelve commits touches `common/arg.cpp`, `tools/server/`, the
speculative helpers or the model loader. The local Vulkan and HIP
`llama-server --help` outputs have the **same SHA-256 as b10948, b10977 and
b11030**, the parsed option set is **415 names / 328 long options**, nothing
was added or removed, and `--rpc` is still absent only because the recipes
keep `GGML_RPC=OFF`. Every bundled profile's `extra_args` token exists in the
manifest and all **168 standard profile × chat/coding commands per backend**
(one more profile than b11030: `bonsai-2-27b.yaml`) parse on the real server.

Commits in the range, checked against AutoTuner:

- `ggml-cpu: add F16 input to the FWHT` ([PR #27779](https://github.com/ggml-org/llama.cpp/pull/27779)):
  mainline's fast Walsh–Hadamard op grows an F16 input path. This is the
  generic `GGML_OP_FWHT`; mainline still has **no** `PQ2_0`/`PTQ1_0` types
  (ids 142/143) and no `prism.hadamard.*` loader, see the Bonsai section.
- `ggml: check for allocation failures to prevent crashes` ([PR #28149](https://github.com/ggml-org/llama.cpp/pull/28149))
  and `ggml: handle graph buffer reservation failure` ([PR #26070](https://github.com/ggml-org/llama.cpp/pull/26070)):
  failed allocations now return errors instead of aborting. AutoTuner's
  plan never relies on a reservation failing; the `sched_reserve` retry
  message ("compute buffer allocation failed, retrying without pipeline
  parallelism") is what the old fork pin printed for PQ2_0 on Vulkan and is
  informational.
- `vulkan: add IQ3_S MMQ matmul kernels` ([PR #28822](https://github.com/ggml-org/llama.cpp/pull/28822)):
  no local IQ3_S model; no planner impact.
- `Model-Saver: write the SWA pattern, 15 more architectures roundtrip`
  ([PR #29042](https://github.com/ggml-org/llama.cpp/pull/29042)),
  `vocab: add ufakzeka pre-tokenizer` ([PR #29033](https://github.com/ggml-org/llama.cpp/pull/29033)),
  `gguf-py: fix Q8_1 block size` ([PR #29036](https://github.com/ggml-org/llama.cpp/pull/29036)),
  `opencl` kernel, `cmake: fix build when GGML_CPU=OFF and GGML_CUDA=ON`,
  CI runner changes: conversion/tooling only, no runtime behaviour that
  AutoTuner plans against.

The standard live chain (runtime, tool call, DFlash2 two-GPU draft, vision,
vision + DFlash2 regression, Flash-Next lazy PLE, Gemma 4 MTP, Maple,
Nemotron 3 Nano Omni) was repeated on the b11042 trees on both backends;
results are in the validation document. The vision + DFlash2 gate and the
DeepSeek-V4.1 block keep their b11030 wording: nothing in this range touches
those paths and neither was re-decided here.

## PrismML Ternary-Bonsai 2 27B on the prism fork

### The files

`I:\models\prism-ml\` holds the three published files: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
(5.95 GB), `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.21 GB) and the reference
`Ternary-Bonsai-2-27B-mmproj-BF16.gguf` (0.93 GB; a Q8_0 projector of 0.63 GB
is also published). Header facts (read with the fork's `gguf-py`, mainline's
reader rejects type 143):

| | PTQ1_0 | PQ2_0 |
|---|---|---|
| `general.architecture` | `qwen35` (Qwen3.8-27B hybrid-attention base, 64 blocks, 262,144 context, `ssm.group_count` 16) | same |
| `general.file_type` | 143 | 141 |
| Ternary tensors | 402 × `PTQ1_0` (ggml type 143, "1.75 bpw ternary, group 128"), 5.47 GiB | 402 × `PQ2_0` (type 142, "2.13 bpw, group 128"), 6.65 GiB |
| Other tensors | 353 × F32, 96 × BF16 (norms, SSM scalars) | same |
| `prism.hadamard.*` | version 1, block size 1024, `normalized-sylvester-walsh-hadamard`, `sign_mode explicit`, 401 folded weight names + `token_embd.weight` inverse, sign widths 5120/6144/17408 | same |

The mmproj is a plain `clip` / `qwen3vl_merger` projector (27 blocks, BF16)
without Prism-private types.

### Which runtime loads what (live, 2026-09-18)

| Runtime | PTQ1_0 | PQ2_0 |
|---|---|---|
| Mainline **b11042** Vulkan and HIP | rejected at `gguf_init_from_reader`: `tensor 'output.weight' has invalid ggml type 143. should be in [0, 43)` (exit 1 after 2.5 s) | same, type 142 |
| Previous fork pin **prism-b10660** (`e311ed3`, the old `2b_b10616` tree) | rejected: `invalid ggml type 143. should be in [0, 143)` (PTQ1_0 was added after that pin) | loads (the pin already had PQ2_0 and the Hadamard fold) |
| Fork **prism-b10687** HIP (`2b_b10687_hip`, ROCm0 = R9700) | **runs on the GPU**: 65/65 layers, 5,395 MiB on ROCm0, decode **34.0 t/s** (310 tokens), prompt 78 t/s, image prompt 266 t/s | **runs on the GPU**: 6,540 MiB on ROCm0, decode **51.3 t/s**, prompt 216 t/s, image prompt 263 t/s |
| Fork **prism-b10687** Vulkan (`2b_b10687_vulkan`, Vulkan0 = RX 9070 XT) | **runs on the GPU**: 5,395 MiB on Vulkan0, decode **10.6 t/s**, prompt 41 t/s, image prompt 131 t/s | loads and answers correctly, but the ternary weights stay **CPU-mapped** (6,862 MiB host, 55 MiB on Vulkan0): decode 3.0 t/s, prompt 5.4 t/s |

Why: the fork's Vulkan backend ships `PTQ1_0` shaders (`dequant_ptq1_0`,
`mul_mat_vec_ptq1_0_*`, MMQ matmul and `mul_mat_id` variants) but no `PQ2_0`
path at all (`ggml-vulkan.cpp` at the pin has zero `PQ2_0` references), so
`PQ2_0` matmuls are not supported on Vulkan and the scheduler keeps those
weights on the CPU. The CUDA/HIP backend has native `PTQ1_0` and `PQ2_0` MMQ
loaders, so HIP runs both packings on the device; PQ2_0 is the faster decode
there, as PrismML's own throughput table says for datacenter cards. Upstream
has an open request to add the two types
([ggml-org/llama.cpp#29058](https://github.com/ggml-org/llama.cpp/issues/29058),
no maintainer reply on 2026-09-18), so both files remain fork-only. The old
`Q1_0` (41) and `Q2_0` (42) types that mainline knows are not used by these
files; the profile for the first-generation Ternary-Bonsai 27B (`Q2_0-g128`)
is unchanged.

All four fork runs went through AutoTuner's real planner (`bonsai_test.py`):
`match_profile` selects `bonsai-2-27b.yaml`, `check_profile_build` reads the
fork build 10687, `check_model_build` passes, the plan is 32k context, Q8/Q8
K/V (1,088 MiB), single GPU, `--mmproj` + `--mmproj-device`, `--jinja`,
`--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0`. Each run answered
`17 * 23 = 391` with thinking off, named the colour of a red test image, and
listed 1…80 for the decode measurement. The loader reports
`loaded 402 Hadamard-folded weight(s) (1 inverse-lookup) using 1 rotation(s)
and 3 sign vector(s)` on every backend.

### What changed in AutoTuner

- **Recipes** `ternary_bonsai_{vulkan,hip}_llama_build.ps1`: `ExpectedCommit`
  moved from `e311ed38…` (prism-b10660, 2026-08-27) to
  `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6` (prism-b10687, 2026-09-17), and
  `-FixedIdentity "b10687"` names the folder after the fork's own build
  number. Without it `Invoke-LlamaPinnedForkBuild` derives the identity from
  the mainline merge-base (`git rev-list --count` of b10615 = 10616), which is
  unchanged since the old pin and would have made the recipe verify the old
  tree and stop with "Source commit mismatch". The `-DLLAMA_OPENSSL=OFF
  -DLLAMA_CURL=OFF` extras still match the fork's CMake options
  (`LLAMA_CURL` is only a deprecation shim there). Both builds took about
  four minutes each (Vulkan 20:50–20:54, HIP 20:54–20:59) and passed
  `Test-LlamaBuildOutput` including the HIP two-GPU semantic check.
- **Profile `bonsai-2-27b.yaml`** (new): patterns `ternary-bonsai-2-27b` and
  `bonsai-2-27b` (both longer than the 8B family's `ternary-bonsai`, so the
  longest-match rule picks it; `Ternary-Bonsai-27B`, `Bonsai-27B` and Qwen3.6
  names keep their profiles), 262,144 context, Q8 K/V first, `server_binary:
  2b_llama/llama-server`, `min_llama_build: 10687`, `--jinja`, official
  thinking-mode sampling from the model card (temperature 1.0, top-p 0.95,
  top-k 20, min-p 0, no presence/repeat penalty; the non-thinking values 0.7 /
  0.80 / presence 1.5 are quoted in the notes). The profile deliberately has
  no `required_runtime_markers`: `match_profile` skips a marker profile when
  the GGUF architecture is claimed by another profile (`qwen35` belongs to
  `qwen3_5-3_6.yaml`), which would have routed the file to the Qwen profile.
- **Loader gate** `_prism_hadamard_block_reason` in `check_model_build`: a
  GGUF with any `prism.hadamard.*` key is refused before launch unless the
  selected `llama-server` or its `llama*.dll` / `libllama` sibling contains
  the `prism.hadamard` loader string. The message names the packing (from
  `general.file_type`), the recipes, the pin and the mainline behaviour
  ("unknown ggml types 142/143, issue #29058"). GGUFs without those keys never
  trigger the scan. Verified against the four local runtimes: both fork trees
  pass, both b11042 trees are refused; the old fork tree passes this gate but
  is refused by `min_llama_build` (reports build 10660).
- Nine language packs carry the new profile note; the README's fork tree,
  binary-selection and recipe sections name the prism fork instead of the
  historical `1b_llama.cpp`.

### Not changed, noted

- On this workstation Vulkan runs PTQ1_0 about three times slower than HIP
  (10.6 vs 34.0 t/s; different cards: Vulkan0 is the RX 9070 XT, which also
  drives the desktop and later that evening a game, ROCm0 the idle R9700,
  both Navi 48 — treat the Vulkan figure as a lower bound). The profile note recommends HIP for Bonsai 2 here;
  AutoTuner does not switch backends on its own.
- PrismML's fork issue #180 (CPU-repack segfault on `output.weight` for
  PQ2_0 with the Linux CPU/Vulkan binaries of prism-b10683/b10685) did not
  reproduce on the Windows Vulkan build at prism-b10687 with `-ngl 999`: the
  PQ2_0 file loads, stays CPU-mapped and answers; `--no-repack` was not
  needed. Not added to the profile.
- The Bonsai-demo repository still describes Vulkan/HIP as needing the
  `Q2_0_g64` file; the HIP result above (both Prism packings on the GPU)
  and the Vulkan PTQ1_0 result are the local evidence that this note is out
  of date for HIP and half right for Vulkan.
