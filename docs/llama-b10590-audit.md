# llama.cpp b10590 integration audit

AutoTuner v5.2.6 was audited against the exact upstream range below.

| Tag | Commit | Date |
|---|---|---|
| `b10549` | `b2e5e9b28b2484fbf94b543432ece638996a8b97` | 2026-08-21 |
| `b10590` | `6657ded4faa3b8450221119fc6b4d002e35104a2` | 2026-08-23 |

The range contains **41 upstream commits**. Evidence came from a full-history
upstream checkout, the exact source diff, generated `llama-server --help`,
profile/command matrices, and bounded real-model starts on Windows/Vulkan.

## Command-line compatibility

No `llama-server` option emitted by AutoTuner was removed or renamed between
b10549 and b10590. The only removed option in `common/arg.cpp` is `-no-cnv`
(`--conversation` off) from the interactive **CLI** example; AutoTuner never
emits it and launches `llama-server`, not that CLI path.

The locally compiled checkout was one UI-only commit after the exact tag and
reported b10591. `git diff b10590..HEAD` contains only Web UI Svelte/TypeScript
files, so its native argument parser and model/runtime code are byte-identical
to b10590 for this audit.

Validation against that binary covered:

- all **60 normal-server YAML profiles**, including every profile `extra_args`
  sequence;
- a Vision + external DFlash + ngram + tools/reasoning command;
- integrated MTP + `ngram-map-k4v`;
- Unlimited-OCR + mmproj, DRY, template, F16-KV, and FA-off inputs;
- the mainline diffusion CLI command.

`prepare_command_for_binary()` retained every argument in all of those
commands: **zero unsupported options and zero orphaned values**. The complete
current surface remains advertised, including `--fit off`, explicit
`--perf`, metrics/slots, load modes, prompt-cache RAM, KV types, GPU placement,
MTMD/mmproj device selection, reasoning, tools, and every supported speculative
parameter.

## Upstream changes that affect AutoTuner behavior

- **MTP + embeddings fix (PR #27400):** the speculative draft context now
  clears embedding/pooling mode and inherits the target context size. This is a
  transparent runtime correction; no AutoTuner flag changed.
- **BailingMoE3 DSpark rollback (PR #27508):** Ling 3.0 can now use DSpark
  rollback state. AutoTuner already recognizes BailingMoE3 and DSpark sidecar
  tensors and emits `draft-dspark` on supported builds.
- **Dots3-Note (PRs #27060 and #27524):** b10590 adds its DSA/SWA language
  architecture plus MTMD vision/audio. AutoTuner's metadata-first generic
  loader, mmproj pairing, explicit `--mmproj-device`, and help-pruned server
  command remain compatible; no new server argument is required. Audio/image
  interaction is provided by llama-server's bundled Web UI/API.
- **Fit accounting (PR #27496):** upstream auto-fit now counts streams. Normal
  AutoTuner launches still emit `--fit off`, because the displayed/planned
  context and placement must remain authoritative.
- **WebP MTMD and backend work:** WebP-through-ffmpeg, Vulkan
  `PAD_REFLECT_1D`, CUDA `POOL_1D`, OpenCL/SYCL/ROCm fixes, JSON abstraction,
  and the 0.2.0 version bump are rebuild benefits and need no new launch flag.

## Qwen3.8 DFlash2: why stock b10590 fails

The Qwen3.8 sidecars are **DFlash2**, not first-generation DFlash. They retain
`general.architecture=dflash` and the existing `draft-dflash` CLI token, but
add grouped local convolution and candidate-selector tensors. Their metadata
includes:

- `dflash.conv_kernel_size=2`
- `dflash.conv_group_size=16`
- `dflash.selector_rank=256`
- `dflash.selector_top_k=16`
- `dflash.block_size=8`

Each tested BF16/Q8/Q4 sidecar contains **81 tensors**. Stock b10590's DFlash
loader still instantiates the older **58-tensor** graph because DFlash2 support
is in the still-open upstream **PR #27342**, not the b10590 tag. A direct
stock-b10590 launch reproduced the reported failure exactly:

```text
done_getting_tensors: wrong number of tensors; expected 81, got 58
```

This is not an AutoTuner argv error: `-md`, `--spec-type draft-dflash`, and the
other speculative flags are accepted before the draft GGUF reaches the missing
loader implementation.

## v5.2.6 DFlash2 handling

AutoTuner now:

1. detects DFlash2 from its GGUF convolution + selector metadata;
2. detects a compatible sibling PR build and routes only that DFlash2 launch
   through it, leaving stock b10590 selected for normal models;
3. rejects known stock b10590/b10591 binaries **before** loading the 27B target
   when no compatible sibling exists, with an actionable PR #27342 message;
4. accepts the reviewed PR commits (`5ecbe1ac`, `1deefcca`), while preserving
   warning-only behavior for unprobeable wrappers;
5. derives `--spec-draft-n-max 7` from `block_size=8` (one anchor position),
   rather than reusing Qwen3.8 MTP's unrelated profile default of 2;
6. uses DFlash2's intended explicit p-min `0.0`; the Qwen MTP profile's `0.75`
   confidence threshold remains on the embedded-MTP path;
7. originally shipped a reproducible Windows/Vulkan PR recipe; v5.2.9 removed
   it after the implementation merged into mainline b10658.

The pinned PR head is commit `1deefcca395743049c3820ab8f9b15043f3e9446`
and reports b10499 because the PR is based on an older mainline revision. A
bounded validation with the real Qwen3.8-27B target + Q4_K_M DFlash2 sidecar
reached `/health`, then completed an eight-token request. llama.cpp reported
10 drafted tokens and 5 accepted tokens. Stock b10590 remains the recommended
build for every normal mainline feature; the separate PR build is required only
while using DFlash2.

## Primary upstream references

- Release: <https://github.com/ggml-org/llama.cpp/releases/tag/b10590>
- Exact comparison: <https://github.com/ggml-org/llama.cpp/compare/b10549...b10590>
- Server options: <https://github.com/ggml-org/llama.cpp/blob/b10590/tools/server/README.md>
- DFlash2 PR: <https://github.com/ggml-org/llama.cpp/pull/27342>
- Qwen3.8 DFlash2 GGUF instructions: <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF>
- MTP embeddings fix: <https://github.com/ggml-org/llama.cpp/pull/27400>
- BailingMoE3 DSpark: <https://github.com/ggml-org/llama.cpp/pull/27508>
- Dots3-Note model: <https://github.com/ggml-org/llama.cpp/pull/27060>
- Dots3-Note MTMD: <https://github.com/ggml-org/llama.cpp/pull/27524>
