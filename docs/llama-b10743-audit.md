# llama.cpp b10743 compatibility audit

Date: 2026-09-01

AutoTuner release: v5.3.6

Audited upstream range: b10717 (`a32af33de`) through b10743 (`8887a48f0`)

## Authority and scope

The audited source is the exact local upstream checkout and the exact Vulkan
and HIP binaries built from it:

```text
L:\LAB\ai-local\b10743_vulkan_llama.cpp
L:\LAB\ai-local\b10743_hip_llama.cpp
version: 0.3.0-dev (build 10743, commit 8887a48f0)
```

Runtime/source behavior is authoritative. Comments, old release notes, and
profile prose were changed when they disagreed with the executable.

The audit covered:

- the full `b10717..b10743` upstream commit and source diff;
- current `llama-server --help` from both local backends;
- every flag emitted by the 67 shipped YAML profiles;
- command assembly, duplicate handling, and binary-aware pruning;
- NextN/MTP model and standalone-draft loading;
- qwen4exp QSA correctness and lazy PLE residency assumptions;
- real GGUF metadata and bounded real-server launches.

## Findings

### 1. No server option was added or removed in b10717..b10743

Extracting option literals from `common/arg.cpp` at both endpoints produced an
empty added/removed set. All 14 option spellings emitted directly by shipped
profile `extra_args` are advertised by exact b10743 Vulkan and HIP help.

The audit nevertheless found an existing mismatch: llama.cpp b10700 replaced
`--tensor-read-lazy` with `--lazy-mode` / `-lzm`. AutoTuner still generated the
old spelling, so its generic unsupported-option filter removed lazy loading on
current builds. This is a rename, not feature removal.

Resolution:

- generated commands use `--lazy-mode auto`;
- `prepare_command_for_binary` translates in either direction according to the
  selected binary's advertised option;
- a runtime with neither spelling has the complete option/value pair pruned;
- lazy tensor accounting is unchanged because upstream residency behavior did
  not change.

### 2. Value-taking options could leave orphan argv tokens

`--samplers` (Gemma 4) and `--pooling` (Granite Embedding R2) take a following
value but were absent from `_ARG_FLAGS_WITH_VALUES`. If a user repeated either
profile-owned flag, duplicate suppression could reject the flag and retain its
value as a stray positional token.

Resolution: classify both options and the current reasoning/lazy aliases as
value-bearing. Focused tests cover profile value precedence and prove that the
rejected override value is consumed too.

### 3. qwen4exp needs the b10737 correctness fixes

The qwen4exp loader first landed in b10660 and AutoTuner's context-by-ubatch
memory coefficients were measured on b10666. The audited range contains two
material QSA follow-ups:

- b10731 `0eadefebd`: recurrent-state rollback (PR #28123);
- b10737 `36b101543`: sequence copies, block-position keying, multimodal QSA
  input, and a CUDA abort fix (PR #27941).

These are serving-correctness fixes rather than optional tuning. The
`qwen3_8_flash_next.yaml` minimum is therefore b10737. The measured
context-by-ubatch buffers and 5% active-row lazy residency remain unchanged:
there is no upstream evidence that their allocation contract changed.

### 4. b10741-b10748 contain a NextN regression

b10741 `9d817213a` (PR #28159) moved
`nextn_predict_layers` loading ahead of calls to `n_layer()`. Two paired cases
were incomplete:

1. generic per-layer FF/head arrays were still read with the reduced main-layer
   count, so a standard `block_count`-length array can fail during hparam load;
2. an all-NextN standalone assistant has `nextn_predict_layers == block_count`,
   contradicting the new global `< block_count` assertion.

The first tag containing both upstream repairs is b10749:

- `d11b3cc7e`, PR #28173: load relevant arrays at `n_layer_all`;
- `73159c303`, PR #28183: repair Gemma 4 assistant handling.

Exact b10743 evidence:

- the real `mtp-gemma-4-12b-it-v2.gguf` aborts in `llama-model.cpp` with
  `GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all) failed`;
- the real Qwen3.6-27B integrated-MTP target loads as a base model, but an
  unadapted `draft-mtp` launch fails while creating the MTP context;
- the same Qwen target loads and decodes after only MTP draft arguments are
  removed.

Resolution is deliberately narrow:

- b10741-b10748 command preparation removes `draft-mtp`, `-md` when it is the
  MTP head, and `--spec-draft-*` values;
- a compatible ngram method and all target-model/core launch arguments remain;
- standalone MTP heads are rejected before launch;
- embedded-NextN targets whose generic metadata arrays do not match the broken
  runtime's reduced count are rejected before launch;
- GUI, terminal launch, and performance benchmark paths share the preflight;
- a benchmark variant whose requested MTP is disabled fails explicitly instead
  of recording base/ngram throughput under an MTP label;
- b10740 and earlier and b10749+ commands are unchanged.

No core model-loading flag is guessed away.

### 5. `--perf` is still required explicitly

Exact b10743 help reports performance timings disabled by default. AutoTuner
already emitted `--perf`, so product behavior was correct; only the stale
comment claiming a true default was fixed.

## Profile and metadata implications

Real Qwen3.6/Qwen3.8 embedded-MTP metadata reports a `block_count` that includes
the appended NextN block (65/41 with `nextn_predict_layers=1`). AutoTuner must
not subtract that layer from file-size, tensor, or lazy-weight accounting. The
existing scanner/tuner calculations already use the on-disk block count and do
not need modification.

The qwen4exp Flash-Next GGUF remains a different case: its 48 blocks are the
actual hybrid target stack and its 26.82 GiB PLE table is architecture-marked
for lazy reads. The b10743 launch retained that table as a host mapping rather
than treating it as splittable GPU layer weight.

## Validation summary

- Exact b10743 Vulkan/HIP version and help probes: pass.
- All profile-emitted flags present in b10743 help: pass.
- Real qwen4exp split GGUF, Vulkan, `--lazy-mode auto`: health ready in 86.54 s;
  57-token prompt plus 8-token decode completed at 8.81 decode tokens/s.
- Real Qwen3.6-27B MTP GGUF, guarded b10743 base fallback: health ready in
  13.74 s; 15-token prompt plus 8-token decode completed at 10.63 tokens/s.
- Real Gemma 4 standalone MTP head: direct b10743 assertion reproduced; new
  Vulkan and HIP preflights reject it and require b10749+ or b10740 and earlier.
- Source test suite: 451 passed, 7 skipped.
- Ruff, Python compilation, and diff checks: pass.

Release-level platform and artifact evidence is recorded in
[`v5.3.6-validation.md`](v5.3.6-validation.md).
