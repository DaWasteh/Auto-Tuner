# llama.cpp b10878 audit — AutoTuner v5.4.5

Date: 2026-09-09. Exact target: **b10878**, commit
`4850c7727fa73bbe3098e10ee369fbc3467c445f`.
[Comparison b10863…b10878](https://github.com/ggml-org/llama.cpp/compare/b10863...b10878):
15 commits, 31 changed files. The official Windows x64 Vulkan release was
SHA-256-verified against GitHub's asset digest and unpacked into an ignored,
isolated audit directory. The user's installed llama.cpp builds were not changed.

## Removed mmap/mlock/Direct-I/O flags

[PR #28334](https://github.com/ggml-org/llama.cpp/pull/28334), **b10875**, is
called “officially deprecate” but actually **removes the parser registrations**:

| Removed spelling | Current spelling |
|---|---|
| `--mmap` | `--load-mode mmap` |
| `--no-mmap` | `--load-mode none` |
| `--mlock` | `--load-mode mlock` |
| `--direct-io`, `-dio` | `--load-mode dio` |
| `--no-direct-io`, `-ndio` | `--load-mode none` |

The corresponding old `LLAMA_ARG_MMAP`, `LLAMA_ARG_MLOCK`, `LLAMA_ARG_DIO`
environment registrations are gone too; use `LLAMA_ARG_LOAD_MODE` if managing
llama.cpp outside AutoTuner. mmap itself **has not been removed**.

AutoTuner's generated argv already used `--load-mode`. v5.4.5 additionally:

- Migrates legacy Extra switches before deduplication/capability pruning.
  The last load switch within one Extra source wins; an explicit modeled
  dropdown choice remains authoritative over Extras. Values resembling option
  names are not interpreted as flags. Invalid legacy `=false` forms fail clearly.
- Preserves supported legacy switches on pre-load-mode runtimes and pre-b10151
  mapped-lock semantics. Existing adaptation of explicitly requested modern
  `mlock` versus `mmap+mlock` stays intact.
- Rejects **unmodeled locking through Extras**. Such flags previously bypassed
  the RAM/OS/old-GPU veto because that veto ran before Extra merging. Select the
  Expert load-mode dropdown or CLI `--force-mlock` instead; neither method
  promises that an unsafe or unauthorized lock will be accepted.
- Rejects nonzero/partial `--help` as capability evidence rather than silently
  pruning valid arguments. Unknown wrappers keep their command unchanged,
  except for the existing narrow, independently version-probed NextN safeguard.
- Labels Expert `auto` as **runtime/device default**, not “mmap default”.

The [exact server flag manifest](llama-b10878-server-flags.json) contains
**416 short/long names, including 329 long options**. Compared with the official
b10863 Vulkan manifest, exactly the seven spellings above disappeared; no new
CLI option names appeared. All **158** standard profile × chat/coding commands
parse on the actual b10878 binary. Parser acceptance does not mean every model
was loaded. K2 Horizon still requires its IFM fork; no new architecture claim.

## Lazy AUTO changed: protect the giant-table memory plan

[PR #28326](https://github.com/ggml-org/llama.cpp/pull/28326), **b10867**, changes
`src/llama-model.cpp`: when any selected device reports `mmap_support == false`,
`LLAMA_LAZY_MODE_AUTO` becomes `OFF`. A previously lazy 26.8-GiB table can thus
become eager even though the AutoTuner budget charges only active rows.

AutoTuner now emits **`--lazy-mode on`** for models with a giant marked table
(or the qwen4exp architecture), independently of ordinary `--load-mode`.
`on` exists in older `--tensor-read-lazy` builds too; only the spelling is
adapted. Stored `auto` Extras migrate to the explicit `on` contract. Eager
`off` overrides still fail with a replan-required message. Unsupported lazy
controls remain in argv so a runtime rejects them instead of silently loading
the table eagerly. Small models without such a table acquire no new requirement.

This deliberately prioritizes the memory plan over upstream's iGPU throughput
heuristic. `on` can also map smaller marked tables, so their existing full-weight
budget is conservative. It is **not an OS residency cap** and not a guarantee
of optimum iGPU speed. mmap must exist on the host; the supported Windows,
Linux and macOS platforms supply it. iGPU hardware inference was not tested.

Live b10878 validation of three-part Qwen3.8-Flash-Next-UD-IQ1_S, 4,096 context:

- The loader explicitly reports `per_layer_token_embd.weight ... lazy read enabled`.
- GGUF tensor offsets locate the **28,800,138,240-byte** table in shard 2.
- Windows `VirtualQueryEx`/`GetMappedFileNameW` identify its file mapping;
  `QueryWorkingSetEx` over that exact tensor range reports **266,240 bytes**
  resident at readiness in this sample. This is a point-in-time measurement,
  not a bound for long prompts or future requests.
- A subsequent request returns **“Lazy loading works”** successfully.
- Whole-process RSS was **34.504 GiB** at readiness and **33.949 GiB** after
  generation. Initial validation incorrectly compared total RSS with table size
  and failed; inspecting the exact mapping resolved the measurement error.
  Total RSS includes other mapped model weights/runtime/driver memory and must
  not be presented as table-only residency. The planner's 5%/minimum 0.5-GiB
  active-row allowance remains a heuristic, not an enforced limit.

Source: [model loader](https://github.com/ggml-org/llama.cpp/blob/b10878/src/llama-model-loader.cpp),
[device AUTO policy](https://github.com/ggml-org/llama.cpp/blob/b10878/src/llama-model.cpp).

## HIP/CUDA build option change

[PR #28079](https://github.com/ggml-org/llama.cpp/pull/28079), **b10876**, replaces
the all-quants boolean with **`GGML_CUDA_FA_QUANTS`**. The old
`GGML_CUDA_FA_ALL_QUANTS=ON` still works but warns and overrides the new list.
The common Windows HIP recipe now inspects the checked-out `ggml/CMakeLists.txt`:

- New trees: `-UGGML_CUDA_FA_ALL_QUANTS -DGGML_CUDA_FA_QUANTS=all`.
- Older stable/pinned forks: retain `-DGGML_CUDA_FA_ALL_QUANTS=ON`.

This preserves all quant combinations, clears a stale legacy cache value and
avoids guessing from a build number. No CUDA/HIP compiler or SDK was installed
or replaced. The helper's syntax and both branches are tested; a full HIP/CUDA
llama.cpp rebuild is **not** claimed. Q8/Q8 and Q4/Q4 remain in upstream's default
kernel list; no AutoTuner KV-policy change is required. Custom missing kernel
pairs now use an upstream F16 conversion fallback, potentially slower and with
extra temporary memory—not evidence for reducing graph reserves.

## Other changes and API coverage

- b10864 improves recurrent checkpoint retention/deduplication; no request
  schema or context-sizing flag change.
- b10865 reverts HIP integrated-device reporting after corruption issues;
  do not infer a new safe shared-memory optimization from this change.
- Vulkan Intel F16 tuning and IQ4_XS mat-vec, SYCL IQ MoE handling and RDNA3 MMQ
  tuning are backend changes, not new sampling parameters.
- Jinja null-membership, video frame cache IDs and Granite MoE size labeling
  need no AutoTuner HTTP/CLI migration. Video was not live-tested.
- b10878 changes a C sampler-chain return type to `int32_t`; AutoTuner uses
  subprocess/HTTP, not the llama C ABI.
- Real Spark-X2.5-4B and MiniCPM5-2B generation and structured tool calls pass
  with Q8 K/V. Qwen3.8-27B + DFlash2 passes on two Vulkan GPUs with the draft
  and shared output head colocated. These b10878 tests are **Windows Vulkan**;
  historical b10863 HIP evidence is not relabeled as a b10878 HIP test.

Historical benchmark data is retained; search schema **6** prevents automatic
reuse of winners measured before this loading-policy change.
See [v5.4.5 validation](v5.4.5-validation.md).
