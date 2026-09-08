# llama.cpp b10863 audit — AutoTuner v5.4.4

Date: 2026-09-08. Exact target: **b10863**, commit
`88ada91c18cd026388be742838d9f27fc12673bc`.
[Comparison from b10839](https://github.com/ggml-org/llama.cpp/compare/b10839...b10863):
24 commits, 90 changed files. No changes were made to the user's llama.cpp builds.

## Upstream changes and launch contracts

- **No parser option was added or removed** in the exact `common/arg.cpp`
  comparison (435 long names in both sources, including non-server tools).
  The local server builds retain 333 long options. The official RPC-enabled
  Windows Vulkan release additionally advertises `--rpc`; this is a build
  feature, not a new b10863 option. The portable
  [server flag manifest](llama-b10863-server-flags.json) records that official
  binary's help. All **158** standard-profile × chat/coding commands parse
  against the actual local b10863 server; this does not claim every model loads.
- [PR #28390](https://github.com/ggml-org/llama.cpp/pull/28390) changes default
  draft/projector device inheritance. AutoTuner already pins the projector.
  External drafts now receive **`--spec-draft-device`**, matching the primary
  GPU on which their complete weight budget was reserved. Aliases consume
  values and cannot override the generated placement behind the planner's back.
- **Pinning alone is insufficient for DFlash/DSpark/EAGLE shared output heads.**
  A real HIP test initially aborted because the target's `output.weight` was
  on ROCm1 while the pinned draft context only had ROCm0. In layer-split mode
  the output belongs to the *last* GPU, not necessarily `--main-gpu`.
  AutoTuner therefore orders the primary last for external drafts on known
  HIP/CUDA/Vulkan device lists, updating visibility, tensor split, primary,
  projector and draft names together. Both AMD backends now pass the same
  two-GPU Qwen3.8-27B + DFlash2 inference test. Unknown multi-GPU draft identity
  and incompatible shared-head placement fail visibly instead of guessing.
  Source: [derived draft params](https://github.com/ggml-org/llama.cpp/blob/b10863/common/speculative.cpp),
  [DFlash shared head](https://github.com/ggml-org/llama.cpp/blob/b10863/src/models/dflash.cpp).
- [Kimi-K3 rollback, PR #28466](https://github.com/ggml-org/llama.cpp/pull/28466),
  b10853, adds bounded recurrent snapshots. Existing AutoTuner state sizing
  already includes `(1 + draft_n_max)`; there is no new KV formula to invent.
  Pre-b10853 external-draft preflight now warns about full host checkpoints
  and different transient memory/performance. The baseline loader minimum
  remains b10749; ordinary inference is not incorrectly prohibited.
- [Server LRU fix, PR #28539](https://github.com/ggml-org/llama.cpp/pull/28539)
  fixes same-model request hangs in optional multi-model routing. AutoTuner
  does not automatically enable that server mode.
- Vulkan HC/unary fusions, HIP/backend fixes and Metal IQ3_XXS corrections
  need no new sampling flags. Throughput improvements do not establish lower
  peak memory: Q8-first KV, Q4 capacity fallback and measured QSA graph reserves
  remain unchanged. Historical benchmark measurements are retained; search
  schema 5 invalidates reuse of pre-fix automatic winners.

## Requested models

| Model | Verified contract and action |
|---|---|
| **Spark-X2.5-4B**, also 1.7B | Both use `spark2_5`, native 1,048,576 context, 3:1 sliding/full attention, 512-token sliding window, head dimension 256. 4B has 36 layers / 4 KV heads; 1.7B has 28 / 2. Existing metadata sizing handles both; the common profile and all language notes now explicitly cover 4B. Sampling remains temperature 1, top-p .95, top-k/min-p disabled. Mainline support starts at b10828. |
| **MiniCPM5-2B** | New `minicpm5-2b.yaml`: temperature **1.0**, top-p .95 in both task presets, embedded template, native 131,072 context. It must not inherit 1B's .7/.9 temperatures. Official GGUF's internal **2.6B** label also matches. Dense `llama`, 42 layers, 16 Q / 2 KV heads, dimension 128. No global `llama` architecture fallback is claimed. |
| **K2-Horizon-MoVA-36B-A4B** | New `k2-horizon.yaml`, **not Kimi-K2/deepseek2**. Mainline b10863 does not support `k2-horizon`; the selected executable/library must contain the IFM architecture marker. No guessed build minimum or automatic fork installation. Native 524,288 context, temperature 1/top-p .95, embedded template, high reasoning/XML tool defaults. |

Spark and MiniCPM5 were tested with the user's completed GGUFs on **both local
b10863 Vulkan and HIP**: successful generation, structured JSON tool calls
following >2,500 input tokens, and actual Q8 K/V allocations. Some old Spark
GGUF cards still say a fork is required; that statement is superseded by
mainline PR #27868 and these exact-runtime tests. No extra Spark YAML is needed.
Thinking remains a template/runtime choice, not an automatic assertion that
selecting a sampling preset changes the model's reasoning mode.

K2's actual downloaded **header** confirms 48 full-attention layers, 100 FFN
experts (8 active), 64 value experts (4 active), three leading dense layers,
8 KV heads and dimension 128. The [IFM implementation](https://github.com/MBZUAI-IFM/llama.cpp/blob/35999d101cf2233fc54f09c3c8d599da7303ce02/src/models/k2-horizon.cpp)
combines value experts before caching, so KV must **not** be multiplied by
4 or 64. Its large `attn_v_exps` weights are not matched by the FFN-only
`--n-cpu-moe` rule. The generic 8%-shared placement estimate would undercount
those GPU weights. Auto uses conservative **whole-layer `-ngl` placement**
(including corresponding KV/split accounting) instead, while still displaying
MoE architecture. This is a safe initial policy, not a claim of optimal
fork-specific expert offload. K2 inference/tool parsing has not been tested
on the required IFM fork. Vision, MTP and RoPE extension are not assumed.

Primary model sources:

- [Spark 4B config](https://huggingface.co/XHToken/Spark-X2.5-4B/blob/main/config.json),
  [1.7B config](https://huggingface.co/XHToken/Spark-X2.5-1.7B/blob/main/config.json),
  [mainline PR #27868](https://github.com/ggml-org/llama.cpp/pull/27868).
- [MiniCPM5-2B config](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json),
  [generation config](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/generation_config.json),
  [OpenBMB deployment recommendations](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/llama_cpp.md).
- [K2 official GGUF/fork requirement](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-GGUF),
  [model config](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/main/config.json).

## Qwen3.8 Flash Next: the 51B NGRAM table

**Already lazy, not eagerly loaded in full.** Local shard headers identify
`per_layer_token_embd.weight`, shape **160 × 320,001,536**: 51,200,245,760
entries, stored in **28,800,138,240 bytes (26.8222 GiB)**. Both local three-part
IQ1_S/Q2_K_XL variants use this table. Scanner aggregation correctly excludes
it from ordinary CPU/GPU-splittable weights. The table is on the user's NVMe.

The exact b10839/b10863 model-loader sources are unchanged: marked tensors
**greater than 4 GiB** are auto-lazy; their mmap ranges are excluded from
prefetch. `--load-mode none/dio/mlock` changes ordinary weights but does not
turn off this separate lazy mapping. Automatic locking stays conservative;
comments now describe the actual upstream behavior. Forced CPU-only locking
also now works when the RAM/privilege/memlock checks permit it.

A real b10863 Vulkan launch of the sharded IQ1_S model at 4,096 context
confirmed `per_layer_token_embd.weight ... lazy read enabled` and generated
`Lazy loading works`. The **entire server** RSS was 21.657 GiB after readiness
and 22.568 GiB after the short request, less than the table alone. This confirms
that the complete table was not resident in that test; it is not a measurement
of table-only RSS or a guarantee that memory stays constant on long workloads.

Lazy still uses RAM for active pages: NVMe pages enter the OS file cache on
access. The 5% / minimum .5-GiB active-row planning reserve is a **heuristic,
not an enforced residency cap**. Available RAM, workload and OS caching can
increase residency or cause paging. No undocumented llama.cpp flag promises
zero RAM or a hard NVMe-only mode.

Additional safeguards:

1. Lazy flags cannot be silently pruned on an incompatible binary while the
   plan continues to exclude 26.8 GiB. Such a runtime rejects the command.
2. Explicit conflicting lazy overrides (including short/legacy/inline forms)
   produce a clear replan-required error; they are neither silently ignored
   nor allowed to make the table eager under a lazy budget.
3. Equivalent option aliases cannot bypass computed context/KV/device settings.
4. GUI, TUI and benchmark paths report these errors before spawning a model.
5. Small Gemma4 models without a large auto-lazy table acquire no unnecessary
   lazy-feature requirement on older builds.

PLE NGRAM embeddings are unrelated to `--spec-type ngram-*` draftless
speculation and to MTP sidecars. Disabling speculative n-grams does not disable
lazy loading. See [release validation](v5.4.4-validation.md) for test boundaries.
