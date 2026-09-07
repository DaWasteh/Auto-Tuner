# llama.cpp b10839 audit — AutoTuner v5.4.3

Date: 2026-09-07. Exact target: **b10839**, commit `0cae43063`.
Comparison: [b10797…b10839](https://github.com/ggml-org/llama.cpp/compare/b10797...b10839), 42 commits.
Local clean Vulkan/HIP source trees and their actual Windows executables were
inspected; this audit does not substitute current master for the requested tag.

## Server flags and features

The exact server help contains **333 long options**, compared with 331 in
b10797. Only `--log-jsonl` and `--no-log-jsonl` were added, by
[PR #28437](https://github.com/ggml-org/llama.cpp/pull/28437). No long option was
removed. Both boolean switches work through Extra CLI flags and are removed
when an older binary does not advertise them. AutoTuner retains ordinary text
logging by default; structured logging is opt-in, not a new inference setting.

The tracked [flag manifest](llama-b10839-server-flags.json) makes the complete
profile-option check portable to CI; it no longer depends on ignored local
`.pi/` help captures. The existing simple/map-k n-gram size/min-hits controls
were already upstream before this range, but their value arity was missing in
AutoTuner's compatibility table. It now strips an unsupported option **and its
value**, without leaving an orphan argument. Quantised KV/Flash-Attention
controls are essential memory contracts: they are no longer silently stripped
into a runtime-default F16 allocation. An incompatible runtime must reject the
launch instead.

`--fit-print` remains an optional upstream diagnostic; `--fuse-qkv` (PR #22780)
is an HF-to-GGUF **conversion** option, not a llama-server launch option. Neither
requires an automatically enabled LLM setting. Existing Expert extra arguments
remain the escape hatch for optional runtime controls; generated placement/KV
flags remain authoritative rather than accepting a conflicting duplicate.

## New models and changed runtime behaviour

| Evidence | AutoTuner action |
|---|---|
| [Spark2_5, PR #27868](https://github.com/ggml-org/llama.cpp/pull/27868), b10828 | Add `spark2_5.yaml`, filename/architecture matching, build gate, native 1M context and official sampling. |
| [HY4, PR #28127](https://github.com/ggml-org/llama.cpp/pull/28127), present in b10813 | Add a mainline `hy_v4` profile, separate from the incompatible AngelSlim `hyv4` patched-format profile. Metadata disambiguates identical filenames. Both converters omit MTP. |
| [Explicit Qwen recurrent layers, PR #28208](https://github.com/ggml-org/llama.cpp/commit/9a7570587) | Already handled: explicit recurrent-layer metadata takes precedence; no new heuristic required. |
| [Gemma4 vision handling](https://github.com/ggml-org/llama.cpp/commit/163a40796) | Runtime fix; existing Gemma/mmproj/assistant routing remains supported. |
| [GDN normalisation correction](https://github.com/ggml-org/llama.cpp/commit/5fdfa6282) | Runtime correctness fix, no tuning switch. |
| [Vulkan TQ1_0](https://github.com/ggml-org/llama.cpp/commit/8fe90e1fb) | Weight quantisation support, not TurboQuant KV and not a new cache type. |
| [Vulkan type-aligned GET_ROWS](https://github.com/ggml-org/llama.cpp/commit/0cae43063) | Backend fix affecting tensor views; no new AutoTuner launch flag. |
| CUDA F16 FA barriers/races, Metal FA tuning, OpenCL/SYCL changes | Transparent backend improvements, not new profile defaults. |

Spark sources: [official model card](https://huggingface.co/XHToken/Spark-X2.5-1.7B)
and [config](https://huggingface.co/XHToken/Spark-X2.5-1.7B/blob/main/config.json).
The 1.7B checkpoint has 28 layers, 3:1 sliding/full attention, head dimension
256, a 512-token sliding window, and native context 1,048,576. Official sampling:
temperature 1, top-p .95, top-k disabled. Use its embedded template, not an
invented Qwen template. No unverified MTP or RoPE extension is enabled.

New profile explanations are available in all nine bundled languages. Existing
architectures and finetunes retain metadata-based profile fallback; adding a
marketing name alone is not evidence of a new architecture or a new sampler.

## Q8 memory/compatibility contract

- Auto targets symmetric Q8_0 and reserves for it **before** weight placement.
  It never upgrades to F16/BF16 merely because memory is idle. Q4 is a fallback
  only if Q8 cannot hold the selected target in the remaining budget.
- The selection budget is per parallel slot. Exact manual pairs and one-sided
  pins use separate K/V byte costs in selection/placement. Manual snapshot
  estimates now rescale K and V independently, including unequal head sizes.
- Upstream `src/llama-context.cpp` requires FA for quantised V and block-aligned
  head dimensions. Auto uses F16 with FA off, Grok's forced non-FA path, or known
  non-32-aligned Q8/Q4 heads. Explicit invalid manual choices are not disguised
  as valid automatic settings.
- FA is now a cascading setting, including legacy saved Auto snapshots and
  benchmark profiles. Turning it off cannot leave automatically quantised V.
- Exhausted GPU peers remain excluded when Q8 causes dense partial offload;
  CPU-layer KV is no longer incorrectly counted as primary-GPU KV in that gate.
- A Qwen4exp host/QSA clamp triggers one bounded Q8 reconsideration at the final
  context, checked against both physical pools, without losing context.
- DiffusionGemma remains its verified F16 runner exception. b10839's generic
  diffusion-cli explicitly copies `cache_type_k/v` and `flash_attn_type` into
  context parameters; AutoTuner now emits those three planned controls there.
- Benchmark search schema 4 invalidates automatic reuse of older measured
  winners. Explicit manual profiles and historical measurement records remain.

## Validation boundaries

Real **Vulkan and HIP** launches passed Q8 generation and a structured tool call
after a 10,057-token synthetic input, using Qwen3.8-27B and a 32,768-token window.
Both logs confirm K(Q8_0)/V(Q8_0), 544 MiB each. This verifies runtime/command
compatibility, not universal long-context accuracy or a Q4-vs-Q8 quality study.
Spark/HY4 were source/profile-tested, not fully loaded: no suitable local Spark
GGUF or datacenter-size HY4 deployment was used. CUDA/Metal hardware inference,
new vision inputs and generic diffusion generation were not measured locally.
Cross-platform source/frozen tests are release gates, not claims of those
hardware measurements. See [v5.4.3 validation](v5.4.3-validation.md).
