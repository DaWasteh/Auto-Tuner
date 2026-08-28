# llama.cpp b10666 integration audit

AutoTuner v5.2.9 was audited against the exact upstream range below.

| Tag | Commit | Date |
|---|---|---|
| `b10590` | `6657ded4faa3b8450221119fc6b4d002e35104a2` | 2026-08-23 |
| `b10666` | `4e97ac86e` | 2026-08-28 |

The range contains **76 upstream commits**. Evidence came from the full-history
upstream checkout, source/argument diffs, the exact locally compiled b10666
Windows/Vulkan binary, generated profile commands, and bounded real-model
server starts.

## Command-line compatibility

The selected binary reports:

```text
version: 0.3.0-dev (build 10666, commit 4e97ac86e)
```

No option emitted by AutoTuner was removed in this range. Current b10666 accepts
its placement, KV, MTMD, reasoning, tools, prompt-cache, slots/metrics, and
speculative arguments. AutoTuner already uses the replacement spellings for
removed legacy draft/ngram options and never emits the CLI-only `-no-cnv` flag.
Help-based compatibility pruning remains the fallback for older forks.

b10653 adds `--tensor-read-lazy {on,auto,off}`. AutoTuner now treats it as a
value-taking option and explicitly emits `auto` for architectures with marked
row-gather tables; older binaries lose the complete flag/value pair during
capability pruning.

## Mainline model/runtime promotions

- **b10658 / `b10f9ca58`:** DFlash2 (grouped local convolution + candidate
  selector) merged. The existing `draft-dflash` token remains correct.
  AutoTuner accepts b10658+ and still recognizes the reviewed PR #27342 commits
  as legacy fallbacks; b10590–b10657 remain blocked for 81-tensor sidecars.
- **b10660 / `6c84c7d5d`:** Qwen3.8 Flash Next merged as `qwen4exp` (PR
  #27742). `settings/qwen3_8_flash_next.yaml` now requires b10660; both this
  and the newly merged DFlash2 dedicated PR build recipes were removed.
- **b10665 / `ca3d5a3e1`:** Nemotron3.5 DSpark support adds causal SWA,
  attention sinks, and confidence-head-free checkpoints. Its new profile uses
  draft length 7 and p-min 0.0, which b10665 requires when no confidence head
  exists.
- **b10644 / `d7a207411`:** Nanbeige4.2 graph support was corrected. A
  dedicated 256k profile now carries the official reasoning/agent sampling.

No other new `general.architecture` value was added between b10590 and b10666.

## Why Qwen3.8 Flash Next looked smaller than its files

The tested split GGUF is:

```text
Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
```

Its three shards total **67.564 GiB**. The unusual part is
`per_layer_token_embd.weight`, an IQ4_NL PLE n-gram hash table with shape
`[160, 320001536]` and storage size **26.822 GiB**. b10666 marks that tensor
`TENSOR_READ_LAZY`; it is a host mmap row-gather table and is not an ordinary
expert/layer weight eligible for GPU tensor splitting. The remaining placement
weight is therefore about **40.742 GiB**.

The previous AutoTuner divided all 67.564 GiB across MoE layers. In Safe mode it
therefore selected `--n-cpu-moe 19`, predicted 43.0 GiB GPU plus 24.6 GiB CPU,
and clamped an explicit 90k request to 36,092. That model was physically wrong:
it attributed part of the fixed host PLE table to GPU/CPU expert layers, leaving
avoidable VRAM free and moving real experts into RAM.

v5.2.9 scans tensor offsets across split shards, records the auto-lazy span,
uses only the remaining bytes for layer placement, and displays the mapped RAM
separately. Windows budgets the whole mapping because b10666 cannot unmap
unused `MapViewOfFile` fragments; POSIX reserves a conservative 5% working set,
matching upstream's approximately 4.4% measured residency for random lazy
rows.

## KV and graph memory that file size does not show

Qwen4exp has 12 normal attention layers (one every four blocks), but b10666 also
creates a second QSA indexer cache. At context 90,112 with Q4_0 K/V, the real
logs report:

| Allocation | b10666 size |
|---|---:|
| normal attention KV | 594.00 MiB |
| QSA indexer KV | 222.75 MiB |
| recurrent R/S/PLE state | 112.57 MiB |

The indexer is one extra K head of 128 dimensions plus one V head of 256
dimensions on each full-attention layer. AutoTuner now adds those separate K/V
parts before choosing asymmetric cache quantization.

QSA graph buffers scale almost exactly with **context × ubatch**. Three otherwise
identical ~90k-context loads measured:

| ubatch | device compute | host compute | prompt | decode |
|---:|---:|---:|---:|---:|
| 1024 | 5,588.78 MiB | 17,977.61 MiB | 75.10 tok/s | 32.56 tok/s |
| 256 | 1,409.64 MiB | 4,498.29 MiB | 67.28 tok/s | 32.15 tok/s |
| 64 | 398.69 MiB | 1,128.46 MiB | 71.92 tok/s | 31.34 tok/s |

Safe ubatch 64 cut the graph pools by about 14–16× while retaining 96.3% of
decode speed and 95.8% of short-prompt throughput. AutoTuner now uses ubatch
64/128/256 for Safe/Balanced/Throughput and applies measured conservative
device/host coefficients to placement and final context clamping.

## Real load validation

All runs used b10666, both AMD Vulkan devices, Q4_0 KV, one slot, no prompt
cache, no warmup, and an eight-token OpenAI-compatible chat request.

1. **90,000 context, full placement, ubatch 1024:** reached `/health` and
   completed. The model buffers were 27,806.97 MiB CPU_Mapped,
   14,558.87 MiB Vulkan0, and 26,809.35 MiB Vulkan1. With the 17.56 GiB host
   graph, available system RAM fell from 32.03 GiB to 0.46 GiB.
2. **90,000 context, full placement, ubatch 256:** reached `/health` and
   completed. Device graph memory fell by about 4.18 GiB and host graph memory
   by about 13.16 GiB; process RSS after completion was 36.19 GiB.
3. **90,000 context, Safe ubatch 64:** reached `/health` and completed. The two
   AMD local-memory counters rose by ~40.69 GiB in total, leaving roughly
   5.35 GiB of the 47.8 GiB physical VRAM pool unused. Process RSS was
   34.87 GiB; because only 33.82 GiB RAM was free before this deliberately
   forced run, Windows had just 0.52 GiB available afterward. With the clean
   post-stop 36.63 GiB available, the same allocation leaves about 3 GiB.
4. **AutoTuner-generated bounded context:** reached `/health` and completed
   with the new full-placement split and explicit `--tensor-read-lazy auto`.
   Context is re-clamped from live free RAM/VRAM, so background applications
   can reduce it rather than forcing an OOM. This confirms that 90k is valid
   when enough launch-time RAM is free, but not from the 68.4 GB file sum alone.

The free-VRAM observation was therefore real, not a llama.cpp accounting bug:
the old planner unnecessarily CPU-offloaded experts because it treated the PLE
host table as splittable model weight. At the same time, the apparent headroom
was partly consumed by previously unbudgeted QSA indexer/graph allocations.

## Primary upstream references

- Release: <https://github.com/ggml-org/llama.cpp/releases/tag/b10666>
- Exact comparison: <https://github.com/ggml-org/llama.cpp/compare/b10590...b10666>
- Qwen3.8 Flash Next PR: <https://github.com/ggml-org/llama.cpp/pull/27742>
- DFlash2 PR: <https://github.com/ggml-org/llama.cpp/pull/27342>
- Nemotron3.5 DSpark commit: <https://github.com/ggml-org/llama.cpp/commit/ca3d5a3e1>
- Tensor lazy-read commit: <https://github.com/ggml-org/llama.cpp/commit/fac889fb3>
- Nanbeige4.2 support: <https://github.com/ggml-org/llama.cpp/commit/b77d64675>
