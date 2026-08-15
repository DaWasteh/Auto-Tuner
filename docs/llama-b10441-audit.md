# llama.cpp b10441 integration audit

AutoTuner v5.1.3 was audited against the exact upstream range below.

| Tag | Commit | Date |
|---|---|---|
| `b10329` | `18f7ad7fc912444acc0f51995a4b8e45fd9a0cd4` | 2026-08-08 |
| `b10441` | `0177dcc7300bad8914bb838baabce87899812491` | 2026-08-15 |

The range contains **112 upstream commits**. Sources were checked from an exact
`b10441` checkout and against the generated `llama-server --help`. The official
Windows b10441 CPU, SYCL, and OpenVINO packages were also run locally with
`--list-devices` (no model load):

```text
CPU:      Available devices: (none)
SYCL:     SYCL0: Intel(R) Graphics (25538 MiB, 25538 MiB free)
OpenVINO: OPENVINO0: OpenVINO Runtime (48504 MiB, 27781 MiB free)
```

## Command-line compatibility

No AutoTuner-emitted server option was removed in this range. The current
b10441 help still advertises the options used by AutoTuner, including:

- `--fit off`, `-fa on|off`, `-ctk/-ctv`, `--cache-ram`, `--parallel`
- `-ngl`, `--n-cpu-moe`, `--tensor-split`, `--main-gpu`, `--device`
- `--metrics`, `--slots/--no-slots`, `--no-kv-offload`
- `--reasoning`, `--reasoning-budget`, `--reasoning-preserve`
- all used speculative types (`draft-mtp`, `draft-eagle3`, `draft-dflash`,
  `draft-dspark`, and the n-gram family)
- `--load-mode` with `auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, and `dio`

b10441 makes `--load-mode auto` the upstream default and migrates old mmap
spelling internally. AutoTuner already represents its default as `auto`; it now
also recognizes the new `--reasoning-effort` value flag in Extra CLI arguments.
System/user llama.cpp `config.ini` files apply before environment variables and
CLI arguments, so AutoTuner's explicit launch arguments remain authoritative.

## Backends and official b10441 packages

The b10441 release publishes native packages for all backends relevant to this
audit:

- macOS arm64 and x64 (Metal is enabled by default)
- Windows x64 CPU, Vulkan, ROCm 7.14, SYCL, OpenVINO 2026.2.1,
  CUDA 12.4, and CUDA 13.3
- Windows arm64 CPU and CUDA 13.4 preview
- Ubuntu x64 CPU, Vulkan, SYCL FP16/FP32, and OpenVINO 2026.2.1
- Ubuntu arm64 CPU and Vulkan

AutoTuner does not link to a backend and therefore remains backend-neutral: it
launches the selected executable and probes that exact executable. v5.1.3 now
preserves backend-qualified device IDs (`CUDA1`, `HIP0`, `Vulkan0`, `SYCL0`,
`Metal0`, `OPENVINO0`) instead of reusing Vulkan order for every backend.
CUDA, HIP, Vulkan, and oneAPI receive their own visibility selector.

OpenVINO is represented as a whole-graph, unified/host-memory accelerator even
though its device name does not match a physical Windows adapter. The inherited
`GGML_OPENVINO_DEVICE` environment variable remains the upstream authority for
`CPU`, `GPU`, `GPU.0`, or `NPU`; model/device coverage is still experimental and
must follow upstream's validated-model table.

## Apple Silicon and integrated-GPU memory

Apple Silicon is detected only on Darwin arm64/aarch64. Installed memory is no
longer reported as fully free VRAM. Live OS-available memory and, when present,
the selected Metal binary's device report are combined conservatively.

Most importantly, CPU and GPU allocations are budgeted from **one** unified
pool. Model weights, KV, projector/draft allocations, prompt cache, and runtime
workspace are subtracted once. Moving KV from Metal to CPU changes compute
placement but does not create extra physical capacity. The GUI and TUI present
one unified-memory total instead of independent RAM and VRAM totals.

The release workflow uses GitHub's Apple-Silicon `macos-15` runner, asserts
`arm64`, publishes `AutoTuner-macOS-arm64.zip`, and the updater matches both OS
and CPU architecture while retaining compatibility with older generic assets.

## Architecture and memory-planning changes

- **Dense/full offload:** KV remains device-resident unless
  `--no-kv-offload` is emitted. Host RAM is no longer silently added to a
  full-offload VRAM budget. If weights plus the selected tier's KV reserve do
  not fit on one GPU, dense layers spill to CPU; established multi-GPU
  full-offload behavior is retained when weights fit across the GPU pool.
- **Asymmetric KV:** key and value dimensions are sized independently. This is
  required for MLA and any model whose K/V head dimensions differ.
- **Metadata aliases:** legacy/HF-style `kv_head_count` and
  `num_key_value_heads` keys now affect sizing as well as diagnostics.
- **MoE:** authoritative MoE architecture names remain in the expert-aware
  path when a community GGUF omits `expert_count`. If even estimated shared
  tensors cannot fit, tuning falls back to CPU instead of reporting an
  impossible GPU footprint.
- **Hybrid/recurrent:** scalar and per-layer boolean
  `attention.recurrent_layers` metadata are supported. Pure Mamba/RWKV models
  use a minimal nonzero KV sentinel rather than a fictitious 25% attention
  cache. Fixed F32 recurrent state is tracked separately from context-growing
  attention KV, scales with parallel slots and model-draft rollback snapshots,
  and follows partial K/Q/V offload placement. New b10441 `minimax-01` support
  counts its Lightning-Attention layout (one full-attention layer per eight
  blocks).
- **Performance targets:** the existing targets remain generic user intent,
  but apply separate dense/MoE placement rules and profile recommendations.
  Large 4096-token MoE throughput batches now reserve their extra workspace;
  user-facing dense reservation descriptions match the configured 64k/32k/16k
  values.

## New upstream architectures in the range

- `graniteswitch` (b10342): existing Granite Switch 4.1 profile.
- `muse-glimmer` (b10349): existing dense multimodal/tool-use profile.
- `pockettts` (b10369): converter/MTMD/TTS support; AutoTuner's normal chat
  server workflow does not claim to replace the dedicated TTS workflow.
- `minimax-01` (b10437): new profile plus MoE/hybrid KV detection for
  MiniMax-Text-01 and MiniMax-M1.

## Residual limits

- Physical M5 Max, NVIDIA CUDA, and an Intel ThinkPad were not available for a
  full model load. Their launch paths are source-, help-, package-, and CI-
  validated, not benchmark-certified on those exact machines.
- The macOS community artifact is native arm64 but is not Apple Developer ID
  signed/notarized. First launch may require Finder **Right-click → Open** or
  **Privacy & Security → Open Anyway**; use only the official GitHub release.
- SYCL supports Intel GPUs from 11th-generation Core onward; older Intel iGPUs
  should use CPU or a separately validated OpenCL/OpenVINO path.
- OpenVINO model, quantization, stateful-execution, GPU, and NPU coverage is
  explicitly work in progress upstream. Unsupported combinations can fail even
  though AutoTuner launches the backend correctly.
- Performance estimates remain conservative approximations. Runtime logs are
  authoritative for actual weight, recurrent-state, compute-buffer, and KV
  allocations.

## Primary upstream references

- Release: <https://github.com/ggml-org/llama.cpp/releases/tag/b10441>
- Server options: <https://github.com/ggml-org/llama.cpp/blob/b10441/tools/server/README.md>
- Build/backends: <https://github.com/ggml-org/llama.cpp/blob/b10441/docs/build.md>
- SYCL: <https://github.com/ggml-org/llama.cpp/blob/b10441/docs/backend/SYCL.md>
- OpenVINO: <https://github.com/ggml-org/llama.cpp/blob/b10441/docs/backend/OPENVINO.md>
- Exact comparison: <https://github.com/ggml-org/llama.cpp/compare/b10329...b10441>
