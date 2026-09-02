# llama.cpp b10760 compatibility audit

Date: 2026-09-02

AutoTuner release: v5.3.8

Audited upstream range: b10743 (`8887a48f0`) through b10760
(`0f3a71be15af836d277c9f918adfafb45732677e`)

## Authority and scope

The audited source and binaries are exact local upstream builds:

```text
L:\LAB\ai-local\b10760_vulkan_llama.cpp
L:\LAB\ai-local\b10760_hip_llama.cpp
version: 0.3.0-dev (build 10760, commit 0f3a71be1)
```

Both source trees resolve to tag `b10760`. The audit covered the complete
17-commit range from b10743, current Vulkan/HIP `llama-server --help`, every
option emitted directly by shipped YAML profiles, model/profile coverage,
NextN and recurrent-state correctness changes, and a clean Windows HIP build.
The earlier [`llama-b10743-audit.md`](llama-b10743-audit.md) remains the
historical v5.3.6 record; it is not rewritten retroactively.

## CLI and AutoTuner feature compatibility

The exact b10743 and b10760 server help each expose 331 distinct long-option
names. The set difference is empty in both directions. None of the 17 upstream
commits changes `common/arg.cpp`, the server CLI, `include/llama.h`,
`src/llama-arch.cpp`, or `src/llama-model-loader.cpp`.

All option names currently emitted by profile `extra_args` are advertised by
both exact b10760 backends. AutoTuner therefore needs no new flag, renamed
option, command-pruning rule, or public feature switch for this range. Existing
support for lazy reads, fit authority, prompt caching, reasoning/Jinja, MTMD,
MTP/draftless speculation, GPU placement, and metrics remains valid.

## Material upstream changes

### NextN regression is fixed at the existing b10749 boundary

The first two relevant commits after b10743 are the paired repairs already
identified by the v5.3.6 audit:

```text
d11b3cc7e  load relevant arrays with n_layer_all (#28173)
73159c303  fix gemma4-assistant (#28183)
```

Together they remove the b10741-b10748 failures for array-backed integrated
NextN and all-NextN standalone assistant heads. b10749 is still the first valid
tag containing both. AutoTuner's narrow quarantine remains correct and is not
broadened or removed for older selected binaries.

### b10749 corrects recurrent SSM tensor lifetime flags

Commit `f28493c78347` changes `ssm_a` to the explicit no-scan tensor name for:

- BailingMoE3 (Ling 3);
- Kimi-K3; and
- Kimi Linear.

This is a model-loading/allocation correctness contract, not a sampling tweak.
The Ling 3 and Kimi-K3 profile minimums are raised to b10749, and a dedicated
Kimi Linear profile uses the same boundary. Their original loaders landed at
b10460, b10448, and b7957 respectively, but those earlier introduction builds
do not contain the corrected tensor classification.

### Remaining changes are transparent runtime improvements

No additional AutoTuner control is required for:

- b10749 custom-YaRN native-context autoscaling;
- b10750 indexed n-gram history lookup (the upstream report measured a 4.9%
  decode gain at 71k context on Qwen3.8 Flash Next);
- CUDA fused weighted expert reduction;
- Vulkan conditional BF16-extension requests;
- Vulkan IQ3_S mat-vector handling for batches greater than four; or
- b10760's Qwen3-TTS-0.6B optional projector/F32 overflow correction.

AutoTuner's existing YaRN, n-gram, batch, Vulkan, and MTMD paths inherit these
changes when the newer binary is selected.

## Popular-model profile review

The review compared the 67 v5.3.7 profiles with official model cards/configs,
llama.cpp b10760 architecture support, and a 2026-09-02 Hugging Face
trending/download snapshot. Existing Qwen3/Qwen3.5/Qwen3.6/Qwen3.8,
DeepSeek-R1/V3/V4, Gemma 4, Granite, GLM, and current Nemotron families were
already covered and were not duplicated.

Seven focused profiles were added, bringing the bundle to 74:

| Profile | Contract added |
|---|---|
| `gemma-3.yaml` | Gemma 3 4B/12B/27B 128k maximum, with GGUF metadata capping 1B to 32k; official temp 1.0/top-k 64/top-p 0.95 |
| `gemma-3n.yaml` | separate 32k E2B/E4B profile; text/vision supported, current b10760 audio limitation documented |
| `mistral-small-3.yaml` | Mistral Small 3.1/3.2 24B, 128k, official low temperature 0.15, Pixtral/mmproj caveat |
| `llama-4.yaml` | Meta temp 0.6/top-p 0.9; profile ceiling 10,485,760 while GGUF metadata caps Maverick to 1,048,576 and base models to 262,144 |
| `kimi-linear.yaml` | dedicated `kimi-linear` architecture, 1M context, b10749 SSM correctness gate, metadata-driven sampling because the card publishes no fixed sampler contract |
| `paddleocr-vl.yaml` | deterministic 131k OCR/table/formula/chart element recognition and the official prompt forms |
| `ornith-1_5.yaml` | longest-pattern override for the current 1.5 family, 256k native/validated YaRN extension, separate general and precise-coding sampling |

`llama-3-4.yaml` is renamed to `llama-3.yaml` and no longer claims Llama 4.
This removes the shared 131,072-token cap that understated Scout by 80x and
Maverick by 8x. The scanner's existing `native_context` cap remains the
per-checkpoint authority, so the larger family ceiling does not inflate a
Maverick or base checkpoint.

Popularity is used only to prioritize supported llama.cpp workloads, not as a
claim that every high-download GGUF belongs in AutoTuner. For example,
Qwen3-TTS/audio.cpp and unrelated ASR artifacts are excluded from
`llama-server` chat profiles. Inkling-Small is also not promoted: b10760 has no
Inkling language-model architecture despite the repository's GGUF tag. This
prevents a popular filename from becoming a false runtime-support claim.

Primary source material included the official Google Gemma documentation,
Mistral Small 3.2 card/config, Meta `llama-models` Llama 4 reference defaults,
Moonshot Kimi Linear card/config, PaddleOCR-VL card/config, Ornith 1.5 card,
and the exact b10760 llama.cpp source. Volatile download counts were used only
for prioritization; architecture and model cards remain authoritative for the
profile values.

## Windows HIP recipe and warning audit

### Recipe correction

The shared recipe previously passed:

```text
-DCMAKE_HIP_COMPILER=<ROCm clang.exe>
```

Current llama.cpp's Windows HIP configuration enables HIP through `GGML_HIP`
and compiles HIP translation units with the selected C/C++ ROCm clang drivers.
CMake retained the extra variable as `UNINITIALIZED` and reported it unused.
The dead argument is removed. The required C/C++ compilers, `gfx1201`, HIP
graph/VMM/RCCL choices, Flash Attention, all-quant FA, and mandatory
`GGML_CUDA_NO_PEER_COPY=ON` contract are unchanged.

A clean configure after the change has no `CMAKE_HIP_COMPILER` cache entry and
no `Manually-specified variables were not used by the project` warning.
Regression tests pin this absence and the safety/performance flags.

### Clean-build warning classification

The exact b10760 HIP recipe rebuilt 630 Ninja targets with ROCm 7.2 / Clang 21,
MSVC 14.51 headers, and warnings left enabled. The log contains 11,417 compiler
warning records across 239 source locations. Template instantiation multiplies
a small set of diagnostics thousands of times:

| Class | Records |
|---|---:|
| GGML HIP/CUDA kernels and template instantiations | 11,395 (99.807%) |
| vendored portability extensions | 10 |
| upstream Windows POSIX-name deprecations | 6 |
| optional examples/tools | 4 |
| upstream sampler ignored-`nodiscard` calls | 2 |

The largest diagnostic groups are sign comparisons (4,786), nested anonymous
types (2,269), HIP intrinsic stubs inferred as `noreturn` (1,562), unused
variables/parameters (1,235 combined), and aggregate-brace/C++98-extra-semicolon
portability diagnostics (1,366 combined). Eight loop/occupancy transformation
warnings come from the AMD optimizer. None is produced by AutoTuner code or a
mis-selected SDK/target.

These warnings are not hidden globally: `LLAMA_ALL_WARNINGS` stays on, and the
recipe adds neither warning-disable flags nor `/WX`/`-Werror`. Patching stock
llama.cpp source in a build recipe would make the source identity dishonest and
would be overwritten by every release. The correct action is therefore to
remove only the avoidable unused CMake input and leave upstream diagnostics
visible for upstream fixes.

### HIP result

The clean build completed and reported:

```text
version: 0.3.0-dev (build 10760, commit 0f3a71be1)
built with Clang 21.0.0 for Windows AMD64
ROCm0: AMD Radeon AI PRO R9700
ROCm1: AMD Radeon RX 9070 XT
HIP multi-GPU semantic output verified: HIP MULTI GPU OK
```

The no-peer-copy cache bit remained `ON`, matching ROCm 7 runtime DLLs and
kernel libraries were bundled/linked, and the deterministic real-model split
passed. This is stronger correctness evidence than treating a warning-free
compile or throughput-only benchmark as sufficient.

## Conclusion

AutoTuner's command/runtime feature layer is already compatible with b10760.
The required v5.3.8 changes are profile correctness/coverage, three b10749 SSM
minimums, removal of one unused HIP CMake input, tests, and documentation. No
new llama-server flag or broad warning suppression is justified.

Release-level source and executable evidence is recorded in
[`v5.3.8-validation.md`](v5.3.8-validation.md).
