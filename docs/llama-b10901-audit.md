# llama.cpp b10901 / model-profile audit — AutoTuner v5.4.6

Checked 2026-09-10/11. Exact runtime: **b10901**, commit
`28ff0958291ce3465fabd7bd679d4b0edd742bd9`.
[Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/b10878...b10901):
23 commits, 54 files. Both installed Windows HIP and Vulkan servers actually
report that version and were executed; no llama.cpp checkout was changed or rebuilt.
See [validation](v5.4.6-validation.md) for the test boundaries.

## DeepSeek-V4.1-Flash is not V4-Flash

Primary sources: [official card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
[config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json),
[conversion-only PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696).

- HF architecture `DeepseekV41ForCausalLM`, model type `deepseek_v41`, 40 layers,
  1,048,576 context. New 552B-backbone causal encoder/decoder with CSA2,
  cross-layer cache reuse, Engram conditional-memory tables and DSpark.
- Official temperature **1.0**, top_p **0.95 or 1.0**; choose 0.95 for the
  new profile. Reasoning effort is numeric **1–100**, not V4's older modes.
- The publisher supplies a Python prompt encoder, **no Jinja template**.
  Advertised FP4-cache savings are not a supported llama.cpp cache contract.
- PR #28696 is explicitly conversion-only: **no V4.1 inference runtime**.
  Its current proposed GGUF architecture is `deepseek41`; early output used
  `deepseek4` incorrectly and lacks tensors demanded by that loader.

`settings/deepseek-v4_1.yaml` therefore recognizes the model but **blocks
launch and command export**. It does not promise support by inventing a build
minimum or accepting a converter's architecture string as runtime evidence.
The new `runtime_block_reason` profile field is checked before numeric/marker
probes. Known blocked architecture metadata cannot be hidden by a generic
filename, and a V4.1 filename cannot fall back to V4. A separate model-level
check catches renamed V4.1 GGUFs and old `deepseek4.engram.*` conversions,
including use as a draft. The block must be revisited when an actual runtime,
its memory layout and template support have been validated.

The original V4 Pro/Flash profile, sampling, build minimum and architecture
fallback remain unchanged. The new Q8 field is only an unvalidated placeholder
in a non-runnable profile, not a claim about V4.1's optimal KV format.

## Nex-AGI Nex-N2.5-mini

Primary sources: [official card](https://huggingface.co/nex-agi/Nex-N2.5-mini),
[config](https://huggingface.co/nex-agi/Nex-N2.5-mini/blob/main/config.json),
[template](https://huggingface.co/nex-agi/Nex-N2.5-mini/blob/main/chat_template.jinja).

- `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`: existing GGUF
  `qwen35moe` family, 35B, 40 layers, attention every fourth layer,
  256 experts/top-8, native context **262,144**.
- Separate filename-only profile: **temperature 0.7 / top_p 0.95 / top_k 40**
  for both chat and coding, neutral additional penalties/min_p, Q8-first KV.
  It must not take over the generic Qwen3.5/3.6 architecture fallback.
- Preserve the **Nex-specific embedded template**. For llama.cpp use
  `chat_template_kwargs: {"reasoning_effort": "none"|"medium"|"high"}`.
  Default/medium opens `<think>` (adaptive); high opens `<think>\n`; none
  emits an empty closed thinking block. `enable_thinking` is not its switch.
- `--jinja` enables structured tools. Do not copy SGLang's parser flags into
  llama.cpp. Vision needs the matching mmproj. MTP needs the converted NextN
  weights; its presence in the HF config is not proof that every GGUF has them.
- No speculative YaRN extension or unsupported runtime parser is invented.

All four reasoning-mode renders and the tool prompt were checked through the
actual b10901 `/apply-template` endpoint using the official Nex template.
This is template validation, **not inference with Nex weights**, which were
not present locally.

## GLM-5.3 Cybersecurity: same architecture, different defaults

Exact variant identified: **dealignai/GLM-5.3-CYBERSECURITY-FP8**, not an
official Z.ai Cyber SKU and not GLM-5.3-Flash.

Sources: [derivative card/files](https://huggingface.co/dealignai/GLM-5.3-CYBERSECURITY-FP8),
[derivative generation config](https://huggingface.co/dealignai/GLM-5.3-CYBERSECURITY-FP8/blob/main/generation_config.json),
[derivative template](https://huggingface.co/dealignai/GLM-5.3-CYBERSECURITY-FP8/blob/main/chat_template.jinja),
[base GLM-5.3 files](https://huggingface.co/zai-org/GLM-5.3/tree/main).

The two fetched `config.json` files are **byte-identical**, SHA-256
`3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69`.
Architecture, context and KV/MTP behavior therefore reuse the existing glm-dsa
contract; native context is **1,048,576**. The card's 131,072-token vLLM launch
example is not a changed architectural maximum.

However, a separate profile is warranted:

| Setting | Base GLM-5.3 | Cybersecurity derivative |
|---|---|---|
| temperature / top_p | 1.0 / 0.95 | 1.0 / 0.95 |
| repetition_penalty | 1.0 default | **1.1 explicitly published** |
| template `clear_thinking` default | false | **true** |

Since b10786 llama.cpp defaults reasoning preservation on; simply omitting
`--reasoning-preserve` would still overwrite the derivative's template default.
`--no-reasoning-preserve` explicitly preserves its intended history behavior
([upstream capability mapping](https://github.com/ggml-org/llama.cpp/blob/b10901/common/jinja/caps.cpp)).
The actual b10901 template endpoint confirms base history retains a marked
reasoning span while the derivative drops it. Other template differences
include null-content handling and removal of loop `break` shortcuts; no
replacement template is injected by AutoTuner.

The new filename-only profile retains AutoTuner's local GLM top_k 40/min_p 0.01
and does not change ordinary GLM or Flash. FP8 Safetensors are not directly
loadable by llama.cpp; compatible GGUF conversion is still required. No model
weights, refusal behavior or safety mechanisms were modified or evaluated.

## Real b10901 failure: Qwen vision + DFlash2

[PR #28587](https://github.com/ggml-org/llama.cpp/pull/28587) changes DFlash's
multimodal prefill: pinned M-RoPE embedding batches are skipped rather than
injected into its draft context. The exact code checks `has_embeddings`,
`n_rows > 1` and equal first/last positions; it is not a CLI/API rename.

Actual **Qwen3.8-27B Q8_K_XL + matching BF16 mmproj + DFlash2 Q4_K_M** image
requests fail on **both HIP and Vulkan b10901** after image encoding:

```text
last position stored ... X = 3
input batch ... starting position ... Y = 18
required ... Y = X + 1
llama_decode(ctx_dft) failed rc=-1
failed to process speculative batch
HTTP 500
```

AutoTuner now rejects this **verified build + qwen35 + DFlash2 + enabled
vision** combination before process launch. The message offers explicit
alternatives: disable Draft for images, or disable Vision for text-only
DFlash2. Both paths were actually retested successfully on both backends.
No silent user-setting change, global DFlash ban, or assertion that a future
build is fixed. Other drafter types/older builds retain their previous behavior;
this is not proof that their multimodal combinations work.

## Other b10901 changes and retained safeguards

- No relevant CLI or HTTP schema migration. Local HIP/Vulkan help exposes
  **415 names / 328 long options**. Against the official b10878 manifest only
  `--rpc` is absent, because these local builds have **GGML_RPC=OFF**, not
  because upstream removed it. [Manifest and binary hashes](llama-b10901-server-flags.json).
- All **162 standard profile × chat/coding commands per backend** parse on
  the real server. DeepSeek-V4.1 is deliberately excluded and its hard gate
  tested. Fork-only model parser acceptance is not architecture support.
- [PR #28330 / b10889](https://github.com/ggml-org/llama.cpp/pull/28330) removes
  the unused **qwen4exp indexer V cache**. AutoTuner's version-independent
  planner deliberately keeps the old reserve for old/unknown runtimes and
  exported configurations. For the 48-layer/12-attention-layer test fixture,
  K remains 15,360 bytes/token; actual new V is 12,288 versus the conservative
  18,432 F16 bytes/token budget. The extra reserve is 0.75 GiB at 131,072
  tokens before V quant scaling. This is a documented capacity cost, **not
  an under-budgeting bug**. No blanket subtraction or automatic enlargement.
- Explicit `--lazy-mode on`, old loading-flag migration, Q8-first policy,
  recurrent state reserves and shared draft/output-head GPU placement remain.
- Vulkan small-matrix/copy/fusion changes, CPU s390x support, Hexagon updates,
  Granite parameter labels and Nemotron conversion fixes need no AutoTuner
  launch migration. Existing HIP build-recipe flags remain valid.

Tests and translated profile notes cover all nine bundled languages. Source
inspection does not establish universal performance or long-context safety;
see the bounded live evidence in the validation report.
