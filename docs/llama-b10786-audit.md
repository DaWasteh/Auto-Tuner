# llama.cpp b10786 compatibility audit

Date: 2026-09-03

AutoTuner release: v5.3.9

Audited upstream range: b10760 (`0f3a71be15af836d277c9f918adfafb45732677e`)
through b10786 (`de8656bd94f1163188125542534e4bcbc9f9fb1f`)

## Authority and scope

The audited source and binaries are exact local upstream builds:

```text
L:\LAB\ai-local\b10786_vulkan_llama.cpp   (MSVC 19.51, build/bin/Release)
L:\LAB\ai-local\b10786_hip_llama.cpp      (Clang 21 / ROCm, build/bin)
version: 0.3.0-dev (build 10786, commit de8656bd9)
```

Both source trees resolve to tag `b10786`. The audit covered the complete
26-commit range from b10760, the current HIP `llama-server --help`, every
option emitted directly by shipped YAML profiles, the architecture table and
classifier functions in `src/llama-arch.cpp`, the loader/hparams changes, the
server request-parsing changes, and the DeepSeek-V4 vision projector. The
earlier [`llama-b10760-audit.md`](llama-b10760-audit.md) remains the v5.3.8
record and is not rewritten retroactively.

## CLI and AutoTuner feature compatibility

The exact b10760 and b10786 server help each expose 331 distinct long-option
names. The set difference is empty in both directions, so no flag was added,
removed, or renamed. All option names currently emitted by profile
`extra_args` are advertised by b10786; a regression test now checks this
against the captured help text.

One **semantic** change does affect AutoTuner's documentation and Expert tooltip
(commit `e750b887a`, PR #28174):

- `--reasoning-preserve` is now **enabled by default** when the chat template
  advertises `supports_preserve_reasoning`. Previously the template default
  applied. The server logs a warning when the default kicks in and
  `--no-reasoning-preserve` disables it explicitly.
- Passing `preserve_reasoning` through `--chat-template-kwargs` is deprecated
  in favour of the dedicated switches.

AutoTuner's behaviour is unchanged: profiles that already emit
`--reasoning-preserve` remain explicit; the Expert checkbox still emits the flag
when ticked and emits nothing when unticked. The tooltip and README now state
that "unticked" means the llama.cpp default (enabled on b10786+) and that
`--no-reasoning-preserve` can be added to the Extra CLI flags to force it off.
Both spellings are in the b10786 help, so command pruning never strips them.

## Material upstream changes

### Nemotron-3-Puzzle-75B-A9B needs per-layer expert arrays (b10786 gate)

Commit `c61b98b87` (PR #25444) turns `n_ff_exp` and `n_expert_used` into
per-layer arrays (`get_key_or_arr` on the existing
`expert_feed_forward_length` / `expert_used_count` keys, scalar values are
broadcast). Nemotron-3-Puzzle uses five distinct expert FFN widths and seven
top-k values across its 40 MoE layers; older builds read a scalar and fail at
tensor creation.

AutoTuner consequences:

- New `nemotron-3-puzzle.yaml` profile with `min_llama_build: 10786`, matching
  only by explicit filename patterns (the shared `nemotron_h_moe` architecture
  is never claimed as a fallback, so Lightning/Nano keep their own profiles).
- New `nemotron-3-super.yaml` profile for the parent Nemotron 3 Super
  120B-A12B (`min_llama_build: 8295`, the first tag with the 88-layer
  `120B.A12B` type mapping), official temp 1.0 / top-p 0.95, three MTP draft
  tokens as NVIDIA recommends.
- The scanner and tuner read only `<arch>.expert_count` (still a scalar) for
  MoE detection and placement, so array-valued `expert_used_count` metadata
  cannot break existing accounting.

### DeepSeek-V4 vision (b10786)

Commits `7798007a2` and `9400c8946` add the DeepSeek-V4-Flash-Vision-Exp
projector (`deepseek4v` mmproj, new `v.token_embd.img_*` tensors), an
image-token routing bias (`blk.N.exp_probs_b_vl`), and a non-causal
sliding-window rule for image spans. `deepseek4` is also listed by
`llm_arch_is_hybrid()`.

AutoTuner consequences:

- `deepseek-v4.yaml` notes describe the new vision path and its mmproj
  requirement; the existing sibling-mmproj pairing applies unchanged.
- `scanner._HYBRID_ARCHS` now contains the exact upstream strings
  `deepseek4`, `falcon-h1`, and `granitehybrid`. DeepSeek-V4 writes no
  `attention.recurrent_layers` key, so KV accounting deliberately stays
  full-layer (conservative), while the hybrid classification matches upstream.

### Remaining changes are transparent runtime improvements

No additional AutoTuner control is required for:

- `5ec4eab69` loader change that stages non-mmap tensors biggest-first and
  frees each staging buffer immediately (lower host-RAM peak for `--load-mode
  none`/`dio`; AutoTuner's RAM planning is already conservative);
- `9cffdcc80` server acceptance of `data:` URLs for `input_video` and
  `input_audio`;
- `0ba6499c3` CUDA concurrent streams per split (no new flag or environment
  variable), `8e93a9773` CUDA sparse FA for DSV4/GLM, `c7bda030e` Vulkan
  FA dequant path fix, Metal sparse FA and M3 tunings, SYCL peer-to-peer copy;
- `cff184438` CI-only ROCm 10.0 release update (AutoTuner's Windows HIP recipes
  keep the locally validated ROCm 7.2 toolchain; nothing in the recipe reads the
  upstream CI version);
- MTMD const-correctness and `mtmd_tokenize_from_parts()` API additions.

## Architecture classification review

The b10786 architecture table lists 149 names. AutoTuner's classifiers were
compared against `llm_arch_is_recurrent`, `llm_arch_is_hybrid`, and
`llm_arch_is_diffusion`:

| Upstream classifier | AutoTuner set | Result |
|---|---|---|
| recurrent (mamba, mamba2, rwkv6, rwkv6qwen2, rwkv7, arwkv7) | `_RECURRENT_ARCHS` | identical |
| hybrid (17 archs) | `_HYBRID_ARCHS` | now identical after adding `deepseek4`, `falcon-h1`, `granitehybrid`; the generic `*.ssm.*` fallback had already covered the latter two |
| diffusion (dream, llada, llada-moe, rnd1) | `_DIFFUSION_ARCHS` | identical plus the fork-only `diffusion-gemma` |
| drafters (eagle3, dflash, dspark) | `_DRAFTER_ARCHS` | identical |

Every architecture in the table therefore either has a dedicated profile, an
architecture fallback, or is a standard full-attention Transformer served by the
generic profile with full-layer KV accounting.

## Conclusion

AutoTuner's command/runtime feature layer is compatible with b10786 without
flag changes. The required v5.3.9 changes are the Nemotron Super/Puzzle
profiles with the b10786 gate, the DeepSeek-V4 vision note, exact hybrid
architecture names, and the reasoning-preserve default documentation.

Release-level evidence is recorded in
[`v5.3.9-validation.md`](v5.3.9-validation.md).
