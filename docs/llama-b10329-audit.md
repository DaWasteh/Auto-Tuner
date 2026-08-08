# llama.cpp b10329 integration audit

AutoTuner v5.1.0 was audited against the exact upstream tags below:

| Tag | Commit | Date |
|---|---|---|
| `b10151` | `8e8681e0e20820a7736960381d71dec06a830163` | 2026-07-27 |
| `b10329` | `18f7ad7fc912444acc0f51995a4b8e45fd9a0cd4` | 2026-08-08 |

The range contains **178 commits**. The local checkout used for runtime tests is
`L:/LAB/ai-local/b10329_llama.cpp`; it is one commit after the tag
(`687e77892`, build `10330`). That extra commit is CUDA-only fusion work
(`rms_norm + mul + rope`) and does not change the tested AMD/Vulkan MTMD/server
path. Source comparisons in this document always use the exact `b10329` tag.

## Integrated changes

| Upstream change | First build | AutoTuner integration |
|---|---:|---|
| DSpark speculative decoding (`draft-dspark`) | b10164 | DSpark filenames, the DFlash architecture plus Markov/confidence tensors, and future explicit `dspark` architecture values are classified as draft sidecars. AutoTuner emits `-md … --spec-type draft-dspark`. On older binaries it removes the entire DSpark path rather than silently running it as ordinary DFlash. |
| DSpark Hugging Face sidecar resolution | b10231 | Local `dspark-*` sibling matching follows the same sidecar convention. Hugging Face downloads remain llama.cpp's responsibility. |
| DeepSeek-OCR multi-row MTMD batching | b10285 | The OCR workflow uses the normal multimodal server path and the profile's large batch. No private endpoint or fork is required. |
| Unlimited-OCR `preproc_max_tiles` fix | b10287 | Unlimited-OCR has a minimum-build gate of b10287, verified `document parsing.` prompt, DeepSeek-OCR chat template, F16 KV cache, deterministic sampling, explicit Flash Attention off, and the b10329 DRY loop guard. AutoTuner also inspects the projector: a stale mmproj without `clip.vision.preproc_max_tiles=32` triggers an explicit reduced-quality warning because a new binary alone cannot add converter metadata. |
| Speculative acceptance counters in `/metrics` | b10282 | AutoTuner already enables `--metrics`; the new counters appear automatically without a new launcher flag. |
| MTP for Qwen3-Next, DeepSeek V3.2, GLM-4.7-Flash/GLM-5.2 | b101xx-b102xx | Existing embedded/sidecar MTP metadata detection and `draft-mtp` command generation apply. Tensor scanning still prevents metadata-only false positives. |
| EAGLE-3 v3 for GPT-OSS and DFlash fixes | b101xx-b102xx | Existing EAGLE-3/DFlash routing is retained; b10329 model-loader fixes are automatic runtime benefits. |
| Tool isolation with `--tools-runtime` | b10328 | The value-bearing flag is parsed, de-duplicated, and capability-pruned correctly in Extra CLI flags. It is deliberately not enabled by default because host/Docker tools are security-sensitive, not tuning controls. |
| Built-in `get_info` tool and `x-tool-cwd` | b102xx | Available whenever users explicitly enable llama.cpp built-in tools. AutoTuner does not grant them implicitly. |
| Stricter repeat/DRY history validation | b102xx | AutoTuner and the OCR profile emit non-negative history lengths and finite, positive repeat penalties. |
| Router LRU scheduler/model modality reporting | b103xx | These are automatic server/Web UI improvements. AutoTuner continues to tune one selected model per managed process because VRAM planning is model-specific. |
| Vulkan submission/device-lost diagnostics and quantized concat | b102xx | Automatic benefits for the Windows AMD/Vulkan build; no command-line contract changed. |

## OCR workflow

llama.cpp accepts images/audio/video through MTMD but does **not** directly read
PDF, Word, OpenDocument, presentation, or spreadsheet containers. AutoTuner now
provides the missing application workflow in both GUI and TUI:

1. Discover supported files without modifying the originals.
2. Convert Office files to PDF through a private, headless LibreOffice profile.
3. Render selected PDF pages with PyMuPDF at 72–600 DPI.
4. Normalize raster images with Pillow, including EXIF orientation and
   multi-frame TIFF/GIF pages, with pixel/request limits.
5. Start the selected OCR model only after conversion is complete, then verify
   both process liveness and the expected model alias through `/v1/models`.
6. Send each page to `/v1/chat/completions` with the image before the canonical
   OCR prompt (`<image>document parsing.` ordering).
7. Atomically write per-page output, a combined Markdown/text file, and a JSON
   manifest containing source hashes, settings, timing, errors, and a redacted
   server command.
8. Cancel cleanly during discovery, LibreOffice conversion, model loading, or
   HTTP inference; job-owned servers are stopped and temporary pages are removed
   unless the user asks to keep them.

The local Baidu files used for validation were:

- `I:/models/Baidu/Unlimited-OCR-BF16.gguf`
- `I:/models/Baidu/mmproj-Unlimited-OCR-F16.gguf`

Runtime checks covered the upstream `tools/mtmd/test-1.jpeg`, a real PDF, and a
real DOCX from `L:/RAW_ARCHIVE`. Originals were opened read-only; all generated
files were written under ignored `logs/` test folders. The locally supplied
F16 projector lacks the new max-tiles key, so those tests correctly exercised
llama.cpp's 9-tile fallback and AutoTuner now warns about that limitation. A
projector converted/re-uploaded with b10287+ is required for reference 32-tile
quality on tall or dense pages.

## Upstream changes that do not become AutoTuner controls

- **Default port notice:** b10329 only announces a future upstream change from
  8080 to 9931. It does not complete that change. AutoTuner continues to pass its
  explicit desktop default (`1234`), so no ambiguity exists.
- **TTS CLI breaking changes:** Qwen3-TTS support changes `llama-tts` options.
  AutoTuner launches `llama-server` (or its documented diffusion runners), not
  the separate TTS executable, and never emitted the removed vocoder arguments.
- **MCP/tool runtimes:** MCP config JSON and Docker tool isolation can execute
  external programs and alter the trust boundary. They remain explicit Extra
  CLI/API choices rather than automatic performance settings.
- **Backend/model loaders:** SYCL, CUDA, Metal, Vulkan, OpenCL, WebGPU, tokenizer,
  conversion, and architecture fixes require a new llama.cpp binary/model but no
  new AutoTuner knob.

## Compatibility policy

AutoTuner probes the **selected** executable, not a global assumed version.
Unsupported value-bearing flags are removed together with their values; aliases
are treated as one group; load-mode values are adapted for pre-b10151 builds;
and enum additions such as DSpark receive numeric build gates because ordinary
`--help` flag-name probing cannot validate an enum value.
