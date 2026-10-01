# Opt-in Windows RDNA4 MoE build workaround

Stock llama.cpp b11302/b11319 regresses Vulkan MoE prompt throughput on the
locally tested R9700 and RX 9070 XT. Controlled source isolation identified
[PR #29182](https://github.com/ggml-org/llama.cpp/pull/29182). The upstream
optimization remains untouched on unqualified devices/platforms.

## Build and select

From the source checkout, with Python available on PATH and the normal
Windows Vulkan build prerequisites:

```powershell
pwsh -File '.\building llama.cpp\llama_prerelease_vulkan_build.ps1' `
  -Tag b11319 -Rdna4MoeWorkaround
```

Output: `L:\LAB\ai-local\rdna4moe_b11319_vulkan_llama.cpp`. Select this clearly
named Vulkan runtime in AutoTuner when using the workaround. Existing stock
builds and persisted runtime/settings selections are **not** replaced or
silently changed. Without the switch the stock recipe is unchanged.

## Fail-closed scope

- Only exact b11319 commit `3ec4df42d9c1d4de896c886ebc65fad2e6e29fa4` and the
  audited full Vulkan source SHA-256 are accepted. Other tags, master, HIP,
  source drift and unrelated tracked edits are rejected without overwriting.
- Runtime dispatch changes only on **Windows + AMD proprietary Vulkan driver
  + PCI vendor 1002/device 7550 or 7551** (tested RX 9070 XT and AI PRO R9700).
  Other GPUs/drivers/OSes keep upstream per-expert tile selection.
- Application is atomic/idempotent. A JSON receipt binds upstream commit,
  scope, patched-source SHA-256 and the built server SHA-256. Existing
  qualified-output mismatches fail rather than being silently restamped.
- Missing-receipt incomplete builds get a clean rebuild before qualification.
  Preserve the receipt beside the source tree; do not relabel this as stock.

## Verification and limits

The hardware-scoped build restores IQ3_XXS/MXFP4/Q6_K prompt throughput.
A fresh ABBA comparison against b11249 on both cards found differences of
**−0.4…+1.2%** for pp128/512/1024 and tg128 (normal run variation), instead
of the stock double-digit loss. Real AutoTuner server plans passed text/tools,
Gemma and native Qwen/GLM MTP, image+DFlash2, vision-only, OCR and Coder-Next.
The source and stock binaries remain preserved.

This is not a universal GPU/driver/model/context optimization. New upstream
versions need renewed source/runtime qualification; `-Tag latest` with the
switch deliberately fails if it resolves outside b11319. The current
[CLI/source audit](llama-b11319-audit.md) and
[release validation record](v5.6.0-validation.md) separate stock measurements,
the diagnostic revert and this maintained, inference-tested option.
