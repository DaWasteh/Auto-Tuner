# Windows / HIP (ROCm) build of the ROCmFPX fork of llama.cpp (ROCmFPX/ROCmFPX,
# formerly charlie12345/ROCmFPX): the only runtime that loads the ROCmFP4 /
# ROCmFPX weight formats (ggml types 100..111, e.g. kingjones777's
# Agnes-3.0-Flash-Preview-MTP-*-ROCmFP4-*.gguf; file_type 106 = ROCmFP4 Strix
# Lean). Source is pinned to the fork's main commit aed0d5fd9 (2026-09-06) so
# the Vulkan/HIP siblings are identical; both Pandaking GPUs target gfx1201
# (the fork's HIP MMQ table mmq-config-rdna3-5.cuh carries Q4_0_ROCMFP4 and
# _FAST entries for RDNA3+). Mainline llama.cpp merge base: b10766 (9400c8946).
# The folder carries the fork's own build number (11544, as printed by
# `llama-server --version`), so AutoTuner's numeric build gates see 11544 even
# though the upstream feature set is that of b10766.
# patches/rocmfpx-rdna4-mmq-fallback.patch: the fork's ROCmFPX MMQ fallback
# table (PR #20, issue #17) is only wired into the RDNA3 selector; without it
# every ROCmFP4 prompt batch on gfx1201 (RDNA4) aborts with "J_best=0" in
# mmq.cuh (reproduced 2026-09-19 with Agnes-3.0-Flash STRIX_LEAN on the R9700).
# The patch routes the RDNA4 selector to the same fallback; HEAD stays pinned.
# Output: fpx_b11544_hip_llama.cpp.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\7.2",
    [ValidateRange(1, 256)][int]$Parallel = 12
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend HIP `
    -Name "ROCmFPX" `
    -RemoteUrl "https://github.com/ROCmFPX/ROCmFPX.git" `
    -ExpectedCommit "aed0d5fd9620ee96a10cb4e6b16c18514ea370e1" `
    -FixedIdentity "b11544" `
    -FolderPrefix "fpx_" `
    -Workspace $Workspace `
    -RocmPath $RocmPath `
    -ExtraCMakeArgs @("-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF") `
    -PatchFiles @((Join-Path $PSScriptRoot "patches\rocmfpx-rdna4-mmq-fallback.patch")) `
    -Parallel $Parallel
