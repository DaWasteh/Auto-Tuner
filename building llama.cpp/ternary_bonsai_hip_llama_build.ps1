# Windows / HIP (ROCm) build of PrismML Ternary/Bonsai (branch prism).
# Source is pinned to the fork release tag prism-b10743-adfffbe (2026-09-25,
# Ternary-Bonsai-2 PTQ1_0 + PQ2_0 kernels, native PQ2_0 MMQ tile unpacking and
# vectorized PTQ1_0 vec_dot on HIP); both Pandaking GPUs target gfx1201.
# The folder carries the fork's own build number (10743, as printed by
# `llama-server --version`), not the mainline merge-base count.
# Output: 2b_bNNNN_hip_llama.cpp.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\7.2",
    [ValidateRange(1, 256)][int]$Parallel = 12
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend HIP `
    -Name "PrismML Ternary/Bonsai" `
    -RemoteUrl "https://github.com/PrismML-Eng/llama.cpp.git" `
    -ExpectedCommit "adfffbe41b2cabcd51fff326ab045662265062bb" `
    -FixedIdentity "b10743" `
    -FolderPrefix "2b_" `
    -Workspace $Workspace `
    -RocmPath $RocmPath `
    -ExtraCMakeArgs @("-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF") `
    -Parallel $Parallel
