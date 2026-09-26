# Windows / Vulkan build of PrismML Ternary/Bonsai (branch prism).
# Source is pinned to the fork release tag prism-b10743-adfffbe (2026-09-25,
# Ternary-Bonsai-2 PTQ1_0 + PQ2_0 kernels; first pin with a Vulkan PQ2_0
# dequant path and the PTQ1_0 integer-dot mat-vec) so the Vulkan/HIP siblings
# are identical. The folder carries the fork's own build number (10743, as
# printed by `llama-server --version`), not the mainline merge-base count.
# Output: 2b_bNNNN_vulkan_llama.cpp.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [ValidateRange(1, 256)][int]$Parallel = 20
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend Vulkan `
    -Name "PrismML Ternary/Bonsai" `
    -RemoteUrl "https://github.com/PrismML-Eng/llama.cpp.git" `
    -ExpectedCommit "adfffbe41b2cabcd51fff326ab045662265062bb" `
    -FixedIdentity "b10743" `
    -FolderPrefix "2b_" `
    -Workspace $Workspace `
    -ExtraCMakeArgs @("-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF") `
    -Parallel $Parallel
