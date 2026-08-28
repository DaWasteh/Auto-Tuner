# Windows / Vulkan build of PrismML Ternary/Bonsai (branch prism).
# Source is pinned as of 2026-08-28 so the Vulkan/HIP siblings are identical.
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
    -ExpectedCommit "e311ed38fe7ab8fb577a5435b049d48b7d040923" `
    -FolderPrefix "2b_" `
    -Workspace $Workspace `
    -ExtraCMakeArgs @("-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF") `
    -Parallel $Parallel
