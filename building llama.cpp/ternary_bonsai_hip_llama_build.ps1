# Windows / HIP (ROCm) build of PrismML Ternary/Bonsai (branch prism).
# Source is pinned as of 2026-08-28; both Pandaking GPUs target gfx1201.
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
    -ExpectedCommit "e311ed38fe7ab8fb577a5435b049d48b7d040923" `
    -FolderPrefix "2b_" `
    -Workspace $Workspace `
    -RocmPath $RocmPath `
    -ExtraCMakeArgs @("-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF") `
    -Parallel $Parallel
