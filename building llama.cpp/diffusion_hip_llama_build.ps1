# Windows / HIP (ROCm) build of Diffusion-Gemma PR #24427.
# HIP avoids Vulkan's roughly 1 GiB single-allocation ceiling for this model.
# Source is pinned as of 2026-08-28; both Pandaking GPUs target gfx1201.
# Output: d_bNNNN_hip_llama.cpp.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\7.2",
    [ValidateRange(1, 256)][int]$Parallel = 12
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend HIP `
    -Name "Diffusion-Gemma PR #24427" `
    -RemoteUrl "https://github.com/ggml-org/llama.cpp.git" `
    -FetchRef "pull/24427/head" `
    -ExpectedCommit "dd0cf04459b0c4f43aa6667dbc0879ac0cd50323" `
    -FolderPrefix "d_" `
    -Workspace $Workspace `
    -RocmPath $RocmPath `
    -Parallel $Parallel
