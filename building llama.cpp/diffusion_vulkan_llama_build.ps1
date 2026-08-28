# Windows / Vulkan build of Diffusion-Gemma PR #24427.
# Source is pinned as of 2026-08-28 so the Vulkan/HIP siblings are identical.
# Output: d_bNNNN_vulkan_llama.cpp.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [ValidateRange(1, 256)][int]$Parallel = 20
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend Vulkan `
    -Name "Diffusion-Gemma PR #24427" `
    -RemoteUrl "https://github.com/ggml-org/llama.cpp.git" `
    -FetchRef "pull/24427/head" `
    -ExpectedCommit "dd0cf04459b0c4f43aa6667dbc0879ac0cd50323" `
    -FolderPrefix "d_" `
    -Workspace $Workspace `
    -Parallel $Parallel
