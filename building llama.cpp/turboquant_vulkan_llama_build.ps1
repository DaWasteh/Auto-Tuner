# Windows / Vulkan build of TheTom TurboQuant KV-cache fork.
# Source is pinned as of 2026-08-28 so the Vulkan/HIP siblings are identical.
# Output: tq_bNNNN_vulkan_llama.cpp.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [ValidateRange(1, 256)][int]$Parallel = 20
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend Vulkan `
    -Name "TurboQuant KV-cache" `
    -RemoteUrl "https://github.com/TheTom/llama-cpp-turboquant.git" `
    -ExpectedCommit "df7f5472949ce37cdc6a2155ef6b8836a8c10bac" `
    -FolderPrefix "tq_" `
    -Workspace $Workspace `
    -Parallel $Parallel
