# Windows / Vulkan recipe for the newest published llama.cpp bNNNN release.
# Output: bNNNN_vulkan_llama.cpp (or bNNNN_dev_COMMIT_vulkan_llama.cpp for -Tag master).
# Run: pwsh -File '.\building llama.cpp\llama_prerelease_vulkan_build.ps1'

param(
    [ValidatePattern('^(latest|master|b\d+)$')]
    [string]$Tag = "latest",
    [string]$Workspace = "L:\LAB\ai-local",
    [ValidateRange(1, 256)][int]$Parallel = 20
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPrereleaseBuild -Backend Vulkan -Tag $Tag -Workspace $Workspace -Parallel $Parallel
