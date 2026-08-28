# Windows / Vulkan recipe for the newest stable llama.cpp X.Y.Z release.
# Output: X.Y.Z_vulkan_llama.cpp with the truthful compatibility build embedded.
# Run: pwsh -File '.\building llama.cpp\llama_stable_vulkan_build.ps1'

param(
    [ValidatePattern('^(latest|v?\d+\.\d+\.\d+)$')]
    [string]$Tag = "latest",
    [string]$Workspace = "L:\LAB\ai-local",
    [ValidateRange(1, 256)][int]$Parallel = 20
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaStableBuild -Backend Vulkan -Tag $Tag -Workspace $Workspace -Parallel $Parallel
