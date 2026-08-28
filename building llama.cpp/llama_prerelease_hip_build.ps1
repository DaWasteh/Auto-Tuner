# Windows / HIP (ROCm) recipe for the newest published llama.cpp bNNNN release.
# Pandaking target: RX 9070 XT + AI PRO R9700, both gfx1201.
# Output: bNNNN_hip_llama.cpp (or bNNNN_dev_COMMIT_hip_llama.cpp for -Tag master).
# Run: pwsh -File '.\building llama.cpp\llama_prerelease_hip_build.ps1'

param(
    [ValidatePattern('^(latest|master|b\d+)$')]
    [string]$Tag = "latest",
    [string]$Workspace = "L:\LAB\ai-local",
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\7.2",
    [ValidateRange(1, 256)][int]$Parallel = 12
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPrereleaseBuild -Backend HIP -Tag $Tag -Workspace $Workspace -RocmPath $RocmPath -Parallel $Parallel
