# Windows / HIP (ROCm) recipe for the newest stable llama.cpp X.Y.Z release.
# Pandaking target: RX 9070 XT + AI PRO R9700, both gfx1201.
# Output: X.Y.Z_hip_llama.cpp with the truthful compatibility build embedded.
# Run: pwsh -File '.\building llama.cpp\llama_stable_hip_build.ps1'

param(
    [ValidatePattern('^(latest|v?\d+\.\d+\.\d+)$')]
    [string]$Tag = "latest",
    [string]$Workspace = "L:\LAB\ai-local",
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\7.2",
    [ValidateRange(1, 256)][int]$Parallel = 12
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaStableBuild -Backend HIP -Tag $Tag -Workspace $Workspace -RocmPath $RocmPath -Parallel $Parallel
