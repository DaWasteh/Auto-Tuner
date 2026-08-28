param(
    [string]$Workspace = "L:\LAB\ai-local",
    [string]$PrereleaseTag = "latest",
    [string]$StableTag = "latest",
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\7.2",
    [ValidateRange(1, 256)][int]$VulkanParallel = 20,
    [ValidateRange(1, 256)][int]$HipParallel = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")

if ($PrereleaseTag -eq "latest") {
    $PrereleaseTag = Get-LatestLlamaPrereleaseTag
}
if ($StableTag -eq "latest") {
    $StableTag = Get-LatestStableSemanticTag
}
Write-Host "==> Frozen mainline inputs for this run: prerelease=$PrereleaseTag stable=$StableTag"
Write-Host "==> HIP SDK for this run: $RocmPath"

function Invoke-Recipe {
    param(
        [Parameter(Mandatory = $true)][string]$File,
        [string[]]$Arguments = @()
    )

    $path = Join-Path $PSScriptRoot $File
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Build recipe not found: $path"
    }
    Write-Host ""
    Write-Host "==============================================================================="
    Write-Host "==> $File $($Arguments -join ' ')"
    Write-Host "==============================================================================="
    & pwsh -NoProfile -File $path @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File failed with exit code $LASTEXITCODE"
    }
}

# Mainline channels. Resolve tags once above so Vulkan and HIP cannot drift.
Invoke-Recipe "llama_prerelease_vulkan_build.ps1" @("-Tag", $PrereleaseTag, "-Workspace", $Workspace, "-Parallel", "$VulkanParallel")
Invoke-Recipe "llama_prerelease_hip_build.ps1" @("-Tag", $PrereleaseTag, "-Workspace", $Workspace, "-RocmPath", $RocmPath, "-Parallel", "$HipParallel")
Invoke-Recipe "llama_stable_vulkan_build.ps1" @("-Tag", $StableTag, "-Workspace", $Workspace, "-Parallel", "$VulkanParallel")
Invoke-Recipe "llama_stable_hip_build.ps1" @("-Tag", $StableTag, "-Workspace", $Workspace, "-RocmPath", $RocmPath, "-Parallel", "$HipParallel")

# Pinned fork pairs. Each pair contains the same exact source commit.
foreach ($stem in ("diffusion", "ocr", "ternary_bonsai", "turboquant")) {
    Invoke-Recipe "${stem}_vulkan_llama_build.ps1" @("-Workspace", $Workspace, "-Parallel", "$VulkanParallel")
    Invoke-Recipe "${stem}_hip_llama_build.ps1" @("-Workspace", $Workspace, "-RocmPath", $RocmPath, "-Parallel", "$HipParallel")
}

Write-Host ""
Write-Host "All backend-qualified Windows llama.cpp builds completed successfully."
Write-Host "Old builds were intentionally preserved; remove them only after independent smoke/benchmark checks."
