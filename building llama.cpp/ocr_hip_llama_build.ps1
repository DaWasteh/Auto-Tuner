# Windows / HIP (ROCm) build of the reviewed legacy DeepSeek-OCR PR #17400.
# Unlimited-OCR uses current mainline instead; this fork remains for OCR v1.
# Output: ocr_b17400_hip_llama.cpp; Pandaking target gfx1201.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\7.2",
    [ValidateRange(1, 256)][int]$Parallel = 12
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend HIP `
    -Name "DeepSeek-OCR PR #17400" `
    -RemoteUrl "https://github.com/ggml-org/llama.cpp.git" `
    -FetchRef "pull/17400/head" `
    -ExpectedCommit "95cc5665859b49d7158c5c4abc9943adf109c6d5" `
    -FolderPrefix "ocr_" `
    -FixedIdentity "b17400" `
    -Workspace $Workspace `
    -RocmPath $RocmPath `
    -BuildUi $false `
    -Targets @("llama-server", "llama-mtmd-cli") `
    -ExtraCMakeArgs @("-DLLAMA_OPENSSL=OFF", "-DCPPHTTPLIB_OPENSSL_SUPPORT=OFF", "-DLLAMA_CURL=OFF") `
    -Parallel $Parallel
