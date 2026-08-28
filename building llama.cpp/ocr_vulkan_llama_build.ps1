# Windows / Vulkan build of the reviewed legacy DeepSeek-OCR PR #17400.
# Unlimited-OCR uses current mainline instead; this fork remains for OCR v1.
# Output: ocr_b17400_vulkan_llama.cpp.

param(
    [string]$Workspace = "L:\LAB\ai-local",
    [ValidateRange(1, 256)][int]$Parallel = 20
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_llama_build_common.ps1")
Invoke-LlamaPinnedForkBuild `
    -Backend Vulkan `
    -Name "DeepSeek-OCR PR #17400" `
    -RemoteUrl "https://github.com/ggml-org/llama.cpp.git" `
    -FetchRef "pull/17400/head" `
    -ExpectedCommit "95cc5665859b49d7158c5c4abc9943adf109c6d5" `
    -FolderPrefix "ocr_" `
    -FixedIdentity "b17400" `
    -Workspace $Workspace `
    -BuildUi $false `
    -Targets @("llama-server", "llama-mtmd-cli") `
    -ExtraCMakeArgs @("-DLLAMA_OPENSSL=OFF", "-DCPPHTTPLIB_OPENSSL_SUPPORT=OFF", "-DLLAMA_CURL=OFF") `
    -Parallel $Parallel
