import os
from pathlib import Path
import shutil
import subprocess
import textwrap
import time

import pytest


ROOT = Path(__file__).resolve().parent
COMMON_RECIPE = ROOT / "building llama.cpp" / "windows_llama_build_common.ps1"


def _common_recipe() -> str:
    return COMMON_RECIPE.read_text(encoding="utf-8")


def _pwsh() -> str:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell 7 is not installed on this runner")
    return executable


def _run_pwsh(
    script: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    process_env["AUTOTUNER_TEST_COMMON_RECIPE"] = str(COMMON_RECIPE)
    if env:
        process_env.update(env)
    return subprocess.run(
        [_pwsh(), "-NoProfile", "-Command", textwrap.dedent(script)],
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def test_windows_hip_recipe_blocks_peer_copy_and_checks_decoded_text() -> None:
    common = _common_recipe()

    assert '"-DGGML_CUDA_NO_PEER_COPY=ON"' in common
    assert "GGML_CUDA_NO_PEER_COPY:BOOL=ON" in common
    assert "Test-HipMultiGpuSemanticOutput" in common
    assert "Resolve-HipSemanticValidationModel" in common
    assert "AUTOTUNER_HIP_VERIFY_MODEL" in common
    assert '"HIP_VISIBLE_DEVICES"] = "0,1"' in common
    assert '"--tensor-split", "0.667,0.333"' in common
    assert 'expected = "HIP MULTI GPU OK"' in common
    assert "assistantContent.Trim() -ne $expected" in common
    assert "Stop-ProcessTreeBounded" in common
    assert "Get-TaskTextBounded" in common


def test_common_recipe_parses_in_powershell() -> None:
    result = _run_pwsh(
        """
        $null = [scriptblock]::Create(
            (Get-Content -LiteralPath $env:AUTOTUNER_TEST_COMMON_RECIPE -Raw)
        )
        "PowerShell parse OK"
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PowerShell parse OK" in result.stdout


def test_process_cleanup_helper_terminates_a_nonterminating_child() -> None:
    started = time.monotonic()
    result = _run_pwsh(
        """
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        $startInfo = [Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = (Get-Process -Id $PID).Path
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        foreach ($argument in @("-NoProfile", "-Command", "Start-Sleep -Seconds 300")) {
            [void]$startInfo.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) { throw "child did not start" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        try {
            Stop-ProcessTreeBounded -Process $process -TimeoutMilliseconds 5000
            [void](Get-TaskTextBounded -Task $stdoutTask -TimeoutMilliseconds 5000)
            [void](Get-TaskTextBounded -Task $stderrTask -TimeoutMilliseconds 5000)
            if (-not $process.HasExited) { throw "child is still running" }
        } finally {
            if (-not $process.HasExited) { $process.Kill($true) }
            $process.Dispose()
        }
        "bounded cleanup OK"
        """
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stdout + result.stderr
    assert "bounded cleanup OK" in result.stdout
    assert elapsed < 20


def test_failed_clone_cleans_every_pre_promotion_staging_tree(tmp_path: Path) -> None:
    result = _run_pwsh(
        """
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        function global:git {
            $rows = @($args)
            if ($rows -contains "ls-remote") {
                $global:LASTEXITCODE = 0
                return "1111111111111111111111111111111111111111`trefs/tags/v0.3.0"
            }
            if ($rows -contains "clone") {
                $destination = [string]$rows[-1]
                New-Item -ItemType Directory -Force -Path $destination | Out-Null
                $global:LASTEXITCODE = 23
                return
            }
            throw "unexpected mocked git call: $($rows -join ' ')"
        }

        $workspace = $env:AUTOTUNER_TEST_WORKSPACE
        $cases = @(
            { Invoke-LlamaPrereleaseBuild -Backend HIP -Tag b1234 -Workspace $workspace },
            { Invoke-LlamaStableBuild -Backend HIP -Tag v0.3.0 -Workspace $workspace },
            {
                Invoke-LlamaPinnedForkBuild `
                    -Backend HIP `
                    -Name "synthetic fork" `
                    -RemoteUrl "https://invalid.example/repo.git" `
                    -ExpectedCommit "2222222222222222222222222222222222222222" `
                    -FolderPrefix "synthetic_" `
                    -Workspace $workspace
            }
        )
        foreach ($case in $cases) {
            try {
                & $case
                throw "synthetic clone unexpectedly succeeded"
            } catch {
                if ($_.Exception.Message -notlike "*failed with exit code 23*") {
                    throw
                }
            }
            $leftovers = @(Get-ChildItem $workspace -Directory -Filter "_tmp_*")
            if ($leftovers.Count -ne 0) {
                throw "staging tree leaked: $($leftovers.FullName -join ', ')"
            }
        }
        "staging cleanup OK"
        """,
        env={"AUTOTUNER_TEST_WORKSPACE": str(tmp_path)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "staging cleanup OK" in result.stdout


def test_stable_recipe_accepts_prefixed_and_unprefixed_exact_tags() -> None:
    common = _common_recipe()

    assert "$remoteTag = $Tag" in common
    assert '$alternateTag = if ($Tag.StartsWith("v"))' in common
    assert "git clone --branch $remoteTag --single-branch" in common
    assert "Checkout is not the requested exact stable tag $remoteTag" in common
