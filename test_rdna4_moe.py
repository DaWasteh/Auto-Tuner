"""Fail-closed recipe/source/receipt tests without network, GPU or CMake."""

import hashlib
import importlib.util
import json

import pytest

from test_build_recipes import ROOT, _run_pwsh


@pytest.fixture
def patcher():
    spec = importlib.util.spec_from_file_location(
        "rdna4_moe_patcher", ROOT / "building llama.cpp/patch_rdna4_moe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(tmp_path, patcher, monkeypatch):
    text = "// synthetic fixture\n" + patcher.ORIGINAL + "\n"
    monkeypatch.setattr(
        patcher, "ORIGINAL_SHA256", hashlib.sha256(text.encode()).hexdigest()
    )
    monkeypatch.setattr(
        patcher,
        "_git",
        lambda repo, *args: patcher.COMMIT if args == ("rev-parse", "HEAD") else "",
    )
    path = tmp_path / patcher.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode("utf8"))
    return tmp_path


def test_scope_is_windows_amd_proprietary_tested_pci_only(patcher):
    assert patcher.BUILD == 11319
    assert patcher.COMMIT == "3ec4df42d9c1d4de896c886ebc65fad2e6e29fa4"
    assert "#ifdef _WIN32" in patcher.PATCHED
    assert "vendor_id == VK_VENDOR_ID_AMD" in patcher.PATCHED
    assert "driver_id == vk::DriverId::eAmdProprietary" in patcher.PATCHED
    assert "deviceID == 0x7550u" in patcher.PATCHED
    assert "deviceID == 0x7551u" in patcher.PATCHED
    assert "const bool autotuner_rdna4_moe = false;" in patcher.PATCHED
    assert "autotuner_rdna4_moe ? nei1 : n_per_expert" in patcher.PATCHED


@pytest.mark.parametrize("crlf", [False, True])
def test_patch_is_atomic_and_idempotent(checkout, patcher, crlf):
    path = checkout / patcher.SOURCE
    if crlf:
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert patcher.apply(checkout)
    result = path.read_bytes()
    assert patcher.PATCHED.encode() in result
    assert not patcher.apply(checkout)
    assert path.read_bytes() == result
    assert not path.with_name(path.name + ".autotuner-tmp").exists()


def test_wrong_commit_never_modifies_source(checkout, patcher, monkeypatch):
    path = checkout / patcher.SOURCE
    before = path.read_bytes()
    monkeypatch.setattr(patcher, "_git", lambda repo, *args: "0" * 40)
    with pytest.raises(ValueError, match="only for exact b11319"):
        patcher.apply(checkout)
    assert path.read_bytes() == before


@pytest.mark.parametrize("staged", [False, True])
def test_unrelated_tracked_work_is_preserved(checkout, patcher, monkeypatch, staged):
    path = checkout / patcher.SOURCE
    before = path.read_bytes()

    def git(repo, *args):
        if args == ("rev-parse", "HEAD"):
            return patcher.COMMIT
        if ("--cached" in args) == staged:
            return "src/user-edited.cpp"
        return ""

    monkeypatch.setattr(patcher, "_git", git)
    with pytest.raises(ValueError, match="unrelated tracked source"):
        patcher.apply(checkout)
    assert path.read_bytes() == before


def test_source_drift_rejected_without_overwriting(checkout, patcher):
    path = checkout / patcher.SOURCE
    path.write_text(path.read_text() + "// other user change\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unrecognized Vulkan source"):
        patcher.apply(checkout)
    assert path.read_bytes() == before


def test_existing_patch_staging_file_is_preserved(checkout, patcher):
    path = checkout / patcher.SOURCE
    temporary = path.with_name(path.name + ".autotuner-tmp")
    temporary.write_text("user data")
    with pytest.raises(ValueError, match="staging file already exists"):
        patcher.apply(checkout)
    assert temporary.read_text() == "user data"
    assert patcher.ORIGINAL in path.read_text()


def test_receipt_binds_source_and_binary(checkout, patcher):
    patcher.apply(checkout)
    binary = patcher._binary(checkout)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic test binary")
    patcher.record(checkout)
    patcher.verify(checkout)
    receipt = json.loads((checkout / patcher.RECEIPT).read_text())
    assert receipt["device_ids"] == ["7550", "7551"]
    assert receipt["build"] == 11319
    binary.write_bytes(b"altered binary")
    with pytest.raises(ValueError, match="receipt does not match"):
        patcher.verify(checkout)


def test_cannot_record_an_unpatched_source(checkout, patcher):
    with pytest.raises(ValueError, match="not present"):
        patcher.record(checkout)
    assert not (checkout / patcher.RECEIPT).exists()


@pytest.mark.parametrize(
    "backend,tag", [("HIP", "b11319"), ("Vulkan", "b11320"), ("Vulkan", "master")]
)
def test_recipe_rejects_unqualified_mode_before_clone(backend, tag):
    result = _run_pwsh(
        f"""
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        function global:git {{ throw 'git must not run for unqualified mode' }}
        try {{
            Invoke-LlamaPrereleaseBuild -Backend {backend} -Tag {tag} -Rdna4MoeWorkaround
            throw 'invalid mode accepted'
        }} catch {{
            if ($_.Exception.Message -notlike '*qualified only for Vulkan -Tag b11319*') {{ throw }}
        }}
        'guard OK'
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "guard OK" in result.stdout


def test_wrapper_exposes_opt_in_without_changing_stock_default():
    wrapper = (
        ROOT / "building llama.cpp/llama_prerelease_vulkan_build.ps1"
    ).read_text()
    common = (ROOT / "building llama.cpp/windows_llama_build_common.ps1").read_text()
    assert "[switch]$Rdna4MoeWorkaround" in wrapper
    assert "-Rdna4MoeWorkaround:$Rdna4MoeWorkaround" in wrapper
    assert '"rdna4moe_${folderVersion}_${backendToken}_llama.cpp"' in common
    assert '"${folderVersion}_${backendToken}_llama.cpp"' in common
    assert "Mode verify" in common and "Mode record" in common
    assert "receipt mismatch" in common
