"""Opt-in, fail-closed b11319 Windows RDNA4 MoE tile workaround.

Only AMD proprietary-driver PCI 7550/7551 devices change dispatch; other
platforms/devices retain upstream PR #29182 behavior. No model weights changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BUILD = 11319
COMMIT = "3ec4df42d9c1d4de896c886ebc65fad2e6e29fa4"
SOURCE = "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
ORIGINAL_SHA256 = "a793c78d67f755c81fe861b2a9d145654055fe8ac351a974b6e6bae95bdde730"
RECEIPT = "autotuner-rdna4-moe-workaround.json"
PATCH_ID = "autotuner-rdna4-moe-v1"

ORIGINAL = """    const uint32_t n_per_expert = (uint32_t)CEIL_DIV(nei0 * nei1, n_as);
    const uint32_t kpad = quantize_y ? 0 : ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, n_per_expert, true));
    const bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && n_per_expert > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, n_per_expert, aligned, true);"""
PATCHED = """    const uint32_t n_per_expert = (uint32_t)CEIL_DIV(nei0 * nei1, n_as);
    // autotuner-rdna4-moe-v1: PR #29182 regresses tested Windows Navi48 boards.
    // Keep upstream dispatch on every other device/platform/driver.
#ifdef _WIN32
    const bool autotuner_rdna4_moe = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
        ctx->device->driver_id == vk::DriverId::eAmdProprietary &&
        (ctx->device->properties.deviceID == 0x7550u || ctx->device->properties.deviceID == 0x7551u);
#else
    const bool autotuner_rdna4_moe = false;
#endif
    const uint32_t n_tile = autotuner_rdna4_moe ? nei1 : n_per_expert;
    const uint32_t kpad = quantize_y ? 0 : ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, n_tile, true));
    const bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && n_tile > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, n_tile, aligned, true);"""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _validate_checkout(repo: Path) -> None:
    if _git(repo, "rev-parse", "HEAD") != COMMIT:
        raise ValueError(
            "RDNA4 workaround is qualified only for exact b11319 commit " + COMMIT
        )
    for args in [("diff", "--name-only"), ("diff", "--cached", "--name-only")]:
        changes = set(_git(repo, *args).splitlines()) - {SOURCE}
        if changes:
            raise ValueError(
                "Refusing unrelated tracked source changes: "
                + ", ".join(sorted(changes))
            )


def validate_source(repo: Path) -> tuple[Path, str, bool]:
    _validate_checkout(repo)
    path = repo / SOURCE
    # Git core.autocrlf varies by workstation; normalize only newline encoding.
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    is_patched = text.count(PATCHED) == 1
    original = text.replace(PATCHED, ORIGINAL, 1) if is_patched else text
    if (
        _digest(original.encode("utf-8")) != ORIGINAL_SHA256
        or original.count(ORIGINAL) != 1
    ):
        raise ValueError(
            "Unrecognized Vulkan source; refusing to overwrite user/upstream changes"
        )
    return path, text, is_patched


def apply(repo: Path) -> bool:
    path, text, is_patched = validate_source(repo)
    if is_patched:
        return False
    target = text.replace(ORIGINAL, PATCHED, 1)
    temporary = path.with_name(path.name + ".autotuner-tmp")
    if temporary.exists():
        raise ValueError(
            "Patch staging file already exists; refusing to overwrite: "
            + str(temporary)
        )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(target)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _binary(repo: Path) -> Path:
    return repo / "build/bin/Release/llama-server.exe"


def _receipt_data(repo: Path) -> dict:
    _, text, patched = validate_source(repo)
    if not patched:
        raise ValueError("RDNA4 workaround is not present in this source")
    return {
        "patch": PATCH_ID,
        "build": BUILD,
        "upstream_commit": COMMIT,
        "platform": "Windows",
        "driver": "AMD proprietary",
        "vendor_id": "1002",
        "device_ids": ["7550", "7551"],
        "source_sha256_lf": _digest(text.encode("utf-8")),
        "server_sha256": _file_digest(_binary(repo)),
    }


def record(repo: Path) -> None:
    data = _receipt_data(repo)
    path = repo / RECEIPT
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def verify(repo: Path) -> None:
    actual = json.loads((repo / RECEIPT).read_text(encoding="utf-8"))
    if actual != _receipt_data(repo):
        raise ValueError(
            "RDNA4 receipt does not match current source/binary; rebuild explicitly"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["apply", "record", "verify"])
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "apply":
            print(
                "RDNA4 patch applied"
                if apply(args.repo)
                else "RDNA4 patch already verified"
            )
        elif args.mode == "record":
            record(args.repo)
            print("RDNA4 build receipt recorded")
        else:
            verify(args.repo)
            print("RDNA4 source/binary receipt verified")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"RDNA4 workaround failed: {exc}\n")


if __name__ == "__main__":
    main()
