"""b11302 audit regressions: authoritative MTP absence and canonical OCR tasks."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles
from test_llama_b11063 import _qwen_vision_dflash2

from ocr_workflow import ocr_model_preset
from scanner import ModelEntry
from test_smoke import _fake_model_md


@pytest.mark.parametrize("declares_nextn", [False, True])
def test_complete_tensor_scan_vetoes_mtp_filename(tmp_path, declares_nextn):
    metadata = {
        "general.architecture": "qwen35",
        "__mtp_scan__": "absent",
        "__tensor_scan_complete__": True,
    }
    if declares_nextn:
        metadata["qwen35.nextn_predict_layers"] = 1
    model = _fake_model_md(tmp_path, "Qwen3.6-27B-MTP-Q4_K_M", 16, metadata)
    assert not model.has_embedded_mtp
    assert not model.has_speculative_draft
    # A separate real head remains available; it is not embedded in the trunk.
    model.draft = tmp_path / "mtp-Qwen3.6-27B-Q4_K_M.gguf"
    assert model.has_speculative_draft
    assert not model.has_embedded_mtp


@pytest.mark.parametrize("scan", [None, "inconclusive", "found"])
def test_legacy_mtp_filename_fallback_survives_without_proven_absence(tmp_path, scan):
    metadata = {"general.architecture": "qwen35"}
    if scan is not None:
        metadata["__mtp_scan__"] = scan
    model = _fake_model_md(tmp_path, "Qwen3.6-27B-MTP-Q4_K_M", 16, metadata)
    assert model.has_embedded_mtp


@pytest.mark.parametrize(
    ("name", "arch", "prompt"),
    [
        ("GLM-OCR-Q8_0", "glm4", "Text Recognition:"),
        ("PaddleOCR-VL-1.6", "paddleocr", "OCR:"),
        ("PaddleOCR-VL-1.5", "paddleocr", "OCR:"),
        ("dots.ocr-Q8_0", "qwen2", "Extract the text content from this image."),
    ],
)
def test_ocr_defaults_follow_the_model_publishers(name, arch, prompt):
    model = ModelEntry(
        path=Path(name + ".gguf"),
        name=name,
        group="audit",
        size_bytes=1024,
        metadata={"general.architecture": arch},
    )
    assert ocr_model_preset(model).prompt == prompt


def test_b11302_cli_manifest_retains_the_b11249_contract():
    root = Path(__file__).resolve().parent
    current = json.loads((root / "docs/llama-b11302-server-flags.json").read_text())
    previous = json.loads((root / "docs/llama-b11249-server-flags.json").read_text())
    assert current["tag"] == "b11302"
    assert current["commit"].startswith("05af0d2b1")
    assert current["flags"] == previous["flags"]
    assert len(current["flags"]) == 416
    assert sum(flag.startswith("--") for flag in current["flags"]) == 329
    for backend in ("hip", "vulkan"):
        assert (
            current["binaries"][backend]["help_sha256"]
            == previous["binaries"][backend]["help_sha256"]
        )
    for profile in load_profiles(root / "settings"):
        assert all(
            arg.split("=", 1)[0] in current["flags"]
            for arg in profile.extra_args
            if arg.startswith("--")
        )


def test_b11302_recheck_does_not_lift_the_vision_dflash2_gate(tmp_path, monkeypatch):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11302)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    with pytest.raises(ValueError, match="b11302.*rechecked on b11302"):
        tuner.build_command(model, config, profile, draft_model=draft)
    assert tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE == 10896
