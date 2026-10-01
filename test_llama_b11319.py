"""b11319's image/DFlash2 recheck bounds the historical combination gate."""

import json
from pathlib import Path

import pytest

import tuner
from test_llama_b11063 import _qwen_vision_dflash2


@pytest.mark.parametrize("build", [10896, 11249, 11302, 11318])
def test_unqualified_affected_builds_still_block_image_dflash2(
    tmp_path, monkeypatch, build
):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    with pytest.raises(ValueError, match="cannot reliably combine"):
        tuner.build_command(model, config, profile, draft_model=draft)


@pytest.mark.parametrize("build", [10895, 11319, 11320])
def test_pre_regression_and_qualified_new_builds_allow_image_dflash2(
    tmp_path, monkeypatch, build
):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    manifest = json.loads(
        (Path(__file__).parent / "docs/llama-b11319-server-flags.json").read_text()
    )
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: set(manifest["flags"])
    )
    command = tuner.build_command(model, config, profile, draft_model=draft)
    assert "--mmproj" in command
    assert command[command.index("--spec-type") + 1] == "draft-dflash"
    assert "-md" in command


def test_b11319_manifest_and_qualification_floor():
    root = Path(__file__).parent
    current = json.loads((root / "docs/llama-b11319-server-flags.json").read_text())
    previous = json.loads((root / "docs/llama-b11302-server-flags.json").read_text())
    assert current["tag"] == "b11319"
    assert current["commit"] == "3ec4df42d9c1d4de896c886ebc65fad2e6e29fa4"
    assert current["flags"] == previous["flags"]
    assert tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE == 10896
    assert tuner.QWEN35_VISION_DFLASH2_FIXED_SINCE == 11319
