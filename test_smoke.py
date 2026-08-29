"""Smoke tests for AutoTuner.

These tests don't need real GGUF models and run on any GitHub Actions
runner. They cover:
  - profile loading + pattern matching against real-world model names
  - mmproj pairing (longest-prefix) on a synthetic models tree
  - compute_config produces sensible values across hardware shapes
  - hardware detection doesn't crash on a runner without GPUs
"""

from __future__ import annotations

import copy
import os
import struct
import sys
import types
from pathlib import Path

import pytest

# Widget-level GUI checks run without requiring a real display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Make the project root importable when tests are run from the repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hardware import GPUInfo, SystemInfo, detect_system, format_system  # noqa: E402
from scanner import (  # noqa: E402
    ModelEntry,
    group_entries,
    metadata_has_embedded_mtp,
    scan_models,
)
from settings_loader import load_profiles, match_profile  # noqa: E402
from tuner import (  # noqa: E402
    build_command,
    compute_config,
    extract_params_billion,
    TunedConfig,
)


SETTINGS_DIR = ROOT / "settings"
_QT_TEST_APP = None


# ---------------------------------------------------------------------------
# Hardware detection


def test_detect_system_does_not_raise() -> None:
    info = detect_system()
    assert info.total_ram_gb > 0
    assert info.cpu_cores_logical >= 1
    # GitHub runners may have no detected GPUs — that's fine
    assert isinstance(info.gpus, list)


def test_format_system_produces_text() -> None:
    info = detect_system()
    out = format_system(info)
    expected_memory_label = "MEM:" if info.has_unified_memory else "RAM:"
    assert "CPU:" in out and expected_memory_label in out


# ---------------------------------------------------------------------------
# Profile loading + pattern matching


def test_all_profiles_load() -> None:
    profiles = load_profiles(SETTINGS_DIR)
    assert len(profiles) >= 8, "expected the 8 shipped YAML profiles"
    files = {p.source_file for p in profiles}
    assert "_default.yaml" in files
    assert "qwen2_5-3.yaml" in files
    assert "gemma-4.yaml" in files
    assert "devstral.yaml" in files


@pytest.mark.parametrize(
    "filename, expected_display",
    [
        ("Qwen3.5-9B-Q8_0", "Qwen3.5 / Qwen3.6 (Alibaba)"),
        ("Qwen3.6-27B-UD-Q3_K_XL", "Qwen3.5 / Qwen3.6 (Alibaba)"),
        ("Qwen3.6-35B-A3B-UD-IQ3_S", "Qwen3.5 / Qwen3.6 (Alibaba)"),
        ("Qwen3.8-27B-Q8_0", "Qwen3.8 (Alibaba)"),
        ("Qwen3.8-2.4T-A95B-UD-IQ2_XXS", "Qwen3.8 (Alibaba)"),
        (
            "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003",
            "Qwen3.8 Flash Next (Alibaba)",
        ),
        ("Gemma-4-26B-A4B-IQ3_M", "Gemma 4 (Google)"),
        ("gemma-4-E2B-it-BF16", "Gemma 4 (Google)"),
        ("Devstral-Small-2-24B-Instruct-2512-Q3_K_L", "Devstral (Mistral, code)"),
        ("Ministral-3-14B-Reasoning-2512-Q6_K", "Ministral 3 (Mistral, reasoning)"),
        ("Mistral-Medium-3.5-128B-UD-IQ3_XXS", "Mistral Medium 3.x"),
        ("m51Lab-MiniMax-M2.7-REAP-139B-A10B.i1-IQ3_M", "MiniMax-M2 (MiniMaxAI)"),
        ("Bonsai-8B", "Bonsai 8B (PrismML, 1-bit)"),
        ("Ternary-Bonsai-8B-Q2_0", "Ternary-Bonsai (PrismML, 1.58-bit)"),
        ("Archon-14B.Q6_K", "Frankenmerger / community merge"),
        ("voldemort-10b-dpo.Q8_0", "Frankenmerger / community merge"),
        ("Some-Random-LLM.gguf", "Generic / fallback"),
    ],
)
def test_pattern_matching(filename, expected_display) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile(filename, profiles)
    assert p.display_name == expected_display, (
        f"{filename!r} matched {p.display_name!r}, expected {expected_display!r}"
    )


def test_qwen38_profile_contract_and_precedence() -> None:
    profiles = load_profiles(SETTINGS_DIR)
    qwen38 = match_profile("Qwen3.8-2.4T-A95B-UD-IQ2_XXS", profiles, arch="qwen35moe")

    assert qwen38.source_file == "qwen3_8.yaml"
    assert qwen38.max_context == 262144
    assert qwen38.rope_scale_max_ctx == 1000000
    assert qwen38.sampling["chat"] == qwen38.sampling["coding"]
    assert qwen38.ngram_method == "ngram-map-k4v"
    assert "--jinja" in qwen38.extra_args
    assert "--reasoning-preserve" in qwen38.extra_args

    # The shared qwen35moe architecture cannot identify a 3.8 checkpoint by
    # itself, so opaque older-family re-quants must keep the existing fallback.
    older_fallback = match_profile(
        "Opaque-Qwen-Community-Merge.gguf", profiles, arch="qwen35moe"
    )
    assert older_fallback.source_file == "qwen3_5-3_6.yaml"


def test_qwen38_flash_next_profile_uses_qwen4exp_metadata_contract() -> None:
    from scanner import metadata_attention_layer_count
    from tuner import _moe_expert_count

    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003",
        profiles,
        arch="qwen4exp",
    )
    assert profile.source_file == "qwen3_8_flash_next.yaml"
    assert profile.max_context == 262144
    assert profile.rope_scale_enabled is False
    assert profile.rope_scale_max_ctx == 262144
    assert profile.sampling["chat"] == profile.sampling["coding"]
    assert profile.ngram_method == "ngram-map-k4v"
    assert profile.recommended_kv_quant == "q4_0"
    assert profile.performance_target == "safe"
    assert profile.min_llama_build == 10660
    assert match_profile("opaque-preview.gguf", profiles, "qwen4exp") is profile

    model = ModelEntry(
        path=Path("Qwen3.8-Flash-Next.gguf"),
        name="Qwen3.8-Flash-Next",
        group=".",
        size_bytes=32 * 1024**3,
        metadata={
            "general.architecture": "qwen4exp",
            "qwen4exp.block_count": 48,
            "qwen4exp.context_length": 262144,
            "qwen4exp.expert_count": 512,
            "qwen4exp.expert_used_count": 10,
            "qwen4exp.ssm.state_size": 128,
            "qwen4exp.full_attention_interval": 4,
        },
    )
    assert model.is_hybrid
    assert metadata_attention_layer_count(model.metadata) == 12
    assert _moe_expert_count(model) == 512


def test_qwen4exp_kv_includes_qsa_indexer_and_metadata_precedence() -> None:
    from scanner import metadata_attention_layer_count
    from tuner import kv_per_token_parts_mb_from_metadata

    metadata = {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 48,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.full_attention_interval": 4,
    }
    key_mb, value_mb = kv_per_token_parts_mb_from_metadata(metadata)
    assert key_mb == pytest.approx(15_360 / 1024**2)
    assert value_mb == pytest.approx(18_432 / 1024**2)

    # b10666 gives the explicit recurrent list precedence over the interval.
    metadata["qwen4exp.attention.recurrent_layers"] = [True] * 24 + [False] * 24
    assert metadata_attention_layer_count(metadata) == 24


def test_read_lazy_table_is_removed_from_weight_placement() -> None:
    from scanner import _read_lazy_tensor_span_bytes

    five_gib = 5 * 1024**3
    assert (
        _read_lazy_tensor_span_bytes(
            [("per_layer_token_embd.weight", 64), ("next.weight", 64 + five_gib)],
            data_start=4096,
            file_size=4096 + 64 + five_gib + 1024,
        )
        == five_gib
    )

    model = ModelEntry(
        path=Path("qwen4exp.gguf"),
        name="qwen4exp",
        group=".",
        size_bytes=68 * 1024**3,
        metadata={"__read_lazy_tensor_bytes__": 27 * 1024**3},
    )
    assert model.read_lazy_size_gb == pytest.approx(27.0)
    assert model.placement_size_gb == pytest.approx(41.0)


def test_new_b10666_model_profiles_are_present() -> None:
    profiles = load_profiles(SETTINGS_DIR)
    nanbeige = match_profile("Nanbeige4.2-3B-Q8_0", profiles, "nanbeige")
    nemotron = match_profile(
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M",
        profiles,
        "nemotron_h",
    )
    assert nanbeige.source_file == "nanbeige-4_2.yaml"
    assert nanbeige.max_context == 262144
    assert nanbeige.min_llama_build == 10153
    assert nemotron.source_file == "nemotron-3_5.yaml"
    assert nemotron.max_context == 1048576
    assert nemotron.min_llama_build == 10665
    assert nemotron.draft_max == 7
    assert nemotron.draft_p_min == pytest.approx(0.0)


def test_ministral_does_not_collide_with_mistral_medium() -> None:
    """The ministral pattern has no overlap with mistral-medium."""
    profiles = load_profiles(SETTINGS_DIR)
    assert (
        match_profile("Ministral-3-14B", profiles).display_name
        == "Ministral 3 (Mistral, reasoning)"
    )
    assert (
        match_profile("Mistral-Medium-3.5-128B", profiles).display_name
        == "Mistral Medium 3.x"
    )


# ---------------------------------------------------------------------------
# Param extraction


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Qwen3.5-9B-Q8_0", 9.0),
        ("Qwen3.6-35B-A3B-UD-IQ3_S", 35.0),  # MoE: total params, not active
        ("Qwen3.8-27B-Q8_0", 27.0),
        ("Qwen3.8-2.4T-A95B-UD-IQ2_XXS", 2400.0),
        ("Mistral-Medium-3.5-128B-UD-IQ3_XXS", 128.0),
        ("gemma-4-E2B-it-BF16", 2.0),  # Gemma "effective" size
        ("gemma-4-E4B-it-Q8_0", 4.0),
        ("Qwen3.5-0.8B-Q8_0", 0.8),
    ],
)
def test_extract_params_billion(name, expected) -> None:
    assert extract_params_billion(name) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Scanner + mmproj pairing


def _write_minimal_gguf(path: Path) -> None:
    """Write a valid empty-metadata GGUF header so the scanner accepts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 0))  # kv count


def test_scanner_pairs_mmproj_by_size(tmp_path) -> None:
    """Each Qwen3.5 sub-model must get its own size-matched mmproj."""
    folder = tmp_path / "Alibaba" / "Qwen3.5"
    files = [
        "mmproj-Qwen3.5-0.8B-BF16.gguf",
        "Qwen3.5-0.8B-Q8_0.gguf",
        "mmproj-Qwen3.5-2B-BF16.gguf",
        "Qwen3.5-2B-Q8_0.gguf",
        "mmproj-Qwen3.5-9B-BF16.gguf",
        "Qwen3.5-9B-Q8_0.gguf",
    ]
    for f in files:
        _write_minimal_gguf(folder / f)

    entries = scan_models(tmp_path)
    by_name = {e.name: e for e in entries}

    expected = [
        ("Qwen3.5-0.8B-Q8_0", "mmproj-Qwen3.5-0.8B-BF16.gguf"),
        ("Qwen3.5-2B-Q8_0", "mmproj-Qwen3.5-2B-BF16.gguf"),
        ("Qwen3.5-9B-Q8_0", "mmproj-Qwen3.5-9B-BF16.gguf"),
    ]
    for stem, expected_name in expected:
        mmproj = by_name[stem].mmproj
        assert mmproj is not None, f"{stem} should have been paired with an mmproj"
        assert mmproj.name == expected_name


def test_scanner_skips_mmproj_from_main_list(tmp_path) -> None:
    _write_minimal_gguf(tmp_path / "MyModel-Q8_0.gguf")
    _write_minimal_gguf(tmp_path / "mmproj-MyModel-F16.gguf")
    entries = scan_models(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "MyModel-Q8_0"
    assert entries[0].has_vision


def test_scanner_hides_large_mtp_prefixed_draft(tmp_path, monkeypatch) -> None:
    """A leading mtp- identifies an external head even above the generic
    size guard; large infix -MTP- targets must remain runnable models."""
    import scanner

    large_size = int(2.9 * 1024**3)
    assert scanner._is_draft_filename("mtp-Tess-4-27B-BF16.gguf", large_size)
    assert not scanner._is_draft_filename("Tess-4-27B-MTP-Q4_K_M.gguf", large_size)

    # Exercise the complete scan without creating a multi-GiB fixture.
    monkeypatch.setattr(scanner, "_DRAFT_MAX_SIZE_BYTES", 32)
    target = tmp_path / "Tess-4-27B-Q4_K_M.gguf"
    draft = tmp_path / "mtp-Tess-4-27B-BF16.gguf"
    _write_minimal_gguf(target)
    _write_minimal_gguf(draft)
    with draft.open("ab") as f:
        f.write(b"\0" * 64)

    entries = scan_models(tmp_path)
    assert [entry.name for entry in entries] == ["Tess-4-27B-Q4_K_M"]
    assert entries[0].draft == draft
    assert entries[0].folder_drafts == [draft]


def test_scanner_handles_empty_folder(tmp_path) -> None:
    assert scan_models(tmp_path) == []


def test_scanner_persists_and_invalidates_metadata_cache(
    tmp_path, monkeypatch
) -> None:
    import scanner

    cache_path = tmp_path / "cache" / "metadata.json"
    model = tmp_path / "CachedModel-Q4_K_M.gguf"
    _write_minimal_gguf(model)
    monkeypatch.setenv("AUTOTUNER_METADATA_CACHE", str(cache_path))
    scanner._reset_metadata_cache_for_tests()
    original = scanner._read_gguf_metadata_uncached
    calls = 0

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(scanner, "_read_gguf_metadata_uncached", counted)
    first = scanner.read_gguf_metadata(model)
    second = scanner.read_gguf_metadata(model)
    assert first == second
    assert calls == 1
    scanner.flush_gguf_metadata_cache()
    assert cache_path.is_file()

    # Simulate the next application process: unchanged metadata must come from
    # disk, while a changed file signature must invoke the parser again.
    scanner._reset_metadata_cache_for_tests()
    assert scanner.read_gguf_metadata(model) == first
    assert calls == 1
    with model.open("ab") as handle:
        handle.write(b"\0")
    assert scanner.read_gguf_metadata(model) == first
    assert calls == 2
    scanner._reset_metadata_cache_for_tests()


def test_scanner_reads_independent_headers_in_parallel(tmp_path, monkeypatch) -> None:
    import threading
    import time

    import scanner

    for index in range(8):
        _write_minimal_gguf(tmp_path / f"Model-{index:02d}-Q4_K_M.gguf")
    monkeypatch.setenv("AUTOTUNER_METADATA_CACHE", "off")
    monkeypatch.setenv("AUTOTUNER_SCAN_WORKERS", "4")
    scanner._reset_metadata_cache_for_tests()
    original = scanner.read_gguf_metadata
    worker_names: set[str] = set()

    def delayed(path):
        worker_names.add(threading.current_thread().name)
        time.sleep(0.02)
        return original(path)

    monkeypatch.setattr(scanner, "read_gguf_metadata", delayed)
    entries = scanner.scan_models(tmp_path)
    assert [entry.name for entry in entries] == [
        f"Model-{index:02d}-Q4_K_M" for index in range(8)
    ]
    assert len(worker_names) >= 2
    scanner._reset_metadata_cache_for_tests()


def test_group_entries_buckets_by_folder(tmp_path) -> None:
    _write_minimal_gguf(tmp_path / "Vendor1" / "ModelA.gguf")
    _write_minimal_gguf(tmp_path / "Vendor2" / "ModelB.gguf")
    entries = scan_models(tmp_path)
    groups = group_entries(entries)
    assert "Vendor1" in groups
    assert "Vendor2" in groups


def test_model_folder_parts_preserve_nested_storage_and_separate_roots(
    tmp_path,
) -> None:
    from qt_launcher import _model_folder_parts

    root_a = tmp_path / "models-a"
    root_b = tmp_path / "models-b"
    entry = _fake_model(root_a / "Alibaba" / "Qwen3.6", "Qwen3.6-27B", 1.0)
    entry.group = "Alibaba/Qwen3.6"

    assert _model_folder_parts(entry, [root_a]) == ("Alibaba", "Qwen3.6")
    assert _model_folder_parts(entry, [root_a, root_b]) == (
        "models-a",
        "Alibaba",
        "Qwen3.6",
    )


def test_model_list_sorts_favorites_first_without_losing_group_order(tmp_path) -> None:
    import app_settings
    from qt_launcher import _sort_model_entries

    alpha = _fake_model(tmp_path, "Alpha", 1.0)
    beta = _fake_model(tmp_path, "Beta", 1.0)
    gamma = _fake_model(tmp_path, "Gamma", 1.0)
    alpha.group = "VendorB"
    beta.group = "VendorA"
    gamma.group = "VendorA"

    favorite_keys = {
        app_settings.favorite_model_key(alpha.path),
        app_settings.favorite_model_key(gamma.path),
    }
    ordered = _sort_model_entries([alpha, gamma, beta], favorite_keys)
    assert [entry.name for entry in ordered] == ["Gamma", "Alpha", "Beta"]


def test_favorite_star_click_selects_row_and_emits_both_states(tmp_path) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_core = pytest.importorskip("PyQt6.QtCore")
    qt_test = pytest.importorskip("PyQt6.QtTest")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

    model_list = qt_widgets.QListWidget()
    delegate = qt_launcher._FavoriteStarDelegate(model_list)
    model_list.setItemDelegate(delegate)
    first_entry = _fake_model(tmp_path, "FirstModel", 1.0)
    target_entry = _fake_model(tmp_path, "FavoriteTarget", 1.0)
    first_item = qt_widgets.QListWidgetItem(first_entry.name)
    first_item.setData(qt_core.Qt.ItemDataRole.UserRole, first_entry)
    first_item.setData(qt_launcher._FAVORITE_ROLE, False)
    target_item = qt_widgets.QListWidgetItem(target_entry.name)
    target_item.setData(qt_core.Qt.ItemDataRole.UserRole, target_entry)
    target_item.setData(qt_launcher._FAVORITE_ROLE, False)
    model_list.addItem(first_item)
    model_list.addItem(target_item)
    model_list.setCurrentItem(first_item)
    model_list.resize(400, 100)
    model_list.show()
    _QT_TEST_APP.processEvents()

    toggles = []
    delegate.favoriteToggled.connect(
        lambda toggled_entry, favorite: toggles.append((toggled_entry.name, favorite))
    )
    rect = model_list.visualItemRect(target_item)
    star_position = qt_core.QPoint(rect.left() + 12, rect.center().y())
    qt_test.QTest.mouseClick(
        model_list.viewport(),
        qt_core.Qt.MouseButton.LeftButton,
        pos=star_position,
    )
    assert model_list.currentItem() is target_item
    assert toggles == [("FavoriteTarget", True)]

    target_item.setData(qt_launcher._FAVORITE_ROLE, True)
    qt_test.QTest.mouseClick(
        model_list.viewport(),
        qt_core.Qt.MouseButton.LeftButton,
        pos=star_position,
    )
    assert toggles[-1] == ("FavoriteTarget", False)
    model_list.close()


def test_model_tree_favorite_interaction_is_deferred_and_crash_safe(
    tmp_path, monkeypatch
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )

    qt_launcher._run_model_tree_interaction_smoke(_QT_TEST_APP, SETTINGS_DIR)


# ---------------------------------------------------------------------------
# Tuner


def _fake_system(
    ram_total: float = 64,
    ram_free: float = 48,
    vram_total: float = 24,
    vram_free: float = 22,
    vendor: str = "amd",
    unified: bool = False,
    runtime_backend: str | None = None,
    runtime_index: int | None = None,
):
    """Build a synthetic SystemInfo for tuner tests."""
    return SystemInfo(
        os_name="Linux test",
        cpu_name="Test CPU",
        cpu_cores_physical=16,
        cpu_cores_logical=32,
        total_ram_gb=ram_total,
        free_ram_gb=ram_free,
        gpus=[
            GPUInfo(
                index=0,
                name="Test GPU",
                vendor=vendor,
                total_vram_mb=int(vram_total * 1024),
                free_vram_mb=int(vram_free * 1024),
                unified_memory=unified,
                is_integrated=True if unified else False,
                runtime_backend=runtime_backend,
                runtime_device=(
                    f"{runtime_backend}{runtime_index}"
                    if runtime_backend is not None and runtime_index is not None
                    else None
                ),
                hip_index=runtime_index,
            )
        ]
        if vram_total > 0
        else [],
    )


def _fake_model(tmp_path, name, size_gb):
    p = tmp_path / f"{name}.gguf"
    _write_minimal_gguf(p)
    from scanner import ModelEntry

    return ModelEntry(
        path=p,
        name=name,
        group=".",
        size_bytes=int(size_gb * 1024**3),
    )


def test_small_model_full_offload(tmp_path) -> None:
    """A small model on a big GPU → full offload, generous context."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    assert cfg.full_offload is True
    assert cfg.ngl == 999
    assert cfg.ctx >= 32768


def test_huge_model_falls_back_to_partial_or_cpu(tmp_path) -> None:
    """A 50 GB model on a 24 GB GPU → not a full offload."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=50.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(ram_total=96, ram_free=80), profile)
    assert cfg.full_offload is False
    assert cfg.ctx >= 2048  # always at least the floor


def _mistral_dense_md(tmp_path, size_gb):
    """A realistic dense hybrid model (Mistral-Medium: 88 layers, GQA-8)."""
    return _fake_model_md(
        tmp_path,
        "Mistral-Medium-3.5-128B-UD-Q3_K_XL",
        size_gb,
        {
            "general.architecture": "mistral3",
            "mistral3.block_count": 88,
            "mistral3.context_length": 262144,
            "mistral3.embedding_length": 12288,
            "mistral3.attention.head_count": 96,
            "mistral3.attention.head_count_kv": 8,
        },
    )


def test_dense_too_big_tier_context_is_monotonic(tmp_path) -> None:
    """A dense model larger than VRAM must give MORE context on safe than on
    balanced than on throughput. Before the fix the dense path packed weights
    until VRAM was full (no KV reservation) and divided context by n_parallel,
    so 'safe' paradoxically produced the SMALLEST context."""
    from performance_target import get_target

    profiles = load_profiles(SETTINGS_DIR)
    model = _mistral_dense_md(tmp_path, size_gb=57.0)
    profile = match_profile(model.name, profiles)
    # 48 GB VRAM (16+32), 45 GB RAM — the reporter's system.
    sys_info = _fake_system(
        ram_total=45.2, ram_free=38.0, vram_total=47.8, vram_free=46.2
    )

    ctxs = {}
    for tier in ("safe", "balanced", "throughput"):
        cfg = compute_config(model, sys_info, profile, perf_target=get_target(tier))
        ctxs[tier] = cfg.ctx
        assert cfg.n_parallel == 1, f"{tier} should default to 1 slot"

    assert ctxs["safe"] > ctxs["balanced"] > ctxs["throughput"], ctxs
    # All three must clear the old broken 3k/12k and actually use the RAM.
    assert ctxs["throughput"] >= 16384, ctxs


def test_dense_that_fits_vram_keeps_full_offload(tmp_path) -> None:
    """A dense model that FITS in VRAM must not have weights spilled to CPU by
    the KV reservation — it stays full-offload and draws KV from leftover VRAM
    (the reservation only trades weights for KV when the model is too big)."""
    from performance_target import get_target

    profiles = load_profiles(SETTINGS_DIR)
    model = _mistral_dense_md(tmp_path, size_gb=20.0)  # fits in 48 GB
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(
        ram_total=45.2, ram_free=38.0, vram_total=47.8, vram_free=46.2
    )
    cfg = compute_config(model, sys_info, profile, perf_target=get_target("safe"))
    assert cfg.full_offload is True
    assert cfg.ngl == 999


def test_no_gpu_falls_back_to_cpu(tmp_path) -> None:
    """No GPU → ngl=0, no full_offload."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(vram_total=0, vram_free=0), profile)
    assert cfg.ngl == 0
    assert cfg.full_offload is False


def test_user_ctx_override_wins(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile, user_ctx=8192)
    assert cfg.ctx == 8192


def test_nvidia_auto_kv_stays_symmetric_for_cuda_flash_attention(tmp_path) -> None:
    """NVIDIA CUDA builds default GGML_CUDA_FA_ALL_QUANTS=OFF, and b9888
    validates V-cache types for FlashAttention too. Since AutoTuner emits
    -fa on, automatic K/V asymmetry must stay off on NVIDIA unless the user
    explicitly pins it in Expert mode.
    """
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-27B-UD-Q8_K_XL", size_gb=18.0)
    profile = match_profile(model.name, profiles)

    amd_cfg = compute_config(
        model,
        _fake_system(
            ram_total=128,
            ram_free=100,
            vram_total=64,
            vram_free=64,
            vendor="amd",
        ),
        profile,
    )
    nvidia_cfg = compute_config(
        model,
        _fake_system(
            ram_total=128,
            ram_free=100,
            vram_total=64,
            vram_free=64,
            vendor="nvidia",
        ),
        profile,
    )

    # v5.3.2 uses one backend-friendly, symmetric capacity default everywhere;
    # manual Expert pins remain the escape hatch for higher precision.
    assert (amd_cfg.cache_k, amd_cfg.cache_v) == ("q4_0", "q4_0")
    assert (nvidia_cfg.cache_k, nvidia_cfg.cache_v) == ("q4_0", "q4_0")


def test_devstral_uses_high_context_when_ram_is_plenty(tmp_path) -> None:
    """Regression test for the v1 bug: Devstral was capped at 16k context
    even with tons of free RAM. With a roomy system it must now reach far
    above that."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(
        tmp_path, "Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL", size_gb=13.5
    )
    profile = match_profile(model.name, profiles)
    cfg = compute_config(
        model,
        _fake_system(ram_total=96, ram_free=71, vram_total=24, vram_free=22.8),
        profile,
    )
    assert cfg.ctx > 16384, f"v1 bug regression — got only {cfg.ctx}"


# ---------------------------------------------------------------------------
# Command builder


def test_build_command_includes_essentials(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, port=12345)
    assert "-m" in cmd and str(model.path) in cmd
    assert "-c" in cmd and str(cfg.ctx) in cmd
    assert "-ngl" in cmd
    assert "--port" in cmd and "12345" in cmd
    assert "-ctk" in cmd and "-ctv" in cmd
    assert "--perf" in cmd
    assert "--metrics" in cmd


def test_build_command_emits_explicit_load_modes(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)

    cfg.load_mode = "mlock"
    cmd = build_command(model, cfg, profile)
    idx = cmd.index("--load-mode")
    assert cmd[idx + 1] == "mlock"
    assert "--mlock" not in cmd and "--no-mmap" not in cmd

    cfg.load_mode = "mmap+mlock"
    cmd = build_command(model, cfg, profile)
    idx = cmd.index("--load-mode")
    assert cmd[idx + 1] == "mmap+mlock"


def test_build_command_dedupes_value_flag_pair_in_extras(tmp_path) -> None:
    """A duplicate --load-mode in Extras must not leak its value token.

    When the dropdown selects a load mode and the free-form Extras field
    repeats --load-mode with a different value, only the modeled pair
    survives; the stray value must not be appended as a bare token."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cfg.load_mode = "mlock"
    cfg.extra_cli_flags = ["--load-mode", "none"]
    cmd = build_command(model, cfg, profile, extra_args=["--load-mode", "dio"])
    # The modeled pair is present exactly once.
    assert cmd.count("--load-mode") == 1
    idx = cmd.index("--load-mode")
    assert cmd[idx + 1] == "mlock"
    # No stray value tokens leaked from the duplicate Extras entries.
    assert "none" not in cmd[idx + 1 :]
    assert "dio" not in cmd[idx + 1 :]


def test_load_mode_b10151_compatibility_adaptation(monkeypatch) -> None:
    import tuner

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10114)
    legacy, notes = tuner._adapt_load_mode_for_binary(
        ["llama-server", "--load-mode", "mmap+mlock", "-m", "model.gguf"]
    )
    assert legacy[1:3] == ["--load-mode", "mlock"]
    assert "legacy equivalent" in notes[0]

    unavailable, notes = tuner._adapt_load_mode_for_binary(
        ["llama-server", "--load-mode", "mlock", "-m", "model.gguf"]
    )
    assert unavailable == ["llama-server", "-m", "model.gguf"]
    assert "requires b10151+" in notes[0]

    # Inline form (--load-mode=MODE / -lm=MODE) is adapted the same way.
    inline_legacy, notes = tuner._adapt_load_mode_for_binary(
        ["llama-server", "--load-mode=mmap+mlock", "-m", "model.gguf"]
    )
    assert inline_legacy == ["llama-server", "--load-mode=mlock", "-m", "model.gguf"]
    assert "legacy equivalent" in notes[0]

    inline_removed, notes = tuner._adapt_load_mode_for_binary(
        ["llama-server", "-lm=mlock", "-m", "model.gguf"]
    )
    assert inline_removed == ["llama-server", "-m", "model.gguf"]
    assert "requires b10151+" in notes[0]

    # b10151+ leaves both locking modes untouched.
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10151)
    current, notes = tuner._adapt_load_mode_for_binary(
        ["llama-server", "--load-mode", "mmap+mlock"]
    )
    assert current == ["llama-server", "--load-mode", "mmap+mlock"]
    assert notes == []


def test_effective_load_mode_normalizes_legacy_and_garbage(tmp_path) -> None:
    from tuner import TunedConfig, effective_load_mode

    cfg = TunedConfig(
        ctx=2048,
        ngl=99,
        threads=4,
        batch_threads=4,
        batch=2048,
        ubatch=512,
        cache_k="f16",
        cache_v="f16",
        flash_attn=True,
    )
    # Explicit modes pass through and are case-normalized.
    for mode in ("none", "mmap", "mlock", "mmap+mlock", "dio"):
        cfg.load_mode = mode.upper()
        assert effective_load_mode(cfg) == mode
    # Legacy coupled booleans migrate to the b10151 split semantics.
    cfg.load_mode = "auto"
    cfg.mlock, cfg.no_mmap = True, False
    assert effective_load_mode(cfg) == "mmap+mlock"
    cfg.mlock, cfg.no_mmap = True, True
    assert effective_load_mode(cfg) == "mlock"
    cfg.mlock, cfg.no_mmap = False, True
    assert effective_load_mode(cfg) == "none"
    cfg.mlock, cfg.no_mmap = False, False
    assert effective_load_mode(cfg) is None
    # Garbage falls back to the legacy-boolean path, then None.
    cfg.load_mode = "nonsense"
    assert effective_load_mode(cfg) is None


def test_build_command_passes_extra_args(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, extra_args=["--metrics", "--log-disable"])
    assert "--metrics" in cmd
    assert "--log-disable" in cmd


def test_build_command_metrics_and_slots_toggles(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)

    cfg.metrics_enabled = False
    cfg.slots_api_enabled = False
    cmd_off = build_command(model, cfg, profile)
    assert "--metrics" not in cmd_off
    assert "--no-slots" in cmd_off
    assert "--slots" not in cmd_off

    cfg.slots_api_enabled = True
    cmd_on = build_command(model, cfg, profile)
    assert "--slots" in cmd_on
    assert "--no-slots" not in cmd_on


def test_parse_llama_build_number_supports_current_and_legacy_output() -> None:
    from tuner import _parse_llama_build_number

    current = (
        "version: 0.2.0-dev (build 10572, commit 3af988fab)\n"
        "built with MSVC 19.51.36256.0 for x64"
    )
    assert _parse_llama_build_number(current) == 10572
    assert (
        _parse_llama_build_number(
            "version: 0.2.0 (build 10566, commit bb4caa754)\nbuilt with MSVC 19.51"
        )
        == 10566
    )
    assert (
        _parse_llama_build_number("version: 10058 (788e07dc9)\nbuilt with MSVC 19.51")
        == 10058
    )
    assert _parse_llama_build_number("  version: b10056 (b85833e)\n") == 10056
    assert _parse_llama_build_number("version: 0.2.0-dev (commit 3af988fab)") is None
    assert _parse_llama_build_number("built with MSVC 19.51") is None


def test_vision_prompt_cache_is_build_gated(tmp_path, monkeypatch) -> None:
    import tuner

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "gemma-4-12b-it-qat-q4_0", size_gb=6.5)
    mmproj = tmp_path / "mmproj-gemma-4-12b-it-qat-q4_0.gguf"
    _write_minimal_gguf(mmproj)
    model.mmproj = mmproj
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: None)
    unknown_cmd = build_command(model, cfg, profile, enable_prompt_cache=True)
    unknown_idx = unknown_cmd.index("--cache-ram")
    assert unknown_cmd[unknown_idx + 1] == "0"

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10044)
    old_cmd = build_command(model, cfg, profile, enable_prompt_cache=True)
    old_idx = old_cmd.index("--cache-ram")
    assert old_cmd[old_idx + 1] == "0"

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10045)
    current_cmd = build_command(model, cfg, profile, enable_prompt_cache=True)
    current_idx = current_cmd.index("--cache-ram")
    assert current_cmd[current_idx + 1] == "2048"

    cfg.prompt_cache_ram_mib = 4096
    bounded_cmd = build_command(model, cfg, profile, enable_prompt_cache=True)
    bounded_idx = bounded_cmd.index("--cache-ram")
    assert bounded_cmd[bounded_idx + 1] == "4096"

    disabled_cmd = build_command(model, cfg, profile, enable_prompt_cache=False)
    disabled_idx = disabled_cmd.index("--cache-ram")
    assert disabled_cmd[disabled_idx + 1] == "0"


def test_mmproj_cpu_offload_moves_memory_and_emits_flag(tmp_path, monkeypatch) -> None:
    """--no-mmproj-offload moves the projector estimate from VRAM to RAM,
    and the command flag is emitted only while a projector is loaded."""
    import tuner

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "gemma-4-12b-it-qat-q4_0", size_gb=6.5)
    mmproj = tmp_path / "mmproj-gemma-4-12b-it-f16.gguf"
    mmproj.write_bytes(b"x" * (4 * 1024 * 1024))
    model.mmproj = mmproj
    profile = match_profile(model.name, profiles)

    gpu_cfg = compute_config(
        model,
        _fake_system(),
        profile,
        no_mmproj_offload=False,
        prompt_cache_ram_mib=0,
    )
    cpu_cfg = compute_config(
        model,
        _fake_system(),
        profile,
        no_mmproj_offload=True,
        prompt_cache_ram_mib=0,
    )
    expected_gb = mmproj.stat().st_size / 1024**3
    assert gpu_cfg.vision_vram_gb == pytest.approx(expected_gb)
    assert gpu_cfg.vision_ram_gb == 0
    assert cpu_cfg.vision_vram_gb == 0
    assert cpu_cfg.vision_ram_gb == pytest.approx(expected_gb)

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10045)
    cmd = build_command(model, cpu_cfg, profile)
    assert cpu_cfg.mmproj_device is None
    assert "--no-mmproj-offload" in cmd
    assert "--mmproj-device" not in cmd

    no_vision = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    no_vision_cfg = compute_config(
        no_vision,
        _fake_system(),
        match_profile(no_vision.name, profiles),
        no_mmproj_offload=True,
        prompt_cache_ram_mib=0,
    )
    no_vision_cmd = build_command(
        no_vision, no_vision_cfg, match_profile(no_vision.name, profiles)
    )
    assert "--no-mmproj-offload" not in no_vision_cmd


def test_prompt_cache_limit_reduces_ram_kv_budget(tmp_path) -> None:
    """A bounded host prompt cache competes with RAM-resident KV and must
    lower the safe context; legacy unlimited mode uses a finite 2 GiB reserve."""
    from performance_target import get_target

    profiles = load_profiles(SETTINGS_DIR)
    model = _mistral_dense_md(tmp_path, size_gb=20.0)
    profile = match_profile(model.name, profiles)
    system = _fake_system(ram_total=40, ram_free=30, vram_total=8, vram_free=7.5)
    target = get_target("low_vram")
    assert target is not None
    uncached = compute_config(
        model, system, profile, perf_target=target, prompt_cache_ram_mib=0
    )
    bounded = compute_config(
        model, system, profile, perf_target=target, prompt_cache_ram_mib=4096
    )
    unlimited = compute_config(
        model, system, profile, perf_target=target, prompt_cache_ram_mib=-1
    )
    assert bounded.prompt_cache_ram_gb == pytest.approx(4.0)
    assert bounded.ctx < uncached.ctx
    assert unlimited.prompt_cache_ram_gb == pytest.approx(2.0)
    assert 2048 <= unlimited.ctx < uncached.ctx


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell-script fake binary")
def test_prepare_command_for_binary_prunes_unsupported_flags(tmp_path) -> None:
    """Older llama.cpp builds abort on unknown flags before loading a model.
    The launch path should remove only flags absent from the selected binary's
    --help while keeping core supported arguments and their values.
    """
    from tuner import prepare_command_for_binary

    fake = tmp_path / ("llama-server.exe" if os.name == "nt" else "llama-server")
    fake.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        "-m, --model\n"
        "-c, --ctx-size\n"
        "-ngl, --gpu-layers\n"
        "--host\n"
        "--port\n"
        "EOF\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        fake.chmod(0o755)

    cmd = [
        str(fake),
        "-m",
        "model.gguf",
        "--fit",
        "off",
        "--cache-ram",
        "-1",
        "--mcp-servers-config",
        "-trusted-mcp.json",
        "--mmproj-device",
        "Vulkan1",
        "--host",
        "127.0.0.1",
        "--metrics",
        "--port",
        "1234",
    ]
    filtered, removed = prepare_command_for_binary(cmd)

    assert "--fit" not in filtered and "off" not in filtered
    assert "--cache-ram" not in filtered and "-1" not in filtered
    assert "--mcp-servers-config" not in filtered
    assert "-trusted-mcp.json" not in filtered
    assert "--metrics" not in filtered
    assert ["-m", "model.gguf"] == filtered[1:3]
    assert "--host" in filtered and "127.0.0.1" in filtered
    assert "--port" in filtered and "1234" in filtered
    assert removed == [
        "--fit off",
        "--cache-ram -1",
        "--mcp-servers-config -trusted-mcp.json",
        "--mmproj-device Vulkan1",
        "--metrics",
    ]


def test_filter_command_keeps_mmproj_device_through_short_alias() -> None:
    """A fork advertising only -mmdev still supports our long spelling."""
    from tuner import _filter_command_for_supported_flags

    cmd = [
        "llama-server",
        "-m",
        "model.gguf",
        "--mmproj-device",
        "Vulkan1",
    ]
    filtered, removed = _filter_command_for_supported_flags(cmd, {"-m", "-mmdev"})
    assert filtered == cmd
    assert removed == []


def test_filter_command_keeps_negative_values_of_supported_flags() -> None:
    """Values like ``--cache-ram -1`` / ``-n -1`` start with '-' but are NOT
    flags. Pruning must never strip them from a supported flag (regression:
    ``--cache-ram -1 -n -1`` became ``--cache-ram -n``, aborting every launch
    on a probed binary)."""
    from tuner import _filter_command_for_supported_flags

    supported = {"-m", "--cache-ram", "-n", "-c", "-ngl"}
    cmd = [
        "llama-server",
        "-m",
        "model.gguf",
        "-c",
        "8192",
        "-ngl",
        "99",
        "--cache-ram",
        "-1",
        "-n",
        "-1",
        "--fit",
        "off",
        "--metrics",
    ]
    filtered, removed = _filter_command_for_supported_flags(cmd, supported)
    assert filtered == cmd[:11]
    assert removed == ["--fit off", "--metrics"]


def test_filter_command_removes_stray_value_of_unknown_flag() -> None:
    """An unknown fork flag with a separate value (e.g. from extra_args) must
    take its value with it — llama-server has no positional arguments, so a
    left-behind value aborts the launch exactly like the unknown flag."""
    from tuner import _filter_command_for_supported_flags

    supported = {"-m", "--port"}
    cmd = [
        "llama-server",
        "-m",
        "model.gguf",
        "--some-fork-knob",
        "42",
        "--port",
        "1234",
    ]
    filtered, removed = _filter_command_for_supported_flags(cmd, supported)
    assert filtered == ["llama-server", "-m", "model.gguf", "--port", "1234"]
    assert removed == ["--some-fork-knob 42"]


def test_setting_tooltip_has_beginner_and_technical_layers() -> None:
    """Every settings hover uses the same readable two-level structure."""
    from qt_launcher import _setting_tooltip

    tooltip = _setting_tooltip(
        "Easy summary with a <beginner> term.",
        "--technical-flag\nSecond line",
    )
    assert "<b>In short:</b>" in tooltip
    assert "<b>Technical details:</b>" in tooltip
    assert "&lt;beginner&gt;" in tooltip
    assert "--technical-flag<br>Second line" in tooltip


def test_about_and_source_update_text_include_running_version_and_github():
    qt_launcher = pytest.importorskip("qt_launcher")
    from autotuner_version import GITHUB_REPO, VERSION

    about = qt_launcher._about_text()
    assert f"Version v{VERSION}" in about
    assert f"https://github.com/{GITHUB_REPO}" in about
    assert qt_launcher._source_update_message("Source update complete").startswith(
        f"AutoTuner v{VERSION}:"
    )


def test_theme_replacement_is_only_authorized_for_selected_unchanged_user_id():
    qt_launcher = pytest.importorskip("qt_launcher")
    from theme_manager import ThemeDefinition

    builtin = ThemeDefinition("dark", "Dark", "", {}, source="builtin")
    user = ThemeDefinition("ocean", "Ocean", "", {}, source="user")
    unchanged = ThemeDefinition("ocean", "Edited", "", {}, source="user")
    renamed = ThemeDefinition("forest", "Renamed", "", {}, source="user")

    assert qt_launcher._theme_replace_id(builtin, unchanged) is None
    assert qt_launcher._theme_replace_id(user, unchanged) == "ocean"
    assert qt_launcher._theme_replace_id(user, renamed) is None


def test_launch_button_recovers_after_selection_and_benchmark_cleanup(
    tmp_path, monkeypatch
) -> None:
    """A scan/benchmark must never leave Launch permanently disabled."""
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )

    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    model = _fake_model(tmp_path, "Qwen3.8-27B-Q4_K_M", 12.0)
    window._system = _fake_system()
    window._all_entries = [model]

    # _start_scan() disables Launch. Selecting a valid model after the scan
    # must be an explicit transition back to enabled.
    window._btn_launch.setEnabled(False)
    window._show_config(model)
    assert window._btn_launch.isEnabled()

    # While a benchmark thread is winding down, Launch stays locked. Once its
    # finished callback clears the worker state, Launch must recover even when
    # the saved pre-benchmark widget state happened to be disabled.
    window._benchmark_thread = object()  # type: ignore[assignment]
    window._set_benchmark_controls_locked(True)
    window._benchmark_locked_states[window._btn_launch] = False
    window._set_benchmark_controls_locked(False)
    assert not window._btn_launch.isEnabled()
    window._on_performance_tuning_thread_finished()
    assert window._btn_launch.isEnabled()
    window.close()


def test_ngram_defaults_on_but_explicit_model_off_survives(tmp_path, monkeypatch) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )
    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    model = _fake_model(tmp_path, "Qwen3.8-27B-Q4_K_M", 12.0)
    window._system = _fake_system()
    window._all_entries = [model]
    window._show_config(model)
    assert window._chk_ngram.isChecked()

    qt_launcher.app_settings.set_model_override(model.name, "ngram", False)
    window._option_overrides.clear()
    window._update_checkboxes(model)
    assert not window._chk_ngram.isChecked()
    window.close()


def test_performance_test_real_validates_qwen38_above_static_estimate(
    tmp_path, monkeypatch
) -> None:
    """The user's 110,592-token setup must be tried, not rejected at ~78k."""
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )

    model = _fake_qwen38_27b_model(
        tmp_path, "Qwen3.8-27B-3.27bpw", size_gb=13.04
    )
    profile = match_profile(
        model.name, load_profiles(SETTINGS_DIR), model.architecture
    )
    system = _fake_system(
        ram_total=64,
        ram_free=55.7,
        vram_total=16,
        vram_free=15.97,
        vendor="amd",
    )
    manual = {
        "mode": "manual",
        "values": {
            "ctx": 110592,
            "cache_k": "q4_0",
            "cache_v": "q4_0",
            "ngl": 999,
            "n_cpu_moe": 0,
            "threads": 8,
            "batch_threads": 8,
            "batch": 512,
            "ubatch": 64,
            "flash_attn": True,
            "draft_n_max": 2,
            "parallel_enabled": True,
            "parallel_count": 1,
        },
    }
    qt_launcher.app_settings.set_expert_override(
        model.name, manual, model.path, "throughput"
    )

    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    window._system = system
    options = window._benchmark_model_options(model, tune_mtp=True)
    static_cfg, static_context, static_trial = window._benchmark_config_for_target(
        model,
        profile,
        system,
        "throughput",
        options,
        desired_context=110592,
        enable_yarn=False,
        real_validation=False,
    )
    assert not static_trial
    assert 70000 <= static_context <= 85000
    assert static_cfg.ctx == static_context

    trial_cfg, safe_context, exact_trial = window._benchmark_config_for_target(
        model,
        profile,
        system,
        "throughput",
        options,
        desired_context=110592,
        enable_yarn=False,
        real_validation=True,
    )
    assert exact_trial is True
    assert safe_context == static_context
    assert trial_cfg.ctx == 110592
    assert (trial_cfg.cache_k, trial_cfg.cache_v) == ("q4_0", "q4_0")
    assert trial_cfg.ngl == 999
    assert (trial_cfg.batch, trial_cfg.ubatch) == (512, 64)

    # RAM-backed KV is tested as its own mode instead of unsafely adding spare
    # RAM to a full-offload VRAM budget.
    ram_cfg, ram_safe, ram_trial = window._benchmark_config_for_target(
        model,
        profile,
        system,
        "low_vram",
        options,
        desired_context=110592,
        enable_yarn=False,
        real_validation=False,
    )
    assert not ram_trial
    assert ram_safe == 110592
    assert ram_cfg.no_kv_offload is True
    assert ram_cfg.kv_ram_gb > 2.0

    validated_snapshot = window._benchmark_snapshot(
        trial_cfg, validated_exact_context=True
    )
    assert validated_snapshot["mode"] == "manual"
    restored = qt_launcher.expert_cfg_from_values(
        static_cfg, validated_snapshot["values"]
    )
    assert restored.ctx == 110592
    assert restored.estimated_kv_gb > static_cfg.estimated_kv_gb
    window.close()


def test_performance_suite_selects_and_reports_fastest_mode(
    tmp_path, monkeypatch
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    from model_benchmark import (
        BenchmarkCandidate,
        BenchmarkResult,
        BenchmarkRunner,
        BenchmarkSample,
        BenchmarkSuiteJob,
        BenchmarkSuiteJobResult,
        BenchmarkSuiteResult,
        CandidateResult,
    )
    from tuner import TunedConfig

    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )
    messages: list[str] = []
    monkeypatch.setattr(
        qt_launcher.QMessageBox,
        "information",
        lambda *args, **_kwargs: messages.append(str(args[-1])),
    )

    model = _fake_model(tmp_path, "Qwen3.8-27B-Q4_K_M", 4.0)
    profile = match_profile(model.name, load_profiles(SETTINGS_DIR))
    system = _fake_system()
    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    window._system = system
    window._all_entries = [model]
    window._show_config(model)

    def outcome(
        target: str,
        prompt: float,
        decode: float,
        *,
        context: int = 8192,
        prompt_tokens: int = 4096,
    ):
        cfg = TunedConfig(
            ctx=context,
            ngl=999,
            threads=8,
            batch_threads=8,
            batch=512,
            ubatch=256,
            cache_k="q4_0",
            cache_v="q4_0",
            flash_attn=True,
            sampling={},
            performance_target=target,
        )
        baseline_candidate = BenchmarkCandidate(
            "baseline", "Auto baseline", 8, 8, 512, 256
        )
        winner_candidate = BenchmarkCandidate(
            f"winner-{target}", f"Winner {target}", 8, 8, 1024, 512
        )
        baseline_sample = BenchmarkSample(
            100.0, 20.0, prompt_tokens, 256, 1.0
        )
        winner_sample = BenchmarkSample(
            prompt, decode, prompt_tokens, 256, 1.0
        )
        measured = BenchmarkResult(
            desired_context=context,
            baseline_id="baseline",
            winner_id=winner_candidate.id,
            candidates=[
                CandidateResult(baseline_candidate, [baseline_sample, baseline_sample]),
                CandidateResult(winner_candidate, [winner_sample, winner_sample]),
            ],
            elapsed_s=10.0,
            runtime_binary="llama-server",
            runtime_build=10572,
            reason="measured winner",
        )
        runner = BenchmarkRunner(
            model=model,
            profile=profile,
            base_config=cfg,
            runtime_binary="llama-server",
            physical_cores=8,
            logical_cores=16,
        )
        job = BenchmarkSuiteJob(
            key=target,
            label=f"{model.name} [{target}]",
            performance_target=target,
            runner=runner,
            metadata={"system": system, "safe_context": context},
        )
        return BenchmarkSuiteJobResult(job=job, result=measured)

    suite_result = BenchmarkSuiteResult(
        jobs=[
            outcome("safe", 180.0, 24.0),
            outcome("throughput", 260.0, 42.0),
        ],
        elapsed_s=20.0,
    )
    window._on_performance_tuning_finished(suite_result)

    assert window._perf_combo.currentText() == "throughput"
    assert qt_launcher.app_settings.get_performance_target() == "throughput"
    assert (
        qt_launcher.app_settings.get_model_performance_target(model.path)
        == "throughput"
    )
    assert messages
    assert "safe: PP" in messages[-1]
    assert "★ throughput:" in messages[-1]
    assert "Selected model now uses the fastest measured mode: throughput" in messages[-1]

    # Per-model choice is restored when returning to a model, not just kept as
    # a one-session global combo value.
    qt_launcher.app_settings.set_model_performance_target(model.path, "safe")
    window._show_config(model)
    assert window._perf_combo.currentText() == "safe"

    # Different contexts/token mixes are not comparable. Keep the user's
    # current mode instead of selecting the numerically larger raw TPS.
    messages.clear()
    window._benchmark_checkpoints = {}
    mismatched = BenchmarkSuiteResult(
        jobs=[
            outcome("safe", 180.0, 24.0, context=8192, prompt_tokens=4096),
            outcome(
                "throughput",
                300.0,
                50.0,
                context=16384,
                prompt_tokens=2048,
            ),
        ],
        elapsed_s=20.0,
    )
    window._on_performance_tuning_finished(mismatched)
    assert window._perf_combo.currentText() == "safe"
    assert (
        qt_launcher.app_settings.get_model_performance_target(model.path) == "safe"
    )
    assert messages
    assert "No auto-selection" in messages[-1]
    window.close()


def test_performance_test_enumerates_external_drafter_quantizations(
    tmp_path, monkeypatch
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )
    monkeypatch.setattr(qt_launcher, "is_draft_compatible", lambda *_args: True)

    model = _fake_model(tmp_path, "Qwen3.8-27B-Q4", 1.0)
    drafts = []
    for quant in ("Q4_K_M", "Q8_0", "BF16"):
        path = tmp_path / f"dflash2-Qwen3.8-27B-{quant}.gguf"
        _write_minimal_gguf(path)
        drafts.append(path)
    model.folder_drafts = drafts
    model.draft = drafts[0]

    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    variants = window._benchmark_option_variants(
        model, tune_mtp=True, include_external_drafters=True
    )
    assert [Path(str(item["drafter_label"])).name for item in variants] == [
        path.name for path in drafts
    ]
    assert len({str(item["drafter_key"]) for item in variants}) == 3
    assert all(item["enable_speculative"] for item in variants)
    window.close()


def test_completed_benchmark_matching_is_exact_to_environment_workload_and_model(
    tmp_path, monkeypatch
) -> None:
    qt_launcher = pytest.importorskip("qt_launcher")
    model = _fake_model(tmp_path, "CompletedModel", 1.0)
    cfg = TunedConfig(
        ctx=32768,
        ngl=99,
        threads=6,
        batch_threads=6,
        batch=512,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
        sampling={},
    )
    system = _fake_system()
    monkeypatch.setattr(
        qt_launcher, "probe_binary_build_number", lambda _binary: 10572
    )
    stat = model.path.stat()
    record = {
        "model_size": stat.st_size,
        "model_mtime_ns": stat.st_mtime_ns,
        "desired_context": cfg.ctx,
        "benchmark_type": "quick",
        "prompt_context_fraction": 0.125,
        "candidates": [{"id": "baseline"}],
        "environment_fingerprint": qt_launcher._benchmark_environment_fingerprint(
            "llama-server", 10572, system, cfg
        ),
    }
    assert qt_launcher.MainWindow._completed_benchmark_matches(
        record,
        model,
        cfg,
        "quick",
        0.125,
        runtime_binary="llama-server",
        system=system,
    )
    assert not qt_launcher.MainWindow._completed_benchmark_matches(
        record,
        model,
        cfg,
        "fast",
        0.03125,
        runtime_binary="llama-server",
        system=system,
    )
    changed = dict(record, desired_context=65536)
    assert not qt_launcher.MainWindow._completed_benchmark_matches(
        changed,
        model,
        cfg,
        "quick",
        0.125,
        runtime_binary="llama-server",
        system=system,
    )
    monkeypatch.setattr(
        qt_launcher, "probe_binary_build_number", lambda _binary: 10679
    )
    assert not qt_launcher.MainWindow._completed_benchmark_matches(
        record,
        model,
        cfg,
        "quick",
        0.125,
        runtime_binary="llama-server",
        system=system,
    )


def test_measured_profile_is_applied_only_to_matching_environment(
    tmp_path, monkeypatch
) -> None:
    qt_launcher = pytest.importorskip("qt_launcher")
    cfg = TunedConfig(
        ctx=32768,
        ngl=99,
        threads=6,
        batch_threads=6,
        batch=512,
        ubatch=256,
        cache_k="q4_0",
        cache_v="q4_0",
        flash_attn=True,
        sampling={},
    )
    system = _fake_system()
    monkeypatch.setattr(
        qt_launcher, "probe_binary_build_number", lambda _binary: 10572
    )
    fake_window = types.SimpleNamespace(
        _system=system,
        _active_llama_binary=lambda: "llama-server",
    )
    environment = qt_launcher._benchmark_environment_fingerprint(
        "llama-server", 10572, system, cfg
    )
    measured = {
        "source": "measured-performance-test",
        "benchmark_environment": environment,
    }
    assert qt_launcher.MainWindow._measured_snapshot_matches_environment(
        fake_window, measured, cfg
    )

    changed_system = copy.deepcopy(system)
    changed_system.gpus[0].runtime_backend = "HIP"
    fake_window._system = changed_system
    assert not qt_launcher.MainWindow._measured_snapshot_matches_environment(
        fake_window, measured, cfg
    )
    assert not qt_launcher.MainWindow._measured_snapshot_matches_environment(
        fake_window, {"source": "measured-performance-test"}, cfg
    )
    assert qt_launcher.MainWindow._measured_snapshot_matches_environment(
        fake_window, {"source": "user-custom"}, cfg
    )


def test_completed_external_drafters_are_reused_when_rerun_is_off(
    tmp_path, monkeypatch
) -> None:
    """Every external head keeps an independent completed-run cache entry."""
    import app_settings

    qt_launcher = pytest.importorskip("qt_launcher")
    monkeypatch.setattr(
        app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    model = _fake_model(tmp_path, "CachedDrafterModel", 1.0)
    cfg = TunedConfig(
        ctx=32768,
        ngl=99,
        threads=6,
        batch_threads=6,
        batch=512,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
        sampling={},
    )
    stat = model.path.stat()
    system = _fake_system()
    monkeypatch.setattr(
        qt_launcher, "probe_binary_build_number", lambda _binary: 10572
    )
    snapshot = {"mode": "auto", "values": {}}
    drafter_keys = []
    for index, quant in enumerate(("Q4_K_M", "Q8_0", "BF16"), start=1):
        draft = tmp_path / f"draft-CachedDrafterModel-{quant}.gguf"
        draft.write_bytes(bytes(index * 7))
        key = qt_launcher._drafter_profile_key(draft)
        drafter_keys.append(key)
        record = {
            "model_name": model.name,
            "model_size": stat.st_size,
            "model_mtime_ns": stat.st_mtime_ns,
            "desired_context": cfg.ctx,
            "benchmark_type": "quick",
            "performance_target": "throughput",
            "prompt_context_fraction": 0.125,
            "drafter_key": key,
            "drafter_label": draft.name,
            "candidates": [{"id": "baseline"}],
            "environment_fingerprint": qt_launcher._benchmark_environment_fingerprint(
                "llama-server", 10572, system, cfg
            ),
        }
        assert app_settings.save_performance_tuning_result(
            model.name,
            model.path,
            record,
            snapshot,
            "throughput",
            "quick",
            key,
        )

    assert len(set(drafter_keys)) == 3
    for key in drafter_keys:
        completed = app_settings.get_performance_tuning_result(
            model.path, "throughput", "quick", key
        )
        assert completed is not None
        assert completed["drafter_key"] == key
        assert qt_launcher.MainWindow._completed_benchmark_matches(
            completed,
            model,
            cfg,
            "quick",
            0.125,
            runtime_binary="llama-server",
            system=system,
        )

    # Read-only cache access must still return caller-owned records; mutating a
    # retrieved dict cannot alter the process cache or on-disk evidence.
    first = app_settings.get_performance_tuning_result(
        model.path, "throughput", "quick", drafter_keys[0]
    )
    assert first is not None
    first["desired_context"] = 1
    unchanged = app_settings.get_performance_tuning_result(
        model.path, "throughput", "quick", drafter_keys[0]
    )
    assert unchanged is not None
    assert unchanged["desired_context"] == cfg.ctx


def test_performance_analysis_keeps_quick_standard_and_custom_tiles_separate(
    tmp_path,
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

    setup = qt_launcher._PerformanceTuneSetupDialog(
        "Example", 32768, 131072, 2
    )
    assert setup.benchmark_type() == "fast"
    assert setup.prompt_context_fraction() == pytest.approx(0.03125)
    assert not setup.custom_percent_spin.isEnabled()
    assert not setup.try_best_settings.isEnabled()
    assert not setup.rerun_all_models.isEnabled()
    setup.all_models.setChecked(True)
    assert setup.rerun_all_models.isEnabled()
    assert not setup.test_external_drafters.isEnabled()
    setup.tune_mtp.setChecked(True)
    assert setup.test_external_drafters.isEnabled()
    setup.test_length_combo.setCurrentIndex(
        setup.test_length_combo.findData("quick")
    )
    assert setup.prompt_context_fraction() == pytest.approx(0.125)
    assert setup.try_best_settings.isEnabled()
    setup.test_length_combo.setCurrentIndex(
        setup.test_length_combo.findData("custom")
    )
    setup.custom_percent_spin.setValue(87.65)
    assert setup.benchmark_type() == "custom"
    assert setup.custom_percent_spin.isEnabled()
    assert setup.prompt_context_fraction() == pytest.approx(0.8765)
    setup.close()

    def record(name: str, test_type: str, value: float) -> dict:
        winner_id = f"{name}-{test_type}"
        return {
            "model_name": name,
            "model_path": f"C:/{name}.gguf",
            "benchmark_type": test_type,
            "performance_target": "balanced",
            "desired_context": 32768,
            "winner_id": winner_id,
            "winner_settings": {"threads": 8, "batch": 512, "ubatch": 256},
            "candidates": [
                {
                    "id": winner_id,
                    "prompt_tps": value * 4,
                    "generation_tps": value,
                    "overall_tps": value * 2,
                }
            ],
        }

    html_path = tmp_path / "report.html"
    html_path.write_text("<!doctype html>", encoding="utf-8")
    dialog = qt_launcher._PerformanceAnalysisDialog(
        {
            "fast": [record("QuickModel", "fast", 8.0)],
            "quick": [record("StandardModel", "quick", 10.0)],
            "custom": [record("CustomModel", "custom", 50.0)],
        },
        html_path=html_path,
    )
    assert dialog.record_count_by_test == {"fast": 1, "quick": 1, "custom": 1}
    assert dialog.model_names_by_test["fast"] == ["QuickModel"]
    assert dialog.model_names_by_test["quick"] == ["StandardModel"]
    assert dialog.model_names_by_test["custom"] == ["CustomModel"]
    assert dialog.test_tiles["fast"].property("benchmarkType") == "fast"
    assert dialog.test_tiles["quick"].property("benchmarkType") == "quick"
    assert dialog.test_tiles["custom"].property("benchmarkType") == "custom"
    assert dialog.findChild(
        qt_widgets.QPushButton, "openPerformanceHtmlReport"
    ) is not None
    # Each tile normalizes its own PP/decode/end-to-end scales. With one model
    # in each tile every non-zero bar reaches that tile's maximum.
    for tile in dialog.test_tiles.values():
        bars = tile.findChildren(qt_widgets.QProgressBar)
        assert len(bars) == 3
        assert all(bar.value() == 1000 for bar in bars)
    text = " ".join(
        label.text() for label in dialog.findChildren(qt_widgets.QLabel)
    )
    assert "Prompt processing (PP)" in text
    assert "n_decode" in text
    assert "End-to-end" in text
    dialog.close()


def test_performance_dialog_exposes_graceful_stop_controls() -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    requested: list[str] = []
    dialog = qt_launcher._PerformanceTuneDialog(
        "All-model suite", 8, allow_model_stop=True
    )
    dialog.stop_after_model_requested.connect(lambda: requested.append("model"))
    dialog.stop_after_mode_requested.connect(lambda: requested.append("mode"))
    assert dialog.stop_after_model_button.text() == "Stop after Model"
    assert dialog.stop_after_model_button.isEnabled()
    assert dialog.stop_after_mode_button.text() == "Stop after Performance Mode"

    dialog.stop_after_model_button.click()
    assert requested == ["model"]
    assert "current model" in dialog.status_label.text()
    dialog.stop_after_mode_button.click()
    assert requested == ["model", "mode"]
    assert "performance mode" in dialog.status_label.text()
    dialog.mark_finished()
    dialog.close()

    single = qt_launcher._PerformanceTuneDialog(
        "Single-model suite", 4, allow_model_stop=False
    )
    assert not single.stop_after_model_button.isEnabled()
    assert single.stop_after_mode_button.isEnabled()
    single.mark_finished()
    single.close()


def test_model_expert_settings_copy_and_paste_is_target_scoped(
    tmp_path, monkeypatch
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )

    source = _fake_model(tmp_path / "source", "SourceModel", 1.0)
    target = _fake_model(tmp_path / "target", "TargetModel", 1.0)
    source_snapshot = {
        "mode": "manual",
        "pins": {},
        "values": {"ctx": 24576, "threads": 6, "batch": 512, "ubatch": 128},
        "saved_at": "2026-01-01T00:00:00",
    }
    qt_launcher.app_settings.set_expert_override(
        source.name, source_snapshot, source.path, "balanced"
    )

    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    window._system = _fake_system()
    window._all_entries = [source, target]
    window._show_config(source)
    window._copy_expert_settings(source)
    assert window._expert_settings_clipboard is not None
    assert window._expert_settings_clipboard["performance_target"] == "balanced"

    window._show_config(target)
    window._paste_expert_settings(target)
    pasted = qt_launcher.app_settings.get_setting_profile_snapshot(
        target.name,
        target.path,
        "balanced",
        "custom1",
        qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
    )
    assert pasted is not None
    assert pasted["mode"] == "manual"
    assert pasted["values"]["ctx"] == 24576
    assert pasted["values"]["threads"] == 6
    assert pasted["source"] == "copied-expert-settings"
    assert pasted["copied_from"] == {
        "model": source.name,
        "performance_target": "balanced",
    }
    # Provenance changes apply only to the pasted deep copy.
    assert "copied_from" not in qt_launcher.app_settings.get_setting_profile_snapshot(
        source.name,
        source.path,
        "balanced",
        "custom1",
        qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
    )

    # A right-click/model switch must flush a pending edit while the source is
    # still current; otherwise the delayed Expert signal could overwrite the
    # destination with the source panel's state.
    window._show_config(source)
    window._enter_expert_mode()
    window._expert_panel._sp_threads.setValue(7)
    assert window._expert_panel._save_timer.isActive()
    window._show_config(target)
    assert qt_launcher.app_settings.get_setting_profile_snapshot(
        source.name,
        source.path,
        "balanced",
        "custom1",
        qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
    )["values"]["threads"] == 7
    assert qt_launcher.app_settings.get_setting_profile_snapshot(
        target.name,
        target.path,
        "balanced",
        "custom1",
        qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
    )["values"]["threads"] == 6
    window.close()


def test_settings_profile_selector_preserves_auto_perform_and_custom(
    tmp_path, monkeypatch
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )

    model = _fake_model(tmp_path, "ProfileModel", 1.0)
    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    window._system = _fake_system()
    window._all_entries = [model]
    window._show_config(model)

    combo = window._setting_profile_combo
    assert combo.count() == 6
    assert combo.currentData() == qt_launcher.app_settings.PROFILE_AUTO
    perform_index = combo.findData(qt_launcher.app_settings.PROFILE_PERFORM)
    assert not combo.model().item(perform_index).isEnabled()

    # Editing immutable Auto transparently forks into the first free Custom.
    window._enter_expert_mode()
    window._expert_panel._sp_threads.setValue(7)
    window._expert_panel.flush_pending_save()
    _QT_TEST_APP.processEvents()
    assert combo.currentData() == "custom1"
    custom = qt_launcher.app_settings.get_setting_profile_snapshot(
        model.name,
        model.path,
        "balanced",
        "custom1",
        qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
    )
    assert custom is not None
    assert custom["values"]["threads"] == 7
    assert (
        qt_launcher.app_settings.get_setting_profile_snapshot(
            model.name,
            model.path,
            "balanced",
            "auto",
            qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
        )
        is None
    )

    monkeypatch.setattr(
        qt_launcher.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("Latency lab", True),
    )
    window._rename_setting_profile()
    assert combo.currentText() == "Latency lab"

    perform = copy.deepcopy(custom)
    perform["values"]["threads"] = 11
    assert qt_launcher.app_settings.set_setting_profile_snapshot(
        model.name,
        model.path,
        "balanced",
        qt_launcher.app_settings.PROFILE_PERFORM,
        perform,
        qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
        select=True,
    )
    window._refresh_setting_profile_selector()
    assert combo.model().item(perform_index).isEnabled()
    assert combo.currentData() == qt_launcher.app_settings.PROFILE_PERFORM
    profile = match_profile(model.name, window._profiles, model.architecture)
    assert window._effective_config(model, profile).threads == 11
    # The measured slot did not overwrite the custom experiment.
    assert qt_launcher.app_settings.get_setting_profile_snapshot(
        model.name,
        model.path,
        "balanced",
        "custom1",
        qt_launcher.app_settings.NO_DRAFTER_PROFILE_KEY,
    )["values"]["threads"] == 7
    window.close()


def test_settings_widgets_have_two_level_hover_help(tmp_path, monkeypatch) -> None:
    """Lock in complete beginner + technical help for settings dialogs."""
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    parent = qt_widgets.QWidget()

    monkeypatch.setattr(
        qt_launcher.startup_manager, "is_autostart_enabled", lambda: False
    )
    monkeypatch.setattr(qt_launcher, "_system_tray_supported", lambda: True)
    monkeypatch.setattr(
        qt_launcher.app_settings, "get_minimize_on_close", lambda: False
    )
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )

    expert = qt_launcher.ExpertPanel(parent)
    app_dialog = qt_launcher._ApplicationSettingsDialog(parent)
    paths_dialog = qt_launcher._PathListDialog(
        parent, "Paths", [(tmp_path, True)], "Pick a folder"
    )

    expert_names = (
        "_btn_auto",
        "_btn_manual",
        "_btn_reset",
        "_btn_close",
        "_sp_ctx",
        "_cb_cache_k",
        "_cb_cache_v",
        "_sp_ngl",
        "_sp_ncpumoe",
        "_sp_threads",
        "_sp_batch_threads",
        "_sp_batch",
        "_sp_ubatch",
        "_chk_parallel",
        "_sp_parallel",
        "_chk_fa",
        "_cb_load_mode",
        "_chk_jinja",
        "_chk_verbose",
        "_chk_metrics",
        "_chk_slots_api",
        "_cb_numa",
        "_chk_rope",
        "_sp_rope_factor",
        "_sp_temp",
        "_sp_top_k",
        "_sp_top_p",
        "_sp_min_p",
        "_sp_rep",
        "_sp_presence",
        "_sp_draft_n_max",
        "_cb_reasoning",
        "_sp_think_budget",
        "_chk_reasoning_preserve",
        "_le_extra",
    )
    widgets = [getattr(expert, name) for name in expert_names]
    widgets.extend(
        [
            app_dialog.autostart_checkbox,
            app_dialog.minimize_checkbox,
            app_dialog.theme_combo,
            app_dialog.reload_themes_button,
            app_dialog.customize_theme_button,
            app_dialog.open_themes_button,
            app_dialog.about_button,
            app_dialog.export_profiles_button,
            app_dialog.import_profiles_button,
        ]
    )
    widgets.extend(
        [
            paths_dialog._master,
            paths_dialog._list,
            paths_dialog._btn_add,
            paths_dialog._btn_edit,
            paths_dialog._btn_remove,
        ]
    )

    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    main_names = (
        "_fork_combo",
        "_btn_fork_folder",
        "_perf_combo",
        "_mode_combo",
        "_gpu_combo",
        "_search",
        "_btn_list_view",
        "_btn_tree_view",
        "_btn_expert",
        "_btn_diagnose",
        "_btn_benchmark",
        "_btn_performance_analysis",
        "_cb_mmproj",
        "_cb_draft",
        "_chk_mmproj_cpu",
        "_chk_ngram",
        "_chk_prompt_cache",
        "_sp_prompt_cache_mib",
        "_chk_thinking",
        "_host_edit",
        "_port_edit",
        "_port_offset_combo",
        "_server_combo",
        "_btn_toggle_log",
        "_btn_ocr",
        "_btn_launch",
        "_btn_stop",
        "_btn_stop_all",
        "_btn_quit",
    )
    widgets.extend(getattr(window, name) for name in main_names)
    assert not hasattr(window, "_chk_turbo_kv")
    assert not hasattr(window, "_chk_vision")
    assert not hasattr(window, "_chk_draft")
    assert "bf16" in expert._KV_QUANT_OPTIONS
    assert "turbo3" in expert._KV_QUANT_OPTIONS
    toolbar_texts = {
        "📂 Models folder",
        "🔄 Refresh",
        "⬆ Update",
        "⚙ Settings",
        "A−",
        "A+",
    }
    widgets.extend(
        button
        for button in window.findChildren(qt_widgets.QPushButton)
        if button.text() in toolbar_texts
    )
    assert toolbar_texts <= {
        button.text() for button in window.findChildren(qt_widgets.QPushButton)
    }

    for widget in widgets:
        tooltip = widget.toolTip()
        assert "<b>In short:</b>" in tooltip, widget.objectName()
        assert "<b>Technical details:</b>" in tooltip, widget.objectName()

    assert "<b>In short:</b>" in app_dialog.theme_combo.toolTip()
    assert app_dialog.about_button.accessibleName() == "About AutoTuner"
    assert (
        app_dialog.theme_combo.itemData(
            app_dialog.theme_combo.currentIndex(),
            qt_launcher.Qt.ItemDataRole.ToolTipRole,
        )
        == app_dialog.theme_combo.currentText()
    )

    alibaba = _fake_model(tmp_path / "Alibaba" / "Qwen3.6", "Qwen3.6-27B-Q4_K_M", 1.0)
    google = _fake_model(tmp_path / "Google" / "Gemma", "Gemma-3-12B-Q4", 1.0)
    window._last_scan_roots = [tmp_path]
    window._all_entries = [google, alibaba]
    window._favorite_models = {
        qt_launcher.app_settings.favorite_model_key(alibaba.path)
    }
    window._populate_list(window._all_entries)
    assert window._model_tree.topLevelItem(0).text(0) == "★ Favoriten (1)"
    assert (
        window._model_tree.topLevelItem(0)
        .child(0)
        .data(0, qt_launcher.Qt.ItemDataRole.UserRole)
        .path
        == alibaba.path
    )
    assert [
        window._model_tree.topLevelItem(index).text(0)
        for index in range(1, window._model_tree.topLevelItemCount())
    ] == ["Alibaba", "Google"]
    assert all(
        window._model_tree.topLevelItem(index).isExpanded()
        for index in range(window._model_tree.topLevelItemCount())
    )
    assert window._model_tree.topLevelItem(1).child(0).text(0) == "Qwen3.6"
    assert window._model_tree.topLevelItem(1).child(0).isExpanded()
    window._set_model_view("tree")
    assert window._model_view_stack.currentWidget() is window._model_tree
    assert qt_launcher.app_settings.get_model_view_mode() == "tree"
    window._apply_filter("Alibaba")
    assert window._model_tree.topLevelItemCount() == 2  # favorites + matching folder
    window._apply_filter("")

    window.resize(1320, 840)
    window.show()
    _QT_TEST_APP.processEvents()
    alibaba_folder = window._model_tree.topLevelItem(1)
    alibaba_folder.setExpanded(False)
    _QT_TEST_APP.processEvents()
    assert window._tree_native_toggle_item is None
    window._on_tree_item_clicked(alibaba_folder, 0)
    assert alibaba_folder.isExpanded()  # clicking the folder name toggles it

    top_split, main_split = window._splitters
    top_split.setSizes([333, 777])
    main_split.setSizes([555, 222])
    _QT_TEST_APP.processEvents()
    expected_top = top_split.sizes()
    expected_main = main_split.sizes()
    window._apply_theme_definition(
        window._theme_manager.get("builtin:dark"), _QT_TEST_APP, lambda: None
    )
    _QT_TEST_APP.processEvents()
    assert all(
        abs(actual - expected) <= 1
        for actual, expected in zip(top_split.sizes(), expected_top)
    )
    assert all(
        abs(actual - expected) <= 1
        for actual, expected in zip(main_split.sizes(), expected_main)
    )
    assert window._launch_options_group.maximumHeight() >= (
        window._launch_options_group.minimumSizeHint().height()
    )

    launch_was_enabled = window._btn_launch.isEnabled()
    fork_was_enabled = window._fork_combo.isEnabled()
    window._set_ocr_controls_locked(True)
    assert not window._btn_launch.isEnabled()
    assert not window._fork_combo.isEnabled()
    window._set_ocr_controls_locked(False)
    assert window._btn_launch.isEnabled() == launch_was_enabled
    assert window._fork_combo.isEnabled() == fork_was_enabled

    window.close()
    paths_dialog.close()
    app_dialog.close()
    parent.close()


def test_mmproj_dropdown_shows_projector_file_size(tmp_path, monkeypatch) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )

    model_path = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    projector = tmp_path / "mmproj-Qwen3.8-27B-BF16.gguf"
    _write_minimal_gguf(model_path)
    _write_minimal_gguf(projector)
    entry = ModelEntry(
        path=model_path,
        name=model_path.stem,
        group=".",
        size_bytes=model_path.stat().st_size,
        mmproj=projector,
        folder_mmprojs=[projector],
    )

    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    window._populate_mmproj_combo(entry, {})
    labels = [window._cb_mmproj.itemText(i) for i in range(window._cb_mmproj.count())]
    projector_label = next(label for label in labels if projector.name in label)
    assert "(0.0 GB)" in projector_label
    assert "(auto)" in projector_label
    window.close()


def test_mmproj_and_embedded_draft_dropdowns_are_authoritative(
    tmp_path, monkeypatch
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )

    model_path = tmp_path / "Ling-3.0-flash-Q4_K_M.gguf"
    projector = tmp_path / "mmproj-Ling-3.0-BF16.gguf"
    _write_minimal_gguf(model_path)
    _write_minimal_gguf(projector)
    entry = ModelEntry(
        path=model_path,
        name=model_path.stem,
        group=".",
        size_bytes=model_path.stat().st_size,
        mmproj=projector,
        folder_mmprojs=[projector],
        metadata={
            "general.architecture": "bailingmoe3",
            "bailingmoe3.nextn_predict_layers": 1,
        },
    )

    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    window._current_entry = entry
    window._update_checkboxes(entry)
    assert window._vision_enabled() is True
    assert window._draft_enabled() is True
    assert (
        window._cb_draft.currentData()
        == qt_launcher.app_settings.DRAFT_EMBEDDED_SENTINEL
    )

    # Both dropdowns can disable and re-enable their feature without a second
    # checkbox or stale override fighting the selected item.
    window._cb_mmproj.setCurrentIndex(0)
    assert window._vision_enabled() is False
    window._cb_mmproj.setCurrentIndex(1)
    assert window._vision_enabled() is True

    window._cb_draft.setCurrentIndex(0)
    assert window._draft_enabled() is False
    embedded_idx = window._cb_draft.findData(
        qt_launcher.app_settings.DRAFT_EMBEDDED_SENTINEL
    )
    assert embedded_idx > 0
    window._cb_draft.setCurrentIndex(embedded_idx)
    assert window._draft_enabled() is True
    assert (
        qt_launcher.app_settings.get_draft_selection(entry.name)
        == qt_launcher.app_settings.DRAFT_EMBEDDED_SENTINEL
    )
    assert not hasattr(window, "_chk_vision")
    assert not hasattr(window, "_chk_draft")
    window.close()


def test_turbo_kv_expert_warning_has_dismiss_and_never_show_again(
    tmp_path, monkeypatch
) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )

    dialogs = []

    class FakeMessageBox:
        class Icon:
            Warning = object()

        class ButtonRole:
            AcceptRole = object()
            DestructiveRole = object()

        def __init__(self, parent):
            self.parent = parent
            self.buttons = {}
            dialogs.append(self)

        def setWindowTitle(self, title):
            self.title = title

        def setIcon(self, icon):
            self.icon = icon

        def setText(self, text):
            self.text = text

        def setInformativeText(self, text):
            self.informative_text = text

        def addButton(self, label, role):
            button = object()
            self.buttons[label] = button
            return button

        def setDefaultButton(self, button):
            self.default_button = button

        def exec(self):
            return 0

        def clickedButton(self):
            return self.buttons["Never Show Again"]

    parent = qt_widgets.QWidget()
    monkeypatch.setattr(qt_launcher, "QMessageBox", FakeMessageBox)
    assert qt_launcher._show_turbo_kv_fork_warning(parent) is True
    assert set(dialogs[0].buttons) == {"Dismiss", "Never Show Again"}
    assert "special" not in dialogs[0].text.lower()  # details live below
    assert "mainline llama.cpp" in dialogs[0].text
    assert "compatible TurboQuant fork" in dialogs[0].informative_text
    assert qt_launcher.app_settings.get_turbo_kv_warning_suppressed() is True
    assert qt_launcher._show_turbo_kv_fork_warning(parent) is False
    assert len(dialogs) == 1
    parent.close()


def test_expert_turbo_kv_warning_is_once_per_session(tmp_path, monkeypatch) -> None:
    global _QT_TEST_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_TEST_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    calls = []
    monkeypatch.setattr(
        qt_launcher,
        "_show_turbo_kv_fork_warning",
        lambda parent: calls.append(parent) or True,
    )

    panel = qt_launcher.ExpertPanel()
    panel._cb_cache_k.setCurrentText("turbo3")
    panel._cb_cache_v.setCurrentText("turbo4")
    assert calls == [panel]
    panel.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX pipe/pump streaming path")
def test_terminal_process_streams_output_live(tmp_path, monkeypatch) -> None:
    """POSIX GUI launches must mirror server output live to the on_output
    callback (GUI log panel) AND persist it in the per-launch log file, so
    tokens/s and prompt-processing progress are visible while running."""
    import time

    import qt_launcher

    monkeypatch.setattr(qt_launcher.app_settings, "app_data_dir", lambda: tmp_path)
    script = tmp_path / "fake-server.sh"
    script.write_text(
        "#!/bin/sh\n"
        "echo 'prompt processing, n_tokens = 1024, 150.2 tokens per second'\n"
        "echo 'generation: 42.0 t/s' 1>&2\n",  # stderr must be merged too
        encoding="utf-8",
    )
    script.chmod(0o755)

    lines: list = []
    proc = qt_launcher._TerminalProcess([str(script)], on_output=lines.append)
    proc.start()
    assert proc.proc is not None
    proc.proc.wait(timeout=10)
    deadline = time.monotonic() + 5
    while len(lines) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)

    assert lines == [
        "prompt processing, n_tokens = 1024, 150.2 tokens per second",
        "generation: 42.0 t/s",
    ]
    assert proc.log_path is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        content = proc.log_path.read_text(encoding="utf-8")
        if "42.0 t/s" in content:
            break
        time.sleep(0.05)
    assert "150.2 tokens per second" in content
    assert "42.0 t/s" in content


def test_mlock_disabled_when_gpu_present(tmp_path, monkeypatch) -> None:
    """llama.cpp b9895 aborts with GGML_ASSERT(addr) in llama_mlock::grow_to
    whenever --mlock is combined with a loaded GPU backend (Vulkan host
    buffer with NULL base) — independent of RLIMIT_MEMLOCK and even with
    -ngl 0. With GPUs in the system the tuner must never auto-emit
    --mlock/--no-mmap."""
    import tuner

    monkeypatch.setattr(tuner, "_memlock_limit_gb", lambda: None)  # unlimited
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=5.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    assert cfg.mlock is False
    assert cfg.no_mmap is False


def test_veto_unsafe_mlock_overrides_gui_reenabled_mlock(tmp_path, monkeypatch) -> None:
    """A stale per-model override / expert checkbox can set cfg.mlock=True on a
    GPU system AFTER compute_config's gate. veto_unsafe_mlock is the final net
    that strips it again (the actual Mistral-Medium GUI crash)."""
    from tuner import veto_unsafe_mlock

    monkeypatch.setattr("tuner._memlock_limit_gb", lambda: None)
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=5.0)
    profile = match_profile(model.name, profiles)
    system = _fake_system()  # has a GPU
    cfg = compute_config(model, system, profile)
    assert cfg.mlock is False  # compute_config already gated

    # Simulate the GUI applying a persisted "mlock": true override.
    cfg.mlock = True
    cfg.no_mmap = True

    assert veto_unsafe_mlock(cfg, system) is True
    assert cfg.mlock is False
    assert cfg.no_mmap is False


def test_veto_unsafe_mlock_allows_b10151_gpu_build(tmp_path, monkeypatch) -> None:
    import tuner

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=5.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cfg.load_mode = "mlock"
    cfg.mlock = True
    cfg.no_mmap = True

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10151)
    assert tuner.veto_unsafe_mlock(cfg, _fake_system(), binary="llama-server") is False
    assert cfg.load_mode == "mlock"

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10114)
    assert tuner.veto_unsafe_mlock(cfg, _fake_system(), binary="llama-server") is True
    assert cfg.load_mode == "auto"


def test_veto_unsafe_mlock_respects_force_and_cpu_only(tmp_path) -> None:
    """--force-mlock bypasses the veto (patched builds); a CPU-only system has
    no GPU host buffer, so mlock is left intact there too."""
    from tuner import veto_unsafe_mlock

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=5.0)
    profile = match_profile(model.name, profiles)

    gpu_cfg = compute_config(model, _fake_system(), profile)
    gpu_cfg.mlock = True
    gpu_cfg.no_mmap = True
    assert veto_unsafe_mlock(gpu_cfg, _fake_system(), force_mlock=True) is False
    assert gpu_cfg.mlock is True

    cpu_cfg = compute_config(model, _fake_system(vram_total=0), profile)
    cpu_cfg.mlock = True
    cpu_cfg.no_mmap = True
    assert veto_unsafe_mlock(cpu_cfg, _fake_system(vram_total=0)) is False
    assert cpu_cfg.mlock is True


@pytest.mark.skipif(os.name == "nt", reason="RLIMIT_MEMLOCK gate is POSIX-only")
def test_mlock_disabled_when_memlock_limit_too_small(tmp_path, monkeypatch) -> None:
    """Desktop Linux defaults RLIMIT_MEMLOCK to 8 MiB; a non-root process
    cannot pin a model then. Checked on a CPU-only system so the RLIMIT gate
    (not the GPU gate) is what decides."""
    import tuner

    monkeypatch.setattr(tuner, "_memlock_limit_gb", lambda: 8 / 1024)
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=5.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(vram_total=0), profile)
    assert cfg.mlock is False
    assert cfg.no_mmap is False


@pytest.mark.skipif(os.name == "nt", reason="RLIMIT_MEMLOCK gate is POSIX-only")
def test_mlock_allowed_cpu_only_with_unlimited_memlock(tmp_path, monkeypatch) -> None:
    """CPU-only systems (no GPU backend, plain malloc buffers) keep the mlock
    feature when RLIMIT_MEMLOCK is unlimited and the RAM checks pass."""
    import tuner

    monkeypatch.setattr(tuner, "_memlock_limit_gb", lambda: None)
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=5.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(vram_total=0), profile)
    assert cfg.mlock is True
    assert cfg.no_mmap is True


# ---------------------------------------------------------------------------
# MTP detection (scanner.metadata_has_embedded_mtp) — tri-state scan logic


def _mtp_meta(**kw):
    return dict(kw)


def test_mtp_single_file_key_and_tensor() -> None:
    """Standard single-file MTP GGUF: metadata key + scan found → detected."""
    md = _mtp_meta(
        **{
            "general.architecture": "qwen3moe",
            "qwen3moe.nextn_predict_layers": 1,
            "__mtp_scan__": "found",
        }
    )
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_sharded_inconclusive_trusts_key() -> None:
    """Sharded MTP model read from shard 1: the nextn block lives in the LAST
    shard, so the scan is 'inconclusive'. The metadata key must still be
    trusted — this was the 'sometimes detected, sometimes not' bug."""
    md = _mtp_meta(
        **{
            "general.architecture": "deepseek2",
            "deepseek2.nextn_predict_layers": 1,
            "__mtp_scan__": "inconclusive",
        }
    )
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_ud_quant_absent_suppresses_false_positive() -> None:
    """UD/unsloth quant keeps the metadata key but strips the MTP weights. A
    complete non-sharded scan reports 'absent' → key is correctly vetoed."""
    md = _mtp_meta(
        **{
            "general.architecture": "qwen2",
            "qwen2.nextn_predict_layers": 1,
            "__mtp_scan__": "absent",
        }
    )
    assert metadata_has_embedded_mtp(md) is False


@pytest.mark.parametrize(("last_block", "expected"), [(41, False), (42, True)])
def test_sharded_mtp_scan_combines_all_part_headers(
    tmp_path, monkeypatch, last_block, expected
) -> None:
    """A stale nextn key must not expose MTP when every shard proves that the
    extra block is absent; a real final-shard block remains detected."""
    import scanner

    part1 = tmp_path / "Ling-3.0-flash-00001-of-00002.gguf"
    part2 = tmp_path / "Ling-3.0-flash-00002-of-00002.gguf"
    _write_minimal_gguf(part1)
    _write_minimal_gguf(part2)

    def fake_metadata(path):
        common = {
            "split.count": 2,
            "__mtp_scan__": "inconclusive",
            "__tensor_scan_complete__": True,
        }
        if path == part1:
            return {
                **common,
                "general.architecture": "bailingmoe3",
                "bailingmoe3.block_count": 42,
                "bailingmoe3.nextn_predict_layers": 1,
                "__max_block_index__": 41,
            }
        return {**common, "__max_block_index__": last_block}

    monkeypatch.setattr(scanner, "read_gguf_metadata", fake_metadata)
    entries = scanner.scan_models(tmp_path)
    assert len(entries) == 1
    assert entries[0].metadata["__mtp_scan__"] == ("found" if expected else "absent")
    assert entries[0].has_embedded_mtp is expected


def test_mtp_name_based_without_key() -> None:
    """No metadata key at all, but the tensor scan saw a nextn-named tensor
    (scan='found'). Detection must succeed without 'MTP' in the filename."""
    md = _mtp_meta(**{"general.architecture": "bailingmoe2", "__mtp_scan__": "found"})
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_external_metadata_no_scan_trusts_key() -> None:
    """Metadata from a source that ran no tensor scan (no __mtp_scan__ key):
    the metadata key is trusted."""
    md = _mtp_meta(
        **{"general.architecture": "qwen3moe", "qwen3moe.nextn_predict_layers": 2}
    )
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_negative_clean_model() -> None:
    md = _mtp_meta(**{"general.architecture": "llama", "__mtp_scan__": "absent"})
    assert metadata_has_embedded_mtp(md) is False
    assert metadata_has_embedded_mtp({}) is False


# ---------------------------------------------------------------------------
# n-gram (ngram-mod) speculative decoding flags


def _spec_tokens(cmd):
    """Return the --spec-type value, or '' if none was emitted."""
    if "--spec-type" in cmd:
        return cmd[cmd.index("--spec-type") + 1]
    return ""


def test_external_qwen_mtp_head_emits_draft_mtp_with_and_without_mmproj(
    tmp_path, monkeypatch
) -> None:
    """Tess's external qwen35 nextn head must use the dedicated MTP path.

    Without ``--spec-type draft-mtp`` llama.cpp treats the sparse 18-tensor
    sidecar as a complete sibling model and crashes while loading it.
    """
    import tuner

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Tess-4-27B-Q6_K", size_gb=20.0)
    model.metadata = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 64,
    }
    draft = _fake_model(tmp_path, "mtp-Tess-4-27B-Q4_K_M", size_gb=1.9)
    draft.metadata = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 65,
        "qwen35.nextn_predict_layers": 1,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile, draft_model=draft)

    cmd = build_command(model, cfg, profile, draft_model=draft)
    assert "-md" in cmd
    assert "draft-mtp" in _spec_tokens(cmd).split(",")

    plain_draft = _fake_model(tmp_path, "Tess-4-27B-Draft-Q4_K_M", size_gb=1.0)
    plain_cmd = build_command(model, cfg, profile, draft_model=plain_draft)
    assert "-md" in plain_cmd
    assert "draft-mtp" not in _spec_tokens(plain_cmd).split(",")

    mmproj = tmp_path / "mmproj-Tess-4-27B-F16.gguf"
    _write_minimal_gguf(mmproj)
    model.mmproj = mmproj
    monkeypatch.setattr(
        tuner,
        "_probe_supported_flags",
        lambda _binary: {"-m", "--model", "-md", "--model-draft", "--spec-type"},
    )
    cmd = build_command(
        model, cfg, profile, draft_model=draft, server_binary="modern-llama-server"
    )
    assert "--mmproj" in cmd and "-md" in cmd
    assert "draft-mtp" in _spec_tokens(cmd).split(",")


def test_qwen38_dflash2_uses_trained_depth_and_requires_capable_build(
    tmp_path, monkeypatch
) -> None:
    """DFlash2 reuses draft-dflash but stock b10590 lacks its 81-tensor graph."""
    import tuner

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.8-27B-Q4_K_M", size_gb=15.0)
    model.metadata = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 65,
        "qwen35.nextn_predict_layers": 1,
        "__mtp_scan__": "found",
    }
    draft = _fake_model(tmp_path, "Qwen3.8-27B-DFlash2-Q4_K_M", size_gb=1.1)
    draft.metadata = {
        "general.architecture": "dflash",
        "dflash.block_count": 5,
        "dflash.block_size": 8,
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": 256,
        "dflash.selector_top_k": 16,
    }
    assert draft.is_dflash2_drafter
    assert draft.drafter_spec_type == "dflash"

    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile, draft_model=draft)
    cmd = build_command(model, cfg, profile, draft_model=draft)
    assert "draft-dflash" in _spec_tokens(cmd).split(",")
    depth_index = cmd.index("--spec-draft-n-max")
    assert cmd[depth_index + 1] == "7"
    p_min_index = cmd.index("--spec-draft-p-min")
    assert cmd[p_min_index + 1] == "0.0"

    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _binary: 10590)
    monkeypatch.setattr(
        tuner,
        "_probe_binary_version_output",
        lambda _binary: "version: 0.2.0-dev (build 10590, commit 6657ded4f)",
    )
    allowed, message, detected = tuner.check_draft_model_build(draft, "server")
    assert not allowed
    assert detected == 10590
    assert "PR #27342" in message
    assert "expected" not in message  # actionable preflight, not raw loader jargon

    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _binary: 10499)
    monkeypatch.setattr(
        tuner,
        "_probe_binary_version_output",
        lambda _binary: "version: 0.2.0-dev (build 10499, commit 1deefcca3)",
    )
    allowed, message, detected = tuner.check_draft_model_build(draft, "server")
    assert allowed
    assert message == ""
    assert detected == 10499

    # Builds immediately before the b10658 mainline merge remain incompatible.
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _binary: 10600)
    monkeypatch.setattr(
        tuner,
        "_probe_binary_version_output",
        lambda _binary: "version: 0.2.0-dev (build 10600, commit deadbeef0)",
    )
    allowed, message, detected = tuner.check_draft_model_build(draft, "server")
    assert not allowed
    assert detected == 10600
    assert "b10658+" in message

    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _binary: 10658)
    monkeypatch.setattr(
        tuner,
        "_probe_binary_version_output",
        lambda _binary: "version: 0.3.0-dev (build 10658, commit b10f9ca58)",
    )
    allowed, message, detected = tuner.check_draft_model_build(draft, "server")
    assert allowed
    assert message == ""
    assert detected == 10658


def test_plain_dflash_uses_block_size_minus_anchor(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-27B-Q4_K_M", size_gb=15.0)
    draft = _fake_model(tmp_path, "Qwen3.6-27B-DFlash-Q4_K_M", size_gb=1.0)
    draft.metadata = {
        "general.architecture": "dflash",
        "dflash.block_size": 16,
    }
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile, draft_model=draft)
    cmd = build_command(model, cfg, profile, draft_model=draft)
    depth_index = cmd.index("--spec-draft-n-max")
    assert cmd[depth_index + 1] == "15"


def test_ngram_disabled_by_default(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Llama-3-8B", size_gb=8.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile)
    assert "--spec-type" not in cmd
    assert "--spec-ngram-mod-n-match" not in cmd


def test_ngram_standalone_on_any_model(tmp_path) -> None:
    """ngram-mod needs no draft model — it must work on a plain model."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Llama-3-8B", size_gb=8.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    assert _spec_tokens(cmd) == "ngram-mod"
    assert "--spec-ngram-mod-n-match" in cmd
    assert "--spec-ngram-mod-n-min" in cmd
    assert "--spec-ngram-mod-n-max" in cmd


def test_ngram_suppressed_when_embedded_mtp(tmp_path) -> None:
    """MTP + ngram → ONLY draft-mtp; ngram-mod is dropped.

    Combining draft-mtp,ngram-mod in one --spec-type list crashes
    mid-generation on MTP models (llama.cpp #23154, open as of b9305). The
    trained MTP head supersedes a generic ngram hash anyway, so draft-mtp wins
    and ngram-mod (plus its --spec-ngram-mod-* flags) is suppressed.
    """
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-MoE-A3B", size_gb=8.0)
    model.metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.nextn_predict_layers": 1,
        "qwen3moe.block_count": 48,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    types = _spec_tokens(cmd).split(",")
    assert "draft-mtp" in types
    assert "ngram-mod" not in types
    assert "--spec-ngram-mod-n-match" not in cmd
    assert "--spec-ngram-mod-n-min" not in cmd
    assert "--spec-ngram-mod-n-max" not in cmd


def test_ngram_survives_speculative_disabled(tmp_path) -> None:
    """Unchecking Draft (enable_speculative=False) suppresses MTP but ngram
    is independent and must still be emitted."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-MoE-A3B", size_gb=8.0)
    model.metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.nextn_predict_layers": 1,
        "qwen3moe.block_count": 48,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(
        model, cfg, profile, enable_speculative=False, enable_ngram=True
    )
    assert _spec_tokens(cmd) == "ngram-mod"


def test_ngram_map_k4v_coexists_with_mtp(tmp_path) -> None:
    """b9334: ngram-map-k4v is the MTP-compatible draftless method.

    Unlike ngram-mod (suppressed next to draft-mtp because of #23154), a
    profile that selects ngram_method=ngram-map-k4v must emit BOTH draft-mtp
    and ngram-map-k4v in --spec-type, plus the k4v parameter flags (and never
    the ngram-mod flags). This is ggerganov's MTP clean-up combo (PR #23269).
    """
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-MoE-A3B", size_gb=8.0)
    model.metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.nextn_predict_layers": 1,
        "qwen3moe.block_count": 48,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    profile.ngram_method = "ngram-map-k4v"
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    types = _spec_tokens(cmd).split(",")
    assert "draft-mtp" in types
    assert "ngram-map-k4v" in types
    assert "ngram-mod" not in types
    assert "--spec-ngram-map-k4v-size-n" in cmd
    assert "--spec-ngram-map-k4v-size-m" in cmd
    assert "--spec-ngram-map-k4v-min-hits" in cmd
    assert "--spec-ngram-mod-n-match" not in cmd


def test_draft_n_max_override_wins(tmp_path) -> None:
    """Expert-panel draft_n_max (config) overrides the profile's draft_max.

    0 = unset → profile value; >0 → the override is emitted verbatim as
    --spec-draft-n-max on the active speculative path (here: embedded MTP).
    """
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-MoE-A3B", size_gb=8.0)
    model.metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.nextn_predict_layers": 1,
        "qwen3moe.block_count": 48,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)

    # Default: 0 = kein Override → Profilwert (draft_max) landet im Flag.
    cmd = build_command(model, cfg, profile)
    idx = cmd.index("--spec-draft-n-max")
    assert cmd[idx + 1] == str(profile.draft_max or 2)

    # Override gesetzt → gewinnt über das Profil.
    cfg.draft_n_max = 5
    cmd = build_command(model, cfg, profile)
    idx = cmd.index("--spec-draft-n-max")
    assert cmd[idx + 1] == "5"


def test_ngram_map_k_emits_type_only(tmp_path) -> None:
    """ngram-map-k / ngram-simple / ngram-cache: emit only the --spec-type
    token and rely on llama.cpp defaults (no guessed sub-param flags)."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Llama-3-8B", size_gb=8.0)
    profile = match_profile(model.name, profiles)
    profile.ngram_method = "ngram-map-k"
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    assert _spec_tokens(cmd) == "ngram-map-k"
    assert "--spec-ngram-mod-n-match" not in cmd
    assert "--spec-ngram-map-k4v-size-n" not in cmd


def test_ngram_method_invalid_falls_back(tmp_path) -> None:
    """An unknown ngram_method in YAML must fall back to ngram-mod at load."""
    from settings_loader import _validate_ngram_method

    assert _validate_ngram_method("totally-bogus", "x.yaml") == "ngram-mod"
    assert _validate_ngram_method("NGRAM-MAP-K4V", "x.yaml") == "ngram-map-k4v"


# ---------------------------------------------------------------------------
# GPU detection / iGPU filtering


def test_filter_drops_igpu_next_to_dgpu() -> None:
    """User's exact scenario: Intel iGPU + AMD RX 9070 XT.
    The iGPU must be ignored so the tuner doesn't underuse the dGPU."""
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(
            index=0,
            name="Intel(R) Graphics",
            vendor="intel",
            total_vram_mb=2048,
            free_vram_mb=1900,
        ),
        GPUInfo(
            index=1,
            name="AMD Radeon RX 9070 XT",
            vendor="amd",
            total_vram_mb=16 * 1024,
            free_vram_mb=int(15.2 * 1024),
        ),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 1
    assert used[0].vendor == "amd"
    assert len(ignored) == 1
    assert ignored[0].vendor == "intel"


def test_filter_keeps_matched_dual_gpus() -> None:
    """Two equal dGPUs (e.g. 2x RTX 4090) must both be kept for tensor-split."""
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(
            index=0,
            name="RTX 4090",
            vendor="nvidia",
            total_vram_mb=24 * 1024,
            free_vram_mb=23 * 1024,
        ),
        GPUInfo(
            index=1,
            name="RTX 4090",
            vendor="nvidia",
            total_vram_mb=24 * 1024,
            free_vram_mb=23 * 1024,
        ),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 2 and not ignored


def test_vendor_inference() -> None:
    from hardware import _vendor_from_name

    assert _vendor_from_name("AMD Radeon RX 9070 XT") == "amd"
    assert _vendor_from_name("NVIDIA GeForce RTX 4090") == "nvidia"
    assert _vendor_from_name("Intel(R) UHD Graphics 770") == "intel"
    assert _vendor_from_name("Intel(R) Arc(TM) A770") == "intel"
    assert _vendor_from_name("some-mystery-card") == "unknown"


def test_filter_3gpu_workstation_setup() -> None:
    """Basti's workstation: 9700 Pro 32GB + 9070 XT 16GB + Intel iGPU.

    Both Radeons must be kept (16 is exactly half of 32 → still a peer
    under the >= half-of-largest rule); the iGPU must be ignored; the
    9700 Pro must end up as the first entry (largest), so main_gpu
    selection later picks it.
    """
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(
            index=0,
            name="Intel(R) Graphics",
            vendor="intel",
            total_vram_mb=2048,
            free_vram_mb=1900,
        ),
        GPUInfo(
            index=1,
            name="AMD Radeon RX 9070 XT",
            vendor="amd",
            total_vram_mb=16 * 1024,
            free_vram_mb=14 * 1024,
        ),
        GPUInfo(
            index=2,
            name="AMD Radeon AI PRO R9700",
            vendor="amd",
            total_vram_mb=32 * 1024,
            free_vram_mb=31 * 1024,
        ),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 2
    assert used[0].total_vram_mb == 32 * 1024  # 9700 Pro first (largest)
    assert used[1].total_vram_mb == 16 * 1024  # 9070 XT kept as peer
    assert len(ignored) == 1
    assert ignored[0].vendor == "intel"


def test_vendor_inference_r9700_workstation_card() -> None:
    """The new Radeon AI PRO R9700 must register as AMD (was untested)."""
    from hardware import _vendor_from_name

    assert _vendor_from_name("AMD Radeon AI PRO R9700") == "amd"
    assert _vendor_from_name("Radeon AI PRO R9700 AI TOP") == "amd"


def _fake_dual_gpu_system(
    large_total: float = 32,
    large_free: float = 31,
    small_total: float = 16,
    small_free: float = 14,
    ram_total: float = 96,
    ram_free: float = 80,
) -> SystemInfo:
    """Synthetic two-GPU SystemInfo (large + small) for placement tests."""
    return SystemInfo(
        os_name="Linux test",
        cpu_name="Test CPU",
        cpu_cores_physical=24,
        cpu_cores_logical=24,
        total_ram_gb=ram_total,
        free_ram_gb=ram_free,
        gpus=[
            GPUInfo(
                index=0,
                name="AMD Radeon AI PRO R9700",
                vendor="amd",
                total_vram_mb=int(large_total * 1024),
                free_vram_mb=int(large_free * 1024),
            ),
            GPUInfo(
                index=1,
                name="AMD Radeon RX 9070 XT",
                vendor="amd",
                total_vram_mb=int(small_total * 1024),
                free_vram_mb=int(small_free * 1024),
            ),
        ],
    )


def test_multi_gpu_pins_to_largest_when_model_and_q4_kv_fit(tmp_path) -> None:
    """Keep the peer GPU free when requested context and Q4 KV fit."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_dual_gpu_system(), profile, user_ctx=32768)

    assert cfg.cache_k == cfg.cache_v == "q4_0"
    assert cfg.tensor_split is not None, "Expected tensor_split to be set"
    assert cfg.main_gpu == 0, f"Expected main_gpu=0 (largest), got {cfg.main_gpu}"

    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    assert parts[0] > 0.99, f"Expected ~1.0 on GPU 0, got {parts[0]}"
    assert parts[1] < 0.01, f"Expected ~0.0 on GPU 1, got {parts[1]}"


def _fake_qwen38_27b_model(tmp_path, name: str, size_gb: float):
    """Qwen3.8-27B metadata shared by the user's Q3–Q8 GGUF variants."""
    return _fake_model_md(
        tmp_path,
        name,
        size_gb,
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 65,
            "qwen35.context_length": 262144,
            "qwen35.embedding_length": 5120,
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.nextn_predict_layers": 1,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.group_count": 16,
            "qwen35.ssm.time_step_rank": 48,
            "qwen35.ssm.inner_size": 6144,
            "qwen35.full_attention_interval": 4,
        },
    )


@pytest.mark.parametrize(
    ("name", "size_gb"),
    [
        ("Qwen3.8-27B-UD-Q3_K_XL", 12.52),
        ("Qwen3.8-27B-UD-Q4_K_XL", 16.69),
        ("Qwen3.8-27B-UD-Q5_K_XL", 18.83),
        ("Qwen3.8-27B-UD-Q6_K_XL", 24.14),
        ("Qwen3.8-27B-UD-Q8_K_XL", 29.30),
    ],
)
def test_qwen38_dual_gpu_preserves_full_context_with_q4_default(
    tmp_path, name, size_gb
) -> None:
    """Every weight quant keeps native context with the global Q4 KV pair."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_qwen38_27b_model(tmp_path, name, size_gb)
    profile = match_profile(model.name, profiles, model.architecture)
    system = _fake_dual_gpu_system_with_vk_order(
        large_free=31,
        small_free=16,
        ram_total=48,
        ram_free=30,
    )
    cfg = compute_config(
        model,
        system,
        profile,
        gpu_priorities={
            "AMD Radeon AI PRO R9700": 2,
            "AMD Radeon RX 9070 XT": 1,
        },
    )

    assert cfg.ctx == 262144
    assert (cfg.cache_k, cfg.cache_v) == ("q4_0", "q4_0")

    # Small variants may now stay on the R9700 because Auto no longer spreads
    # solely to upgrade KV precision. Larger variants still use both cards.
    from tuner import _gpu_usable_cap_gb

    footprint = (
        cfg.estimated_model_vram_gb
        + cfg.kv_vram_gb
        + cfg.recurrent_state_vram_gb
        + cfg.runtime_vram_overhead_gb
        + cfg.batch_vram_overhead_gb
    )
    small_cap = _gpu_usable_cap_gb(system.gpus[1], False)
    large_cap = _gpu_usable_cap_gb(system.gpus[0], True)
    if cfg.tensor_split is None:
        assert footprint <= large_cap + 0.05
    else:
        # tensor_split is runtime order: RX 9070 XT, then R9700.
        parts = [float(part) for part in cfg.tensor_split.split(",")]
        assert footprint * parts[0] <= small_cap + 0.05
        assert footprint * parts[1] <= large_cap + 0.05


def test_qwen4exp_plans_130k_with_lazy_ple_active_residency(
    tmp_path, monkeypatch
) -> None:
    import tuner
    from performance_target import PERFORMANCE_TARGETS

    model = _fake_model_md(
        tmp_path,
        "Qwen3.8-Flash-Next-UD-IQ1_S",
        67.564157158,
        {
            "general.architecture": "qwen4exp",
            "qwen4exp.block_count": 48,
            "qwen4exp.context_length": 262144,
            "qwen4exp.embedding_length": 2560,
            "qwen4exp.attention.head_count": 24,
            "qwen4exp.attention.head_count_kv": 2,
            "qwen4exp.attention.key_length": 256,
            "qwen4exp.attention.value_length": 256,
            "qwen4exp.attention.indexer.key_length": 128,
            "qwen4exp.expert_count": 512,
            "qwen4exp.expert_used_count": 10,
            "qwen4exp.ssm.conv_kernel": 4,
            "qwen4exp.ssm.state_size": 128,
            "qwen4exp.ssm.group_count": 16,
            "qwen4exp.ssm.time_step_rank": 48,
            "qwen4exp.ssm.inner_size": 6144,
            "qwen4exp.full_attention_interval": 4,
            "__read_lazy_tensor_bytes__": 28_800_138_240,
        },
    )
    monkeypatch.setattr(tuner.platform, "system", lambda: "Windows")
    profile = match_profile(model.name, load_profiles(SETTINGS_DIR), model.architecture)
    cfg = compute_config(
        model,
        _fake_dual_gpu_system_with_vk_order(
            large_free=31,
            small_free=16,
            ram_total=48,
            ram_free=30.6,
        ),
        profile,
        user_ctx=130000,
        perf_target=PERFORMANCE_TARGETS["safe"],
        prompt_cache_ram_mib=2048,
    )

    assert cfg.ctx == 130000
    assert cfg.ubatch == 64
    assert cfg.mapped_model_ram_gb == pytest.approx(model.read_lazy_size_gb)
    assert cfg.mapped_model_resident_gb == pytest.approx(
        model.read_lazy_size_gb * 0.05
    )
    assert cfg.estimated_model_vram_gb + cfg.estimated_model_ram_gb == pytest.approx(
        model.placement_size_gb
    )
    assert (cfg.cache_k, cfg.cache_v) == ("q4_0", "q4_0")
    assert cfg.estimated_kv_gb == pytest.approx(1.319, rel=0.03)
    assert cfg.runtime_vram_overhead_gb == pytest.approx(0.558, rel=0.05)
    assert cfg.runtime_ram_overhead_gb == pytest.approx(7.17, rel=0.05)
    assert "file-backed" in (cfg.warning or "")

    cmd = build_command(model, cfg, profile)
    lazy_index = cmd.index("--tensor-read-lazy")
    assert cmd[lazy_index + 1] == "auto"


def test_dense_split_reserves_primary_only_mmproj_vram(tmp_path) -> None:
    """A dense tensor split must not pretend the mmproj is split by layer."""
    from tuner import _gpu_usable_cap_gb

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_qwen38_27b_model(tmp_path, "Qwen3.8-27B-UD-Q6_K_XL", size_gb=24.14)
    model.mmproj = types.SimpleNamespace(
        stat=lambda: types.SimpleNamespace(st_size=int(3.585 * 1024**3))
    )
    profile = match_profile(model.name, profiles, model.architecture)
    system = _fake_dual_gpu_system_with_vk_order(large_free=31, small_free=16)
    cfg = compute_config(
        model,
        system,
        profile,
        gpu_priorities={
            "AMD Radeon AI PRO R9700": 2,
            "AMD Radeon RX 9070 XT": 1,
        },
    )

    assert cfg.tensor_split is not None
    small_share, large_share = map(float, cfg.tensor_split.split(","))
    distributed = (
        cfg.estimated_model_vram_gb + cfg.kv_vram_gb + cfg.recurrent_state_vram_gb
    )
    primary_only = (
        cfg.vision_vram_gb
        + cfg.draft_vram_gb
        + cfg.runtime_vram_overhead_gb
        + cfg.batch_vram_overhead_gb
    )
    small_cap = _gpu_usable_cap_gb(system.gpus[1], False)
    large_cap = _gpu_usable_cap_gb(system.gpus[0], True)
    assert distributed * small_share <= small_cap + 0.05
    assert distributed * large_share + primary_only <= large_cap + 0.05


def test_low_vram_multi_gpu_does_not_pin_oversized_weights_to_one_card(
    tmp_path,
) -> None:
    """--no-kv-offload cannot make oversized fixed weights look pin-safe."""
    from performance_target import PERFORMANCE_TARGETS

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(
        model,
        _fake_dual_gpu_system_with_vk_order(large_free=31, small_free=16),
        profile,
        user_ctx=32768,
        perf_target=PERFORMANCE_TARGETS["low_vram"],
    )

    assert cfg.no_kv_offload is True
    assert cfg.estimated_model_vram_gb > 32.0
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "0,1"
    assert cfg.tensor_split is not None
    assert all(float(part) > 0 for part in cfg.tensor_split.split(","))


def test_low_vram_ignores_full_peer_and_reduces_primary_gpu_weights(
    tmp_path,
) -> None:
    """An unusable peer cannot make oversized weights look single-card safe."""
    from performance_target import PERFORMANCE_TARGETS
    from tuner import _gpu_usable_cap_gb

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    system = _fake_dual_gpu_system_with_vk_order(large_free=31, small_free=1)
    cfg = compute_config(
        model,
        system,
        profile,
        user_ctx=32768,
        perf_target=PERFORMANCE_TARGETS["low_vram"],
    )

    primary_cap = _gpu_usable_cap_gb(system.gpus[0], True)
    assert cfg.no_kv_offload is True
    assert cfg.full_offload is False
    assert (
        cfg.estimated_model_vram_gb
        + PERFORMANCE_TARGETS["low_vram"].dense_vram_safety_gb
        <= primary_cap + 0.01
    )
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "1"
    assert cfg.tensor_split is None


def test_multi_gpu_spreads_when_model_too_large_for_single_gpu(tmp_path) -> None:
    """Smart placement: a model too big for the 9700 Pro alone (with KV +
    safety) must spread across both GPUs proportionally to total VRAM."""
    profiles = load_profiles(SETTINGS_DIR)
    # 40 GB model — well over the 32 GB single-GPU budget once KV/safety
    # are added; must use both GPUs.
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_dual_gpu_system(), profile)

    assert cfg.tensor_split is not None
    assert cfg.main_gpu == 0  # still the largest

    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    # Proportional split: 32/48 ≈ 0.667 and 16/48 ≈ 0.333. Either both
    # are non-trivial (true spread), OR the model didn't fit at all and
    # we offloaded to RAM — but in that case the split is still set,
    # so we only assert it's not the pinned form.
    pinned = parts[0] > 0.99 and parts[1] < 0.01
    assert not pinned, f"Expected proportional split, got pinned: {cfg.tensor_split}"


def _fake_dual_gpu_system_with_vk_order(
    large_total: float = 32,
    large_free: float = 31,
    small_total: float = 16,
    small_free: float = 14,
    ram_total: float = 96,
    ram_free: float = 80,
    large_vk_idx: int = 1,
    small_vk_idx: int = 0,
) -> SystemInfo:
    """Two-GPU system with hip_index set (simulating vulkaninfo resolution).

    Default: small GPU is Vulkan device 0, large is device 1 — the
    typical case on Windows where the primary display adapter (the
    gaming GPU) enumerates first in Vulkan.
    """
    return SystemInfo(
        os_name="Windows 11 test",
        cpu_name="Test CPU",
        cpu_cores_physical=24,
        cpu_cores_logical=24,
        total_ram_gb=ram_total,
        free_ram_gb=ram_free,
        gpus=[
            GPUInfo(
                index=0,
                name="AMD Radeon AI PRO R9700",
                vendor="amd",
                total_vram_mb=int(large_total * 1024),
                free_vram_mb=int(large_free * 1024),
                hip_index=large_vk_idx,
            ),
            GPUInfo(
                index=1,
                name="AMD Radeon RX 9070 XT",
                vendor="amd",
                total_vram_mb=int(small_total * 1024),
                free_vram_mb=int(small_free * 1024),
                hip_index=small_vk_idx,
            ),
        ],
    )


def test_mmproj_device_tracks_post_visibility_main_gpu(tmp_path) -> None:
    """MTMD must use the same physical GPU that owns its VRAM budget."""
    profiles = load_profiles(SETTINGS_DIR)
    mmproj = tmp_path / "mmproj-gemma-4-f16.gguf"
    mmproj.write_bytes(b"x" * 1024)

    def exact_vulkan_system() -> SystemInfo:
        system = _fake_dual_gpu_system_with_vk_order()
        for gpu in system.gpus:
            gpu.runtime_backend = "Vulkan"
            gpu.runtime_device = f"Vulkan{gpu.hip_index}"
        return system

    # Pin: Vulkan1 is hidden/remapped to Vulkan0, so the projector must use 0.
    pinned_model = _fake_model(tmp_path, "gemma-4-12b-it-q4", size_gb=9.0)
    pinned_model.mmproj = mmproj
    pinned_profile = match_profile(
        pinned_model.name, profiles, pinned_model.architecture
    )
    pinned = compute_config(
        pinned_model, exact_vulkan_system(), pinned_profile, user_ctx=32768
    )
    assert pinned.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "1"
    assert pinned.main_gpu == 0
    assert pinned.mmproj_device == "Vulkan0"

    pinned_cmd = build_command(pinned_model, pinned, pinned_profile)
    mmdev_idx = pinned_cmd.index("--mmproj-device")
    assert pinned_cmd[mmdev_idx + 1] == "Vulkan0"

    # Spread: both devices remain visible in Vulkan order and the R9700 is 1.
    spread_model = _fake_model(
        tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0
    )
    spread_model.mmproj = mmproj
    spread_profile = match_profile(
        spread_model.name, profiles, spread_model.architecture
    )
    spread = compute_config(spread_model, exact_vulkan_system(), spread_profile)
    assert spread.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "0,1"
    assert spread.main_gpu == 1
    assert spread.mmproj_device == "Vulkan1"


def test_vulkan_env_var_emitted_for_single_gpu_pinning(tmp_path) -> None:
    """When hip_index is known and the model fits on one GPU, both
    HIP_VISIBLE_DEVICES and GGML_VK_VISIBLE_DEVICES must be emitted
    so the config works on both ROCm and Vulkan backends."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order()
    cfg = compute_config(model, sys_info, profile, user_ctx=32768)

    assert cfg.full_offload is True
    assert cfg.main_gpu == 0  # remapped: only one device visible
    # Both env vars must point to the R9700's Vulkan index (1).
    assert "HIP_VISIBLE_DEVICES" in cfg.env_overrides
    assert "GGML_VK_VISIBLE_DEVICES" in cfg.env_overrides
    assert cfg.env_overrides["HIP_VISIBLE_DEVICES"] == "1"
    assert cfg.env_overrides["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_vulkan_env_var_emitted_for_multi_gpu_spread(tmp_path) -> None:
    """When a large model must spread across both GPUs, both
    HIP_VISIBLE_DEVICES and GGML_VK_VISIBLE_DEVICES must list
    all devices in Vulkan order."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order()
    cfg = compute_config(model, sys_info, profile)

    assert "GGML_VK_VISIBLE_DEVICES" in cfg.env_overrides
    # Both env vars must list devices in ascending Vulkan order.
    expected = cfg.env_overrides["HIP_VISIBLE_DEVICES"]
    assert cfg.env_overrides["GGML_VK_VISIBLE_DEVICES"] == expected


def test_priority_weighted_tensor_split(tmp_path) -> None:
    """With gpu_priorities R9700=2 / 9070 XT=1, the R9700 must receive
    a significantly larger share of the tensor_split than a pure
    VRAM-proportional split would give. This keeps the gaming GPU
    (9070 XT) mostly free for OBS/desktop while the AI GPU (R9700)
    does the heavy lifting."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order(
        large_free=31,
        small_free=14,
    )
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(model, sys_info, profile, gpu_priorities=prio)

    assert cfg.tensor_split is not None
    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    # Vulkan order: device 0 = 9070 XT (small_vk_idx=0), device 1 = R9700 (large_vk_idx=1)
    xt_share = parts[0]  # 9070 XT
    r97_share = parts[1]  # R9700
    # R9700 receives the lion's share, but the split must not exceed its
    # physical usable cap once KV is included. A hard >75% requirement is
    # impossible for a 40 GB model when the card's safe cap is ~29 GB.
    assert r97_share > xt_share
    footprint = (
        cfg.estimated_model_vram_gb
        + cfg.kv_vram_gb
        + cfg.recurrent_state_vram_gb
        + cfg.runtime_vram_overhead_gb
        + cfg.batch_vram_overhead_gb
    )
    assert footprint * r97_share <= 31.0
    assert footprint * xt_share <= 14.0


def test_second_server_avoids_full_primary(tmp_path) -> None:
    """Regression (Basti report #2): launching a second server must NOT
    pile onto the high-priority R9700 when server #1 has already filled it.

    Scenario: the 32 GB R9700 is down to ~1 GB free (server #1 holds it),
    while the 16 GB RX 9070 XT still has ~13 GB free. A new ~10 GB model
    scored purely by priority×VRAM would still pick the R9700 and OOM.
    The free-VRAM demotion must instead route it to the 9070 XT, which is
    the only card that can actually hold the weights.
    """
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=10.0)
    profile = match_profile(model.name, profiles)
    # R9700 nearly full (1 GB free), 9070 XT mostly free (13 GB free).
    sys_info = _fake_dual_gpu_system_with_vk_order(
        large_free=1,
        small_free=13,
    )
    # Even with the R9700 given a higher priority, free VRAM must win:
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(model, sys_info, profile, gpu_priorities=prio)

    # The model fits on the 9070 XT alone → exclusive pin to its Vulkan
    # index (small_vk_idx=0), and the (full) R9700 must be hidden.
    assert cfg.full_offload is True
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "0", (
        "second server must pin to the only card with free VRAM (9070 XT, "
        f"Vulkan 0); got {cfg.env_overrides.get('GGML_VK_VISIBLE_DEVICES')!r}"
    )
    assert cfg.env_overrides.get("HIP_VISIBLE_DEVICES") == "0"


def test_second_server_launch_balance_keeps_available_gpu_pin(
    tmp_path, monkeypatch
) -> None:
    """Exercise live re-detect → choose → final pin, not only compute_config."""
    import qt_launcher

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=10.0)
    profile = match_profile(model.name, profiles)
    system = _fake_dual_gpu_system_with_vk_order(large_free=1, small_free=13)
    cfg = compute_config(
        model,
        system,
        profile,
        user_ctx=16384,
        gpu_priorities={
            "AMD Radeon AI PRO R9700": 2,
            "AMD Radeon RX 9070 XT": 1,
        },
    )
    logs = []
    window = types.SimpleNamespace(
        _system=system,
        _log=logs.append,
        _active_llama_binary=lambda: "llama-server",
        _last_pinned_gpu=None,
    )
    monkeypatch.setattr(qt_launcher, "detect_system", lambda _binary: system)

    chosen, refusal = qt_launcher.MainWindow._choose_gpu_for_launch(
        window, cfg, model, "llama-server"
    )
    assert refusal is None
    assert chosen is system.gpus[1]
    qt_launcher.MainWindow._pin_cfg_to_gpu(window, cfg, chosen)
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "0"
    assert cfg.env_overrides.get("HIP_VISIBLE_DEVICES") == "0"
    assert cfg.tensor_split is None
    assert window._last_pinned_gpu == chosen.name


def test_force_gpu_pins_to_named_card(tmp_path) -> None:
    """force_gpu must pin the server to the named card EXCLUSIVELY, even
    when another GPU scores higher by priority×VRAM. This is the manual
    "boot only on the GPU I choose" override."""
    profiles = load_profiles(SETTINGS_DIR)
    # Small model that would otherwise auto-pin to the 32 GB R9700.
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order()  # both cards have room

    # Force onto the 9070 XT via a substring of its name.
    cfg = compute_config(model, sys_info, profile, force_gpu="9070")

    assert cfg.full_offload is True
    # 9070 XT is Vulkan device 0 (small_vk_idx=0) — it must be the sole
    # visible device, hiding the R9700 the auto-logic would have chosen.
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "0"
    assert cfg.env_overrides.get("HIP_VISIBLE_DEVICES") == "0"
    forced_footprint = (
        cfg.estimated_model_vram_gb
        + cfg.kv_vram_gb
        + cfg.recurrent_state_vram_gb
        + cfg.runtime_vram_overhead_gb
        + cfg.batch_vram_overhead_gb
    )
    assert forced_footprint <= sys_info.gpus[1].free_vram_gb


def test_force_gpu_unknown_name_falls_back_to_auto(tmp_path) -> None:
    """An unknown force_gpu name is ignored — the config must fall back to
    automatic selection rather than crashing or producing no placement."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order()

    cfg = compute_config(model, sys_info, profile, user_ctx=32768, force_gpu="RTX 5090")

    # Falls back to auto: small model pins to the R9700 (Vulkan index 1).
    assert cfg.full_offload is True
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "1"


# ---------------------------------------------------------------------------
# llama-server resolver


def test_prerelease_build_recipes_keep_truthful_backend_qualified_identity() -> None:
    build_dir = ROOT / "building llama.cpp"
    common = (build_dir / "windows_llama_build_common.ps1").read_text(
        encoding="utf-8"
    )
    vulkan = (build_dir / "llama_prerelease_vulkan_build.ps1").read_text(
        encoding="utf-8"
    )
    hip = (build_dir / "llama_prerelease_hip_build.ps1").read_text(
        encoding="utf-8"
    )

    assert "Get-LatestLlamaPrereleaseTag" in common
    assert "Get-ExactRemotePrereleaseTag" in common
    assert '"${buildTag}_dev_${shortCommit}"' in common
    assert '"${folderVersion}_${backendToken}_llama.cpp"' in common
    assert 'return "bUNKNOWN"' not in common
    assert '-BuildIsDev "ON"' in common
    assert "$expectedRuntime = \"$semantic-dev\"" in common
    assert "-ExpectedBuild $build" in common
    assert "--single-branch" in common and "--depth" not in common
    assert "-Backend Vulkan" in vulkan
    assert "-Backend HIP" in hip
    assert "bNNNN_vulkan_llama.cpp" in vulkan
    assert "bNNNN_hip_llama.cpp" in hip


def test_stable_build_recipes_resolve_latest_and_preserve_build_number() -> None:
    build_dir = ROOT / "building llama.cpp"
    common = (build_dir / "windows_llama_build_common.ps1").read_text(
        encoding="utf-8"
    )
    vulkan = (build_dir / "llama_stable_vulkan_build.ps1").read_text(
        encoding="utf-8"
    )
    hip = (build_dir / "llama_stable_hip_build.ps1").read_text(encoding="utf-8")

    assert "Get-LatestStableSemanticTag" in common
    assert "$version = $Tag -replace '^v', ''" in common
    assert '"${version}_${backendToken}_llama.cpp"' in common
    assert '-BuildIsDev "OFF"' in common
    assert "Get-LlamaBuildCount" in common
    assert "Stable/pre-release mismatch" in common
    assert "--single-branch" in common and "--depth" not in common
    assert "-ExpectedBuild $build" in common
    assert "X.Y.Z_vulkan_llama.cpp" in vulkan
    assert "X.Y.Z_hip_llama.cpp" in hip


def test_windows_vulkan_and_hip_recipe_matrix_is_complete_and_pinned() -> None:
    build_dir = ROOT / "building llama.cpp"
    pairs = [
        ("llama_prerelease_vulkan_build.ps1", "llama_prerelease_hip_build.ps1"),
        ("llama_stable_vulkan_build.ps1", "llama_stable_hip_build.ps1"),
        ("diffusion_vulkan_llama_build.ps1", "diffusion_hip_llama_build.ps1"),
        ("ocr_vulkan_llama_build.ps1", "ocr_hip_llama_build.ps1"),
        (
            "ternary_bonsai_vulkan_llama_build.ps1",
            "ternary_bonsai_hip_llama_build.ps1",
        ),
        ("turboquant_vulkan_llama_build.ps1", "turboquant_hip_llama_build.ps1"),
    ]
    for vulkan_name, hip_name in pairs:
        vulkan = (build_dir / vulkan_name).read_text(encoding="utf-8")
        hip = (build_dir / hip_name).read_text(encoding="utf-8")
        assert "-Backend Vulkan" in vulkan
        assert "-Backend HIP" in hip
        assert "_vulkan_llama.cpp" in vulkan
        assert "_hip_llama.cpp" in hip
        vulkan_commit = next(
            (line for line in vulkan.splitlines() if "-ExpectedCommit" in line),
            None,
        )
        hip_commit = next(
            (line for line in hip.splitlines() if "-ExpectedCommit" in line),
            None,
        )
        if vulkan_commit is not None:
            assert vulkan_commit.replace("Vulkan", "HIP") == hip_commit

    common = (build_dir / "windows_llama_build_common.ps1").read_text(
        encoding="utf-8"
    )
    assert '"-DGGML_VULKAN=ON"' in common
    assert '"-DGGML_HIP=OFF"' in common
    assert '"-DGGML_HIP=ON"' in common
    assert '"-DGGML_VULKAN=OFF"' in common
    assert '"-DGPU_TARGETS=gfx1201"' in common
    assert '"-DGGML_AVX_VNNI=ON"' in common
    assert '"-DGGML_BMI2=ON"' in common
    assert '"-G", "Ninja"' in common
    assert '"-DCMAKE_CXX_COMPILER=$clangxx"' in common
    assert '"-DCMAKE_HIP_COMPILER=$clang"' in common
    assert "Get-HipCompatibilityResourceDir" in common
    assert "llvm-pr201563" in common
    assert "Copy-HipRuntimeDependencies" in common
    assert 'New-Item -ItemType Junction' in common
    for runtime_dll in (
        "amdhip64_7.dll",
        "amd_comgr_3.dll",
        "hipblas.dll",
        "rocblas.dll",
        "hipblaslt.dll",
    ):
        assert runtime_dll in common


def test_merged_qwen4exp_and_dflash2_pr_recipes_are_removed() -> None:
    build_dir = ROOT / "building llama.cpp"
    assert not (build_dir / "qwen38_flash_next_llama_build.txt").exists()
    assert not (build_dir / "dflash2_llama_build.txt").exists()


def test_cuda_setup_never_updates_a_stale_versioned_folder_in_place() -> None:
    recipe = (ROOT / "building llama.cpp" / "setup_llamacpp_cuda.ps1").read_text(
        encoding="utf-8-sig"
    )

    assert "Get-ExactLlamaBuildTag" in recipe
    assert "_tmp_llama_$PID" in recipe
    assert '"${buildTag}_dev_${shortCommit}"' in recipe
    assert "git pull --ff-only origin" not in recipe
    assert "-DLLAMA_BUILD_IS_DEV=ON" in recipe
    assert "reportedVersion -ne $expectedRuntimeVersion" in recipe
    assert "reportedBuild -ne $expectedBuild" in recipe
    assert "$env:ComSpec" in recipe


def _fake_llama_server_path(fork_root: Path) -> Path:
    binary = "llama-server.exe" if os.name == "nt" else "llama-server"
    return fork_root / "build" / "bin" / "Release" / binary


def _write_fake_server(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(path.stat().st_mode | 0o755)


def test_resolver_returns_input_when_nothing_matches(tmp_path, monkeypatch) -> None:
    """When the binary can't be found anywhere, the resolver echoes the
    original input — `launch()` then prints a clean 'not found' error
    rather than us silently swallowing the failure."""
    from auto_tuner import _resolve_server_binary

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    result = _resolve_server_binary("definitely-not-a-binary-anywhere")
    assert result == "definitely-not-a-binary-anywhere"


def test_resolver_finds_binary_in_sibling_llama_cpp(tmp_path, monkeypatch) -> None:
    """Simulate the user's layout: auto_tuner sits beside an ai-local/
    folder that contains the llama.cpp checkout."""
    from auto_tuner import _resolve_server_binary

    # Build the tree: tmp/Auto Tuner/, tmp/ai-local/llama.cpp/build/...
    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    server = _fake_llama_server_path(tmp_path / "ai-local" / "llama.cpp")
    _write_fake_server(server)

    monkeypatch.chdir(auto_dir)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

    resolved = _resolve_server_binary("llama-server")
    assert Path(resolved).resolve() == server.resolve(), (
        f"expected {server}, got {resolved}"
    )


def test_discovery_lists_backend_twins_only_after_server_build(
    tmp_path, monkeypatch
) -> None:
    """Vulkan/HIP siblings coexist; source-only or CLI-only trees are skipped."""
    from auto_tuner import _discover_llama_forks

    container = tmp_path / "ai-local"
    # Use deliberately synthetic versions so a developer's real documented
    # L:/LAB/ai-local builds cannot collide with this temp container scan.
    runnable_roots = [
        container / "ocr_b17499_vulkan_llama.cpp",
        container / "ocr_b17499_hip_llama.cpp",
        container / "9.9.9_vulkan_llama.cpp",
        container / "9.9.9_hip_llama.cpp",
        container / "v9.8.7_llama.cpp",  # legacy backend-neutral folder
    ]
    for root in runnable_roots:
        _write_fake_server(_fake_llama_server_path(root))

    cli_only_root = container / "ocr_cli_only_llama.cpp"
    mtmd_name = "llama-mtmd-cli.exe" if os.name == "nt" else "llama-mtmd-cli"
    mtmd = cli_only_root / "build" / "bin" / "Release" / mtmd_name
    _write_fake_server(mtmd)

    monkeypatch.setenv("LLAMA_CPP_DIR", str(container))
    monkeypatch.chdir(tmp_path)

    found = _discover_llama_forks()
    found_map = {name: path.resolve() for name, path in found}
    for root in runnable_roots:
        assert found_map.get(root.name) == root.resolve()
    assert "ocr_cli_only_llama.cpp" not in found_map

    # GUI has a separate container scanner; keep its server-only contract in
    # lockstep with terminal discovery.
    from qt_launcher import MainWindow

    gui_found = MainWindow._expand_fork_container(container)
    gui_map = {name: path.resolve() for name, path in gui_found}
    for root in runnable_roots:
        assert gui_map.get(root.name) == root.resolve()
    assert "ocr_cli_only_llama.cpp" not in gui_map


def test_dflash2_launch_finds_compatible_sibling_build(
    tmp_path, monkeypatch
) -> None:
    import auto_tuner

    regular_root = tmp_path / "b10590_llama.cpp"
    dflash_root = tmp_path / "pr27342_dflash2_llama.cpp"
    regular = _fake_llama_server_path(regular_root)
    dflash = _fake_llama_server_path(dflash_root)
    _write_fake_server(regular)
    _write_fake_server(dflash)
    draft = _fake_model(tmp_path, "Qwen3.8-27B-DFlash2-Q4_K_M", 1.1)
    draft.metadata = {
        "general.architecture": "dflash",
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": 256,
        "dflash.selector_top_k": 16,
    }

    monkeypatch.setattr(
        auto_tuner,
        "_discover_llama_forks",
        lambda: [(regular_root.name, regular_root), (dflash_root.name, dflash_root)],
    )

    def compatibility(_draft, binary):
        supported = Path(binary).resolve() == dflash.resolve()
        return supported, "" if supported else "requires PR #27342", 10499

    monkeypatch.setattr(auto_tuner, "check_draft_model_build", compatibility)
    selected = auto_tuner._find_compatible_draft_server(draft, str(regular))
    assert selected is not None
    assert Path(selected).resolve() == dflash.resolve()


def test_resolver_distinguishes_between_llama_and_1b_llama(
    tmp_path, monkeypatch
) -> None:
    """The Bonsai-Ternary profile uses a relative path starting with the
    fork's directory name. The resolver must respect that and pick the
    1b_llama.cpp checkout, not the regular one sitting next to it."""
    from auto_tuner import _resolve_server_binary

    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    regular = _fake_llama_server_path(tmp_path / "ai-local" / "llama.cpp")
    bitnet = _fake_llama_server_path(tmp_path / "ai-local" / "1b_llama.cpp")
    for s in (regular, bitnet):
        _write_fake_server(s)

    monkeypatch.chdir(auto_dir)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

    # Default resolves to the regular fork
    res1 = _resolve_server_binary("llama-server")
    assert Path(res1).resolve() == regular.resolve()

    # Profile-style relative path must hit the BitNet fork
    res2 = _resolve_server_binary(f"1b_llama.cpp/build/bin/Release/{bitnet.name}")
    assert Path(res2).resolve() == bitnet.resolve()


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only executable filtering")
def test_resolver_ignores_windows_exe_on_posix(tmp_path, monkeypatch) -> None:
    """A shared llama.cpp folder may contain only Windows .exe builds.
    Linux/macOS must not auto-select them, otherwise model launch fails with
    PermissionError / Exec format error and looks like a model-load crash.
    """
    from auto_tuner import _resolve_server_binary

    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    win_only = (
        tmp_path
        / "ai-local"
        / "llama.cpp"
        / "build"
        / "bin"
        / "Release"
        / "llama-server.exe"
    )
    win_only.parent.mkdir(parents=True)
    win_only.write_text("not a native binary", encoding="utf-8")
    win_only.chmod(0o755)

    monkeypatch.chdir(auto_dir)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "ai-local"))
    monkeypatch.setenv("PATH", "")

    assert _resolve_server_binary("llama-server") == "llama-server"


def test_resolver_matches_backend_qualified_fork_twins(tmp_path, monkeypatch) -> None:
    """Backend-neutral hints preserve the selected backend; explicit wins."""
    from auto_tuner import _fork_backend, _fork_family, _resolve_server_binary

    expected_families = {
        "2b_b9551_vulkan_llama.cpp": "2b_llama",
        "2b_b9551_hip_llama.cpp": "2b_llama",
        "b10678_vulkan_llama.cpp": "llama",
        "b10678_hip_llama.cpp": "llama",
        "d_b10000_hip_llama.cpp": "d_llama",
        "ocr_b17400_vulkan_llama.cpp": "ocr_llama",
        "tq_0.3.0_vulkan_llama.cpp": "tq_llama",
        "b10548_dev_a298422da_llama.cpp": "llama",  # legacy
        "1b_llama.cpp": "1b_llama",  # legacy
    }
    for name, family in expected_families.items():
        assert _fork_family(name) == family
    assert _fork_backend("2b_b9551_vulkan_llama.cpp") == "vulkan"
    assert _fork_backend("2b_b9551_hip_llama.cpp") == "hip"
    assert _fork_backend("2b_b9551_llama.cpp") is None

    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    container = tmp_path / "ai-local"
    mainline_hip = _fake_llama_server_path(container / "b10678_hip_llama.cpp")
    fork2b_vulkan = _fake_llama_server_path(
        container / "2b_b9551_vulkan_llama.cpp"
    )
    fork2b_hip = _fake_llama_server_path(container / "2b_b9551_hip_llama.cpp")
    fork1b = _fake_llama_server_path(container / "1b_llama.cpp")
    for server in (mainline_hip, fork2b_vulkan, fork2b_hip, fork1b):
        _write_fake_server(server)

    monkeypatch.chdir(auto_dir)
    # Point at the selected HIP build, not merely the container. A neutral
    # profile switch must retain HIP even though Vulkan sorts first by default.
    monkeypatch.setenv("LLAMA_CPP_DIR", str(mainline_hip.parents[3]))

    neutral = _resolve_server_binary("2b_llama/llama-server")
    assert Path(neutral).resolve() == fork2b_hip.resolve()
    explicit = _resolve_server_binary("2b_vulkan_llama/llama-server")
    assert Path(explicit).resolve() == fork2b_vulkan.resolve()
    one_bit = _resolve_server_binary("1b_llama/llama-server")
    assert Path(one_bit).resolve() == fork1b.resolve()


def test_gui_auto_select_preserves_backend_for_required_family() -> None:
    from qt_launcher import MainWindow

    class Combo:
        def __init__(self) -> None:
            self.items = [
                "b10678_hip_llama.cpp",
                "2b_b9551_vulkan_llama.cpp",
                "2b_b9551_hip_llama.cpp",
            ]
            self.index = 0

        def count(self) -> int:
            return len(self.items)

        def itemText(self, index: int) -> str:
            return self.items[index]

        def currentText(self) -> str:
            return self.items[self.index]

        def currentIndex(self) -> int:
            return self.index

        def setCurrentIndex(self, index: int) -> None:
            self.index = index

        def blockSignals(self, _blocked: bool) -> None:
            pass

    combo = Combo()
    applied: list[int] = []
    fake_window = types.SimpleNamespace(
        _fork_combo=combo,
        _fork_manual_override=False,
        _apply_fork=lambda index: applied.append(index),
        _log=lambda _message: None,
    )

    neutral = types.SimpleNamespace(server_binary="2b_llama/llama-server")
    MainWindow._auto_select_fork(fake_window, neutral)
    assert combo.index == 2  # selected mainline HIP -> required-family HIP
    assert applied == [2]

    combo.index = 0
    explicit = types.SimpleNamespace(
        server_binary="2b_vulkan_llama/llama-server"
    )
    MainWindow._auto_select_fork(fake_window, explicit)
    assert combo.index == 1
    assert applied[-1] == 1


def test_eagle3_dflash_and_dspark_drafter_detection(tmp_path) -> None:
    """Current external draft families are classified and routed correctly."""
    from scanner import (
        metadata_is_drafter_file,
        metadata_is_standalone_drafter,
        read_gguf_metadata,
        _is_draft_filename,
        _DRAFT_MARKER_RE,
    )

    # Architecture-based detection
    assert metadata_is_drafter_file({"general.architecture": "eagle3"})
    assert metadata_is_drafter_file({"general.architecture": "dflash"})
    assert metadata_is_drafter_file({"general.architecture": "dspark"})
    assert metadata_is_drafter_file({"general.architecture": "gemma4-assistant"})
    assert not metadata_is_drafter_file({"general.architecture": "qwen3"})
    # eagle3/dflash are NOT 'standalone drafter' (that term is reserved for
    # the MTP assistant heads) — they must not trigger the draft-mtp path.
    assert not metadata_is_standalone_drafter({"general.architecture": "eagle3"})

    # Filename token detection
    assert _is_draft_filename("Qwen3-4B-eagle3.gguf")
    assert _is_draft_filename("Qwen3-4B-eagle3-Q8_0.gguf")
    assert _is_draft_filename("Qwen3-4B-dflash.gguf")
    assert _is_draft_filename("dspark-Qwen3-4B.gguf")
    assert _DRAFT_MARKER_RE.search("Qwen3-4B-eagle3.gguf")
    assert _DRAFT_MARKER_RE.search("Qwen3-4B-dflash.gguf")
    assert _DRAFT_MARKER_RE.search("dspark-Qwen3-4B.gguf")

    def entry(arch: str, name: str, **metadata) -> ModelEntry:
        return ModelEntry(
            path=Path(name + ".gguf"),
            name=name,
            group=".",
            size_bytes=1024,
            metadata={"general.architecture": arch, **metadata},
        )

    assert entry("eagle3", "eagle").drafter_spec_type == "eagle3"
    assert entry("dflash", "dflash").drafter_spec_type == "dflash"
    assert entry("dflash", "dspark-model").drafter_spec_type == "dspark"
    assert (
        entry("dflash", "opaque", __dspark_scan__="found").drafter_spec_type == "dspark"
    )
    assert entry("gemma4-assistant", "mtp").drafter_spec_type == "mtp"
    assert entry("qwen3", "plain").drafter_spec_type is None

    # Exercise the real tensor-table scan instead of only injecting the
    # synthetic marker used by ModelEntry.
    dspark_file = tmp_path / "opaque-sidecar.gguf"
    key = b"general.architecture"
    arch = b"dflash"
    tensor = b"markov_w1.weight"
    with dspark_file.open("wb") as stream:
        stream.write(b"GGUF")
        stream.write(struct.pack("<I", 3))
        stream.write(struct.pack("<Q", 1))  # tensor count
        stream.write(struct.pack("<Q", 1))  # KV count
        stream.write(struct.pack("<Q", len(key)) + key)
        stream.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
        stream.write(struct.pack("<Q", len(arch)) + arch)
        stream.write(struct.pack("<Q", len(tensor)) + tensor)
        stream.write(struct.pack("<I", 1))  # n_dims
        stream.write(struct.pack("<Q", 1))  # dim[0]
        stream.write(struct.pack("<I", 0))  # tensor type
        stream.write(struct.pack("<Q", 0))  # offset
    metadata = read_gguf_metadata(dspark_file)
    assert metadata["__dspark_scan__"] == "found"
    assert entry("dflash", "opaque", **metadata).drafter_spec_type == "dspark"


def test_dspark_command_and_pre_b10164_compatibility(tmp_path, monkeypatch) -> None:
    import tuner

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3-4B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    draft = ModelEntry(
        path=tmp_path / "dspark-Qwen3-4B.gguf",
        name="dspark-Qwen3-4B",
        group=".",
        size_bytes=1024,
        metadata={
            "general.architecture": "dflash",
            "__dspark_scan__": "found",
        },
    )
    monkeypatch.setattr(
        tuner,
        "_probe_supported_flags",
        lambda _binary: {"-m", "--model", "--spec-type"},
    )
    cmd = build_command(model, cfg, profile, draft_model=draft)
    idx = cmd.index("--spec-type")
    assert cmd[idx + 1] == "draft-dspark"
    assert "-md" in cmd

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10163)
    adapted, notes = tuner._adapt_spec_types_for_binary(cmd)
    assert "draft-dspark" not in " ".join(adapted)
    assert "-md" not in adapted
    assert not any(token.startswith("--spec-draft-") for token in adapted)
    assert "requires llama.cpp b10164+" in notes[0]

    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _binary: 10164)
    current, notes = tuner._adapt_spec_types_for_binary(cmd)
    assert current == cmd
    assert notes == []


def test_turbo_quant_mode_selection(tmp_path, monkeypatch) -> None:
    """Test that the turbo mode selection logic works (mocking input)."""
    from auto_tuner import main
    import sys
    from unittest.mock import patch

    # Mocking sys.argv to avoid command line arguments
    sys.argv = ["auto_tuner.py"]

    # We can't easily test the interactive input in a pure unit test
    # without complex mocking, but we can verify the logic if we
    # were to refactor main. For now, we ensure no crashes occur
    # when simulating different inputs.
    with patch("builtins.input", side_effect=["2", KeyboardInterrupt]):
        try:
            main([])
        except (KeyboardInterrupt, SystemExit):
            pass  # Expected behavior for testing exit


# ---------------------------------------------------------------------------
# Profile schema (server_binary field)


def test_profile_supports_server_binary_field() -> None:
    """Bonsai-Ternary should declare its preferred server binary.

    Ternary-Bonsai (1.58-bit, Q2_0) uses the 2-bit PrismML fork (``2b_…``),
    NOT the 1-bit Bonsai fork (``1b_…``). The two must stay distinct.
    """
    profiles = load_profiles(SETTINGS_DIR)
    by_name = {p.source_file: p for p in profiles}
    assert "bonsai-ternary.yaml" in by_name
    p = by_name["bonsai-ternary.yaml"]
    assert p.server_binary, "Bonsai-Ternary profile must set server_binary"
    assert "2b_llama" in p.server_binary.lower()


def test_ternary_bonsai_pattern_beats_regular_bonsai() -> None:
    """Longest-pattern-wins: a Ternary-Bonsai filename must match the
    BitNet profile, not the generic Bonsai one."""
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile("Ternary-Bonsai-8B-Q2_0", profiles)
    assert p.source_file is not None, "matched profile must come from a YAML file"
    assert "ternary" in (p.display_name + " " + p.source_file).lower()
    assert p.server_binary, "Ternary profile must override the server"


# ---------------------------------------------------------------------------
# Performance targets


def test_performance_target_registry_has_four_tiers() -> None:
    """Sanity: the documented tiers exist and are well-ordered.

    The first three (safe / balanced / throughput) are the performance
    tiers; ``low_vram`` is a special-purpose LOW-VRAM escape hatch kept
    last so it never shifts the default (``balanced``).
    """
    from performance_target import PERFORMANCE_TARGETS, list_target_names

    names = list_target_names()
    assert names == ["safe", "balanced", "throughput", "low_vram"]
    safe = PERFORMANCE_TARGETS["safe"]
    bal = PERFORMANCE_TARGETS["balanced"]
    thr = PERFORMANCE_TARGETS["throughput"]
    lv = PERFORMANCE_TARGETS["low_vram"]
    # KV reservation should shrink monotonically: safe ≥ balanced ≥ throughput
    assert safe.moe_placement_ctx_target >= bal.moe_placement_ctx_target
    assert bal.moe_placement_ctx_target >= thr.moe_placement_ctx_target
    # Same for VRAM safety bands
    assert safe.moe_vram_safety_gb >= bal.moe_vram_safety_gb
    assert bal.moe_vram_safety_gb >= thr.moe_vram_safety_gb
    # kv_to_ram is the defining lever of the low_vram tier, and ONLY it.
    assert lv.kv_to_ram is True
    assert PERFORMANCE_TARGETS["safe"].kv_to_ram is False
    assert PERFORMANCE_TARGETS["balanced"].kv_to_ram is False
    assert PERFORMANCE_TARGETS["throughput"].kv_to_ram is False


def test_resolve_performance_target_priority() -> None:
    """CLI choice beats profile choice beats default."""
    from performance_target import resolve_performance_target

    # CLI wins
    assert resolve_performance_target("safe", "throughput").name == "safe"
    # Profile wins when CLI is empty
    assert resolve_performance_target(None, "throughput").name == "throughput"
    # Default kicks in when both empty
    assert resolve_performance_target(None, None).name == "balanced"
    # Unknown values are silently skipped
    assert resolve_performance_target("typo", "alsotypo").name == "balanced"
    # Mixed case + whitespace tolerated
    assert resolve_performance_target("  Throughput ", None).name == "throughput"


def test_yaml_performance_target_is_loaded() -> None:
    """qwen3_5-3_6.yaml ships with performance_target: throughput."""
    profiles = load_profiles(SETTINGS_DIR)
    by_file = {p.source_file: p for p in profiles}
    assert "qwen3_5-3_6.yaml" in by_file
    assert by_file["qwen3_5-3_6.yaml"].performance_target == "throughput"


def test_throughput_places_more_moe_layers_on_gpu_than_safe(tmp_path) -> None:
    """The whole point of the perf_target switch: on a constrained MoE
    setup, 'throughput' should reserve less KV and therefore put MORE
    expert layers on the GPU than 'safe'.

    We simulate Basti's actual situation: Qwen3.6-A3B (~28 GB Q6_K),
    16 GB VRAM, 48 GB free RAM. Different ctx targets in the two tiers
    must produce different cpu_moe counts.
    """
    from performance_target import PERFORMANCE_TARGETS

    # Use a real MoE profile — Qwen3.6 ships expert_count via GGUF in the
    # wild, but our fake_model has empty metadata. So we hand-craft a
    # ModelEntry with metadata that tells the tuner "this is MoE".
    from scanner import ModelEntry

    p = tmp_path / "Qwen3.6-35B-A3B-UD-Q6_K.gguf"
    _write_minimal_gguf(p)
    model = ModelEntry(
        path=p,
        name="Qwen3.6-35B-A3B-UD-Q6_K",
        group=".",
        size_bytes=int(28.0 * 1024**3),
        metadata={
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_count": 128,
            "qwen3moe.block_count": 64,
            "qwen3moe.context_length": 262144,
        },
    )

    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(ram_total=64, ram_free=48, vram_total=16, vram_free=14)

    cfg_safe = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["safe"]
    )
    cfg_thr = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["throughput"]
    )

    # Both should detect the MoE
    assert cfg_safe.is_moe and cfg_thr.is_moe

    # Throughput places fewer experts on CPU (= more on GPU). Allow equal
    # for edge cases where the model fully fits or fully doesn't fit, but
    # in this constrained scenario throughput must beat safe strictly.
    safe_cpu = cfg_safe.n_cpu_moe or 0
    thr_cpu = cfg_thr.n_cpu_moe or 0
    assert thr_cpu < safe_cpu, (
        f"throughput should keep more experts on GPU than safe; "
        f"got safe={safe_cpu} CPU experts, throughput={thr_cpu}"
    )

    # And the metadata field round-trips into TunedConfig
    assert cfg_safe.performance_target == "safe"
    assert cfg_thr.performance_target == "throughput"


def test_low_vram_unlocks_huge_context_on_tiny_gpu(tmp_path) -> None:
    """The LOW-VRAM colleague scenario: 8 GB VRAM, 64 GB RAM, a 20 GB MoE.

    With the default (balanced) tier the MoE KV budget is VRAM-only, so
    the leftover VRAM after expert placement throttles context to a few
    thousand tokens — useless for agentic coding (needs 90-130k). The
    ``low_vram`` tier moves the KV cache into system RAM via
    --no-kv-offload, drawing context headroom from the abundant RAM
    instead, and must reach dramatically higher context.
    """
    from performance_target import PERFORMANCE_TARGETS
    from scanner import ModelEntry

    p = tmp_path / "Qwen3-30B-A3B-Q4_K_M.gguf"
    _write_minimal_gguf(p)
    model = ModelEntry(
        path=p,
        name="Qwen3-30B-A3B-Q4_K_M",
        group=".",
        size_bytes=int(20.0 * 1024**3),
        metadata={
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_count": 128,
            "qwen3moe.block_count": 48,
            "qwen3moe.context_length": 262144,
            "qwen3moe.attention.head_count_kv": 8,
            "qwen3moe.attention.key_length": 128,
            "qwen3moe.attention.value_length": 128,
        },
    )

    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    # 8 GB VRAM (~7.4 GB free), 64 GB RAM (~58 GB free).
    sys_info = _fake_system(ram_total=64, ram_free=58, vram_total=8, vram_free=7.4)

    cfg_bal = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["balanced"]
    )
    cfg_lv = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["low_vram"]
    )

    # low_vram must unlock context an order of magnitude larger — enough
    # for agentic coding (>= 90k), and here capped by the model's native
    # 262k rather than by RAM.
    assert cfg_lv.ctx >= 90000, f"low_vram ctx too small: {cfg_lv.ctx}"
    assert cfg_lv.ctx > cfg_bal.ctx * 5, (
        f"low_vram should beat balanced by >5×; "
        f"balanced={cfg_bal.ctx}, low_vram={cfg_lv.ctx}"
    )
    # The lever is active and the KV cache is accounted as RAM-resident.
    assert cfg_lv.no_kv_offload is True
    assert cfg_lv.kv_ram_gb > 1.0 and cfg_lv.kv_vram_gb < 0.1
    # balanced is untouched by the new path (no regression on the default).
    assert cfg_bal.no_kv_offload is False

    # build_command must emit the actual flag.
    cmd = build_command(model, cfg_lv, profile, server_binary="llama-server")
    assert "--no-kv-offload" in cmd
    # And it composes with --n-cpu-moe (experts on CPU, KV in RAM).
    assert "--n-cpu-moe" in cmd


def test_low_vram_does_not_regress_high_end_gpu(tmp_path) -> None:
    """On a comfortable GPU the default tiers must be byte-for-byte
    unchanged by the low_vram addition: no_kv_offload stays False for
    safe / balanced / throughput, so their KV budget path is untouched."""
    from performance_target import PERFORMANCE_TARGETS

    model = _fake_model(tmp_path, "Qwen3-30B-A3B-Q4_K_M", size_gb=20.0)
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(ram_total=64, ram_free=58, vram_total=16, vram_free=15)

    for tier in ("safe", "balanced", "throughput"):
        cfg = compute_config(
            model, sys_info, profile, perf_target=PERFORMANCE_TARGETS[tier]
        )
        assert cfg.no_kv_offload is False, f"{tier} must not enable --no-kv-offload"


# ---------------------------------------------------------------------------
# RoPE-Scaling (YaRN) auto-activation — regression suite for the v4.8.0
# fix. Previously RoPE never activated in ANY mode: the gate checked
# `desired_ctx > native_ctx` with desired_ctx = profile.max_context, which
# every profile sets to the NATIVE context → structurally never true.
# Even rope_scale.enabled: true was dead code. Only the GUI Expert
# checkbox (force_rope_scale) worked.
# ---------------------------------------------------------------------------


def _qwen3_moe_model(tmp_path):
    """20 GB Qwen3-MoE, native 256k, supports RoPE scaling (qwen prefix)."""
    p = tmp_path / "Qwen3-30B-A3B-Q4_K_M.gguf"
    _write_minimal_gguf(p)
    from scanner import ModelEntry

    return ModelEntry(
        path=p,
        name="Qwen3-30B-A3B-Q4_K_M",
        group=".",
        size_bytes=int(20.0 * 1024**3),
        metadata={
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_count": 128,
            "qwen3moe.block_count": 48,
            "qwen3moe.context_length": 262144,
            "qwen3moe.attention.head_count_kv": 8,
            "qwen3moe.attention.key_length": 128,
            "qwen3moe.attention.value_length": 128,
        },
    )


def test_rope_activates_when_profile_enabled_low_vram(tmp_path) -> None:
    """rope_scale.enabled: true must actually switch YaRN on.

    Before v4.8.0 the profile switch was dead code: the activation gate
    compared profile.max_context (== native) to native_ctx and was never
    satisfied. On a low_vram box (8 GB VRAM / 64 GB RAM) the abundant RAM
    lets the KV budget reach beyond native → RoPE must fire.
    """
    import dataclasses
    from performance_target import PERFORMANCE_TARGETS

    model = _qwen3_moe_model(tmp_path)
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    profile_on = dataclasses.replace(profile, rope_scale_enabled=True)
    sys_info = _fake_system(ram_total=64, ram_free=58, vram_total=8, vram_free=7.4)

    cfg = compute_config(
        model, sys_info, profile_on, perf_target=PERFORMANCE_TARGETS["low_vram"]
    )
    cmd = build_command(model, cfg, profile_on, server_binary="llama-server")

    assert cfg.rope_scaling is True, "enabled=True must activate RoPE"
    assert "--rope-scaling" in cmd
    assert "--rope-scale" in cmd
    assert cfg.ctx > model.native_context, "YaRN must extend context past native"


def test_rope_activates_on_user_ctx_beyond_native(tmp_path) -> None:
    """Pinning ctx > native activates RoPE at the budget-supported level.

    The old all-or-nothing check required the FULL pinned value × 1.1 to
    fit; otherwise it silently clamped to native WITHOUT emitting YaRN,
    so the user got less context than asked for and no idea why. Now RoPE
    activates as soon as the budget supports anything beyond native.
    """
    from performance_target import PERFORMANCE_TARGETS

    model = _qwen3_moe_model(tmp_path)
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(ram_total=64, ram_free=58, vram_total=8, vram_free=7.4)

    cfg = compute_config(
        model,
        sys_info,
        profile,
        perf_target=PERFORMANCE_TARGETS["low_vram"],
        user_ctx=524288,  # 512k > native 256k
    )
    cmd = build_command(model, cfg, profile, server_binary="llama-server")

    assert cfg.rope_scaling is True
    assert "--rope-scaling" in cmd
    # Partial: must exceed native even if it can't reach the full 512k.
    assert cfg.ctx > model.native_context


def test_rope_stays_off_when_profile_disabled_and_auto(tmp_path) -> None:
    """enabled=False + no user pin → RoPE must NOT silently activate.

    The profile default is OFF deliberately (YaRN beyond native has a
    quality cost). We must not turn it on for every Qwen model just
    because RAM is available — that would surprise users.
    """
    from performance_target import PERFORMANCE_TARGETS

    model = _qwen3_moe_model(tmp_path)
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    assert profile.rope_scale_enabled is False  # precondition
    sys_info = _fake_system(ram_total=64, ram_free=58, vram_total=8, vram_free=7.4)

    cfg = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["low_vram"]
    )
    assert cfg.rope_scaling is False


def test_rope_factor_derived_from_context_not_fixed(tmp_path) -> None:
    """YaRN scale factor tracks the reached context (ceil(ctx/native)),
    capped at the profile's max factor — not always the flat 4.0.

    Over-stretching RoPE to 1M when only ~300k is used degrades quality at
    the positions actually exercised. The factor must fit the context.
    """
    import dataclasses
    from performance_target import PERFORMANCE_TARGETS

    model = _qwen3_moe_model(tmp_path)
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    profile_on = dataclasses.replace(profile, rope_scale_enabled=True)
    sys_info = _fake_system(ram_total=64, ram_free=58, vram_total=8, vram_free=7.4)

    cfg = compute_config(
        model, sys_info, profile_on, perf_target=PERFORMANCE_TARGETS["low_vram"]
    )
    # native is 256k; if ctx lands in (256k, 512k] the factor is 2,
    # in (512k, 768k] it's 3, in (768k, 1M] it's 4. Either way it must be
    # <= profile max (4.0) and > 1, and consistent with the context chosen.
    assert 1 < cfg.rope_scale_factor <= profile.rope_scale_factor
    import math

    expected = math.ceil(cfg.ctx / model.native_context)
    assert cfg.rope_scale_factor == min(expected, profile.rope_scale_factor)


def test_low_vram_dense_model_near_vram_limit_not_full_offload(tmp_path) -> None:
    """A dense model that fits the WEIGHTS but not weights+compute-buffer
    must use PARTIAL offload (not full_off), so the GUI pre-launch check
    doesn't refuse it on an 8 GB card (the gemma-4-12b report).

    Before v4.8.0 a ~7 GB model on a 7.4 GB-free card was marked full_off
    (weights fit in usable VRAM), then the +1 GB pre-launch margin
    refused it. FULL_OFF_HEADROOM_GB makes it spill a layer or two to CPU.
    """
    from performance_target import PERFORMANCE_TARGETS
    from scanner import ModelEntry

    p = tmp_path / "gemma-4-12b-Q4_0.gguf"
    _write_minimal_gguf(p)
    model = ModelEntry(
        path=p,
        name="gemma-4-12b-Q4_0",
        group=".",
        size_bytes=int(7.0 * 1024**3),
        metadata={
            "general.architecture": "gemma4",
            "gemma4.block_count": 48,
            "gemma4.context_length": 32768,
            "gemma4.attention.head_count_kv": 8,
            "gemma4.attention.key_length": 256,
            "gemma4.attention.value_length": 256,
        },
    )
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(ram_total=64, ram_free=58, vram_total=8, vram_free=7.4)

    cfg = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["low_vram"]
    )
    # The whole point: not flagged full-offload, so the pre-launch check
    # (which only hard-refuses full_off) lets it through.
    assert cfg.full_offload is False
    assert cfg.ngl < 48 or cfg.full_offload is False  # spilt to CPU or tight partial
    # And the GPU footprint (weights only — KV is in RAM in low_vram) fits.
    assert cfg.estimated_model_vram_gb <= 7.4


def test_perf_target_default_balanced_when_profile_has_none(tmp_path) -> None:
    """A profile without performance_target: falls back to balanced."""
    profiles = load_profiles(SETTINGS_DIR)
    # Pick a profile that we know has no perf_target set
    p_default = next((pr for pr in profiles if pr.source_file == "_default.yaml"), None)
    assert p_default is not None
    assert p_default.performance_target == ""

    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    cfg = compute_config(model, _fake_system(), p_default)
    assert cfg.performance_target == "balanced"


# ---------------------------------------------------------------------------
# Hybrid Mamba/Transformer detection (Punkt 3)


def test_hybrid_architecture_detection_by_name() -> None:
    """Architecture name matches a known hybrid → True."""
    from scanner import metadata_is_hybrid_architecture

    assert (
        metadata_is_hybrid_architecture({"general.architecture": "nemotron_h"}) is True
    )
    assert metadata_is_hybrid_architecture({"general.architecture": "jamba"}) is True
    assert (
        metadata_is_hybrid_architecture({"general.architecture": "bailingmoe3"}) is True
    )
    assert metadata_is_hybrid_architecture({"general.architecture": "kimi-k3"}) is True
    # Pure Transformer
    assert (
        metadata_is_hybrid_architecture({"general.architecture": "qwen3moe"}) is False
    )
    assert metadata_is_hybrid_architecture({"general.architecture": "llama"}) is False


def test_hybrid_detection_via_ssm_keys() -> None:
    """Generic SSM key catches new hybrid archs not on the allow-list."""
    from scanner import metadata_is_hybrid_architecture

    md = {
        "general.architecture": "future_hybrid_arch",
        "future_hybrid_arch.ssm.state_size": 16,
    }
    assert metadata_is_hybrid_architecture(md) is True


def test_attention_layer_count_pure_transformer() -> None:
    """For pure Transformer, attention count == block count."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "qwen3moe",
        "qwen3moe.block_count": 64,
    }
    assert metadata_attention_layer_count(md) == 64


def test_laguna_sliding_window_counts_only_global_kv_layers() -> None:
    """Laguna defaults to FULL/SWA/SWA/SWA when the pattern key is absent."""
    from scanner import metadata_attention_layer_count
    from tuner import kv_per_token_mb_from_metadata

    md = {
        "general.architecture": "laguna",
        "laguna.block_count": 48,
        "laguna.embedding_length": 3072,
        "laguna.attention.head_count": 72,
        "laguna.attention.head_count_kv": 8,
        "laguna.attention.key_length": 128,
        "laguna.attention.value_length": 128,
        "laguna.attention.sliding_window": 512,
    }
    assert metadata_attention_layer_count(md) == 12
    # 12 global layers × 8 KV heads × (128 K + 128 V) × f16.
    assert kv_per_token_mb_from_metadata(md) == pytest.approx(0.046875)


def test_sliding_window_pattern_array_overrides_default_period() -> None:
    """An explicit per-layer SWA pattern remains authoritative."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "laguna",
        "laguna.block_count": 8,
        "laguna.attention.sliding_window": 512,
        "laguna.attention.sliding_window_pattern": [False, True] * 4,
    }
    assert metadata_attention_layer_count(md) == 4


def test_attention_layer_count_hybrid_with_explicit_metadata() -> None:
    """Hybrid with explicit attention count uses that, not the heuristic."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "nemotron_h",
        "nemotron_h.block_count": 50,
        "nemotron_h.attention.block_count": 12,
    }
    assert metadata_attention_layer_count(md) == 12


def test_attention_layer_count_hybrid_with_heuristic() -> None:
    """Hybrid without explicit count falls back to per-arch ratio."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "nemotron_h",
        "nemotron_h.block_count": 50,
    }
    # Nemotron heuristic: ~25%
    n = metadata_attention_layer_count(md)
    assert 10 <= n <= 16, f"expected ~25% of 50, got {n}"


def test_new_hybrid_architecture_fallback_counts() -> None:
    """Current Kimi-K3 and Ling-3 layouts get their published MLA counts."""
    from scanner import metadata_attention_layer_count

    assert (
        metadata_attention_layer_count(
            {
                "general.architecture": "kimi-k3",
                "kimi-k3.block_count": 93,
            }
        )
        == 24
    )
    assert (
        metadata_attention_layer_count(
            {
                "general.architecture": "bailingmoe3",
                "bailingmoe3.block_count": 42,
            }
        )
        == 7
    )


def test_kv_per_token_estimate_smaller_for_hybrid(tmp_path) -> None:
    """A hybrid model should produce a smaller per-token KV estimate
    than the same-shaped pure Transformer would."""
    from tuner import kv_per_token_mb_from_metadata

    common = {
        "attention.head_count": 32,
        "attention.head_count_kv": 8,
        "embedding_length": 4096,
        "block_count": 50,
    }
    pure = {
        "general.architecture": "llama",
        **{f"llama.{k}": v for k, v in common.items()},
    }
    hybrid = {
        "general.architecture": "nemotron_h",
        **{f"nemotron_h.{k}": v for k, v in common.items()},
    }

    pure_kv = kv_per_token_mb_from_metadata(pure)
    hybrid_kv = kv_per_token_mb_from_metadata(hybrid)
    assert pure_kv > 0 and hybrid_kv > 0
    # Hybrid should be roughly 25% of pure (Nemotron heuristic).
    ratio = hybrid_kv / pure_kv
    assert 0.15 <= ratio <= 0.35, f"expected hybrid KV ≈ 25% of pure, got {ratio:.2%}"


def test_two_pass_placement_recovers_when_first_pass_dumps_to_cpu(tmp_path) -> None:
    """Defensive net: if pass 1 forces all experts to CPU but VRAM is
    free, pass 2 with a halved target_ctx must recover at least one
    layer onto the GPU."""
    from performance_target import PERFORMANCE_TARGETS
    from scanner import ModelEntry

    # Construct a tight scenario: 28 GB MoE on 14 GB VRAM, but we lie
    # about KV-per-token by NOT supplying metadata, so the params-based
    # heuristic kicks in (which over-estimates for MoE) and pass 1 dumps
    # everything to CPU. The Two-Pass should retry and pull at least
    # one layer back onto the GPU once the target_ctx is halved.
    p = tmp_path / "FakeMoE-35B-A3B.gguf"
    _write_minimal_gguf(p)
    model = ModelEntry(
        path=p,
        name="FakeMoE-35B-A3B",
        group=".",
        size_bytes=int(28.0 * 1024**3),
        metadata={
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_count": 128,
            "qwen3moe.block_count": 64,
            "qwen3moe.context_length": 262144,
            # Deliberately no attention.* keys → forces params heuristic.
        },
    )
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(ram_total=64, ram_free=48, vram_total=16, vram_free=14)

    cfg_safe = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["safe"]
    )
    # Even safe (128k target) should not dump all 64 layers to CPU when
    # the two-pass kicks in.
    assert (cfg_safe.n_cpu_moe or 0) < 64, (
        f"two-pass should have rescued at least one layer; "
        f"got n_cpu_moe={cfg_safe.n_cpu_moe}"
    )


# ---------------------------------------------------------------------------
# Reasoning / thinking detection (Punkt 2)


def test_thinking_detection_qwen3_coder_is_false() -> None:
    """Qwen3-Coder must NOT be flagged as a thinking model just because
    its filename contains 'qwen3' — that's the bug we're fixing."""
    from scanner import metadata_supports_thinking

    # No metadata at all — pure filename heuristic.
    assert metadata_supports_thinking({}, "Qwen3-Coder-30B-A3B-Q6_K") is False


def test_thinking_detection_uses_chat_template() -> None:
    """When the chat template contains thinking markers, return True."""
    from scanner import metadata_supports_thinking

    md_with_think = {
        "tokenizer.chat_template": "{% if enable_thinking %}<think>\n{% endif %}"
    }
    assert metadata_supports_thinking(md_with_think, "SomeModel") is True


def test_thinking_detection_template_without_marker_is_false() -> None:
    """A model that ships a chat template without <think> markers is not
    a thinking model — even if its name contains 'qwen3'."""
    from scanner import metadata_supports_thinking

    md_no_think = {
        "tokenizer.chat_template": "<|im_start|>user\n{{ messages }}<|im_end|>"
    }
    # Filename hint says "qwen3" → would have been True under old heuristic.
    assert metadata_supports_thinking(md_no_think, "Qwen3.6-35B-Q6") is False


def test_thinking_detection_falls_back_to_filename_when_no_template() -> None:
    """No metadata at all → use filename keywords."""
    from scanner import metadata_supports_thinking

    assert metadata_supports_thinking({}, "Gemma-3-27B-it") is True
    assert metadata_supports_thinking({}, "DeepSeek-R1-Distill-Llama-70B") is True
    assert metadata_supports_thinking({}, "QwQ-32B-Preview") is True
    # Pure non-thinking
    assert metadata_supports_thinking({}, "Mistral-7B-Instruct") is False


def test_thinking_detection_excludes_qwen3_2507_instruct() -> None:
    """Qwen3-2507-Instruct is the explicitly non-thinking branch — even
    if it inherits a generic template that mentions <think>, the filename
    exclusion must win."""
    from scanner import metadata_supports_thinking

    md_with_think = {"tokenizer.chat_template": "<think>{{ content }}</think>"}
    assert (
        metadata_supports_thinking(md_with_think, "Qwen3-7B-Instruct-2507-Q6_K")
        is False
    )


# ---------------------------------------------------------------------------
# mmproj detection — token-anywhere + .mmproj extension (regression guards)


def test_mmproj_token_detected_mid_name(tmp_path) -> None:
    """A projector whose 'mmproj' marker sits mid-name (after the quant
    label) must still be classified as a projector and paired with its
    model. Regression for qwen3.6-…-mxfp4-moe-mmproj-f16.gguf, which the
    old prefix-only check ('mmproj-' / 'mmproj_') missed entirely."""
    folder = tmp_path / "Alibaba" / "Abliterated"
    _write_minimal_gguf(folder / "qwen3.6-35b-a3b-mxfp4_moe.gguf")
    _write_minimal_gguf(folder / "qwen3.6-35b-a3b-mxfp4-moe-mmproj-f16.gguf")

    entries = scan_models(tmp_path)
    # Only the model is a choosable entry; the projector is attached.
    assert len(entries) == 1, [e.name for e in entries]
    model = entries[0]
    assert model.name == "qwen3.6-35b-a3b-mxfp4_moe"
    assert model.has_vision, "mid-name mmproj projector was not paired"
    assert model.mmproj is not None
    assert model.mmproj.name == "qwen3.6-35b-a3b-mxfp4-moe-mmproj-f16.gguf"


def test_mmproj_extension_file_is_paired(tmp_path) -> None:
    """A projector saved with a literal '.mmproj' extension (e.g. some
    audio projectors) is not matched by the '*.gguf' glob; the scanner
    must pick it up via the dedicated '*.mmproj' pass and attach it."""
    folder = tmp_path / "Liquid AI"
    _write_minimal_gguf(folder / "LFM2.5-Audio-1.5B-bf16.gguf")
    # ".mmproj" extension projector (content irrelevant — never parsed as model)
    (folder).mkdir(parents=True, exist_ok=True)
    (folder / "LFM2.5-Audio-1.5B-f32.mmproj").write_bytes(b"GGUF")

    entries = scan_models(tmp_path)
    assert len(entries) == 1, [e.name for e in entries]
    model = entries[0]
    assert model.has_vision, ".mmproj-extension projector was not paired"
    assert model.mmproj is not None
    assert model.mmproj.name.endswith(".mmproj")


def test_mmproj_pairing_still_size_specific(tmp_path) -> None:
    """The looser (bidirectional, separator-tolerant) matcher must NOT
    let a model grab a projector for a different size in the same dir."""
    folder = tmp_path / "X"
    _write_minimal_gguf(folder / "Qwen3.5-2B-Q8_0.gguf")
    _write_minimal_gguf(folder / "mmproj-Qwen3.5-0.8B-BF16.gguf")
    entries = scan_models(tmp_path)
    by_name = {e.name: e for e in entries}
    # The 2B model must NOT be paired with the 0.8B projector.
    assert by_name["Qwen3.5-2B-Q8_0"].mmproj is None


# ---------------------------------------------------------------------------
# GGUF sampling metadata fallback (loops / tool-call fix)


def _fake_model_md(tmp_path, name, size_gb, metadata):
    """Like _fake_model but attaches GGUF metadata."""
    from scanner import ModelEntry

    p = tmp_path / f"{name}.gguf"
    _write_minimal_gguf(p)
    return ModelEntry(
        path=p,
        name=name,
        group=".",
        size_bytes=int(size_gb * 1024**3),
        metadata=metadata,
    )


def test_metadata_sampling_reader() -> None:
    """general.sampling.* keys are read and typed correctly."""
    from scanner import metadata_sampling

    md = {
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 20,
        "general.sampling.top_p": 0.95,
    }
    s = metadata_sampling(md)
    assert s["temperature"] == 1.0
    assert s["top_k"] == 20.0
    assert s["top_p"] == 0.95
    assert "min_p" not in s  # absent key stays absent
    # Empty / missing metadata yields an empty dict, never raises.
    assert metadata_sampling({}) == {}


def test_gguf_sampling_used_when_profile_is_fallback(tmp_path) -> None:
    """A model that matches only the generic fallback profile must adopt
    the author-recommended GGUF samplers instead of the generic defaults.
    This is the loop / broken-tool-call fix for unprofiled models: any
    GGUF that ships general.sampling.* but has no tailored YAML profile."""
    profiles = load_profiles(SETTINGS_DIR)
    md = {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 40,
        "general.sampling.top_p": 0.95,
    }
    # A name no shipped profile pattern matches → fallback profile.
    model = _fake_model_md(tmp_path, "SomeUnprofiledFinetune-13B-Q6_K", 13.0, md)
    profile = match_profile(model.name, profiles)
    assert not profile.patterns, "expected the fallback profile for this name"
    cfg = compute_config(model, _fake_system(), profile)
    assert cfg.sampling["temperature"] == 1.0
    assert cfg.sampling["top_p"] == 0.95


def test_matched_profile_wins_over_gguf_sampling(tmp_path) -> None:
    """When a real family profile matches, its explicit sampler values
    take precedence over the GGUF recommendation (Basti hand-tuned these)."""
    profiles = load_profiles(SETTINGS_DIR)
    md = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 40,
        "qwen35moe.expert_count": 256,
        # Deliberately absurd GGUF value that must be overridden.
        "general.sampling.top_k": 99,
    }
    model = _fake_model_md(tmp_path, "Qwen3.6-35B-A3B-UD-Q6_K", 30.0, md)
    profile = match_profile(model.name, profiles)
    assert profile.patterns, "expected a matched (non-fallback) profile"
    cfg = compute_config(model, _fake_system(), profile)
    # qwen3_5-3_6.yaml sets top_k 20 for both chat and coding.
    assert cfg.sampling["top_k"] == 20


# ---------------------------------------------------------------------------
# MoE multi-GPU capacity-fill (don't strand VRAM on the secondary card)


def test_moe_spread_fills_both_gpus_by_capacity(tmp_path) -> None:
    """An MoE too large for the primary alone must spread proportionally to
    each card's usable capacity, NOT priority-weighted — so the secondary
    GPU is filled to roughly the same utilisation as the primary. Stranding
    several GB on the 9070 XT (the old priority-weighted behaviour) forced
    extra expert layers onto the CPU and slowed the model down."""
    profiles = load_profiles(SETTINGS_DIR)
    md = {
        "general.architecture": "qwen3moe",
        "qwen3moe.block_count": 48,
        "qwen3moe.expert_count": 128,
        "qwen3moe.attention.head_count": 40,
        "qwen3moe.attention.head_count_kv": 8,
        "qwen3moe.attention.key_length": 128,
        "qwen3moe.attention.value_length": 128,
        "qwen3moe.embedding_length": 5120,
    }
    model = _fake_model_md(tmp_path, "BigMoE-40B-A10B-Q4", 40.0, md)
    profile = match_profile(model.name, profiles)
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(
        model, _fake_dual_gpu_system_with_vk_order(), profile, gpu_priorities=prio
    )
    assert cfg.tensor_split is not None, "expected a spread, got pin/None"
    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    # MoE splits are emitted as LAYER COUNTS (llama.cpp normalises them),
    # so normalise before comparing shares.
    total = sum(parts)
    assert total > 0
    # Vulkan order: device 0 = 9070 XT (cap ~13 GB), device 1 = R9700 (cap ~30 GB).
    xt_share, r97_share = parts[0] / total, parts[1] / total
    # Capacity-proportional shares: 13/43 ≈ 0.30 and 30/43 ≈ 0.70.
    # The 9070 XT must get a MEANINGFUL share — well above the ~0.20 the
    # old priority-weighting produced. (This model fits fully on GPU —
    # n_cpu_moe == 0 — so layer shares equal byte shares here.)
    assert xt_share > 0.25, (
        f"9070 XT under-filled (capacity-fill regressed): share={xt_share:.3f}"
    )
    # Sanity: the larger card still carries the larger share.
    assert r97_share > xt_share


def test_moe_cpu_offload_split_is_byte_aware(tmp_path) -> None:
    """Regression: step-3.7-Flash (74.5 GB MoE, 45 layers) on R9700+9070 XT.

    llama.cpp maps --tensor-split onto LAYER COUNTS, and --n-cpu-moe strips
    the expert tensors of the FIRST N layers to CPU. The old byte-FRACTION
    split therefore handed Vulkan device 0 (the 9070 XT) only expert-
    stripped front layers: ~8/16 GB used while the R9700 sat at 30.9/32 GB.
    The fix emits layer counts computed from per-layer GPU byte weights, so
    reconstructing the byte load per device from the emitted counts must
    show BOTH cards filled to a comparable fraction of their usable caps —
    and neither card over its cap."""
    profiles = load_profiles(SETTINGS_DIR)
    md = {
        "general.architecture": "step35",
        "step35.block_count": 45,
        "step35.expert_count": 288,
        "step35.attention.head_count": 64,
        "step35.attention.head_count_kv": 8,
        "step35.attention.key_length": 128,
        "step35.attention.value_length": 128,
        "step35.embedding_length": 4096,
        "step35.context_length": 262144,
    }
    model = _fake_model_md(tmp_path, "Step-3.7-Flash-UD-IQ3_S", 74.5, md)
    profile = match_profile(model.name, profiles)
    assert "step" in profile.display_name.lower()
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(
        model, _fake_dual_gpu_system_with_vk_order(), profile, gpu_priorities=prio
    )
    # 74.5 GB on 48 GB VRAM → experts MUST spill to CPU.
    assert cfg.n_cpu_moe is not None and cfg.n_cpu_moe > 0
    assert cfg.tensor_split is not None, "expected a spread, got pin/None"
    counts = [int(round(float(x))) for x in cfg.tensor_split.split(",")]
    assert len(counts) == 2 and sum(counts) == 45, (
        f"expected layer counts summing to 45, got {cfg.tensor_split}"
    )
    xt_layers, r97_layers = counts  # Vulkan order: device 0 = 9070 XT

    # The 9070 XT gets the light (expert-stripped) FRONT layers — it must
    # take MORE than just those to carry a real byte share.
    assert xt_layers > cfg.n_cpu_moe, (
        f"9070 XT got only expert-stripped layers ({xt_layers} ≤ "
        f"n_cpu_moe={cfg.n_cpu_moe}) — byte-aware split regressed"
    )

    # Reconstruct the byte load per device with the same per-layer model
    # the tuner uses (light front layers + heavy expert layers + KV slice).
    n_layers = 45
    shared_gb = model.size_gb * 0.08
    per_layer_expert = (model.size_gb - shared_gb) / n_layers
    light = shared_gb / n_layers
    kv_l = 0.0 if cfg.no_kv_offload else cfg.estimated_kv_gb / n_layers
    layer_bytes = [
        light + kv_l + (0.0 if i < cfg.n_cpu_moe else per_layer_expert)
        for i in range(n_layers)
    ]
    xt_gb = sum(layer_bytes[:xt_layers])
    r97_gb = sum(layer_bytes[xt_layers:])
    # Usable caps mirror _gpu_usable_cap_gb on this fake system:
    #   9070 XT: min(16 − 1.6, 14 free) = 14.0; R9700: min(32 − 1.92, 31) = 30.08.
    # Layers are indivisible, so a residual overshoot of strictly less than
    # ONE layer may remain on a device when no contiguous split satisfies
    # every cap exactly — it eats reserved headroom, not physical VRAM.
    one_layer = per_layer_expert + kv_l + 1e-6
    xt_cap, r97_cap = 14.0, 30.08
    assert xt_gb <= xt_cap + one_layer, f"9070 XT over cap: {xt_gb:.1f} GB"
    assert r97_gb <= r97_cap + one_layer, f"R9700 over cap: {r97_gb:.1f} GB"
    # Balanced fill: the 9070 XT must be filled to a fraction of its cap
    # comparable to the R9700's — the whole point of the fix. The old
    # fraction-based split left the XT at ~0.15 fill while the R9700 ran
    # past 0.95.
    xt_fill = xt_gb / xt_cap
    r97_fill = r97_gb / r97_cap
    assert xt_fill > 0.55, f"9070 XT still stranded: fill={xt_fill:.2f}"
    assert abs(xt_fill - r97_fill) < 0.30, (
        f"unbalanced fill: XT={xt_fill:.2f} vs R9700={r97_fill:.2f}"
    )


def test_dense_spread_stays_priority_weighted(tmp_path) -> None:
    """Dense models keep the priority-weighted split: the high-priority AI
    card carries the bulk so the gaming GPU stays as free as possible. This
    must NOT change to capacity-fill (that's MoE-only)."""
    profiles = load_profiles(SETTINGS_DIR)
    # Dense (no expert_count) 40 GB model — too big for the 32 GB primary.
    md = {
        "general.architecture": "llama",
        "llama.block_count": 80,
        "llama.attention.head_count": 64,
        "llama.attention.head_count_kv": 8,
        "llama.embedding_length": 8192,
    }
    model = _fake_model_md(tmp_path, "Dense-70B-Q4", 40.0, md)
    profile = match_profile(model.name, profiles)
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(
        model, _fake_dual_gpu_system_with_vk_order(), profile, gpu_priorities=prio
    )
    assert cfg.tensor_split is not None
    parts = [float(x) for x in cfg.tensor_split.split(",")]
    xt_share, r97_share = parts[0], parts[1]
    # Priority×VRAM (2×32 vs 1×16) → R9700 ≥ ~0.75; 9070 XT ≤ ~0.25.
    assert r97_share > 0.70, f"dense split not priority-weighted: r97={r97_share:.3f}"
    assert xt_share < 0.30, f"dense split puts too much on the 9070 XT: {xt_share:.3f}"


# ---------------------------------------------------------------------------
# v5.1.3 architecture/backend memory regressions


def test_full_offload_kv_never_uses_unflagged_host_ram(tmp_path) -> None:
    """Full-offload + default KV offload must fit model and KV in real VRAM."""
    from performance_target import PERFORMANCE_TARGETS

    profiles = load_profiles(SETTINGS_DIR)
    model = _mistral_dense_md(tmp_path, size_gb=5.0)
    profile = match_profile(model.name, profiles)
    system = _fake_system(
        ram_total=128,
        ram_free=100,
        vram_total=16,
        vram_free=14,
    )
    target = PERFORMANCE_TARGETS["balanced"]
    cfg = compute_config(model, system, profile, perf_target=target)

    assert cfg.full_offload is True
    assert cfg.no_kv_offload is False
    assert cfg.kv_ram_gb == pytest.approx(0.0)
    assert (
        cfg.estimated_model_vram_gb
        + cfg.kv_vram_gb
        + cfg.runtime_vram_overhead_gb
        + cfg.batch_vram_overhead_gb
        + cfg.recurrent_state_vram_gb
        + target.dense_vram_safety_gb
        <= system.free_vram_gb + 1e-6
    )


def test_unified_memory_is_budgeted_once(tmp_path) -> None:
    """Apple/iGPU shared RAM must not become independent RAM + VRAM pools."""
    from performance_target import PERFORMANCE_TARGETS

    profiles = load_profiles(SETTINGS_DIR)
    model = _mistral_dense_md(tmp_path, size_gb=20.0)
    profile = match_profile(model.name, profiles)
    system = _fake_system(
        ram_total=64,
        ram_free=40,
        vram_total=64,
        vram_free=40,
        vendor="apple",
        unified=True,
        runtime_backend="Metal",
        runtime_index=0,
    )
    target = PERFORMANCE_TARGETS["balanced"]
    cfg = compute_config(model, system, profile, perf_target=target)
    total = (
        cfg.estimated_model_vram_gb
        + cfg.estimated_model_ram_gb
        + cfg.estimated_kv_gb
        + cfg.recurrent_state_vram_gb
        + cfg.recurrent_state_ram_gb
        + cfg.vision_vram_gb
        + cfg.vision_ram_gb
        + cfg.draft_vram_gb
        + cfg.runtime_vram_overhead_gb
        + cfg.batch_vram_overhead_gb
        + cfg.prompt_cache_ram_gb
        + max(target.dense_vram_safety_gb, target.ram_safety_gb)
    )
    assert cfg.unified_memory is True
    assert total <= system.free_ram_gb + 1e-6
    assert "MEM:" in format_system(system)


def test_alternate_kv_head_key_is_used_by_sizing() -> None:
    from diagnostics import audit_model_metadata
    from scanner import ModelEntry
    from tuner import kv_per_token_mb_from_metadata

    md = {
        "general.architecture": "future",
        "future.block_count": 32,
        "future.embedding_length": 4096,
        "future.attention.head_count": 32,
        "future.attention.kv_head_count": 8,
        "future.attention.key_length": 128,
        "future.attention.value_length": 128,
    }
    expected = 32 * 8 * (128 + 128) * 2 / (1024 * 1024)
    assert kv_per_token_mb_from_metadata(md) == pytest.approx(expected)
    model = ModelEntry(
        path=Path("future.gguf"),
        name="future",
        group=".",
        size_bytes=1,
        metadata=md,
    )
    assert not any(w.id == "KV-HEAD-COUNT-MISSING" for w in audit_model_metadata(model))


def test_auto_kv_uses_q4_even_when_full_precision_fits() -> None:
    """Unused memory no longer upgrades Auto beyond the Q4 capacity default."""
    from tuner import _pick_kv_quant

    pair = _pick_kv_quant(
        "q5_0",
        262144,
        0.0625,
        20.0,
        262144,
        asymmetric=True,
        base_k_per_token_mb=0.03125,
        base_v_per_token_mb=0.03125,
    )
    assert pair == ("q4_0", "q4_0")


def test_auto_kv_stays_symmetric_with_unequal_kv_dimensions() -> None:
    from tuner import (
        _pick_kv_quant,
        kv_per_token_mb_from_metadata,
        kv_per_token_parts_mb_from_metadata,
    )

    md = {
        "general.architecture": "future",
        "future.block_count": 10,
        "future.attention.head_count": 16,
        "future.attention.head_count_kv": 2,
        "future.embedding_length": 2048,
        "future.attention.key_length": 256,
        "future.attention.value_length": 64,
    }
    k_mb, v_mb = kv_per_token_parts_mb_from_metadata(md)
    assert k_mb / v_mb == pytest.approx(4.0)
    pair = _pick_kv_quant(
        "q8_0",
        32768,
        kv_per_token_mb_from_metadata(md),
        0.20,
        32768,
        asymmetric=True,
        base_k_per_token_mb=k_mb,
        base_v_per_token_mb=v_mb,
    )
    assert pair == ("q4_0", "q4_0")


def test_pure_mamba_recurrent_layers_use_minimal_kv_sentinel(tmp_path) -> None:
    from diagnostics import audit_model_metadata, format_diagnostic_report
    from scanner import ModelEntry, metadata_attention_layer_count
    from tuner import (
        _resolve_kv_per_token_parts_mb,
        recurrent_state_gb_from_metadata,
    )

    md = {
        "general.architecture": "mamba",
        "mamba.block_count": 64,
        "mamba.ssm.state_size": 16,
        "mamba.ssm.inner_size": 4096,
        "mamba.ssm.conv_kernel": 4,
        "mamba.ssm.group_count": 1,
        "mamba.attention.recurrent_layers": 64,
    }
    assert metadata_attention_layer_count(md) == 1
    assert recurrent_state_gb_from_metadata(md) > 0.01
    model = ModelEntry(
        path=tmp_path / "Mamba-7B.gguf",
        name="Mamba-7B",
        group=".",
        size_bytes=7 * 1024**3,
        metadata=md,
    )
    assert sum(_resolve_kv_per_token_parts_mb(model, 7.0)) < 0.00001
    assert not any(
        warning.id == "KV-HEAD-COUNT-MISSING" for warning in audit_model_metadata(model)
    )
    assert "pure recurrent" in format_diagnostic_report(model)


def test_minimax_01_bool_recurrent_array_is_counted() -> None:
    from scanner import metadata_attention_layer_count
    from tuner import recurrent_state_gb_from_metadata

    recurrent = [((i + 1) % 8 != 0) for i in range(80)]
    md = {
        "general.architecture": "minimax-01",
        "minimax-01.block_count": 80,
        "minimax-01.embedding_length": 8192,
        "minimax-01.attention.head_count": 64,
        "minimax-01.attention.key_length": 128,
        "minimax-01.attention.recurrent_layers": recurrent,
    }
    assert metadata_attention_layer_count(md) == 10
    state_gb = recurrent_state_gb_from_metadata(md)
    assert 0.25 < state_gb < 0.30
    assert recurrent_state_gb_from_metadata(md, snapshot_count=2) == pytest.approx(
        state_gb * 3
    )


def test_draft_n_max_resizes_hybrid_recurrent_snapshots(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model_md(
        tmp_path,
        "Qwen3.5-9B-MTP-Q4_K_M",
        8.0,
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 40,
            "qwen35.context_length": 131072,
            "qwen35.embedding_length": 4096,
            "qwen35.attention.head_count": 32,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 128,
            "qwen35.attention.value_length": 128,
            "qwen35.attention.recurrent_layers": 30,
            "qwen35.nextn_predict_layers": 1,
            "qwen35.ssm.inner_size": 2048,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.group_count": 8,
        },
    )
    profile = match_profile(model.name, profiles, model.architecture)
    system = _fake_system(vram_total=24, vram_free=22)
    auto = compute_config(model, system, profile)
    pinned = compute_config(model, system, profile, force_draft_n_max=8)
    auto_state = auto.recurrent_state_vram_gb + auto.recurrent_state_ram_gb
    pinned_state = pinned.recurrent_state_vram_gb + pinned.recurrent_state_ram_gb
    assert pinned.draft_n_max == 8
    assert pinned_state == pytest.approx(auto_state * 3)


def test_known_moe_architecture_survives_missing_expert_count(tmp_path) -> None:
    from diagnostics import audit_model_metadata
    from scanner import (
        ModelEntry,
        metadata_attention_layer_count,
        metadata_is_hybrid_architecture,
    )
    from tuner import _moe_expert_count

    path = tmp_path / "opaque.gguf"
    _write_minimal_gguf(path)
    model = ModelEntry(
        path=path,
        name="opaque-checkpoint",
        group=".",
        size_bytes=1024,
        metadata={
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 32,
            "qwen35moe.attention.head_count_kv": 8,
        },
    )
    assert _moe_expert_count(model) == 2
    assert any(w.id == "MOE-ARCH-FALLBACK" for w in audit_model_metadata(model))

    for arch in (
        "bailingmoe2.5",
        "bailingmoe2_5",
        "bailing_hybrid",
        "bailingmoe3",
        "kimi-k3",
        "kimi_k3",
        "ernie4_5_moe",
        "hunyuan_moe",
        "llada_moe",
        "nemotron-h-moe",
        "granitemoehybrid",
    ):
        alias_model = ModelEntry(
            path=path,
            name="opaque-checkpoint",
            group=".",
            size_bytes=1024,
            metadata={"general.architecture": arch, f"{arch}.block_count": 32},
        )
        assert _moe_expert_count(alias_model) == 2, arch
    bailing_md = {
        "general.architecture": "bailing_hybrid",
        "bailing_hybrid.block_count": 32,
    }
    assert metadata_is_hybrid_architecture(bailing_md)
    assert metadata_attention_layer_count(bailing_md) == 8


def test_moe_alias_diagnostic_is_not_called_filename_fallback(tmp_path) -> None:
    from diagnostics import format_diagnostic_report
    from scanner import ModelEntry

    model = ModelEntry(
        path=tmp_path / "opaque.gguf",
        name="opaque",
        group=".",
        size_bytes=1024,
        metadata={
            "general.architecture": "futuremoe",
            "futuremoe.block_count": 16,
            "futuremoe.attention.head_count": 16,
            "futuremoe.attention.head_count_kv": 4,
            "futuremoe.embedding_length": 2048,
            "futuremoe.num_local_experts": 8,
        },
    )
    report = format_diagnostic_report(model)
    assert "expert_count (md)     : 8" in report
    assert "from filename fallback" not in report


def test_metadata_inventory_and_support_report_are_bounded_and_redacted(
    tmp_path, monkeypatch
) -> None:
    import scanner
    from diagnostics import format_support_report
    from get_metadata import write_metadata_report

    root = tmp_path / "private-model-root"
    model_path = root / "Vendor" / "SupportModel-Q4_K_M.gguf"
    _write_minimal_gguf(model_path)
    monkeypatch.setenv("AUTOTUNER_METADATA_CACHE", "off")
    scanner._reset_metadata_cache_for_tests()
    inventory_path, count = write_metadata_report(
        [root], tmp_path / "reports" / "metadata.md"
    )
    inventory = inventory_path.read_text(encoding="utf-8")
    assert count == 1
    assert "SupportModel-Q4_K_M.gguf" in inventory
    assert "Vendor" in inventory
    assert str(root) not in inventory

    app_log = tmp_path / "autotuner-app.log"
    app_log.write_text(
        f"Scanning {root}\\Vendor\nAuthorization: Bearer abc.def.secret\n"
        "api_key=super-secret-value\n",
        encoding="utf-8",
    )
    model = ModelEntry(
        path=model_path,
        name="SupportModel-Q4_K_M",
        group="Vendor",
        size_bytes=1024,
        metadata={
            "general.architecture": "future",
            "future.block_count": 8,
            "future.context_length": 8192,
            "future.embedding_length": 1024,
            "future.attention.head_count": 8,
            "future.attention.head_count_kv": 2,
        },
    )
    report = format_support_report(
        [model],
        system=_fake_system(),
        model_roots=[root],
        app_log_path=app_log,
        debug_enabled=True,
    )
    assert "AutoTuner support report" in report
    assert "SupportModel-Q4_K_M" in report
    assert "Debug mode: on" in report
    assert str(root) not in report
    assert "super-secret-value" not in report
    assert "abc.def.secret" not in report
    assert "<redacted>" in report
    scanner._reset_metadata_cache_for_tests()


def test_qwen4exp_diagnostic_no_longer_warns_about_merged_loader() -> None:
    from diagnostics import audit_model_metadata

    model = ModelEntry(
        path=Path("Qwen3.8-Flash-Next.gguf"),
        name="Qwen3.8-Flash-Next",
        group=".",
        size_bytes=1,
        metadata={
            "general.architecture": "qwen4exp",
            "qwen4exp.block_count": 48,
            "qwen4exp.attention.head_count": 24,
            "qwen4exp.attention.head_count_kv": 2,
            "qwen4exp.expert_count": 512,
        },
    )
    warning_ids = {warning.id for warning in audit_model_metadata(model)}
    assert "RUNTIME-EXPERIMENTAL-ARCH" not in warning_ids


def test_moe_shared_tensors_that_do_not_fit_fall_back_to_cpu() -> None:
    from tuner import _decide_moe_offload

    ngl, n_cpu, model_vram, model_ram, full = _decide_moe_offload(
        model_size_gb=100.0,
        free_vram_gb=4.0,
        free_ram_gb=128.0,
        n_layers=40,
        expert_count=128,
        params_billion=100.0,
        target_ctx=32768,
        base_kv_per_token_mb=0.05,
    )
    assert ngl == 0
    assert n_cpu == 40
    assert model_vram == 0.0
    assert model_ram == pytest.approx(100.0)
    assert full is False


def test_backend_specific_visibility_selectors() -> None:
    from tuner import _visibility_env_for_gpus

    cuda = GPUInfo(
        0,
        "RTX 5090",
        "nvidia",
        32768,
        32000,
        hip_index=1,
        runtime_backend="CUDA",
    )
    sycl = GPUInfo(
        0,
        "Intel Arc A770",
        "intel",
        16384,
        15000,
        hip_index=1,
        runtime_backend="SYCL",
    )
    cuda_env, cuda_remap = _visibility_env_for_gpus([cuda], [1])
    sycl_env, sycl_remap = _visibility_env_for_gpus([sycl], [1])
    assert cuda_env == {"CUDA_VISIBLE_DEVICES": "1"}
    assert sycl_env == {"ONEAPI_DEVICE_SELECTOR": "level_zero:1"}
    assert cuda_remap and sycl_remap


def test_qt_balance_pin_replaces_stale_backend_selector() -> None:
    import qt_launcher

    logs = []
    window = types.SimpleNamespace(
        _log=logs.append,
        _last_pinned_gpu=None,
    )
    cfg = _fake_diffusion_config()
    cfg.env_overrides = {
        "HIP_VISIBLE_DEVICES": "0",
        "GGML_VK_VISIBLE_DEVICES": "0",
    }
    gpu = GPUInfo(
        0,
        "RTX 5090",
        "nvidia",
        32768,
        30000,
        hip_index=1,
        runtime_backend="CUDA",
    )
    qt_launcher.MainWindow._pin_cfg_to_gpu(window, cfg, gpu)
    assert cfg.env_overrides == {"CUDA_VISIBLE_DEVICES": "1"}
    assert cfg.main_gpu == 0
    assert cfg.tensor_split is None


def test_multi_gpu_cuda_pinning_does_not_emit_hip_or_vulkan(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    system = _fake_dual_gpu_system_with_vk_order()
    for gpu in system.gpus:
        gpu.vendor = "nvidia"
        gpu.runtime_backend = "CUDA"
    cfg = compute_config(model, system, profile, user_ctx=32768)
    assert cfg.env_overrides == {"CUDA_VISIBLE_DEVICES": "1"}
    assert cfg.main_gpu == 0


# ---------------------------------------------------------------------------
# Qt launcher helpers


def test_open_local_folder_uses_qt_desktop_services(tmp_path, monkeypatch) -> None:
    """The model context-menu action must use Qt's cross-platform opener."""
    pytest.importorskip("PyQt6")
    import qt_launcher

    opened = []

    class FakeDesktopServices:
        @staticmethod
        def openUrl(url):
            opened.append(url)
            return True

    monkeypatch.setattr(qt_launcher, "QDesktopServices", FakeDesktopServices)
    assert qt_launcher._open_local_folder(tmp_path)
    assert len(opened) == 1
    assert Path(opened[0].toLocalFile()) == tmp_path.resolve()


# ---------------------------------------------------------------------------
# GPU pin dropdown — short-label derivation (qt_launcher._gpu_short_label)
#
# The GUI GPU pin dropdown persists a short, stable token (e.g. "R9700",
# "9070") that compute_config(force_gpu=...) matches case-insensitively as a
# substring of the full driver name. These tests pin that derivation so the
# click-path keeps producing tokens the tuner actually resolves. The Qt import
# is guarded: on a headless CI runner without PyQt6 the test simply skips.


def test_gpu_short_label_derivation() -> None:
    """Driver names must reduce to a distinctive, digit-bearing token that is
    a case-insensitive substring of the original (so force_gpu matches)."""
    pytest.importorskip("PyQt6")
    from qt_launcher import MainWindow

    cases = {
        "AMD Radeon AI PRO R9700": "R9700",
        "AMD Radeon RX 9070 XT": "9070",
        "NVIDIA GeForce RTX 5090": "5090",
        "Intel Arc A770": "A770",
    }
    for full_name, expected in cases.items():
        token = MainWindow._gpu_short_label(full_name)
        assert token == expected, f"{full_name!r} → {token!r}, expected {expected!r}"
        # The whole point: the token must match the card name the way
        # compute_config does (case-insensitive substring).
        assert token.lower() in full_name.lower()


def test_gpu_short_label_fallbacks() -> None:
    """Names without a digit fall back to the last word; empty → stripped."""
    pytest.importorskip("PyQt6")
    from qt_launcher import MainWindow

    assert MainWindow._gpu_short_label("Some Fancy GPU") == "GPU"
    assert MainWindow._gpu_short_label("   ") == ""


def test_manual_next_port_not_shifted_by_running_server_count() -> None:
    """Regression: main server on 8080 must not make a manually entered
    chatbot port 1235 turn into 1236 just because one server is running."""
    pytest.importorskip("PyQt6")
    from qt_launcher import MainWindow

    # _requested_start_port intentionally ignores len(self._servers). It is
    # called before _next_free_port, which still handles actual collisions.
    dummy = object()
    assert MainWindow._requested_start_port(dummy, 8080, 0) == 8080
    assert MainWindow._requested_start_port(dummy, 1235, 0) == 1235
    assert MainWindow._requested_start_port(dummy, 1235, 1) == 1236


# ---------------------------------------------------------------------------
# Authoritative VRAM from `llama-server --list-devices`
#
# Regression for the dual-AMD-GPU mis-attribution: WMI reported the full
# RX 9070 XT as empty (15.9/15.9 "free") and the empty R9700 as half-full,
# so a second model was refused. llama-server --list-devices gives the
# correct, per-card, live numbers — these tests pin its parsing.


def test_llama_device_probe_preserves_backend_without_memory(monkeypatch) -> None:
    import hardware

    out = """Available devices:
  Metal0: Apple M5 Max
  SYCL1: Intel(R) Arc(TM) A770 Graphics
"""
    monkeypatch.setattr(hardware, "_run", lambda *a, **k: out)
    authoritative, records = hardware._probe_llama_devices("llama-server")
    assert authoritative is True
    assert [(r.backend, r.index, r.identifier) for r in records] == [
        ("Metal", 0, "Metal0"),
        ("SYCL", 1, "SYCL1"),
    ]


def test_exact_runtime_order_beats_vulkan_fallback(monkeypatch) -> None:
    import hardware

    runtime = """Available devices:
  CUDA0: GPU-B (16000 MiB, 15000 MiB free)
  CUDA1: GPU-A (24000 MiB, 23000 MiB free)
"""

    def fake_run(cmd, *args, **kwargs):
        if "--list-devices" in cmd:
            return runtime
        return """Devices:
GPU0:\n deviceName = GPU-A\n deviceID = 0x1000
GPU1:\n deviceName = GPU-B\n deviceID = 0x1001
"""

    monkeypatch.setattr(hardware, "_run", fake_run)
    gpus = [
        GPUInfo(0, "GPU-A", "nvidia", 24000, 23000, pci_device_id=0x1000),
        GPUInfo(1, "GPU-B", "nvidia", 16000, 15000, pci_device_id=0x1001),
    ]
    hardware._assign_hip_indices(gpus, "llama-server")
    assert [(g.hip_index, g.runtime_backend) for g in gpus] == [
        (1, "CUDA"),
        (0, "CUDA"),
    ]


def test_apple_detection_uses_live_available_unified_memory(monkeypatch) -> None:
    import hardware

    class VM:
        total = 64 * 1024**3
        available = 40 * 1024**3

    def fake_run(cmd, *args, **kwargs):
        if "hw.memsize" in cmd:
            return str(64 * 1024**3)
        return "Apple M5 Max"

    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware.psutil, "virtual_memory", lambda: VM())
    monkeypatch.setattr(hardware, "_run", fake_run)
    gpu = hardware._detect_apple()[0]
    assert gpu.total_vram_mb == 64 * 1024
    assert gpu.free_vram_mb == 40 * 1024
    assert gpu.unified_memory is True

    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware._detect_apple() == []


def test_integrated_amd_apu_is_unified_and_filtered_beside_dgpu() -> None:
    import hardware

    assert (
        hardware._infer_integrated_gpu("AMD Radeon 780M Graphics", "amd", 16384) is True
    )
    for name in (
        "AMD Radeon(TM) 840M Graphics",
        "AMD Radeon(TM) 860M Graphics",
        "AMD Radeon(TM) 880M Graphics",
    ):
        assert hardware._infer_integrated_gpu(name, "amd", 16384) is True
    apu = GPUInfo(
        0,
        "AMD Radeon 780M Graphics",
        "amd",
        16384,
        12000,
        unified_memory=True,
        is_integrated=True,
    )
    dgpu = GPUInfo(1, "AMD Radeon RX 9070 XT", "amd", 16384, 15000, is_integrated=False)
    used, ignored = hardware._filter_inference_gpus([apu, dgpu])
    assert used == [dgpu]
    assert apu in ignored


def test_discrete_intel_arc_is_not_blanket_filtered() -> None:
    import hardware

    arc = GPUInfo(
        0,
        "Intel Arc A770",
        "intel",
        16384,
        15000,
        is_integrated=False,
    )
    amd = GPUInfo(1, "AMD Radeon", "amd", 24576, 23000, is_integrated=False)
    used, ignored = hardware._filter_inference_gpus([arc, amd])
    assert arc in used
    assert arc not in ignored
    assert (
        hardware._infer_integrated_gpu("Intel(R) Arc(TM) Pro A60 Graphics", "intel", 0)
        is False
    )


def test_windows_registry_gpu_scan_survives_sparse_subkeys(monkeypatch) -> None:
    import hardware

    subkeys = ["0000", "0002"]
    desc = {"0000": "GPU Zero", "0002": "GPU Two"}

    def open_key(_root, path, *_args):
        return path.rsplit("\\", 1)[-1] if path.endswith(tuple(subkeys)) else "base"

    def enum_key(_base, index):
        if index >= len(subkeys):
            raise OSError
        return subkeys[index]

    def query_value(key, value):
        if value == "DriverDesc":
            return desc[key], 1
        if value == "HardwareInformation.qwMemorySize":
            return (8 if key == "0000" else 16) * 1024**3, 1
        raise FileNotFoundError

    fake = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=object(),
        KEY_READ=1,
        OpenKey=open_key,
        EnumKey=enum_key,
        QueryValueEx=query_value,
        CloseKey=lambda _key: None,
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    result = hardware._get_vram_from_registry()
    assert result == {"GPU Zero": 8 * 1024**3, "GPU Two": 16 * 1024**3}


def test_rocm_text_fallback_subtracts_used_memory(monkeypatch) -> None:
    import hardware

    text = """GPU[0] : VRAM Total Memory (B): 17179869184
GPU[0] : VRAM Total Used Memory (B): 4294967296
"""
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: "rocm-smi")
    monkeypatch.setattr(hardware, "_run", lambda *args, **kwargs: text)
    gpu = hardware._detect_amd_rocm()[0]
    assert gpu.total_vram_mb == 16384
    assert gpu.free_vram_mb == 12288


def test_openvino_runtime_is_preserved_as_unified_accelerator(monkeypatch) -> None:
    import hardware

    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: [])
    monkeypatch.setattr(hardware, "_detect_amd_rocm", lambda: [])
    monkeypatch.setattr(hardware, "_detect_apple", lambda: [])
    monkeypatch.setattr(hardware, "_detect_windows_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_drm_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_other_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_get_pci_device_ids", lambda: {})
    record = hardware._LlamaDeviceRecord(
        "OPENVINO", 0, "openvino runtime", 48000, 27000
    )
    monkeypatch.setattr(
        hardware, "_probe_llama_devices", lambda _binary: (True, [record])
    )
    system = hardware.detect_system("openvino-llama-server")
    assert len(system.gpus) == 1
    assert system.gpus[0].runtime_backend == "OPENVINO"
    assert system.gpus[0].unified_memory is True


def test_cpu_only_binary_intersects_out_os_visible_gpu(monkeypatch) -> None:
    import hardware

    gpu = GPUInfo(0, "RTX 5090", "nvidia", 32768, 32000, is_integrated=False)
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: [gpu])
    monkeypatch.setattr(hardware, "_detect_amd_rocm", lambda: [])
    monkeypatch.setattr(hardware, "_detect_apple", lambda: [])
    monkeypatch.setattr(hardware, "_detect_windows_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_drm_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_detect_linux_other_gpus", lambda **kwargs: [])
    monkeypatch.setattr(hardware, "_get_pci_device_ids", lambda: {})
    monkeypatch.setattr(hardware, "_probe_llama_devices", lambda _binary: (True, []))
    system = hardware.detect_system("cpu-only-llama-server")
    assert system.gpus == []
    assert system.ignored_gpus == [gpu]


_LIST_DEVICES_REAL = """Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15416 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (32624 MiB, 31704 MiB free)
  Vulkan2: Intel(R) Graphics (27647 MiB, 26879 MiB free)
"""


def test_list_devices_vram_parses_real_output(monkeypatch) -> None:
    import hardware

    monkeypatch.setattr(hardware, "_run", lambda *a, **k: _LIST_DEVICES_REAL)
    vram = hardware._detect_llama_device_vram("llama-server")

    # Correct per-card attribution (the whole point of the fix).
    assert vram["amd radeon rx 9070 xt"] == (16304, 15416)
    assert vram["amd radeon ai pro r9700"] == (32624, 31704)
    # Name with parentheses must not break the regex.
    assert vram["intel(r) graphics"] == (27647, 26879)


def test_list_devices_vram_handles_gib_units(monkeypatch) -> None:
    import hardware

    out = "  Vulkan0: Some GPU (16 GiB, 8 GiB free)\n"
    monkeypatch.setattr(hardware, "_run", lambda *a, **k: out)
    vram = hardware._detect_llama_device_vram("llama-server")
    assert vram["some gpu"] == (16 * 1024, 8 * 1024)


def test_list_devices_vram_empty_without_binary() -> None:
    import hardware

    assert hardware._detect_llama_device_vram(None) == {}


def test_list_devices_vram_empty_on_unparsable(monkeypatch) -> None:
    import hardware

    monkeypatch.setattr(hardware, "_run", lambda *a, **k: "no devices here")
    assert hardware._detect_llama_device_vram("llama-server") == {}


def test_list_devices_free_never_exceeds_total(monkeypatch) -> None:
    import hardware

    # Defensive: a malformed free > total is clamped to total.
    out = "  Vulkan0: Weird GPU (8000 MiB, 9999 MiB free)\n"
    monkeypatch.setattr(hardware, "_run", lambda *a, **k: out)
    vram = hardware._detect_llama_device_vram("llama-server")
    total, free = vram["weird gpu"]
    assert free <= total
    assert (total, free) == (8000, 8000)


def test_linux_lspci_names_match_llama_device_names() -> None:
    import hardware

    llama_names = ["amd radeon rx 9070 xt", "amd radeon ai pro r9700"]
    assert (
        hardware._best_gpu_name_match(
            "Radeon RX 9070 XT",
            llama_names,
        )
        == 0
    )
    assert (
        hardware._best_gpu_name_match(
            "Radeon AI PRO R9700",
            llama_names,
        )
        == 1
    )


def test_intel_lspci_name_is_shortened() -> None:
    import hardware

    parsed = hardware._parse_lspci_mm_line(
        '0000:00:02.0 "VGA compatible controller [0300]" '
        '"Intel Corporation [8086]" '
        '"Arrow Lake-S [Intel Graphics] [7d67]" -r06 -p00 '
        '"Micro-Star International Co., Ltd. [MSI] [1462]" "Device [7e20]"'
    )
    assert parsed is not None
    assert parsed[2] == "Intel Graphics"
    assert parsed[3] == 0x7D67


def test_lspci_nnmm_parser_extracts_linux_pci_device_id() -> None:
    import hardware

    parsed = hardware._parse_lspci_mm_line(
        '0000:08:00.0 "VGA compatible controller [0300]" '
        '"Advanced Micro Devices, Inc. [AMD/ATI] [1002]" '
        '"Navi 48 [Radeon AI PRO R9700] [7551]" -rc0 -p00 '
        '"Gigabyte Technology Co., Ltd [1458]" "Device [242f]"'
    )
    assert parsed is not None
    slot, cls, name, dev_id = parsed
    assert slot == "0000:08:00.0"
    assert cls == "VGA compatible controller"
    assert name == "Radeon AI PRO R9700"
    assert dev_id == 0x7551


def test_vulkan_summary_maps_basti_gpu_pci_ids(monkeypatch) -> None:
    """Basti's workstation: Windows lists R9700 first, but Vulkan/llama.cpp
    lists RX 9070 XT first. PCI device IDs must resolve the correct indices."""
    import hardware
    from hardware import GPUInfo

    summary = """
Devices:
========
GPU0:
        deviceID           = 0x7550
        deviceName         = AMD Radeon RX 9070 XT
GPU1:
        deviceID           = 0x7551
        deviceName         = AMD Radeon AI PRO R9700
GPU2:
        deviceID           = 0x7d67
        deviceName         = Intel(R) Graphics
"""
    monkeypatch.setattr(hardware, "_run", lambda *a, **k: summary)
    monkeypatch.setattr(hardware, "_detect_llama_device_order", lambda _binary: [])
    monkeypatch.setattr(hardware, "_detect_vulkan_device_order", lambda: [])

    gpus = [
        GPUInfo(
            index=0,
            name="AMD Radeon AI PRO R9700",
            vendor="amd",
            total_vram_mb=32624,
            free_vram_mb=31704,
            pci_device_id=0x7551,
        ),
        GPUInfo(
            index=1,
            name="AMD Radeon RX 9070 XT",
            vendor="amd",
            total_vram_mb=16304,
            free_vram_mb=15416,
            pci_device_id=0x7550,
        ),
    ]

    hardware._assign_hip_indices(gpus, None)

    assert gpus[0].hip_index == 1  # R9700 is Vulkan1
    assert gpus[1].hip_index == 0  # 9070 XT is Vulkan0


# ---------------------------------------------------------------------------
# Regression tests for b9672 hybrid-arch KV sizing + gpt-oss arch fallback
# ---------------------------------------------------------------------------


def test_hybrid_detection_includes_b9672_linear_archs() -> None:
    """b9672 classifies the linear-/gated-delta-net families as hybrid.

    Qwen3.5/3.6, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next and Kimi-Linear all
    return True from llama_arch_is_hybrid() in mainline, so only their
    full-attention layers carry KV. Mirror that here.
    """
    from scanner import metadata_is_hybrid_architecture

    for arch in (
        "qwen35",
        "qwen35moe",
        "lfm2moe",
        "nemotron_h_moe",
        "qwen3next",
        "kimi-linear",
    ):
        assert (
            metadata_is_hybrid_architecture({"general.architecture": arch}) is True
        ), f"{arch} should be detected as hybrid (b9672)"

    # Plain qwen3moe / gpt-oss remain pure Transformer.
    assert (
        metadata_is_hybrid_architecture({"general.architecture": "qwen3moe"}) is False
    )
    assert metadata_is_hybrid_architecture({"general.architecture": "gpt-oss"}) is False


def test_attention_layer_count_uses_recurrent_layers_key() -> None:
    """b9672 `<arch>.attention.recurrent_layers` gives an exact KV-layer
    count: full-attention layers == block_count - recurrent_layers."""
    from scanner import metadata_attention_layer_count

    # Qwen3.6-A3B MoE: 48 blocks, 36 recurrent → 12 full-attention.
    md = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 48,
        "qwen35moe.attention.recurrent_layers": 36,
    }
    assert metadata_attention_layer_count(md) == 12

    # Qwen3.5 dense: 36 blocks, 27 recurrent → 9.
    md = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 36,
        "qwen35.attention.recurrent_layers": 27,
    }
    assert metadata_attention_layer_count(md) == 9


def test_recurrent_layers_key_takes_priority_over_ratio() -> None:
    """When the recurrent_layers key is present it must override the
    coarse per-arch ratio (which would give a different number)."""
    from scanner import metadata_attention_layer_count

    # 40 blocks, ratio 0.25 would give 10; recurrent=10 gives 30. The key
    # must win → 30, not 10.
    md = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 40,
        "qwen35moe.attention.recurrent_layers": 10,
    }
    assert metadata_attention_layer_count(md) == 30


def test_recurrent_layers_generic_key_scan() -> None:
    """A *.attention.recurrent_layers key with a non-matching arch prefix
    (community re-convert) is still honoured."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "nemotron_h_moe",
        "nemotron_h_moe.block_count": 56,
        # Note the deliberately wrong prefix:
        "foo.attention.recurrent_layers": 48,
    }
    assert metadata_attention_layer_count(md) == 8


def test_recurrent_layers_out_of_range_falls_back_to_ratio() -> None:
    """An impossible recurrent count (>= total) must not produce a
    nonsensical value; it falls through to the per-arch ratio."""
    from scanner import metadata_attention_layer_count

    # recurrent == total → invalid, must fall back to ratio (0.25 → 12).
    md = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 48,
        "qwen35moe.attention.recurrent_layers": 48,
    }
    n = metadata_attention_layer_count(md)
    assert n > 0
    assert 10 <= n <= 14, f"expected ratio fallback ~12, got {n}"

    # recurrent == 0 is valid (all-attention) → total - 0 == total.
    md = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 36,
        "qwen35.attention.recurrent_layers": 0,
    }
    assert metadata_attention_layer_count(md) == 36


def test_attention_layer_count_new_hybrid_ratios() -> None:
    """Without explicit keys, the new hybrid families use sensible ratios."""
    from scanner import metadata_attention_layer_count

    # qwen3next ~25%
    n = metadata_attention_layer_count(
        {"general.architecture": "qwen3next", "qwen3next.block_count": 48}
    )
    assert 10 <= n <= 14, f"qwen3next ~25% of 48, got {n}"

    # qwen35moe ~25%
    n = metadata_attention_layer_count(
        {"general.architecture": "qwen35moe", "qwen35moe.block_count": 48}
    )
    assert 10 <= n <= 14, f"qwen35moe ~25% of 48, got {n}"

    # lfm2moe ~20%
    n = metadata_attention_layer_count(
        {"general.architecture": "lfm2moe", "lfm2moe.block_count": 30}
    )
    assert 4 <= n <= 8, f"lfm2moe ~20% of 30, got {n}"


def test_gpt_oss_arch_fallback_for_unrecognised_filename() -> None:
    """A gpt-oss GGUF whose filename matches no pattern must still resolve
    to the gpt-oss profile via arch_fallback (so it gets --jinja), not
    _default."""
    profiles = load_profiles(SETTINGS_DIR)

    # Unrecognised name, gpt-oss architecture.
    p = match_profile("Some-Weird-MXFP4-Merge.gguf", profiles, "gpt-oss")
    assert p.display_name == "gpt-oss (OpenAI)", (
        f"gpt-oss arch fallback failed, got {p.display_name!r}"
    )
    assert "--jinja" in p.extra_args


def test_vibecoder_matches_gpt_oss_by_pattern() -> None:
    """VibeCoder (gpt-oss arch MXFP4 merge) matches the gpt-oss profile by
    its filename pattern alone."""
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile("VibeCoder-20b-RL1_0_MXFP4_MOE.gguf", profiles)
    assert p.display_name == "gpt-oss (OpenAI)"
    assert "--jinja" in p.extra_args


def test_arch_fallback_does_not_swallow_unrelated_models() -> None:
    """arch_fallback must only fire when NO pattern matched, and only for
    the exact arch. A non-gpt-oss unknown must still go to _default."""
    profiles = load_profiles(SETTINGS_DIR)

    # llama-arch unknown → generic fallback, NOT gpt-oss (and NOT minicpm5,
    # which is also a llama-arch profile but declares no arch_fallback).
    p = match_profile("kafkalm-70b-german-v0.1.Q5_K_M.gguf", profiles, "llama")
    assert p.display_name == "Generic / fallback"

    # An unknown filename whose arch nobody claims as a fallback → generic.
    # gemma2 is a real arch but no profile declares arch_fallback: gemma2.
    p = match_profile("Some-Random-Gemma2-Merge.Q6_K.gguf", profiles, "gemma2")
    assert p.display_name == "Generic / fallback"


def test_qwen35_arch_fallback_and_qwable_qwopus_patterns() -> None:
    """Qwen3.5/3.6 community finetunes must route to the Qwen profile:
    via the qwable/qwopus filename patterns, or via the qwen35 / qwen35moe
    arch_fallback for otherwise-unrecognised filenames. Their model cards
    explicitly use Qwen3.5's recommended sampling, so this profile is the
    correct target rather than _default."""
    profiles = load_profiles(SETTINGS_DIR)
    qwen = "Qwen3.5 / Qwen3.6 (Alibaba)"

    # Filename patterns (work even without an arch hint).
    assert match_profile("Qwable-27b_Q4_K_M.gguf", profiles).display_name == qwen
    assert (
        match_profile("Qwopus3.5-9B-coder-Exp-Q8_0.gguf", profiles).display_name == qwen
    )

    # arch_fallback: an unrecognised filename whose arch is qwen35 / qwen35moe.
    assert (
        match_profile("Merged-Hf-Final.gguf", profiles, "qwen35").display_name == qwen
    )
    assert (
        match_profile("some-a3b-moe-requant.gguf", profiles, "qwen35moe").display_name
        == qwen
    )

    # A real qwen3.5 filename still matches by pattern (unchanged).
    assert match_profile("Qwen3.5-9B-Q8_0.gguf", profiles).display_name == qwen


def test_cohere2moe_and_vibethinker_and_minicpm5_profiles_load() -> None:
    """The three new profiles load with the expected match behaviour and
    each carries the mandatory --jinja extra arg."""
    profiles = load_profiles(SETTINGS_DIR)

    # cohere2moe: pattern + arch_fallback (North-Mini-Code, fork-only arch).
    north = match_profile("North-Mini-Code-1.0-UD-Q6_K_XL.gguf", profiles, "cohere2moe")
    assert north.display_name == "North-Mini-Code (Cohere, cohere2moe)"
    assert "cohere2moe" in north.arch_fallback
    assert "--jinja" in north.extra_args
    # Re-quant with an unrecognised filename still caught by arch_fallback.
    assert (
        match_profile("weird-name.gguf", profiles, "cohere2moe").display_name
        == "North-Mini-Code (Cohere, cohere2moe)"
    )

    # VibeThinker: pattern only, NO arch_fallback (arch is generic "qwen2").
    vt = match_profile("VibeThinker-3B.F32.gguf", profiles, "qwen2")
    assert vt.display_name == "VibeThinker (WeiboAI, reasoning)"
    assert vt.arch_fallback == []
    assert "--jinja" in vt.extra_args
    # A plain qwen2 model must NOT be swallowed by the VibeThinker profile.
    assert (
        match_profile("Qwen2.5-7B-Instruct-Q8_0.gguf", profiles, "qwen2").display_name
        != "VibeThinker (WeiboAI, reasoning)"
    )

    # MiniCPM5: pattern only, NO arch_fallback (arch is generic "llama").
    mc = match_profile("MiniCPM5-1B-F16.gguf", profiles, "llama")
    assert mc.display_name == "MiniCPM5 (OpenBMB, on-device)"
    assert mc.arch_fallback == []
    assert "--jinja" in mc.extra_args
    # A plain llama model must NOT be swallowed by the MiniCPM5 profile.
    assert (
        match_profile("Llama-3.1-8B-Instruct-Q8_0.gguf", profiles, "llama").display_name
        != "MiniCPM5 (OpenBMB, on-device)"
    )


def test_arch_fallback_is_backward_compatible() -> None:
    """Calling match_profile without an arch (old 2-arg form) is unchanged:
    pattern matching still works and unknowns still hit _default."""
    profiles = load_profiles(SETTINGS_DIR)

    # Pattern still wins with no arch supplied.
    assert (
        match_profile("gpt-oss-20b-UD-Q6_K_XL.gguf", profiles).display_name
        == "gpt-oss (OpenAI)"
    )
    # Unknown with no arch → generic fallback (exactly as before).
    assert (
        match_profile("Some-Random-LLM.gguf", profiles).display_name
        == "Generic / fallback"
    )


def test_filename_pattern_beats_arch_fallback() -> None:
    """If a filename pattern matches one profile but the arch_fallback of
    another would also match, the filename pattern must win."""
    profiles = load_profiles(SETTINGS_DIR)

    # gpt-oss filename pattern + a (hypothetical) mismatched arch arg:
    # the pattern match must take precedence over any arch fallback.
    p = match_profile("gpt-oss-20b-UD-Q6_K_XL.gguf", profiles, "some-other-arch")
    assert p.display_name == "gpt-oss (OpenAI)"


# ---------------------------------------------------------------------------
# Regression tests for diffusion-LLM support (Dream/LLaDA/RND1/DiffusionGemma)
# ---------------------------------------------------------------------------


def test_diffusion_architecture_detection() -> None:
    """Diffusion archs (mainline + fork) are detected; autoregressive aren't."""
    from scanner import metadata_is_diffusion_architecture

    for arch in (
        "dream",
        "llada",
        "llada-moe",
        "rnd1",
        "diffusion-gemma",
        "diffusiongemma",
    ):
        assert (
            metadata_is_diffusion_architecture({"general.architecture": arch}) is True
        ), f"{arch} should be detected as diffusion"

    for arch in ("gemma4", "qwen35moe", "llama", "gpt-oss"):
        assert (
            metadata_is_diffusion_architecture({"general.architecture": arch}) is False
        ), f"{arch} must NOT be detected as diffusion"


def test_model_entry_is_diffusion_property() -> None:
    """ModelEntry.is_diffusion reflects the architecture."""
    from scanner import ModelEntry

    diff = ModelEntry(
        path=Path("/m/diffusiongemma-26B.gguf"),
        name="diffusiongemma-26B",
        group=".",
        size_bytes=1,
        metadata={"general.architecture": "diffusion-gemma"},
    )
    assert diff.is_diffusion is True

    normal = ModelEntry(
        path=Path("/m/gemma-4-12b.gguf"),
        name="gemma-4-12b",
        group=".",
        size_bytes=1,
        metadata={"general.architecture": "gemma4"},
    )
    assert normal.is_diffusion is False


def test_diffusion_algorithm_normalization() -> None:
    """Algorithm names and ints map to the CLI integer; junk → None."""
    from tuner import _diffusion_algorithm_value

    assert _diffusion_algorithm_value("confidence") == 4
    assert _diffusion_algorithm_value("entropy") == 1
    assert _diffusion_algorithm_value("margin") == 2
    assert _diffusion_algorithm_value("random") == 3
    assert _diffusion_algorithm_value("origin") == 0
    assert _diffusion_algorithm_value(0) == 0
    assert _diffusion_algorithm_value(4) == 4
    assert _diffusion_algorithm_value("3") == 3
    # out of range / junk / bool
    assert _diffusion_algorithm_value(7) is None
    assert _diffusion_algorithm_value("nope") is None
    assert _diffusion_algorithm_value(None) is None
    assert _diffusion_algorithm_value("") is None
    assert _diffusion_algorithm_value(True) is None


def test_diffusiongemma_kv_broadcast_not_undersized() -> None:
    """DiffusionGemma stores head_count_kv=[2] as a broadcast scalar (a
    single-element list applying to all 30 layers), unlike Gemma-4's
    30-element per-layer array. The KV estimate must expand the broadcast
    to the full layer count, otherwise it is ~30x too small and
    compute_config picks a context that OOMs on KV allocation.

    Regression for the Vulkan ``alloc_tensor_range: failed to allocate
    Vulkan0 buffer of size 1073741824`` crash on DiffusionGemma.
    """
    from tuner import kv_per_token_mb_from_metadata

    # Real DiffusionGemma GGUF metadata: broadcast scalars in 1-element lists.
    md = {
        "general.architecture": "diffusion-gemma",
        "diffusion-gemma.block_count": 30,
        "diffusion-gemma.attention.head_count": 16,
        "diffusion-gemma.attention.head_count_kv": [2],  # broadcast
        "diffusion-gemma.attention.key_length": 512,
        "diffusion-gemma.attention.value_length": 512,
        "diffusion-gemma.attention.sliding_window": 1024,
        "diffusion-gemma.attention.sliding_window_pattern": [False],  # broadcast
        "diffusion-gemma.attention.key_length_swa": 256,
        "diffusion-gemma.attention.value_length_swa": 256,
        "diffusion-gemma.embedding_length": 2816,
    }
    kvt = kv_per_token_mb_from_metadata(md)
    # 30 layers * 2 kv_heads * (512+512) * 2 bytes = 122880 B = 0.1172 MB
    assert kvt > 0.10, f"KV/token far too small ({kvt:.4f} MB) — broadcast not expanded"
    assert kvt < 0.15, f"KV/token unexpectedly large ({kvt:.4f} MB)"
    # Sanity: at max_context 262144 the cache is huge (this is why it OOMs).
    assert kvt * 262144 >= 29 * 1024, "expected ~30 GiB KV at full context"


def test_gemma4_per_layer_kv_array_not_broken_by_broadcast_fix() -> None:
    """Gemma-4's 30-element per-layer KV-head array must still be summed
    correctly (only full-attention layers) after the DiffusionGemma
    broadcast expansion was added — it is a regression guard for that fix.
    """
    from tuner import kv_per_token_mb_from_metadata

    md = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 30,
        "gemma4.attention.head_count": 16,
        # 5 full-attention (8 heads) + 1 SWA (2 heads), repeated 5x = 30 entries
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2] * 5,
        # 5 SWA layers (True) + 1 full-attention layer (False) per 6-group;
        # only the False (full-attention) layers carry scaling KV.
        "gemma4.attention.sliding_window_pattern": [True, True, True, True, True, False]
        * 5,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.embedding_length": 2816,
    }
    kvt = kv_per_token_mb_from_metadata(md)
    # Only the False (full-attention) layers count: 1 per group * 5 = 5, each 2 kv_heads
    # 5 * 2 * (512+512) * 2 = 20480 B = 0.0195 MB
    assert 0.015 < kvt < 0.025, f"Gemma-4 KV/token changed unexpectedly ({kvt:.4f} MB)"


def test_diffusion_resolver_picks_gemma_binary_for_diffusiongemma() -> None:
    """The diffusion binary resolver must prefer llama-diffusion-gemma-cli
    for the diffusion-gemma architecture (PR #24427 fork binary), while
    mainline diffusion archs (dream/llada/rnd1) keep the generic
    llama-diffusion-cli. Without this, DiffusionGemma would launch with
    the wrong (mainline) binary.
    """
    import auto_tuner as at

    assert (
        at._diffusion_binary_for_arch("diffusion-gemma") == "llama-diffusion-gemma-cli"
    )
    assert (
        at._diffusion_binary_for_arch("diffusion_gemma") == "llama-diffusion-gemma-cli"
    )
    # Mainline diffusion archs keep the generic CLI.
    for arch in ("dream", "llada", "rnd1", None):
        assert at._diffusion_binary_for_arch(arch) == "llama-diffusion-cli"

    # Subpath builder produces the native gemma binary name when asked.
    subs = at._diffusion_subpaths_for("llama-diffusion-gemma-cli")
    expected_name = (
        "llama-diffusion-gemma-cli.exe"
        if os.name == "nt"
        else "llama-diffusion-gemma-cli"
    )
    assert any(s.endswith(expected_name) for s in subs)
    if os.name != "nt":
        assert not any(s.endswith(".exe") for s in subs)


def test_build_diffusion_server_command_gemma_server() -> None:
    """DiffusionGemma runs via llama-diffusion-gemma-server (PR #24427 HTTP
    server). The dedicated builder emits ONLY flags the fork's manual arg
    parser understands — it must NOT contain llama-server-only flags
    (--fit/--jinja/--spec-type/--cache-ram) or the binary aborts with
    'unknown argument'. It must bind --host/--port and forward the GPU
    placement so the model boots on the 32 GB card.
    """
    from tuner import build_diffusion_server_command

    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile("diffusiongemma-26B-A4B-it-Q8_0.gguf", profiles)
    assert profile.runner == "llama-diffusion-gemma-server"

    cfg = _fake_diffusion_config()
    cfg.main_gpu = 1
    cfg.tensor_split = "0.000,1.000"
    cmd = build_diffusion_server_command(
        _fake_diffusion_model(),
        cfg,
        profile,
        server_binary="d_b9781/llama-diffusion-gemma-server",
        host="127.0.0.1",
        port=8080,
        alias="diffusiongemma",
    )

    assert cmd[0] == "d_b9781/llama-diffusion-gemma-server"
    # Core load flags
    assert "-m" in cmd and "-c" in cmd and "-ngl" in cmd
    # Diffusion + HTTP binding
    assert "--diffusion-steps" in cmd
    assert "--perf" in cmd
    assert "--host" in cmd
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "8080"
    # GPU placement forwarded
    assert "--main-gpu" in cmd
    assert cmd[cmd.index("--main-gpu") + 1] == "1"
    assert "--tensor-split" in cmd
    # Prompt (-p) must NOT be present — the server takes requests via HTTP
    assert "-p" not in cmd
    # Forbidden llama-server-only flags
    for forbidden in ("--fit", "--jinja", "--spec-type", "--cache-ram"):
        assert forbidden not in cmd, f"{forbidden} must not reach the gemma-server"


def _fake_diffusion_config():
    from tuner import TunedConfig

    return TunedConfig(
        ctx=8192,
        ngl=99,
        threads=8,
        batch_threads=8,
        batch=2048,
        ubatch=512,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
    )


def _fake_diffusion_model():
    from scanner import ModelEntry

    return ModelEntry(
        path=Path("/models/diffusiongemma-26B-A4B-it-Q8_0.gguf"),
        name="diffusiongemma-26B-A4B-it-Q8_0",
        group=".",
        size_bytes=25 * 1024**3,
        metadata={
            "general.architecture": "diffusion-gemma",
            "diffusion-gemma.block_count": 30,
            "diffusion-gemma.context_length": 262144,
            "diffusion-gemma.embedding_length": 2816,
            "diffusion-gemma.attention.head_count": 16,
            "diffusion-gemma.attention.head_count_kv": [2],
            "diffusion-gemma.attention.key_length": 512,
            "diffusion-gemma.attention.value_length": 512,
            "diffusion-gemma.attention.sliding_window_pattern": [False],
            "diffusion-gemma.expert_count": 128,
        },
    )


def test_diffusiongemma_auto_memory_contract() -> None:
    """The dedicated PR #24427 server runs F16 KV and cannot apply
    expert-only CPU offload, so Auto must report that exact runtime contract."""
    from hardware import GPUInfo, SystemInfo

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_diffusion_model()
    profile = match_profile(model.name, profiles)
    system = SystemInfo(
        os_name="Windows test",
        cpu_name="Test CPU",
        cpu_cores_physical=24,
        cpu_cores_logical=24,
        total_ram_gb=48,
        free_ram_gb=40,
        gpus=[
            GPUInfo(
                index=0,
                name="RX 9070 XT",
                vendor="amd",
                total_vram_mb=16 * 1024,
                free_vram_mb=15 * 1024,
                hip_index=0,
            ),
            GPUInfo(
                index=1,
                name="Radeon AI PRO R9700",
                vendor="amd",
                total_vram_mb=32 * 1024,
                free_vram_mb=31 * 1024,
                hip_index=1,
            ),
        ],
    )
    cfg = compute_config(
        model,
        system,
        profile,
        prompt_cache_ram_mib=0,
        gpu_priorities={"Radeon AI PRO R9700": 2, "RX 9070 XT": 1},
    )
    assert cfg.ctx == 4096
    assert (cfg.cache_k, cfg.cache_v) == ("f16", "f16")
    assert cfg.estimated_kv_gb == pytest.approx(0.46875, rel=0.02)
    assert cfg.n_cpu_moe is None
    assert cfg.runtime_vram_overhead_gb == pytest.approx(1.5)
    total_gpu = (
        cfg.estimated_model_vram_gb
        + cfg.kv_vram_gb
        + cfg.vision_vram_gb
        + cfg.draft_vram_gb
        + cfg.runtime_vram_overhead_gb
    )
    assert total_gpu >= model.size_gb + 1.9
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "1"

    # The 4096 limit is an Auto safety default, not an unconditional model
    # cap: an explicit pin remains available for the HIP build and is still
    # clamped by the real F16 VRAM budget.
    pinned = compute_config(
        model,
        system,
        profile,
        user_ctx=8192,
        prompt_cache_ram_mib=0,
        gpu_priorities={"Radeon AI PRO R9700": 2, "RX 9070 XT": 1},
    )
    assert pinned.ctx == 8192
    assert pinned.estimated_kv_gb == pytest.approx(0.9375, rel=0.02)

    from performance_target import get_target

    low_vram = compute_config(
        model,
        system,
        profile,
        perf_target=get_target("low_vram"),
        prompt_cache_ram_mib=0,
    )
    assert low_vram.no_kv_offload is False
    assert low_vram.kv_vram_gb == pytest.approx(low_vram.estimated_kv_gb)
    assert low_vram.kv_ram_gb == 0


def test_build_diffusion_command_mainline_flags() -> None:
    """The diffusion command uses llama-diffusion-cli with mainline b9700
    flags and NO server flags (host/port/fit)."""
    from tuner import build_diffusion_command

    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile("diffusiongemma-26B-A4B-it-Q8_0.gguf", profiles)
    cmd = build_diffusion_command(
        _fake_diffusion_model(),
        _fake_diffusion_config(),
        profile,
        diffusion_binary="d_b96/llama-diffusion-cli",
        prompt="hello",
    )

    assert cmd[0] == "d_b96/llama-diffusion-cli"
    assert "-m" in cmd
    assert "--diffusion-steps" in cmd
    assert "--diffusion-algorithm" in cmd
    assert "--perf" in cmd
    # confidence → 4
    assert cmd[cmd.index("--diffusion-algorithm") + 1] == "4"
    assert "-p" in cmd

    # No server-only flags must leak into the diffusion command.
    for forbidden in ("--host", "--port", "--fit", "--metrics", "--jinja"):
        assert forbidden not in cmd, f"{forbidden} must not be in a diffusion cmd"


def test_build_diffusion_command_gpu_placement_passthrough() -> None:
    """compute_config's multi-GPU placement must reach the diffusion CLI.

    Without --main-gpu/--tensor-split the binary defaults to Vulkan device
    0 (often the smaller card) and OOMs on KV allocation. The fix for the
    DiffusionGemma ``alloc_tensor_range: failed to allocate Vulkan0 buffer
    of size 1073741824`` crash forwards these from the TunedConfig.
    """
    from tuner import build_diffusion_command
    from settings_loader import ModelProfile

    cfg = _fake_diffusion_config()
    cfg.main_gpu = 1  # pin to the 32 GB card
    cfg.tensor_split = "0.000,1.000"
    p = ModelProfile(display_name="x", diffusion={"steps": 48})
    cmd = build_diffusion_command(_fake_diffusion_model(), cfg, p)

    assert "--main-gpu" in cmd, "--main-gpu must be forwarded to pin the large card"
    assert cmd[cmd.index("--main-gpu") + 1] == "1"
    assert "--tensor-split" in cmd
    assert cmd[cmd.index("--tensor-split") + 1] == "0.000,1.000"

    # When unset (single-GPU system), neither flag is emitted.
    cfg2 = _fake_diffusion_config()
    cfg2.main_gpu = None
    cfg2.tensor_split = None
    cmd2 = build_diffusion_command(_fake_diffusion_model(), cfg2, p)
    assert "--main-gpu" not in cmd2
    assert "--tensor-split" not in cmd2


def test_build_diffusion_command_eps_xor_block_length() -> None:
    """eps and block_length are mutually exclusive; block_length wins if
    both are set, and a profile with only eps emits --diffusion-eps."""
    from tuner import build_diffusion_command
    from settings_loader import ModelProfile

    base_model = _fake_diffusion_model()
    cfg = _fake_diffusion_config()

    # Only eps → --diffusion-eps present, block-length absent.
    p_eps = ModelProfile(display_name="x", diffusion={"steps": 128, "eps": 0.001})
    cmd = build_diffusion_command(base_model, cfg, p_eps)
    assert "--diffusion-eps" in cmd
    assert "--diffusion-block-length" not in cmd

    # Both → block-length wins, eps dropped.
    p_both = ModelProfile(
        display_name="x", diffusion={"eps": 0.001, "block_length": 32}
    )
    cmd = build_diffusion_command(base_model, cfg, p_both)
    assert "--diffusion-block-length" in cmd
    assert "--diffusion-eps" not in cmd


def test_build_diffusion_command_fork_args_passthrough() -> None:
    """Fork-only flags in diffusion.fork_args are appended verbatim."""
    from tuner import build_diffusion_command
    from settings_loader import ModelProfile

    p = ModelProfile(
        display_name="x",
        diffusion={
            "steps": 64,
            "fork_args": ["--diffusion-eb", "--diffusion-kv-cache"],
        },
    )
    cmd = build_diffusion_command(_fake_diffusion_model(), _fake_diffusion_config(), p)
    assert "--diffusion-eb" in cmd
    assert "--diffusion-kv-cache" in cmd


def test_diffusion_gemma_profile_loads_with_runner() -> None:
    """The shipped diffusion-gemma profile uses the DiffusionGemma HTTP server
    runner (PR #24427), not the single-shot CLI."""
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile("diffusiongemma-26B-A4B-it-Q8_0.gguf", profiles)
    assert p.runner == "llama-diffusion-gemma-server"
    assert p.display_name.startswith("DiffusionGemma")
    # Diffusion block carries the expected keys.
    # steps: Doku-Empfehlung max 48 (adaptives Early-Stop meist 12-16).
    assert p.diffusion.get("steps") == 48
    assert "block_length" in p.diffusion


def test_diffusion_runner_does_not_affect_server_profiles() -> None:
    """Normal profiles keep runner == '' (server path)."""
    profiles = load_profiles(SETTINGS_DIR)
    for name in (
        "gpt-oss-20b-UD-Q6_K_XL.gguf",
        "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
        "gemma-4-12b-it-BF16.gguf",
    ):
        p = match_profile(name, profiles)
        assert p.runner == "", f"{name} should use the server path, got {p.runner!r}"


def test_diffusion_gemma_profile_has_no_hardcoded_binary() -> None:
    """The diffusion-gemma profile must NOT hard-code a server_binary path.

    The diffusion binary is resolved from the fork selected in the GUI
    dropdown (LLAMA_CPP_DIR), so a daily-rebuilt fork needs no code/profile
    edits. A pinned server_binary would override that choice.
    """
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile("diffusiongemma-26B-A4B-it-Q8_0.gguf", profiles)
    assert p.server_binary is None, (
        "diffusion-gemma must not pin server_binary; it relies on the fork "
        f"dropdown / LLAMA_CPP_DIR. Got: {p.server_binary!r}"
    )


def test_diffusion_binary_request_falls_through_without_pin() -> None:
    """With no server_binary pin, the diffusion request defaults to the bare
    'llama-diffusion-cli' name — which the resolver then searches for inside
    the selected fork. This is the indirection that avoids hard-coding."""
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile("diffusiongemma-26B-A4B-it-Q8_0.gguf", profiles)
    request = p.server_binary or "llama-diffusion-cli"
    assert request == "llama-diffusion-cli"


def test_large_and_special_model_profiles_route_correctly() -> None:
    """The big/special model profiles (GLM-5, Seed-OSS, Qwen3-VL,
    Hunyuan-A13B, ERNIE, Ling/Ring, dots1, Kimi-K2, Grok-2, SmolLM3,
    Ling-2.6 fork) each match their model — via filename pattern and, where
    declared, via arch_fallback for unrecognised filenames."""
    profiles = load_profiles(SETTINGS_DIR)

    def dn(stem: str, arch: str | None = None) -> str:
        return match_profile(stem, profiles, arch).display_name

    # Filename-pattern matches.
    assert "GLM-5" in dn("GLM-5-UD-IQ2_XXS-00001-of-00006.gguf")
    assert "Seed-OSS" in dn("Seed-OSS-36B-Instruct-Q4_K_M.gguf")
    assert "Qwen3-VL" in dn("Qwen3-VL-32B-Instruct-Q8_0.gguf")
    assert "Hunyuan-A13B" in dn("Hunyuan-A13B-Instruct-Q4_K_M.gguf")
    assert "ERNIE" in dn("ERNIE-4.5-21B-A3B-Thinking-Q4_K_M.gguf")
    assert "Ling / Ring" in dn("Ling-mini-2.0-Q4_K_M.gguf")
    assert "dots.llm1" in dn("dots.llm1.inst-Q4_K_M.gguf")
    assert "Kimi-K2" in dn("Kimi-K2-Instruct-Q2_K_XL-00001-of-00008.gguf")
    assert "Grok-2" in dn("grok-2-Q4_K_M-00001-of-00010.gguf")
    assert "SmolLM3" in dn("SmolLM3-3B-Q8_0.gguf")
    assert "Ling-2.6-flash" in dn("inclusionAI__Ling-2.6-flash-IQ2_XS.gguf")

    # arch_fallback matches (unrecognised filename, known arch).
    assert "GLM-5" in dn("weird-glm-requant.gguf", "glm5")
    assert "Seed-OSS" in dn("random-name.gguf", "seed_oss")
    assert "Qwen3-VL" in dn("vl-merge.gguf", "qwen3vlmoe")
    assert "Hunyuan-A13B" in dn("hy-moe-merge.gguf", "hunyuan_moe")
    assert "ERNIE" in dn("baidu-moe-requant.gguf", "ernie4_5-moe")
    assert "Ling / Ring" in dn("bailing-requant.gguf", "bailingmoe2")
    assert "dots.llm1" in dn("dots-requant.gguf", "dots1")
    assert "Grok-2" in dn("grok-requant.gguf", "grok")


def test_deepseek2_family_does_not_collide() -> None:
    """DeepSeek-V3/R1, GLM-4.7-Flash and Kimi-K2 ALL carry
    general.architecture 'deepseek2'. Each must route to its own profile via
    filename pattern, and none may declare arch_fallback: deepseek2 (which
    would hijack the others)."""
    profiles = load_profiles(SETTINGS_DIR)

    def dn(stem: str) -> str:
        # arch hint is deepseek2 for all three — only the filename disambiguates
        return match_profile(stem, profiles, "deepseek2").display_name

    assert "DeepSeek" in dn("DeepSeek-V3.2-Q4_K_M.gguf")
    assert "DeepSeek" in dn("DeepSeek-R1-Q4_K_M.gguf")
    assert "GLM-4.7" in dn("GLM-4.7-Flash-UD-Q6_K_XL.gguf")
    assert "GLM-4.7" in dn("GLM-4.7-Flash-REAP-23B-A3B-UD-Q8_K_XL.gguf")
    assert "Kimi-K2" in dn("Kimi-K2-Instruct-0905-Q2_K.gguf")

    # No profile may claim deepseek2 as an arch_fallback (that would make the
    # disambiguation order-dependent and fragile).
    for p in profiles:
        assert "deepseek2" not in p.arch_fallback, (
            f"{p.source_file} must not declare arch_fallback: deepseek2"
        )


def test_hunyuan_dense_vs_moe_split() -> None:
    """Hunyuan-MT (dense, translation, top_p 0.6), Hunyuan-A13B (MoE, chat,
    top_p 0.8) and the new Hy-MT2 family (hy_v3, top_p 1.0) must all use
    different profiles."""
    profiles = load_profiles(SETTINGS_DIR)

    mt = match_profile("Hunyuan-MT-7B-UD-Q8_K_XL.gguf", profiles, "hunyuan-dense")
    assert "Hunyuan-MT" in mt.display_name
    # dense MT profile keeps the translation sampling (top_p 0.6)
    assert mt.sampling["chat"]["top_p"] == 0.6

    a13b = match_profile("Hunyuan-A13B-Instruct-Q4_K_M.gguf", profiles, "hunyuan_moe")
    assert "A13B" in a13b.display_name
    # MoE chat profile uses top_p 0.8 (NOT the translation 0.6)
    assert a13b.sampling["chat"]["top_p"] == 0.8

    # Hy-MT2 (May 2026, arch hy_v3): the longer "hy-mt2" pattern must beat
    # the old "hy-mt" pattern in hunyuan.yaml, for all sizes incl. the
    # 30B-A3B MoE. Official sampling is top_p 1.0 / top_k off / rep 1.0.
    for fname in ("Hy-MT2-30B-A3B-Q4_K_M.gguf", "Hy-MT2-7B-UD-Q8_K_XL.gguf"):
        mt2 = match_profile(fname, profiles, "hy_v3")
        assert "Hy-MT2" in mt2.display_name, fname
        assert mt2.sampling["chat"]["top_p"] == 1.0, fname
        assert mt2.sampling["chat"]["repeat_penalty"] == 1.0, fname

    # arch_fallback: a hy_v3 re-quant without a matching filename must still
    # land on the Hy-MT2 profile (not hunyuan-dense, not _default).
    requant = match_profile("some-translation-requant.gguf", profiles, "hy_v3")
    assert "Hy-MT2" in requant.display_name


def test_agents_a1_profile() -> None:
    """Agents-A1 (Qwen3.5-35B-A3B based) gets its own profile via filename
    pattern; generic qwen35moe re-quants must still land on qwen3_5-3_6."""
    profiles = load_profiles(SETTINGS_DIR)

    a1 = match_profile("Agents-A1-Q4_K_M.gguf", profiles, "qwen35moe")
    assert "Agents-A1" in a1.display_name
    # official card sampling: temp 0.85 / top_p 0.95 / top_k 20
    assert a1.sampling["chat"]["temperature"] == 0.85
    assert a1.sampling["chat"]["top_p"] == 0.95
    # must NOT claim the shared qwen35moe arch (would hijack generic
    # Qwen3.5 re-quants from qwen3_5-3_6.yaml — same rule as ornith.yaml)
    assert "qwen35moe" not in a1.arch_fallback
    assert "qwen35" not in a1.arch_fallback

    generic = match_profile("random-qwen-requant.gguf", profiles, "qwen35moe")
    assert "Qwen3.5" in generic.display_name


def test_ling_2_0_mainline_vs_2_6_fork() -> None:
    """Ling/Ring 2.0 (bailingmoe2, mainline) and Ling-2.6-flash
    (bailingmoe2.5/bailing_hybrid, fork) must use different profiles."""
    profiles = load_profiles(SETTINGS_DIR)

    v20 = match_profile("Ling-flash-2.0-Q4_K_M.gguf", profiles, "bailingmoe2")
    assert "Ling / Ring" in v20.display_name

    v26 = match_profile(
        "inclusionAI__Ling-2.6-flash-IQ2_XS.gguf", profiles, "bailing_hybrid"
    )
    assert "Ling-2.6-flash" in v26.display_name
    # the fork profile must NOT claim the mainline arch
    assert "bailingmoe2" not in v26.arch_fallback


def test_v511_new_model_profiles_and_architecture_fallbacks() -> None:
    """New 2026 profiles match their own families without hijacking older ones."""
    profiles = load_profiles(SETTINGS_DIR)

    muse = match_profile("muse-glimmer-30B-kquant-17gb.gguf", profiles, "muse-glimmer")
    assert muse.source_file == "muse-glimmer.yaml"
    assert muse.max_context == 131072
    assert muse.min_llama_build == 10353
    assert muse.sampling["chat"]["top_k"] == 64
    assert muse.draft_max == 15

    ling3 = match_profile("Ling-3.0-flash-int4.gguf", profiles, "bailingmoe3")
    assert ling3.source_file == "ling-3.yaml"
    assert ling3.max_context == 262144
    assert ling3.min_llama_build == 10460
    assert ling3.sampling["coding"]["temperature"] == 0.6
    assert "b10460" in ling3.notes
    assert "bailingmoe3" in ling3.arch_fallback
    # Ling 2.6 keeps its separate filename profile despite the broad shared
    # HF model_type; neither generation may steal the other's named GGUF.
    ling26 = match_profile("Ling-2.6-flash-Q4_K_M.gguf", profiles, "bailing_hybrid")
    assert ling26.source_file == "bailingmoe2-5.yaml"

    shield = match_profile("Shieldstral-1.0-3B-Q8_0.gguf", profiles, "ministral3")
    assert shield.source_file == "shieldstral.yaml"
    assert shield.max_context == 32768
    assert shield.sampling["chat"]["temperature"] == 0.0
    assert shield.sampling["chat"]["top_k"] == 0

    minimax = match_profile(
        "MiniMax-M3-UD-IQ3_XXS-00001-of-00005.gguf", profiles, "minimax-m3"
    )
    assert minimax.source_file == "minimax-m3.yaml"
    assert minimax.max_context == 1048576
    assert minimax.min_llama_build == 10142
    assert minimax.flash_attn is True
    # M2 must remain on its older, smaller family profile.
    assert (
        match_profile("MiniMax-M2.7-Q4_K_M.gguf", profiles, "minimax-m2").source_file
        == "minimax-m2.yaml"
    )

    kimi3 = match_profile("Kimi-K3-UD-Q2_K-00001-of-00034.gguf", profiles, "kimi-k3")
    assert kimi3.source_file == "kimi-k3.yaml"
    assert kimi3.max_context == 1048576
    assert kimi3.min_llama_build == 10448
    assert "b10448" in kimi3.notes
    assert "TEXT loader" in kimi3.notes
    assert (
        match_profile("Kimi-K2.5-Q4_K_M.gguf", profiles, "deepseek2").source_file
        == "kimi-k2.yaml"
    )

    glm52 = match_profile("GLM-5.2-UD-Q4_K_XL.gguf", profiles, "glm-dsa")
    assert glm52.source_file == "glm-5_2.yaml"
    assert glm52.max_context == 1048576
    assert glm52.min_llama_build == 10174
    assert glm52.ngram_method == "ngram-map-k4v"
    assert match_profile("opaque-glm.gguf", profiles, "glm-dsa") is glm52
    glm51 = match_profile("GLM-5.1-Q4_K_M.gguf", profiles, "glm5")
    assert glm51.source_file == "glm-5.yaml"
    assert glm51.min_llama_build == 0

    dsv4 = match_profile("DeepSeek-V4-Flash-Q4_K_M.gguf", profiles, "deepseek4")
    assert dsv4.source_file == "deepseek-v4.yaml"
    assert dsv4.max_context == 1048576
    assert dsv4.min_llama_build == 10254
    assert dsv4.sampling["chat"]["top_p"] == 1.0
    assert match_profile("opaque-deepseek.gguf", profiles, "deepseek4") is dsv4
    assert (
        match_profile("DeepSeek-V3.2-Q4_K_M.gguf", profiles, "deepseek2").source_file
        == "deepseek-r1-v3.yaml"
    )

    switch = match_profile("opaque-adapter-model.gguf", profiles, "graniteswitch")
    assert switch.source_file == "granite-switch-4_1.yaml"
    assert switch.min_llama_build == 10342

    # Every unique new architecture must work even when a community requant
    # has an opaque filename, and no profile may claim the same exact fallback.
    for opaque, arch, expected in (
        ("opaque-a.gguf", "muse-glimmer", "muse-glimmer.yaml"),
        ("opaque-b.gguf", "minimax-m3", "minimax-m3.yaml"),
        ("opaque-c.gguf", "kimi-k3", "kimi-k3.yaml"),
        ("opaque-c2.gguf", "bailingmoe3", "ling-3.yaml"),
        ("opaque-d.gguf", "glm-dsa", "glm-5_2.yaml"),
        ("opaque-e.gguf", "deepseek4", "deepseek-v4.yaml"),
        ("opaque-f.gguf", "graniteswitch", "granite-switch-4_1.yaml"),
        ("opaque-g.gguf", "qwen4exp", "qwen3_8_flash_next.yaml"),
    ):
        assert match_profile(opaque, profiles, arch).source_file == expected

    owners: dict[str, str] = {}
    for candidate in profiles:
        for arch in candidate.arch_fallback:
            assert arch not in owners, (
                f"duplicate arch_fallback {arch!r}: "
                f"{owners[arch]} and {candidate.source_file}"
            )
            owners[arch] = str(candidate.source_file)


def test_profile_minimum_build_gate(monkeypatch) -> None:
    """Known-old builds are rejected; current and unprobeable shims are safe."""
    import tuner
    from settings_loader import ModelProfile

    profile = ModelProfile(display_name="Future model", min_llama_build=12345)

    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _binary: 12000)
    allowed, message, detected = tuner.check_profile_build(profile, "server")
    assert allowed is False
    assert detected == 12000
    assert "b12345+" in message and "b12000" in message

    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _binary: 12345)
    assert tuner.check_profile_build(profile, "server") == (True, "", 12345)

    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _binary: None)
    allowed, message, detected = tuner.check_profile_build(profile, "wrapper")
    assert allowed is True
    assert detected is None
    assert "Could not verify" in message


def test_new_big_profiles_have_jinja_where_required() -> None:
    """Profiles for chat-template / tool-call / thinking models must emit
    --jinja (GLM-5, Seed-OSS, Qwen3-VL, Hunyuan-A13B, ERNIE, Ling/Ring,
    Kimi-K2, SmolLM3)."""
    profiles = load_profiles(SETTINGS_DIR)
    by_file = {p.source_file: p for p in profiles}
    for fname in (
        "glm-5.yaml",
        "glm-5_2.yaml",
        "granite-switch-4_1.yaml",
        "seed-oss.yaml",
        "qwen3-vl.yaml",
        "hunyuan-moe.yaml",
        "ernie-4_5.yaml",
        "ling-ring.yaml",
        "kimi-k2.yaml",
        "smollm3.yaml",
        "bailingmoe2-5.yaml",
        "deepseek-v4.yaml",
        "kimi-k3.yaml",
        "ling-3.yaml",
        "minimax-m3.yaml",
        "muse-glimmer.yaml",
        "shieldstral.yaml",
    ):
        assert fname in by_file, f"{fname} not loaded"
        assert "--jinja" in by_file[fname].extra_args, f"{fname} missing --jinja"


def test_agentworld_overrides_qwen35_chat_settings() -> None:
    """Qwen-AgentWorld carries arch 'qwen35moe' (it is Qwen3.5-35B-A3B
    retrained as a world model). The qwen3_5-3_6 profile declares
    arch_fallback [qwen35, qwen35moe], so without a dedicated profile
    AgentWorld would inherit the generic Qwen3.5 chat/coder sampling. The
    'agentworld' filename pattern must win (longest-substring beats
    arch_fallback) and apply AgentWorld's own world-model sampling
    (temp=0.6, top_p=0.95, top_k=20) in both modes."""
    profiles = load_profiles(SETTINGS_DIR)

    def p(stem: str):
        # arch hint is qwen35moe — the same arch the qwen3.5 profile claims
        return match_profile(stem, profiles, "qwen35moe")

    aw = p("Qwen-AgentWorld-35B-A3B-UD-Q8_K_XL.gguf")
    assert "AgentWorld" in aw.display_name
    # world-model sampling, NOT the qwen3.5 thinking temp 1.0
    for mode in ("chat", "coding"):
        assert aw.sampling[mode]["temperature"] == 0.6
        assert aw.sampling[mode]["top_p"] == 0.95
        assert aw.sampling[mode]["top_k"] == 20
    assert "--jinja" in aw.extra_args
    # No arch_fallback: arch qwen35moe is shared with regular Qwen3.5/3.6, so
    # claiming it would collide with qwen3_5-3_6.yaml (order-dependent). The
    # filename pattern is the only intended match.
    assert aw.arch_fallback == []

    # The 397B-A17B variant routes to AgentWorld too (filename pattern).
    assert "AgentWorld" in p("Qwen-AgentWorld-397B-A17B-IQ2_XXS.gguf").display_name

    # Regular Qwen3.5/3.6 with the SAME arch must still go to the qwen3.5
    # profile (AgentWorld must not swallow them).
    assert "Qwen3.5 / Qwen3.6" in p("Qwen3.5-35B-A3B-UD-Q6_K.gguf").display_name
    assert "Qwen3.5 / Qwen3.6" in p("Qwen3.6-35B-A3B-Q4_K_M.gguf").display_name

    # An AgentWorld requant WITHOUT 'agentworld' in the name is
    # indistinguishable from plain Qwen3.5-A3B and correctly lands on the
    # qwen3.5 profile via its arch_fallback — not on AgentWorld.
    assert "Qwen3.5 / Qwen3.6" in p("some-a3b-moe-requant.gguf").display_name


# ---------------------------------------------------------------------------
# Expert-panel per-model persistence (autosave + Reset)
#
# A low-VRAM user reported having to re-enter their manual Expert settings
# on every launch. We now persist the full Expert state per model and
# apply it automatically (like the vision/draft/thinking checkbox
# overrides), with a Reset button to revert to Auto. These tests pin the
# storage round-trip and the value↔config translation that the live panel
# and the disk-restore path share.


@pytest.fixture
def _isolated_settings(tmp_path, monkeypatch):
    """Point app_settings at a throwaway JSON file for the duration of a test."""
    import app_settings

    target = tmp_path / "autotuner_settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: target)
    return target


def test_multi_folder_settings_round_trip(_isolated_settings, tmp_path) -> None:
    import app_settings

    a = tmp_path / "models-a"
    b = tmp_path / "models-b"
    app_settings.set_model_paths([(a, True), (b, False), (a, True)])

    got = app_settings.get_model_paths()
    assert got == [(a.resolve(strict=False), True), (b.resolve(strict=False), False)]
    assert app_settings.get_models_path() is None  # folders need not exist in UI list

    llama = tmp_path / "ai-local"
    app_settings.set_llama_build_paths([(llama, False)])
    assert app_settings.get_llama_build_paths() == [
        (llama.resolve(strict=False), False)
    ]


def test_path_settings_are_os_namespaced(_isolated_settings, tmp_path) -> None:
    """The settings JSON is shared between the Windows and Linux boots of a
    dual-boot machine. Path settings must live under per-OS keys so one OS's
    absolute paths never clobber the other's (the '/run/media/…' models_path
    read on Windows report)."""
    import json

    import app_settings

    models = tmp_path / "models"
    models.mkdir()
    app_settings.set_models_path(models)

    raw = json.loads(_isolated_settings.read_text(encoding="utf-8"))
    os_key = f"models_path.{app_settings._OS_KEY_SUFFIX}"
    assert raw[os_key] == str(models.resolve())
    # Legacy mirror kept for older AutoTuner versions on the same OS.
    assert raw["models_path"] == str(models.resolve())
    assert app_settings.get_models_path() == models.resolve()

    # Simulate the OTHER OS overwriting the legacy key with its own path
    # (old-version behaviour): the per-OS key must win unchanged.
    raw["models_path"] = "/run/media/dawasteh/OneDrive/models"
    _isolated_settings.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert app_settings.get_models_path() == models.resolve()

    # Legacy-only file (pre-namespacing) still resolves as fallback.
    legacy_only = {"models_path": str(models.resolve())}
    _isolated_settings.write_text(
        json.dumps(legacy_only, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert app_settings.get_models_path() == models.resolve()


def test_model_view_mode_defaults_validates_and_persists(_isolated_settings) -> None:
    import json

    import app_settings

    assert app_settings.get_model_view_mode() == "list"
    app_settings.set_model_view_mode("tree")
    assert app_settings.get_model_view_mode() == "tree"
    app_settings.set_model_view_mode("invalid")
    assert app_settings.get_model_view_mode() == "tree"

    _isolated_settings.write_text(
        json.dumps({"model_view_mode": "corrupt"}), encoding="utf-8"
    )
    assert app_settings.get_model_view_mode() == "list"


def test_model_tree_collapsed_paths_round_trip_and_ignore_invalid_values(
    _isolated_settings,
) -> None:
    import json

    import app_settings

    collapsed = {"favorites", "folder:Alibaba", "folder:Alibaba\x1fQwen3.6"}
    assert app_settings.get_model_tree_collapsed_paths() == set()
    app_settings.set_model_tree_collapsed_paths(collapsed)
    assert app_settings.get_model_tree_collapsed_paths() == collapsed

    raw = json.loads(_isolated_settings.read_text(encoding="utf-8"))
    setting_key = f"model_tree_collapsed_paths.{app_settings._OS_KEY_SUFFIX}"
    assert raw[setting_key] == sorted(collapsed, key=str.casefold)

    raw[setting_key] = ["favorites", "folder:Google", "folder:", "", 42, None]
    _isolated_settings.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert app_settings.get_model_tree_collapsed_paths() == {
        "favorites",
        "folder:Google",
    }

    app_settings.set_model_tree_collapsed_paths(set())
    assert app_settings.get_model_tree_collapsed_paths() == set()


def test_application_close_preference_is_opt_in(_isolated_settings) -> None:
    import app_settings

    assert app_settings.get_minimize_on_close() is False
    app_settings.set_minimize_on_close(True)
    assert app_settings.get_minimize_on_close() is True
    app_settings.set_minimize_on_close(False)
    assert app_settings.get_minimize_on_close() is False


def test_debug_mode_is_opt_in_and_persists(_isolated_settings) -> None:
    import app_settings

    assert app_settings.get_debug_mode() is False
    app_settings.set_debug_mode(True)
    assert app_settings.get_debug_mode() is True
    app_settings.set_debug_mode(False)
    assert app_settings.get_debug_mode() is False


def test_prompt_cache_limit_and_mmproj_cpu_override_persist(
    _isolated_settings,
) -> None:
    import app_settings

    assert app_settings.get_prompt_cache_ram_mib() == 2048
    app_settings.set_prompt_cache_ram_mib(4096)
    assert app_settings.get_prompt_cache_ram_mib() == 4096
    app_settings.set_prompt_cache_ram_mib(-99)
    assert app_settings.get_prompt_cache_ram_mib() == -1

    app_settings.set_model_override("VisionModel", "mmproj_cpu", True)
    assert app_settings.get_model_overrides("VisionModel")["mmproj_cpu"] is True


def test_model_favorites_persist_and_ignore_invalid_values(
    _isolated_settings, tmp_path
) -> None:
    import json

    import app_settings

    model_a = tmp_path / "root-a" / "SharedName.gguf"
    model_b = tmp_path / "root-b" / "SharedName.gguf"
    key_a = app_settings.favorite_model_key(model_a)
    key_b = app_settings.favorite_model_key(model_b)

    assert app_settings.get_favorite_models() == set()
    app_settings.set_model_favorite(model_b, True)
    app_settings.set_model_favorite(model_a, True)
    app_settings.set_model_favorite(model_b, True)
    assert app_settings.get_favorite_models() == {key_a, key_b}

    raw = json.loads(_isolated_settings.read_text(encoding="utf-8"))
    setting_key = f"favorite_models.{app_settings._OS_KEY_SUFFIX}"
    assert raw[setting_key] == sorted([key_a, key_b], key=str.casefold)

    app_settings.set_model_favorite(model_a, False)
    assert app_settings.get_favorite_models() == {key_b}

    raw = json.loads(_isolated_settings.read_text(encoding="utf-8"))
    raw[setting_key] = [key_b, " ", 42, None]
    _isolated_settings.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert app_settings.get_favorite_models() == {key_b}


def test_system_tray_support_is_native_on_windows_and_macos(monkeypatch) -> None:
    import types

    import qt_launcher

    # Qt may transiently report False while Explorer/Finder restarts. Icons
    # created in that interval are registered automatically when it returns.
    unavailable = types.SimpleNamespace(isSystemTrayAvailable=lambda: False)
    monkeypatch.setattr(qt_launcher, "QSystemTrayIcon", unavailable)
    monkeypatch.setattr(qt_launcher.sys, "platform", "win32")
    assert qt_launcher._system_tray_supported() is True
    monkeypatch.setattr(qt_launcher.sys, "platform", "darwin")
    assert qt_launcher._system_tray_supported() is True
    monkeypatch.setattr(qt_launcher.sys, "platform", "linux")
    assert qt_launcher._system_tray_supported() is False


def test_linux_autostart_desktop_round_trip(tmp_path, monkeypatch) -> None:
    import startup_manager

    desktop = tmp_path / "autostart" / "AutoTuner.desktop"
    monkeypatch.setattr(startup_manager.sys, "platform", "linux")
    monkeypatch.setattr(startup_manager, "_linux_autostart_path", lambda: desktop)
    monkeypatch.setattr(
        startup_manager,
        "launch_arguments",
        lambda: ["/opt/Auto Tuner/python3", "/opt/Auto Tuner/qt_launcher.py"],
    )

    assert startup_manager.is_autostart_enabled() is False
    startup_manager.set_autostart_enabled(True)
    text = desktop.read_text(encoding="utf-8")
    assert 'Exec="/opt/Auto Tuner/python3" "/opt/Auto Tuner/qt_launcher.py"' in text
    assert startup_manager.is_autostart_enabled() is True
    startup_manager.set_autostart_enabled(False)
    assert desktop.exists() is False


def test_windows_autostart_registry_round_trip(monkeypatch) -> None:
    import subprocess
    import sys
    import types

    import startup_manager

    values = {}

    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _query_value(_key, name):
        if name not in values:
            raise FileNotFoundError(name)
        return values[name], 1

    fake_winreg = types.SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        KEY_SET_VALUE=2,
        REG_SZ=1,
        OpenKey=lambda *_args: _Key(),
        CreateKey=lambda *_args: _Key(),
        QueryValueEx=_query_value,
        SetValueEx=lambda _key, name, _reserved, _kind, value: values.__setitem__(
            name, value
        ),
        DeleteValue=lambda _key, name: values.pop(name),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    monkeypatch.setattr(startup_manager.sys, "platform", "win32")
    args = [r"C:\Program Files\AutoTuner\AutoTuner.exe"]
    monkeypatch.setattr(startup_manager, "launch_arguments", lambda: args)

    assert startup_manager.is_autostart_enabled() is False
    startup_manager.set_autostart_enabled(True)
    assert values["AutoTuner"] == subprocess.list2cmdline(args)
    assert startup_manager.is_autostart_enabled() is True
    startup_manager.set_autostart_enabled(False)
    assert startup_manager.is_autostart_enabled() is False


def test_macos_autostart_launch_agent_round_trip(tmp_path, monkeypatch) -> None:
    import plistlib

    import startup_manager

    launch_agent = tmp_path / "LaunchAgents" / "com.dawasteh.autotuner.plist"
    args = ["/Applications/AutoTuner.app/Contents/MacOS/AutoTuner"]
    monkeypatch.setattr(startup_manager.sys, "platform", "darwin")
    monkeypatch.setattr(
        startup_manager, "_macos_launch_agent_path", lambda: launch_agent
    )
    monkeypatch.setattr(startup_manager, "launch_arguments", lambda: args)

    assert startup_manager.is_autostart_enabled() is False
    startup_manager.set_autostart_enabled(True)
    with launch_agent.open("rb") as fh:
        payload = plistlib.load(fh)
    assert payload["Label"] == "com.dawasteh.autotuner"
    assert payload["ProgramArguments"] == args
    assert payload["RunAtLoad"] is True
    startup_manager.set_autostart_enabled(False)
    assert launch_agent.exists() is False


def test_expert_override_round_trip(_isolated_settings) -> None:
    import app_settings

    assert app_settings.get_expert_override("MyModel") is None

    snap = {
        "mode": "manual",
        "pins": {},
        "values": {"ctx": 32768, "cache_k": "q8_0", "threads": 8},
        "saved_at": "2026-06-30T12:00:00",
    }
    app_settings.set_expert_override("MyModel", snap)
    got = app_settings.get_expert_override("MyModel")
    assert got is not None
    assert got["mode"] == "manual"
    assert got["values"]["ctx"] == 32768
    assert got["values"]["cache_k"] == "q8_0"

    # Clearing removes it (Reset button).
    app_settings.clear_expert_override("MyModel")
    assert app_settings.get_expert_override("MyModel") is None


def test_expert_override_isolated_per_model(_isolated_settings) -> None:
    """Two models keep independent Expert states."""
    import app_settings

    app_settings.set_expert_override("A", {"mode": "auto", "values": {"ctx": 4096}})
    app_settings.set_expert_override("B", {"mode": "manual", "values": {"ctx": 131072}})
    assert app_settings.get_expert_override("A")["values"]["ctx"] == 4096
    assert app_settings.get_expert_override("B")["values"]["ctx"] == 131072
    # Clearing one leaves the other intact.
    app_settings.clear_expert_override("A")
    assert app_settings.get_expert_override("A") is None
    assert app_settings.get_expert_override("B")["values"]["ctx"] == 131072


def test_expert_override_rejects_invalid_snapshot(_isolated_settings) -> None:
    """A snapshot missing mode/values must never land on disk."""
    import app_settings

    app_settings.set_expert_override("Bad", {"mode": "manual"})  # no 'values'
    app_settings.set_expert_override("Bad2", {"values": {}})  # no 'mode'
    app_settings.set_expert_override("Bad3", "not-a-dict")  # wrong type
    assert app_settings.get_expert_override("Bad") is None
    assert app_settings.get_expert_override("Bad2") is None
    assert app_settings.get_expert_override("Bad3") is None


def test_expert_override_survives_corrupt_blob(_isolated_settings) -> None:
    """A structurally invalid entry on disk is treated as missing, not a crash."""
    import app_settings

    # Write a blob whose entry for 'X' is the wrong shape.
    _isolated_settings.write_text(
        '{"expert_overrides": {"X": "garbage"}}', encoding="utf-8"
    )
    assert app_settings.get_expert_override("X") is None
    # A valid entry next to it still loads.
    app_settings.set_expert_override("Y", {"mode": "auto", "values": {"ctx": 8192}})
    assert app_settings.get_expert_override("Y")["values"]["ctx"] == 8192


def test_expert_override_empty_name_noop(_isolated_settings) -> None:
    import app_settings

    app_settings.set_expert_override("", {"mode": "auto", "values": {}})
    app_settings.clear_expert_override("")
    assert app_settings.get_expert_override("") is None


# ---- Expert value ↔ config translation (shared by live panel + disk) -------


def test_reasoning_flags_from_values_mapping() -> None:
    """auto→none, off→--reasoning off, <level>→kwargs, budget≥0→--reasoning-budget."""
    from qt_launcher import _reasoning_flags_from_values

    assert _reasoning_flags_from_values("auto", -1) == []
    assert _reasoning_flags_from_values("off", -1) == ["--reasoning", "off"]
    flags = _reasoning_flags_from_values("high", -1)
    assert flags == ["--chat-template-kwargs", '{"reasoning_effort":"high"}']
    # Think budget emits the b9625+ flag name.
    assert _reasoning_flags_from_values("auto", 2048) == [
        "--reasoning-budget",
        "2048",
    ]
    # off + budget compose.
    assert _reasoning_flags_from_values("off", 0) == [
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    # Defensive: junk budget falls back to -1 (no flag).
    assert _reasoning_flags_from_values("auto", "oops") == []


def test_expert_extras_assembles_all_flag_sources() -> None:
    """jinja + verbose + reasoning dropdown + free text all land in extras."""
    from qt_launcher import _expert_extras_from_values

    vals = {
        "jinja": True,
        "verbose": True,
        "reasoning": "medium",
        "think_budget": 512,
        "reasoning_preserve": True,
        "extras": "--some-flag value",
    }
    out = _expert_extras_from_values(vals)
    assert out[0] == "--jinja"
    assert out[1] == "--verbose"
    assert "--chat-template-kwargs" in out
    assert '{"reasoning_effort":"medium"}' in out
    assert "--reasoning-budget" in out and "512" in out
    assert "--reasoning-preserve" in out
    assert "--some-flag" in out and "value" in out

    vals["reasoning_preserve"] = False
    assert "--reasoning-preserve" not in _expert_extras_from_values(vals)


def _base_cfg(tmp_path) -> TunedConfig:
    """A real TunedConfig to feed the value helpers."""
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    sysinfo = _fake_system()
    profile = match_profile(
        model.name, load_profiles(SETTINGS_DIR), getattr(model, "architecture", "")
    )
    cfg = compute_config(model=model, system=sysinfo, profile=profile)
    assert cfg is not None
    return cfg


def test_apply_expert_values_only_overlays_noncascading(tmp_path) -> None:
    """Cascading fields (ctx/KV/ngl/n_cpu_moe/rope) must be left to compute_config."""
    from qt_launcher import apply_expert_values

    base = _base_cfg(tmp_path)
    orig_ctx = base.ctx
    orig_k = base.cache_k
    orig_ngl = base.ngl

    vals = {
        "ctx": 999999,  # must be IGNORED by apply_expert_values
        "cache_k": "f16",  # must be IGNORED
        "ngl": 0,  # must be IGNORED
        "threads": 42,  # must be APPLIED
        "ubatch": 1234,  # must be APPLIED
        "flash_attn": True,
        "load_mode": "mlock",
        "metrics_enabled": False,
        "slots_api_enabled": True,
        "numa": "isolate",
        "temperature": 0.33,
        "top_k": 7,
        "reasoning": "off",
        "think_budget": 100,
        "reasoning_preserve": True,
        "extras": "--jinja",
        "draft_n_max": 5,  # must be APPLIED
    }
    out = apply_expert_values(base, vals)
    # Cascading untouched
    assert out.ctx == orig_ctx
    assert out.cache_k == orig_k
    assert out.ngl == orig_ngl
    # Non-cascading applied
    assert out.threads == 42
    assert out.ubatch == 1234
    assert out.flash_attn is True
    assert out.load_mode == "mlock"
    assert out.mlock is True and out.no_mmap is True
    assert out.metrics_enabled is False
    assert out.slots_api_enabled is True
    assert out.numa == "isolate"
    assert out.sampling["temperature"] == 0.33
    assert out.sampling["top_k"] == 7
    assert "--jinja" in out.extra_cli_flags
    assert "--reasoning" in out.extra_cli_flags  # from reasoning=off
    assert "--reasoning-preserve" in out.extra_cli_flags
    assert out.draft_n_max == 5


def test_expert_load_mode_migrates_legacy_checkbox_snapshots(tmp_path) -> None:
    from qt_launcher import apply_expert_values

    base = _base_cfg(tmp_path)
    cfg = apply_expert_values(base, {"mlock": True, "no_mmap": False})
    assert cfg.load_mode == "mmap+mlock"
    cfg = apply_expert_values(base, {"mlock": True, "no_mmap": True})
    assert cfg.load_mode == "mlock"
    cfg = apply_expert_values(base, {"mlock": False, "no_mmap": True})
    assert cfg.load_mode == "none"


def test_expert_cfg_from_values_is_frozen_manual(tmp_path) -> None:
    """expert_cfg_from_values sets EVERY field (incl. cascading) from the snapshot."""
    from qt_launcher import expert_cfg_from_values

    base = _base_cfg(tmp_path)
    vals = {
        "ctx": 65536,
        "cache_k": "q5_0",
        "cache_v": "q4_0",
        "ngl": 17,
        "n_cpu_moe": 3,
        "threads": 5,
        "ubatch": 256,
        "rope_scaling": True,
        "rope_factor": 2.0,
        "flash_attn": False,
        "metrics_enabled": False,
        "slots_api_enabled": True,
        "numa": "off",
        "temperature": 0.8,
        "top_k": 40,
        "reasoning": "auto",
        "think_budget": -1,
        "extras": "",
    }
    cfg = expert_cfg_from_values(base, vals)
    assert cfg.ctx == 65536
    assert cfg.cache_k == "q5_0"
    assert cfg.cache_v == "q4_0"
    assert cfg.ngl == 17
    assert cfg.n_cpu_moe == 3
    assert cfg.threads == 5
    assert cfg.ubatch == 256
    assert cfg.rope_scaling is True
    assert cfg.rope_scale_factor == 2.0
    assert cfg.flash_attn is False
    assert cfg.metrics_enabled is False
    assert cfg.slots_api_enabled is True
    assert cfg.numa is None  # 'off' → None
    assert cfg.sampling["temperature"] == 0.8
    assert cfg.kv_quant_strategy == "manual"
    # Unmodelled fields inherited from base (the build must stay complete).
    assert cfg.env_overrides == base.env_overrides


def test_expert_cfg_from_values_tolerates_partial_snapshot(tmp_path) -> None:
    """A snapshot missing some keys falls back to the base value (defensive)."""
    from qt_launcher import expert_cfg_from_values

    base = _base_cfg(tmp_path)
    # Only ctx provided — everything else defaults from base.
    cfg = expert_cfg_from_values(base, {"ctx": 2048})
    assert cfg.ctx == 2048
    assert cfg.cache_k == base.cache_k
    assert cfg.threads == base.threads


def test_manual_config_helper_matches_disk_restore(tmp_path) -> None:
    """Round-trip: live-manual build and disk-restore build must agree.

    This is the invariant that keeps the on-screen panel and the
    no-panel launch path from drifting apart: both translate a values
    dict via the same helpers, so a model launches with exactly what the
    user last saw.
    """
    from qt_launcher import expert_cfg_from_values

    base = _base_cfg(tmp_path)
    vals = {
        "ctx": 32768,
        "cache_k": "q8_0",
        "cache_v": "q8_0",
        "ngl": 999,
        "n_cpu_moe": 0,
        "threads": 12,
        "batch_threads": 12,
        "batch": 2048,
        "ubatch": 512,
        "flash_attn": True,
        "mlock": False,
        "no_mmap": False,
        "jinja": True,
        "verbose": False,
        "metrics_enabled": True,
        "slots_api_enabled": True,
        "numa": "off",
        "rope_scaling": False,
        "rope_factor": 1.0,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
        "presence_penalty": 0.0,
        "reasoning": "high",
        "think_budget": -1,
        "extras": "",
    }
    cfg = expert_cfg_from_values(base, vals)
    # Re-applying the same snapshot must be idempotent.
    cfg2 = expert_cfg_from_values(base, vals)
    assert cfg.ctx == cfg2.ctx
    assert cfg.extra_cli_flags == cfg2.extra_cli_flags
    assert "--chat-template-kwargs" in cfg.extra_cli_flags
    assert "--jinja" in cfg.extra_cli_flags


# ---------------------------------------------------------------------------
# Auto-mode context pin round-trip (the low-VRAM "stay in Auto, remember
# my context" use case). A user who keeps the Expert panel in Auto mode and
# only nudges the context slider expects that value to be restored next time
# AND applied at launch — without having to switch to Manual.


def _auto_override_for_ctx(pinned_ctx: int) -> dict:
    """Build the exact snapshot ExpertPanel._make_snapshot() emits when the
    user pins a context in Auto mode (mode=auto, pins={'user_ctx': N})."""
    return {
        "mode": "auto",
        "pins": {"user_ctx": pinned_ctx},
        "values": {"ctx": pinned_ctx, "threads": 8, "temperature": 0.7},
        "saved_at": "2026-06-30T12:00:00",
    }


def test_auto_ctx_pin_re_cascades_through_compute_config(tmp_path) -> None:
    """A saved Auto-mode ctx pin must re-flow through compute_config at launch.

    This is the heart of the _effective_config() Auto branch: the pin is
    NOT a frozen value — it is re-cascaded so it adapts to the current VRAM
    / checkbox state. We prove it by showing compute_config(user_ctx=N)
    produces the pinned context (when it fits), and that this is exactly
    what the saved override drives.
    """
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    sysinfo = _fake_system()
    profile = match_profile(
        model.name, load_profiles(SETTINGS_DIR), getattr(model, "architecture", "")
    )

    base = compute_config(model=model, system=sysinfo, profile=profile)
    assert base is not None

    # Pin a SMALLER context than the auto default to avoid VRAM clamping —
    # compute_config will always honour a pin it can fit.
    pinned = max(2048, base.ctx // 2)
    pinned_cfg = compute_config(
        model=model, system=sysinfo, profile=profile, user_ctx=pinned
    )
    assert pinned_cfg is not None
    assert pinned_cfg.ctx == pinned, (
        f"user_ctx pin {pinned} not honoured; got ctx={pinned_cfg.ctx}"
    )


# NOTE: The full ExpertPanel round-trip (open panel, edit context, reopen)
# is exercised by a headless script in development; it is NOT replicated
# here because instantiating QWidget + QApplication inside pytest hangs on
# teardown (the suite deliberately avoids live Qt objects). The two pure-
# Python tests below cover the SAME transformation the panel delegates to:
# compute_config(user_ctx=N) for the cascade, and apply_expert_values for the
# non-cascading overlay — i.e. both the Auto-restore and the launch paths.


def test_auto_ctx_pin_applied_at_launch_without_panel_open(
    tmp_path, monkeypatch
) -> None:
    """A saved Auto ctx pin applies at launch even when the Expert panel is
    closed — via the _effective_config() path.

    We exercise the exact transformation _effective_config performs for an
    Auto-mode override: re-cascade the base config through compute_config
    with the saved pins, then overlay the saved non-cascading values. The
    result must carry the pinned context.
    """
    import app_settings
    from qt_launcher import apply_expert_values

    target = tmp_path / "autotuner_settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: target)

    base = _base_cfg(tmp_path)
    pinned = max(2048, base.ctx // 2)

    # Persist an Auto-mode override pinning the context (what the panel saved).
    app_settings.set_expert_override("MyModel", _auto_override_for_ctx(pinned))

    # Replay _effective_config's Auto branch by hand (it lives on MainWindow
    # and reads live checkboxes; here we test the pure cfg transformation it
    # delegates to, using the real compute_config + apply_expert_values).
    override = app_settings.get_expert_override("MyModel")
    pins = {k: v for k, v in (override.get("pins") or {}).items() if v is not None}
    assert pins == {"user_ctx": pinned}

    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    sysinfo = _fake_system()
    profile = match_profile(
        model.name, load_profiles(SETTINGS_DIR), getattr(model, "architecture", "")
    )
    cascaded = compute_config(
        model=model, system=sysinfo, profile=profile, user_ctx=pins["user_ctx"]
    )
    assert cascaded is not None
    cascaded = apply_expert_values(cascaded, override.get("values") or {})
    assert cascaded.ctx == pinned, (
        f"launch config lost the pinned context; got {cascaded.ctx}"
    )


# ---------------------------------------------------------------------------
# Cross-OS GPU identity: Linux short names, Mesa RADV suffixes, priorities
# ---------------------------------------------------------------------------
# Regressions for the Ubuntu 26.04 report "AUTO prioritises the 16 GB
# RX 9070 XT instead of the 32 GB R9700":
#   1. gpu_overrides priorities are keyed by the WINDOWS driver names
#      ("AMD Radeon AI PRO R9700"); Linux lspci/DRM calls the same card
#      "Radeon AI PRO R9700" and Mesa appends "(RADV NAVI48)". The exact
#      dict lookup silently dropped every priority after an OS switch.
#   2. Mesa can report a GENERIC name for a very new card, breaking the
#      name-based hip_index resolution; VRAM-total matching + elimination
#      must take over so the placement never falls back to positional
#      (DRM-order) indices.


_LIST_DEVICES_RADV = """Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (RADV NAVI48) (16304 MiB, 15416 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV NAVI48) (32624 MiB, 31704 MiB free)
"""

_LIST_DEVICES_RADV_GENERIC = """Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (RADV NAVI48) (16304 MiB, 15416 MiB free)
  Vulkan1: AMD Radeon Graphics (RADV NAVI48) (32624 MiB, 31704 MiB free)
"""


def _linux_gpu_pair():
    return [
        GPUInfo(
            index=0,
            name="Radeon RX 9070 XT",  # Linux DRM/lspci short name
            vendor="amd",
            total_vram_mb=16304,
            free_vram_mb=15400,
            pci_device_id=0x7550,
        ),
        GPUInfo(
            index=1,
            name="Radeon AI PRO R9700",
            vendor="amd",
            total_vram_mb=32624,
            free_vram_mb=31700,
            pci_device_id=0x7551,
        ),
    ]


def test_list_devices_regex_survives_radv_suffix(monkeypatch) -> None:
    """Mesa's '(RADV NAVI48)' suffix must not truncate names or break VRAM."""
    import hardware

    monkeypatch.setattr(hardware, "_run", lambda *a, **k: _LIST_DEVICES_RADV)
    devices = hardware._detect_llama_devices("llama-server")
    assert [(d[0], d[2], d[3]) for d in devices] == [
        (0, 16304, 15416),
        (1, 32624, 31704),
    ]
    assert devices[0][1] == "amd radeon rx 9070 xt (radv navi48)"
    assert devices[1][1] == "amd radeon ai pro r9700 (radv navi48)"


def test_hip_index_resolves_by_name_with_radv_suffix(monkeypatch) -> None:
    """Linux short names must map onto RADV device names (no vulkaninfo)."""
    import hardware

    def fake_run(cmd, *a, **k):
        if "--list-devices" in cmd:
            return _LIST_DEVICES_RADV
        return ""  # vulkaninfo unavailable (fresh Ubuntu w/o vulkan-tools)

    monkeypatch.setattr(hardware, "_run", fake_run)
    gpus = _linux_gpu_pair()
    hardware._assign_hip_indices(gpus, "llama-server")
    assert gpus[0].hip_index == 0  # 9070 XT is Vulkan0
    assert gpus[1].hip_index == 1  # R9700 is Vulkan1


def test_hip_index_resolves_generic_radv_name_by_vram(monkeypatch) -> None:
    """When Mesa reports a GENERIC name for the R9700, the unique 32 GB VRAM
    total (and finally elimination) must still resolve the correct index —
    hip_index=None here caused the positional fallback that loaded models
    onto the wrong card on Ubuntu."""
    import hardware

    def fake_run(cmd, *a, **k):
        if "--list-devices" in cmd:
            return _LIST_DEVICES_RADV_GENERIC
        return ""

    monkeypatch.setattr(hardware, "_run", fake_run)
    gpus = _linux_gpu_pair()
    hardware._assign_hip_indices(gpus, "llama-server")
    assert gpus[0].hip_index == 0
    assert gpus[1].hip_index == 1, (
        "generic RADV name must resolve via VRAM-total match/elimination"
    )


def test_duplicate_radv_names_never_share_a_device(monkeypatch) -> None:
    """Two cards with IDENTICAL Mesa names must not both resolve to device 0
    (and their VRAM must not collapse into one dict entry)."""
    import hardware

    out = (
        "  Vulkan0: AMD Radeon Graphics (RADV NAVI48) (16304 MiB, 15416 MiB free)\n"
        "  Vulkan1: AMD Radeon Graphics (RADV NAVI48) (32624 MiB, 31704 MiB free)\n"
    )

    def fake_run(cmd, *a, **k):
        if "--list-devices" in cmd:
            return out
        return ""

    monkeypatch.setattr(hardware, "_run", fake_run)
    gpus = _linux_gpu_pair()
    hardware._assign_hip_indices(gpus, "llama-server")
    assert {gpus[0].hip_index, gpus[1].hip_index} == {0, 1}
    assert gpus[0].hip_index == 0  # 16 GB card ↔ 16304 MiB device
    assert gpus[1].hip_index == 1  # 32 GB card ↔ 32624 MiB device


def test_gpu_vram_mapping_prefers_names_then_totals(monkeypatch) -> None:
    """_map_gpus_to_llama_devices: name first, unique-VRAM second, and every
    device claimed at most once."""
    import hardware

    devices = [
        (0, "amd radeon rx 9070 xt (radv navi48)", 16304, 15416),
        (1, "amd radeon graphics (radv navi48)", 32624, 31704),
    ]
    gpus = _linux_gpu_pair()
    mapped = hardware._map_gpus_to_llama_devices(gpus, devices)
    assert mapped[id(gpus[0])][0] == 0
    assert mapped[id(gpus[1])][0] == 1


def test_gpu_priorities_survive_os_switch(tmp_path) -> None:
    """Windows-keyed priorities ("AMD Radeon …") must apply to the Linux
    short names AND the Mesa RADV names, so AUTO keeps preferring the R9700
    after booting into Ubuntu. Regression for the 'AUTO picks the 16 GB
    9070 XT on Ubuntu' report."""
    prio_windows_keys = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }

    for names in (
        ("Radeon RX 9070 XT", "Radeon AI PRO R9700"),  # Linux lspci/DRM
        (
            "AMD Radeon RX 9070 XT (RADV NAVI48)",  # Mesa RADV
            "AMD Radeon AI PRO R9700 (RADV NAVI48)",
        ),
        ("AMD Radeon RX 9070 XT", "AMD Radeon AI PRO R9700"),  # Windows WMI
    ):
        sysinfo = SystemInfo(
            os_name="Linux test",
            cpu_name="Test CPU",
            cpu_cores_physical=16,
            cpu_cores_logical=32,
            total_ram_gb=48,
            free_ram_gb=40,
            gpus=[
                GPUInfo(
                    index=0,
                    name=names[0],
                    vendor="amd",
                    total_vram_mb=16304,
                    free_vram_mb=15400,
                    hip_index=0,
                ),
                GPUInfo(
                    index=1,
                    name=names[1],
                    vendor="amd",
                    total_vram_mb=32624,
                    free_vram_mb=31700,
                    hip_index=1,
                ),
            ],
        )
        model = _fake_model(tmp_path, f"Dense-22B-Q6_{names[0][:3]}", size_gb=22.5)
        profile = match_profile(
            model.name, load_profiles(SETTINGS_DIR), getattr(model, "architecture", "")
        )
        cfg = compute_config(
            model=model,
            system=sysinfo,
            profile=profile,
            gpu_priorities=prio_windows_keys,
        )
        # The whole model fits the R9700 → exclusive pin on ITS hip index (1).
        assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "1", (
            f"R9700 must stay the primary for names={names}: {cfg.env_overrides}"
        )
        assert cfg.main_gpu == 0  # sole visible device after remapping


def test_forced_gpu_token_matches_both_os_name_styles(tmp_path) -> None:
    """The GUI pin token ('9070' / 'R9700') must select the same physical
    card under Windows AND Linux name styles."""
    for names in (
        ("Radeon RX 9070 XT", "Radeon AI PRO R9700"),
        ("AMD Radeon RX 9070 XT", "AMD Radeon AI PRO R9700"),
    ):
        sysinfo = SystemInfo(
            os_name="test",
            cpu_name="Test CPU",
            cpu_cores_physical=16,
            cpu_cores_logical=32,
            total_ram_gb=48,
            free_ram_gb=40,
            gpus=[
                GPUInfo(
                    index=0,
                    name=names[0],
                    vendor="amd",
                    total_vram_mb=16304,
                    free_vram_mb=15400,
                    hip_index=0,
                ),
                GPUInfo(
                    index=1,
                    name=names[1],
                    vendor="amd",
                    total_vram_mb=32624,
                    free_vram_mb=31700,
                    hip_index=1,
                ),
            ],
        )
        model = _fake_model(tmp_path, f"Pin-9B-Q8_{names[0][:3]}", size_gb=9.0)
        profile = match_profile(
            model.name, load_profiles(SETTINGS_DIR), getattr(model, "architecture", "")
        )
        cfg = compute_config(
            model=model, system=sysinfo, profile=profile, force_gpu="9070"
        )
        assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "0", (
            f"pin '9070' must select the 9070 XT (Vulkan0) for names={names}"
        )


def test_match_gpu_by_token_cross_os_styles() -> None:
    """The shared pin matcher must resolve short tokens AND a full driver
    string from the other OS's name style to the same physical card — the
    one-directional substring it replaces silently dropped the pin after an
    OS switch (launch fell back to auto-placement)."""
    from tuner import match_gpu_by_token

    gpus = [
        GPUInfo(
            index=0,
            name="AMD Radeon RX 9070 XT",
            vendor="amd",
            total_vram_mb=16304,
            free_vram_mb=15400,
            hip_index=0,
        ),
        GPUInfo(
            index=1,
            name="Radeon AI PRO R9700 (RADV NAVI48)",
            vendor="amd",
            total_vram_mb=32624,
            free_vram_mb=31700,
            hip_index=1,
        ),
    ]
    # Short tokens (what the GUI dropdown persists).
    assert match_gpu_by_token("9070", gpus) is gpus[0]
    assert match_gpu_by_token("R9700", gpus) is gpus[1]
    # Full Windows-WMI name against the Mesa/RADV style: neither is a
    # substring of the other → the shared model-number token must hit.
    assert match_gpu_by_token("AMD Radeon AI PRO R9700", gpus) is gpus[1]
    # Unknown / empty pins mean auto behaviour.
    assert match_gpu_by_token("4090", gpus) is None
    assert match_gpu_by_token("", gpus) is None
    assert match_gpu_by_token(None, gpus) is None


def test_forced_gpu_full_other_os_name_still_pins(tmp_path) -> None:
    """force_gpu persisted as a full Windows driver name must still pin the
    right card when the detected names use the Linux/Mesa style."""
    sysinfo = SystemInfo(
        os_name="test",
        cpu_name="Test CPU",
        cpu_cores_physical=16,
        cpu_cores_logical=32,
        total_ram_gb=48,
        free_ram_gb=40,
        gpus=[
            GPUInfo(
                index=0,
                name="Radeon RX 9070 XT (RADV NAVI48)",
                vendor="amd",
                total_vram_mb=16304,
                free_vram_mb=15400,
                hip_index=0,
            ),
            GPUInfo(
                index=1,
                name="Radeon AI PRO R9700 (RADV NAVI48)",
                vendor="amd",
                total_vram_mb=32624,
                free_vram_mb=31700,
                hip_index=1,
            ),
        ],
    )
    model = _fake_model(tmp_path, "Pin-9B-Q8_crossos", size_gb=9.0)
    profile = match_profile(
        model.name, load_profiles(SETTINGS_DIR), getattr(model, "architecture", "")
    )
    cfg = compute_config(
        model=model,
        system=sysinfo,
        profile=profile,
        force_gpu="AMD Radeon AI PRO R9700",
    )
    assert cfg.env_overrides.get("GGML_VK_VISIBLE_DEVICES") == "1", (
        f"full Windows name must pin the R9700 (Vulkan1): {cfg.env_overrides}"
    )


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell-script fake binary")
def test_gemma_draft_ik_fallback_only_for_pre_spec_type_builds(tmp_path) -> None:
    """Gemma-4 + external drafter must use the SELECTED fork when its binary
    advertises --spec-type (mainline b9190+/PR #23398); only a build without
    the flag still needs the legacy ik_llama.cpp redirect. The old
    unconditional redirect broke Gemma drafting on current mainline."""
    from tuner import gemma_draft_needs_ik_fork

    modern = tmp_path / "llama-server"
    modern.write_text(
        "#!/bin/sh\ncat <<'EOF'\n-m, --model\n-md, --model-draft\n--spec-type\nEOF\n",
        encoding="utf-8",
    )
    modern.chmod(0o755)

    legacy = tmp_path / "legacy" / "llama-server"
    legacy.parent.mkdir()
    legacy.write_text(
        "#!/bin/sh\ncat <<'EOF'\n-m, --model\n-md, --model-draft\nEOF\n",
        encoding="utf-8",
    )
    legacy.chmod(0o755)

    # Modern build (advertises --spec-type) → stay on the selected fork.
    assert not gemma_draft_needs_ik_fork("gemma-4-12B-it-Q8_0", True, str(modern))
    # Old build without --spec-type → legacy ik_llama.cpp fallback.
    assert gemma_draft_needs_ik_fork("gemma-4-12B-it-Q8_0", True, str(legacy))
    # No draft / non-Gemma model → never redirect.
    assert not gemma_draft_needs_ik_fork("gemma-4-12B-it-Q8_0", False, str(legacy))
    assert not gemma_draft_needs_ik_fork("Qwen3.6-27B-Q8_0", True, str(legacy))
    # Unprobeable binary → keep the selected fork (no blind redirect).
    assert not gemma_draft_needs_ik_fork(
        "gemma-4-12B-it-Q8_0", True, str(tmp_path / "missing" / "llama-server")
    )


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell-script fake binary")
def test_vision_no_longer_suppresses_external_draft_on_spec_type_builds(
    tmp_path,
) -> None:
    """-md + --mmproj coexist on --spec-type builds (verified against b9940
    server sources, unchanged through b9963). The old unconditional 'vision wins' skip silently
    disabled the Gemma-4 drafter whenever Vision was on; only builds without
    --spec-type may still drop Path A."""
    modern = tmp_path / "llama-server"
    modern.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        "-m, --model\n"
        "-md, --model-draft\n"
        "--mmproj\n"
        "--spec-type\n"
        "EOF\n",
        encoding="utf-8",
    )
    modern.chmod(0o755)

    legacy = tmp_path / "legacy" / "llama-server"
    legacy.parent.mkdir()
    legacy.write_text(
        "#!/bin/sh\ncat <<'EOF'\n-m, --model\n-md, --model-draft\n--mmproj\nEOF\n",
        encoding="utf-8",
    )
    legacy.chmod(0o755)

    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "gemma-4-12B-it-Q8_0", size_gb=13.0)
    mmproj = tmp_path / "mmproj-gemma-4-12B-F16.gguf"
    _write_minimal_gguf(mmproj)
    model.mmproj = mmproj
    draft = _fake_model(tmp_path, "mtp-gemma-4-12B-it-qat-UD", size_gb=0.6)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)

    # Modern build: vision AND external drafter together.
    cmd = build_command(
        model, cfg, profile, draft_model=draft, server_binary=str(modern)
    )
    assert "-md" in cmd and "--mmproj" in cmd

    # Legacy build (no --spec-type): conservative skip stays — -md dropped.
    cmd = build_command(
        model, cfg, profile, draft_model=draft, server_binary=str(legacy)
    )
    assert "-md" not in cmd and "--mmproj" in cmd

    # Vision off: drafter active regardless of the build.
    model_novis = _fake_model(tmp_path, "gemma-4-12B-it-Q8_0-novis", size_gb=13.0)
    cmd = build_command(
        model_novis, cfg, profile, draft_model=draft, server_binary=str(legacy)
    )
    assert "-md" in cmd


# ---------------------------------------------------------------------------
# GUI updater


def test_update_worker_archive_overlay_preserves_settings(
    tmp_path, monkeypatch
) -> None:
    """Downloaded ZIP installs must update code without clobbering settings."""
    import shutil
    import zipfile

    import qt_launcher

    app = tmp_path / "Auto-Tuner-main"
    app.mkdir()
    (app / "autotuner_settings.json").write_text("USER", encoding="utf-8")
    (app / "requirements.txt").write_text("old", encoding="utf-8")

    source_root = tmp_path / "src" / "Auto-Tuner-main"
    source_root.mkdir(parents=True)
    (source_root / "qt_launcher.py").write_text("NEW", encoding="utf-8")
    (source_root / "requirements.txt").write_text("new", encoding="utf-8")
    (source_root / "autotuner_settings.json").write_text("REMOTE", encoding="utf-8")
    (source_root / "settings").mkdir()
    (source_root / "settings" / "profile.yaml").write_text("profile", encoding="utf-8")

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in source_root.rglob("*"):
            zf.write(path, path.relative_to(tmp_path / "src"))

    shared_data = tmp_path / ".autotuner"
    shared_data.mkdir()
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: app / "autotuner_settings.json",
    )
    monkeypatch.setattr(
        qt_launcher.app_settings, "app_data_dir", lambda: shared_data
    )
    worker = qt_launcher._UpdateWorker(app)
    # Force the release-ZIP path regardless of where pytest's tmpdir lives.
    worker._repo_root = None
    monkeypatch.setattr(
        worker,
        "_github_archive_info",
        lambda: ("main", "abcdef1234567890", "local"),
    )
    monkeypatch.setattr(
        worker,
        "_download_file",
        lambda _url, dest: shutil.copy2(archive, dest),
    )
    pip_calls = []
    monkeypatch.setattr(
        worker,
        "_run",
        lambda cmd, check=True, timeout=600.0, cwd=None: (
            pip_calls.append((cmd, cwd)) or ""
        ),
    )

    finished = []
    worker.finished.connect(lambda ok, msg: finished.append((ok, msg)))
    worker.run()

    assert finished and finished[0][0]
    assert "GitHub archive" in finished[0][1]
    assert f"v{qt_launcher.VERSION}" in finished[0][1]
    assert (app / "qt_launcher.py").read_text(encoding="utf-8") == "NEW"
    assert (app / "settings" / "profile.yaml").read_text(encoding="utf-8") == "profile"
    assert (app / "autotuner_settings.json").read_text(encoding="utf-8") == "USER"
    assert (shared_data / ".autotuner_update.json").exists()
    assert pip_calls, "requirements.txt change should reinstall dependencies"


def test_close_refuses_while_an_update_thread_is_running(monkeypatch) -> None:
    """Closing cannot destroy an updater that is still replacing files."""
    import qt_launcher

    class RunningThread:
        def isRunning(self):  # noqa: N802
            return True

    class CloseEvent:
        ignored = False

        def ignore(self):
            self.ignored = True

    class Window:
        _update_thread = RunningThread()
        _update_worker = object()
        _force_quit = True

    messages = []
    monkeypatch.setattr(
        qt_launcher.QMessageBox,
        "information",
        lambda *_args: messages.append(_args[2]),
    )
    event = CloseEvent()
    window = Window()
    qt_launcher.MainWindow.closeEvent(window, event)

    assert event.ignored
    assert window._force_quit is False
    assert messages and "still updating" in messages[0]


def test_prune_dead_server_finalizes_pending_ocr() -> None:
    import qt_launcher

    class Process:
        def is_running(self):
            return False

        def returncode(self):
            return 9

    record = {"proc": Process(), "ocr_started": False}

    class Window:
        _servers = [record]
        _ocr_server_record = record
        _ocr_worker = None
        failures = []

        def _fail_pending_ocr(self, failed_record, message):
            self.failures.append((failed_record, message))

    window = Window()
    qt_launcher.MainWindow._prune_dead_servers(window)
    assert window._servers == []
    assert window.failures == [
        (record, "llama-server exited while loading the OCR model (code 9).")
    ]


def test_update_reference_cleanup_only_clears_the_finished_thread() -> None:
    import qt_launcher

    class Window:
        def __init__(self):
            self._update_thread = object()
            self._update_worker = object()

    window = Window()
    other_thread = object()
    qt_launcher.MainWindow._clear_update_references(window, other_thread)
    assert window._update_thread is not None and window._update_worker is not None
    qt_launcher.MainWindow._clear_update_references(window, window._update_thread)
    assert window._update_thread is None and window._update_worker is None


def test_git_update_rejects_ahead_only_branch(tmp_path, monkeypatch) -> None:
    """An ahead-only checkout must not be described as matching upstream."""
    import qt_launcher

    worker = qt_launcher._UpdateWorker(tmp_path)
    worker._repo_root = tmp_path
    worker._git_bin = "git"
    monkeypatch.setattr(worker, "_launcher_is_tracked_by_git", lambda: True)
    monkeypatch.setattr(worker, "_backup_settings", lambda: {})

    def run_git(*args, **_kwargs):
        values = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "--abbrev-ref", "HEAD"): "main",
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{u}",
            ): "origin/main",
            ("status", "--porcelain", "--untracked-files=no"): "",
            ("ls-files", "--error-unmatch", "autotuner_settings.json"): "",
            ("fetch", "--prune", "origin"): "",
            ("rev-list", "--left-right", "--count", "HEAD...origin/main"): "1 0",
        }
        return values[args]

    monkeypatch.setattr(worker, "_run_git", run_git)
    finished = []
    worker.finished.connect(lambda ok, msg: finished.append((ok, msg)))
    worker.run()

    assert finished == [
        (
            False,
            "Local branch has 1 commit(s) not on origin/main; refusing to auto-merge.",
        )
    ]


def test_qt_frozen_smoke_entrypoint_returns_without_gui() -> None:
    import qt_launcher

    assert (
        qt_launcher.main(["--smoke-test", "--settings-path", str(SETTINGS_DIR)]) is None
    )


def test_release_workflow_publishes_arm64_macos_asset() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "runner: macos-15" in workflow
    assert "artifact: AutoTuner-macOS-arm64.zip" in workflow
    assert "assets/AutoTuner-macOS-arm64.zip" in workflow
    assert 'expected = "${{ matrix.arch }}"' in workflow
    assert "assets/AutoTuner-macOS.zip" not in workflow


def test_binary_update_worker_picks_macos_asset_not_linux(monkeypatch) -> None:
    """A frozen macOS build must not download the Linux release asset."""
    import qt_launcher

    worker = qt_launcher._BinaryUpdateWorker()
    assets = [
        {"name": "AutoTuner-Windows-x64.zip"},
        {"name": "AutoTuner-Linux-x64.zip"},
        {"name": "AutoTuner-macOS-arm64.zip"},
        {"name": "AutoTuner-macOS-x64.zip"},
    ]
    monkeypatch.setattr(qt_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(qt_launcher.platform, "machine", lambda: "arm64")

    assert worker._pick_asset(assets) == {"name": "AutoTuner-macOS-arm64.zip"}


def test_binary_update_worker_macos_missing_asset_returns_none(monkeypatch) -> None:
    import qt_launcher

    worker = qt_launcher._BinaryUpdateWorker()
    monkeypatch.setattr(qt_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(qt_launcher.platform, "machine", lambda: "arm64")

    assert worker._pick_asset([{"name": "AutoTuner-Linux-x64.zip"}]) is None
    assert worker._pick_asset([{"name": "AutoTuner-macOS-x64.zip"}]) is None
