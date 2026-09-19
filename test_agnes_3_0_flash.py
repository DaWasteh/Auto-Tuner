"""Agnes-3.0-Flash Preview (Agnes AI, 33B hybrid, folded ``qwen35`` GGUFs):
the filename-gated profile, and the ROCmFPX tensor-type gate that keeps
kingjones777's ``MTP-ROCmFP4`` / ``ROCmFPX`` files (ggml types 100+, file
types 100..124) off mainline runtimes before llama-server is even started."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import scanner
import tuner
from settings_loader import load_profiles, match_profile
from test_kv_policy import _model
from test_smoke import _fake_model_md, _fake_system, _spec_tokens

ROOT = Path(__file__).resolve().parent
PROFILE = "agnes-3_0-flash.yaml"

STRIX_LEAN = "Agnes-3.0-Flash-Preview-MTP-imatrix-Q4_0-ROCmFP4-STRIX_LEAN"
COHERENT = "Agnes-3.0-Flash-Preview-MTP-Q4_0-ROCmFP4-COHERENT"
Q8_ROCMFPX = "Agnes-3.0-Flash-Preview-MTP-Q8_0-ROCmFPX"
Q8_AGENT = "Agnes-3.0-Flash-Preview-MTP-Q8_0-ROCmFPX-AGENT"


def _profiles():
    return load_profiles(ROOT / "settings")


def _agnes(tmp_path, name=STRIX_LEAN, file_type=106, ggml_types=(0, 1, 8, 100)):
    """The header as the real STRIX_LEAN GGUF carries it (read on 2026-09-19
    from the first 32 MiB of the HF file with scanner._read_gguf_metadata_uncached:
    qwen35, 73 blocks incl. the nextn block, file_type 106)."""
    metadata = {
        "general.architecture": "qwen35",
        "general.name": "Agnes-3.0-Flash-Preview",
        "general.file_type": file_type,
        "general.size_label": "33B",
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 20,
        "general.sampling.top_p": 0.95,
        "qwen35.block_count": 73,
        "qwen35.nextn_predict_layers": 1,
        "qwen35.context_length": 262144,
        "qwen35.embedding_length": 5120,
        "qwen35.feed_forward_length": 19456,
        "qwen35.full_attention_interval": 4,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.inner_size": 6144,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.time_step_rank": 48,
        "__mtp_scan__": "found",
        "__max_block_index__": 72,
        "__tensor_scan_complete__": bool(ggml_types),
        "__ggml_types__": sorted(ggml_types),
    }
    return _fake_model_md(tmp_path, name, 16.82, metadata)


# ---------------------------------------------------------------- profile


@pytest.mark.parametrize(
    "name",
    [
        STRIX_LEAN,
        COHERENT,
        Q8_ROCMFPX,
        Q8_AGENT,
        "Agnes-3.0-Flash-Q4_K_M",  # 0xKitkat / ngquocvinh folded quants
        "Agnes-3.0-Flash-abliterated-Q4_K_M",
        "Agnes-3.0-Flash-Q4.5-v4-XYZ",  # quimmedes cafe-llama.cpp fork
        "agnes_3.0_flash-preview-bf16",
        "Agnes3.0-Flash-IQ4_XS",
    ],
)
def test_agnes_profile_matches_the_community_filenames(name):
    profiles = _profiles()
    # Filename precedence: the qwen35 architecture must not pull these to the
    # generic Qwen3.5/3.6 profile.
    assert match_profile(name, profiles).source_file == PROFILE
    assert match_profile(name, profiles, "qwen35").source_file == PROFILE


def test_agnes_profile_leaves_other_qwen35_models_alone():
    profiles = _profiles()
    assert (
        match_profile("Qwen3.5-27B-Q4_K_M", profiles, "qwen35").source_file
        == "qwen3_5-3_6.yaml"
    )
    assert (
        match_profile("opaque-requant-Q4_K_M", profiles, "qwen35").source_file
        == "qwen3_5-3_6.yaml"
    )
    assert (
        match_profile("Ornith-1.5-9B-Q8_0", profiles).source_file == "ornith-1_5.yaml"
    )
    assert (
        match_profile("Ternary-Bonsai-2-27B-PTQ1_0", profiles, "qwen35").source_file
        == "bonsai-2-27b.yaml"
    )


def test_agnes_profile_contract():
    profile = match_profile(STRIX_LEAN, _profiles())
    assert profile.max_context == 262144
    assert profile.recommended_kv_quant == "q8_0"
    assert profile.performance_target == "throughput"
    # generation_config.json: temp 1.0 / top_k 20 / top_p 0.95.
    chat = profile.sampling["chat"]
    assert chat["temperature"] == 1.0 and chat["top_k"] == 20
    assert chat["top_p"] == 0.95 and chat["presence_penalty"] == 0.0
    assert profile.sampling["coding"]["temperature"] == 0.6
    # Model-card optimum for the in-file MTP head: n-max 4, p-min 0.
    assert profile.draft_max == 4
    assert profile.draft_p_min == pytest.approx(0.0)
    assert "--jinja" in profile.extra_args
    assert "--reasoning-preserve" in profile.extra_args
    # Filename-gated: qwen35 stays with the generic profile, no fork marker
    # on the profile itself (the ROCmFPX gate is metadata-driven).
    assert profile.arch_fallback == []
    assert profile.required_runtime_markers == []
    assert profile.min_llama_build == 0
    assert profile.runtime_block_reason == ""


def test_every_language_pack_explains_the_agnes_profile():
    packs = sorted((ROOT / "assets/languages").glob("*.json"))
    assert len(packs) == 9
    for pack in packs:
        notes = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"]
        note = notes[PROFILE]
        assert "ROCmFPX" in note and "qwen35" in note, pack.name
        assert "#24185" in note and "ffn_*_par" in note, pack.name


# ------------------------------------------------------------ draft p-min


def test_profile_draft_p_min_honours_an_explicit_zero():
    profiles = _profiles()
    agnes = match_profile(STRIX_LEAN, profiles)
    assert tuner._profile_draft_p_min(agnes) == 0.0
    nemotron = next(p for p in profiles if p.source_file == "nemotron-3_5.yaml")
    assert tuner._profile_draft_p_min(nemotron) == 0.0
    qwen = match_profile("Qwen3.5-27B", profiles)
    assert tuner._profile_draft_p_min(qwen) == 0.75
    for bad in (None, "x", -0.1, 1.5, float("nan")):
        assert tuner._profile_draft_p_min(SimpleNamespace(draft_p_min=bad)) == 0.75
    assert tuner._profile_draft_p_min(SimpleNamespace()) == 0.75


# ------------------------------------------------------------ scanner


def _write_gguf_with_tensors(path: Path, tensors) -> None:
    """A GGUF v3 header with two KV pairs and the given (name, ggml_type)
    tensor infos, no tensor data (the scanner never reads it)."""

    def string(value: str) -> bytes:
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", 2))
        f.write(
            string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
        )
        f.write(
            string("qwen35.block_count") + struct.pack("<I", 4) + struct.pack("<I", 2)
        )
        offset = 0
        for name, ggml_type in tensors:
            f.write(string(name))
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<Q", 32))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", offset))
            offset += 64


def test_scanner_records_the_distinct_ggml_tensor_types(tmp_path):
    path = tmp_path / "Agnes-3.0-Flash-Preview-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf"
    _write_gguf_with_tensors(
        path,
        [
            ("token_embd.weight", 14),  # Q6_K embeddings (LEAN tier)
            ("blk.0.attn_q.weight", 100),  # Q4_0_ROCMFP4
            ("blk.0.attn_norm.weight", 0),  # F32
            ("blk.1.ffn_down.weight", 100),
            ("output.weight", 8),  # Q8_0
        ],
    )
    md = scanner._read_gguf_metadata_uncached(path)
    assert md["general.architecture"] == "qwen35"
    assert md["__tensor_scan_complete__"] is True
    assert md["__ggml_types__"] == [0, 8, 14, 100]
    assert tuner._rocmfpx_fork_types(md) == ([100], None)

    empty = tmp_path / "empty.gguf"
    _write_gguf_with_tensors(empty, [])
    assert scanner._read_gguf_metadata_uncached(empty)["__ggml_types__"] == []


# ---------------------------------------------------------------- gate


@pytest.mark.parametrize(
    "name,file_type,types,packing,tier",
    [
        (
            STRIX_LEAN,
            106,
            (0, 8, 13, 14, 100, 101),  # as read from the real file
            "Q4_0_ROCMFP4 (ggml type 100), Q4_0_ROCMFP4_FAST (ggml type 101)",
            "ROCmFP4 Strix Lean",
        ),
        (
            COHERENT,
            102,
            (0, 14, 100),
            "Q4_0_ROCMFP4 (ggml type 100)",
            "ROCmFP4 Coherent",
        ),
        (Q8_ROCMFPX, 111, (0, 8, 103), "Q8_0_ROCMFPX (ggml type 103)", "ROCmFP8"),
        (Q8_AGENT, 115, (0, 8, 103), "Q8_0_ROCMFPX (ggml type 103)", "ROCmFP8 Agent"),
    ],
)
def test_rocmfpx_gguf_is_refused_without_the_rocmfp_loader(
    tmp_path, monkeypatch, name, file_type, types, packing, tier
):
    model = _agnes(tmp_path, name, file_type, types)
    assert model.has_embedded_mtp
    seen = []

    def markers(binary, required):
        seen.append((binary, tuple(required)))
        return False

    monkeypatch.setattr(tuner, "_runtime_has_required_markers", markers)
    monkeypatch.setattr(
        tuner,
        "probe_binary_build_number",
        lambda _: pytest.fail("the type gate must fire before any build probe"),
    )
    ok, message, detected = tuner.check_model_build(model, "llama-server")
    assert not ok and detected is None
    assert seen == [("llama-server", ("rocmfp",))]
    assert model.name in message and packing in message and tier in message
    # Mainline b11042 (Vulkan and HIP) really answers "tensor
    # 'blk.0.attn_gate.weight' has invalid ggml type 101. should be in [0, 43)"
    # for the STRIX_LEAN file (2026-09-19 live run); every fork type is named.
    fork = "/".join(str(t) for t in types if t >= 100)
    assert f"invalid ggml type {fork}. should be in [0, 43)" in message
    assert "ROCmFPX fork" in message and "#24185" in message
    assert "Q4_K_M / Q8_0" in message


def test_rocmfpx_file_type_alone_refuses_a_shard_without_a_tensor_scan(
    tmp_path, monkeypatch
):
    # Shard 1 of a split GGUF (or an unreadable tensor section) has no
    # __ggml_types__ evidence; general.file_type 106 still identifies it.
    model = _agnes(tmp_path, STRIX_LEAN, 106, ())
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: False)
    ok, message, detected = tuner.check_model_build(model, "llama-server")
    assert not ok and detected is None
    assert "file_type 106" in message and "ROCmFP4 Strix Lean" in message
    assert "invalid ggml type 106" in message


def test_rocmfpx_gguf_loads_on_a_rocmfpx_fork_runtime(tmp_path, monkeypatch):
    model = _agnes(tmp_path)
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: True)
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11100)
    # The in-file MTP head then goes through the ordinary NextN regression
    # check (scalar metadata, fixed build): allowed, build reported.
    assert tuner.check_model_build(model, "rocmfpx/llama-server") == (True, "", 11100)


def test_ordinary_ggufs_never_scan_for_the_rocmfp_marker(tmp_path, monkeypatch):
    def explode(*_):
        raise AssertionError("marker scan must not run for ordinary GGUFs")

    monkeypatch.setattr(tuner, "_runtime_has_required_markers", explode)
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11042)
    assert tuner.check_model_build(_model(tmp_path), "llama-server")[0]
    # 0xKitkat's folded Q4_K_M (file_type 15, no MTP block): mainline-loadable.
    folded = _fake_model_md(
        tmp_path,
        "Agnes-3.0-Flash-Q4_K_M",
        18.9,
        {
            "general.architecture": "qwen35",
            "general.file_type": 15,
            "qwen35.block_count": 72,
            "__mtp_scan__": "absent",
            "__ggml_types__": [0, 1, 12, 14],
        },
    )
    assert tuner.check_model_build(folded, "llama-server") == (True, "", None)
    # Mainline's own newest types (MXFP4 39, NVFP4 40, Q1_0 41, Q2_0 42) and
    # the PrismML range (141..143, handled by the prism gate) are not ROCmFPX.
    assert tuner._rocmfpx_fork_types({"__ggml_types__": [39, 40, 41, 42, 143]}) == (
        [],
        None,
    )
    assert tuner._rocmfpx_fork_types({"general.file_type": 143}) == ([], None)
    assert tuner._rocmfpx_fork_types({"general.file_type": "106"}) == ([], 106)


# ------------------------------------------------------------- command


def test_agnes_command_uses_the_in_file_mtp_head_at_depth_4_and_p_min_0(
    tmp_path, monkeypatch
):
    model = _agnes(tmp_path)
    model.mmproj = tmp_path / "mmproj-Agnes-3.0-Flash-Preview-BF16.gguf"
    profile = match_profile(model.name, _profiles(), "qwen35")
    assert profile.source_file == PROFILE
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: True)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11100)
    monkeypatch.setattr(
        tuner,
        "_probe_supported_flags",
        lambda _: {
            "--jinja",
            "--mmproj",
            "--spec-type",
            "--spec-draft-n-max",
            "--spec-draft-p-min",
            "--spec-draft-ngl",
            "--reasoning-preserve",
        },
    )
    config = tuner.compute_config(
        model, _fake_system(vram_total=32, vram_free=31), profile, user_ctx=32768
    )
    assert config.ctx == 32768
    cmd = tuner.build_command(
        model, config, profile, server_binary="rocmfpx/llama-server"
    )
    assert cmd[0] == "rocmfpx/llama-server"
    assert "--jinja" in cmd and "--mmproj" in cmd and "-md" not in cmd
    assert "draft-mtp" in _spec_tokens(cmd).split(",")
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "4"
    assert cmd[cmd.index("--spec-draft-p-min") + 1] == "0.0"
    assert cmd[cmd.index("--temp") + 1] == "1.0"
    assert cmd[cmd.index("--top-p") + 1] == "0.95"
    assert cmd[cmd.index("--top-k") + 1] == "20"


# ------------------------------------------------------------- recipes


ROCMFPX_PIN = "aed0d5fd9620ee96a10cb4e6b16c18514ea370e1"


def test_rocmfpx_recipes_pin_the_fork_main_commit_with_its_own_build_number():
    build_dir = ROOT / "building llama.cpp"
    for name in ("rocmfpx_vulkan_llama_build.ps1", "rocmfpx_hip_llama_build.ps1"):
        text = (build_dir / name).read_text(encoding="utf-8")
        assert 'RemoteUrl "https://github.com/ROCmFPX/ROCmFPX.git"' in text, name
        assert f'-ExpectedCommit "{ROCMFPX_PIN}"' in text, name
        # `llama-server --version` prints the fork's rev-list count (11544);
        # the mainline merge base is b10766, which the header documents.
        assert '-FixedIdentity "b11544"' in text, name
        assert 'FolderPrefix "fpx_"' in text, name
        assert "b10766" in text, name
        # RDNA4 selector patch (fork PR #20 covers RDNA3 only): both siblings
        # apply it so the source trees stay identical.
        assert "patches\\rocmfpx-rdna4-mmq-fallback.patch" in text, name
        assert "-PatchFiles @(" in text, name
    patch = (build_dir / "patches" / "rocmfpx-rdna4-mmq-fallback.patch").read_text(
        encoding="utf-8"
    )
    assert "ggml/src/ggml-cuda/mmq-config-rdna4.cuh" in patch
    assert '+#include "../../rocmfpx/rocmfpx_mmq_rdna3.cuh"' in patch
    assert "+    return ggml_rocmfpx_mmq_get_config_rdna3(type, J, fallback);" in patch
    common = (build_dir / "windows_llama_build_common.ps1").read_text(encoding="utf-8")
    assert "[string[]]$PatchFiles = @()" in common
    assert "git -C $tmp apply --whitespace=nowarn $patch" in common
    assert "apply --reverse --check --whitespace=nowarn $patch" in common
    hip = (build_dir / "rocmfpx_hip_llama_build.ps1").read_text(encoding="utf-8")
    assert "-Backend HIP" in hip and "-RocmPath $RocmPath" in hip
    chain = (build_dir / "build_all_windows_llama.ps1").read_text(encoding="utf-8")
    assert '"ternary_bonsai", "rocmfpx", "turboquant"' in chain
