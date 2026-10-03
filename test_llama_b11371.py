"""b11371 audit: --spec-draft-sampling, Qwen3.8 Flash Next MTP heads,
single-batch decision/encoder models and the new model profiles."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_smoke import _fake_model_md, _fake_system

ROOT = Path(__file__).resolve().parent


def _profiles():
    return load_profiles(ROOT / "settings")


def _manifest(tag: str) -> dict:
    return json.loads(
        (ROOT / f"docs/llama-{tag}-server-flags.json").read_text(encoding="utf-8")
    )


def test_b11371_manifest_only_adds_spec_draft_sampling():
    current = _manifest("b11371")
    previous = _manifest("b11319")
    assert current["tag"] == "b11371"
    assert current["commit"] == "99b95488c"
    assert set(current["flags"]) - set(previous["flags"]) == {"--spec-draft-sampling"}
    assert set(previous["flags"]) <= set(current["flags"])
    assert len(current["flags"]) == 417
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"]


def test_spec_draft_sampling_is_a_value_flag_and_follows_the_draft_path(monkeypatch):
    assert "--spec-draft-sampling" in tuner._ARG_FLAGS_WITH_VALUES
    old = set(_manifest("b11319")["flags"])
    cmd = [
        "llama-server",
        "-m",
        "model.gguf",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "3",
        "--spec-draft-sampling",
        "probabilistic",
        "-c",
        "4096",
    ]
    # b11319 does not know the option: flag and value are pruned together.
    kept, removed = tuner._filter_command_for_supported_flags(cmd, old)
    assert "--spec-draft-sampling" not in kept and "probabilistic" not in kept
    assert removed == ["--spec-draft-sampling probabilistic"]
    assert tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b11371")["flags"])
    ) == (cmd, [])
    # Removing the MTP path on the NextN-regression builds takes it along.
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10745)
    adapted, notes = tuner._adapt_nextn_regression_for_binary(cmd)
    assert adapted == ["llama-server", "-m", "model.gguf", "-c", "4096"]
    assert notes and "draft-mtp disabled" in notes[0]


def _flash_next(tmp_path, *, embedded_mtp=False):
    metadata = {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 49 if embedded_mtp else 48,
        "__tensor_scan_complete__": True,
        "__root_tensors__": [
            "token_embd.weight",
            "output.weight",
            "output_hc_norm.weight",
            "output_hc_down.weight",
            "output_hc_up.weight",
            "per_layer_token_embd.weight",
        ],
    }
    if embedded_mtp:
        metadata["qwen4exp.nextn_predict_layers"] = 1
        metadata["__mtp_scan__"] = "found"
    return _fake_model_md(tmp_path, "Qwen3.8-Flash-Next-UD-Q2_K_XL", 70, metadata)


def _flash_next_head(tmp_path, name, roots):
    return _fake_model_md(
        tmp_path,
        name,
        2.6,
        {
            "general.architecture": "qwen4exp",
            "qwen4exp.block_count": 49,
            "qwen4exp.nextn_predict_layers": 1,
            "__mtp_scan__": "found",
            "__tensor_scan_complete__": True,
            "__root_tensors__": roots,
        },
    )


@pytest.mark.parametrize("build", [11249, 11329, 11330, 11371, 11400])
def test_separate_flash_next_mtp_head_stays_blocked(tmp_path, monkeypatch, build):
    target = _flash_next(tmp_path)
    head = _flash_next_head(
        tmp_path,
        "mtp-Qwen3.8-Flash-Next-Q4_K_M",
        ["token_embd.weight", "output.weight"],
    )
    assert tuner.is_qwen4exp_mtp_sidecar(head)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    ok, reason, detected = tuner.check_draft_model_build(head, "llama-server", target)
    assert not ok and detected == build
    assert "b11330" in reason and "n-gram" in reason and "embedded" in reason
    if build < 11330:
        # No qwen4exp MTP graph at all before PR #29761.
        assert "#29761" in reason and "GGML_ASSERT" not in reason
    else:
        # b11371 HIP/Vulkan: the head loads, then the draft context aborts.
        assert "GGML_ASSERT(buffer)" in reason and "b11371" in reason


def test_unprobeable_runtime_keeps_the_root_tensor_preflight(tmp_path):
    target = _flash_next(tmp_path)
    head = _flash_next_head(
        tmp_path, "mtp-Qwen3.8-Flash-Next-Q4_K_M", ["token_embd.weight"]
    )
    ok, reason, detected = tuner.check_draft_model_build(head, "server", target)
    assert not ok and detected is None
    assert "output_hc_norm.weight" in reason


def test_flash_next_head_gate_ignores_trunks_and_other_architectures(tmp_path):
    trunk = _flash_next(tmp_path)
    assert not tuner.is_qwen4exp_mtp_sidecar(trunk)
    stale = _flash_next_head(tmp_path, "stale", ["token_embd.weight"])
    stale.metadata["__mtp_scan__"] = "absent"
    assert not tuner.is_qwen4exp_mtp_sidecar(stale)
    mimo = _fake_model_md(
        tmp_path,
        "mtp-MiMo-V2.6",
        1,
        {"general.architecture": "mimo2", "mimo2.nextn_predict_layers": 1},
    )
    assert not tuner.is_qwen4exp_mtp_sidecar(mimo)
    assert not tuner.is_qwen4exp_mtp_sidecar(None)


@pytest.mark.parametrize(
    "build,allowed", [(11319, False), (11329, False), (11330, True)]
)
def test_flash_next_with_embedded_mtp_needs_b11330(
    tmp_path, monkeypatch, build, allowed
):
    model = _flash_next(tmp_path, embedded_mtp=True)
    assert model.has_embedded_mtp
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    ok, reason, detected = tuner.check_model_build(model, "llama-server")
    assert ok is allowed and detected == build
    if not allowed:
        assert "wrong number of tensors" in reason and "b11330" in reason
    if allowed:
        # Loadable, but draft-mtp is withheld with an explanatory warning.
        assert "GGML_ASSERT(buffer)" in reason and "not used" in reason
    # The trunk-only GGUF shipped before the MTP conversion is unaffected.
    assert tuner.check_model_build(_flash_next(tmp_path), "llama-server")[0]


def _laya(tmp_path, **extra):
    metadata = {
        "general.architecture": "modern-bert",
        "modern-bert.block_count": 24,
        "modern-bert.context_length": 8192,
        "modern-bert.embedding_length": 384,
        "modern-bert.attention.head_count": 6,
        "modern-bert.attention.causal": False,
        "modern-bert.decision.type": "laya",
    }
    metadata.update(extra)
    return _fake_model_md(tmp_path, "Laya-Q8_0", 0.42, metadata)


def test_decision_and_encoder_models_get_one_batch_for_the_whole_prompt(tmp_path):
    laya = _laya(tmp_path)
    assert tuner.decision_model_type(laya) == "laya"
    assert tuner.single_batch_prompt_tokens(laya) == tuner.SINGLE_BATCH_PROMPT_TOKENS
    profile = match_profile(laya.name, _profiles(), laya.architecture)
    assert profile.source_file == "decision-systemone.yaml"
    config = tuner.compute_config(laya, _fake_system(), profile, prompt_cache_ram_mib=0)
    assert config.batch == config.ubatch == min(config.ctx, 8192)
    assert config.ubatch > 512

    # A plain embedding encoder has the same single-ubatch requirement.
    encoder = _fake_model_md(
        tmp_path,
        "granite-embedding-311m-multilingual-r2-Q8_0",
        0.3,
        {
            "general.architecture": "modern-bert",
            "modern-bert.block_count": 22,
            "modern-bert.context_length": 32768,
            "modern-bert.attention.causal": False,
        },
    )
    assert tuner.decision_model_type(encoder) == ""
    assert tuner.single_batch_prompt_tokens(encoder) == 8192

    clef = _fake_model_md(
        tmp_path,
        "Clef-Flash-Q4_K_M",
        6,
        {"general.architecture": "clef", "clef.decision.type": "clef"},
    )
    assert tuner.single_batch_prompt_tokens(clef) == 8192

    # Causal decision heads share a prompt prefix across ubatches; ordinary
    # chat models are untouched.
    kev = _fake_model_md(
        tmp_path,
        "Kev-4B-Q4_K_M",
        2.8,
        {"general.architecture": "qwen35", "qwen35.decision.type": "kev"},
    )
    assert tuner.decision_model_type(kev) == "kev"
    assert tuner.single_batch_prompt_tokens(kev) == 0
    chat = _fake_model_md(
        tmp_path, "Qwen3.5-9B-Q4_K_M", 6, {"general.architecture": "qwen35"}
    )
    assert tuner.decision_model_type(chat) == ""
    assert tuner.single_batch_prompt_tokens(chat) == 0


@pytest.mark.parametrize(
    "name,arch,expected,min_build",
    [
        ("OpenJev-Q4_K_M", "gemma3", "decision-systemone.yaml", 11364),
        ("Laya-Q8_0", "modern-bert", "decision-systemone.yaml", 11364),
        ("Julia-1-Q8_0", "modern-bert", "decision-systemone.yaml", 11364),
        ("Kev-4B-Q4_K_M", "qwen35", "decision-systemone.yaml", 11364),
        ("lev-Q8_0", "qwen3", "decision-systemone.yaml", 11364),
        ("Bespoke-Nimble-9B-v3-Q4_K_M", "qwen35", "decision-systemone.yaml", 11364),
        ("Clef-Flash-Q4_K_M", "clef", "clef.yaml", 11371),
        ("Cloudflare_clef-Q4_K_M", "clef", "clef.yaml", 11371),
        ("renamed-decision-head", "clef", "clef.yaml", 11371),
        ("llm-jp-4.1-8b-thinking-Q4_K_M", "llama", "llm-jp-4_1.yaml", 11320),
        ("llm-jp-4.1-32b-a3b-thinking-Q4_K_M", "llama", "llm-jp-4_1.yaml", 11320),
        ("GLM-OCR-Q8_0", "glm4", "glm-ocr.yaml", 0),
        ("dots.ocr-Q8_0", "qwen2", "dots-ocr.yaml", 0),
    ],
)
def test_new_models_have_their_own_profiles(name, arch, expected, min_build):
    profile = match_profile(name, _profiles(), arch)
    assert profile.source_file == expected
    assert profile.min_llama_build == min_build


@pytest.mark.parametrize(
    "name,arch,expected",
    [
        ("Qwen3.5-9B-Q4_K_M", "qwen35", "qwen3_5-3_6.yaml"),
        ("llm-jp-4-8b-thinking-Q4_K_M", "llama", "llama-3.yaml"),
        ("GLM-5.2-Q4_K_M", "glm4", "glm-5_2.yaml"),
        ("Eleven-Q4_K_M", "llama", "llama-3.yaml"),
    ],
)
def test_new_profiles_do_not_capture_neighbouring_models(name, arch, expected):
    assert (
        match_profile(name, _profiles(), arch).source_file != "decision-systemone.yaml"
    )
    profile = match_profile(name, _profiles(), arch)
    assert profile.source_file not in {
        "clef.yaml",
        "llm-jp-4_1.yaml",
        "glm-ocr.yaml",
        "dots-ocr.yaml",
    }, (name, expected)


def test_ocr_profiles_are_deterministic_without_repetition_penalty():
    profiles = _profiles()
    glm = match_profile("GLM-OCR-Q8_0", profiles, "glm4")
    dots = match_profile("dots.ocr-Q8_0", profiles, "qwen2")
    assert glm.sampling["chat"]["temperature"] == 0.0
    assert dots.sampling["chat"]["temperature"] == 0.1
    for profile in (glm, dots):
        assert profile.sampling["chat"]["repeat_penalty"] == 1.0
        assert "--jinja" in profile.extra_args
    assert dots.max_context == 32768


def test_llm_jp_profile_follows_the_cookbook_sampling():
    profile = match_profile("llm-jp-4.1-33b-thinking-Q4_K_M", _profiles(), "llama")
    assert profile.sampling["chat"] == {
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 0.9,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    }
    assert profile.extra_args == ["--jinja"]
    assert profile.max_context == 65536
    ok, reason, detected = tuner.check_profile_build(profile, "llama-server")
    assert ok  # unprobeable wrapper: warning only
    assert "b11320" in reason and detected is None


def _mmproj(tmp_path, monkeypatch, name, metadata):
    import scanner

    path = tmp_path / name
    path.write_bytes(b"GGUF" + name.encode())
    known = getattr(_mmproj, "known", {})
    known[str(path)] = metadata
    _mmproj.known = known
    monkeypatch.setattr(
        scanner, "read_gguf_metadata", lambda p: _mmproj.known.get(str(p), {})
    )
    return path


def test_noncausal_gemma4_projectors_report_their_image_budget(tmp_path, monkeypatch):
    uv = _mmproj(
        tmp_path,
        monkeypatch,
        "mmproj-12b.gguf",
        {"clip.vision.projector_type": "gemma4uv"},
    )
    big = _mmproj(
        tmp_path,
        monkeypatch,
        "mmproj-31b.gguf",
        {"clip.vision.projector_type": "gemma4v", "clip.vision.projection_dim": 5376},
    )
    e4b = _mmproj(
        tmp_path,
        monkeypatch,
        "mmproj-e4b.gguf",
        {"clip.vision.projector_type": "gemma4v", "clip.vision.projection_dim": 2560},
    )
    qwen = _mmproj(
        tmp_path,
        monkeypatch,
        "mmproj-qwen.gguf",
        {"clip.vision.projector_type": "qwen3vl_merger"},
    )
    assert tuner.noncausal_vision_token_budget(uv) == 1120
    assert tuner.noncausal_vision_token_budget(big) == 1120
    # E2B/E4B decode images causally; llama.cpp may split them across ubatches.
    assert tuner.noncausal_vision_token_budget(e4b) == 0
    assert tuner.noncausal_vision_token_budget(qwen) == 0
    assert tuner.noncausal_vision_token_budget(None) == 0
    assert tuner.noncausal_vision_token_budget(tmp_path / "missing.gguf") == 0


def test_gemma4_vision_plan_keeps_the_whole_image_in_one_ubatch(tmp_path, monkeypatch):
    profiles = _profiles()
    metadata = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 48,
        "gemma4.context_length": 262144,
        "gemma4.embedding_length": 3840,
        "gemma4.attention.head_count": 16,
        "gemma4.attention.head_count_kv": 8,
    }
    model = _fake_model_md(tmp_path, "gemma-4-12b-it-v2-Q4_K_M", 7, dict(metadata))
    profile = match_profile(model.name, profiles, model.architecture)
    system = _fake_system()
    text_only = tuner.compute_config(
        model, system, profile, user_ctx=16384, prompt_cache_ram_mib=0
    )
    assert text_only.ubatch == 512  # small dense model, short context

    model.mmproj = _mmproj(
        tmp_path,
        monkeypatch,
        "mmproj-gemma-4-12b.gguf",
        {"clip.vision.projector_type": "gemma4uv"},
    )
    vision = tuner.compute_config(
        model, system, profile, user_ctx=16384, prompt_cache_ram_mib=0
    )
    assert vision.ubatch == tuner.GEMMA4_IMAGE_MAX_TOKENS == 1120
    assert vision.batch >= vision.ubatch

    # A causal projector leaves the regular batch tiers alone.
    model.mmproj = _mmproj(
        tmp_path,
        monkeypatch,
        "mmproj-causal.gguf",
        {"clip.vision.projector_type": "qwen3vl_merger"},
    )
    causal = tuner.compute_config(
        model, system, profile, user_ctx=16384, prompt_cache_ram_mib=0
    )
    assert causal.ubatch == 512


def test_embedded_flash_next_mtp_is_loaded_but_never_drafted(tmp_path, monkeypatch):
    model = _flash_next(tmp_path, embedded_mtp=True)
    assert tuner.qwen4exp_mtp_draft_unusable(model)
    assert not tuner.qwen4exp_mtp_draft_unusable(_flash_next(tmp_path))
    profile = match_profile(model.name, _profiles(), model.architecture)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11371)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: set(_manifest("b11371")["flags"])
    )
    config = tuner.compute_config(
        model, _fake_system(ram_total=96, ram_free=80), profile, user_ctx=8192
    )
    command = tuner.build_command(
        model, config, profile, enable_speculative=True, enable_ngram=True
    )
    spec = command[command.index("--spec-type") + 1]
    assert "draft-mtp" not in spec and spec == profile.ngram_method
    assert "--spec-draft-n-max" not in command

    # Other architectures keep their embedded MTP path.
    qwen = _fake_model_md(
        tmp_path,
        "Qwen3.8-27B-Q4_K_M",
        16,
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 65,
            "qwen35.nextn_predict_layers": 1,
            "__mtp_scan__": "found",
        },
    )
    assert not tuner.qwen4exp_mtp_draft_unusable(qwen)


def test_embedded_mtp_target_does_not_default_to_a_sibling_mtp_head(
    tmp_path, monkeypatch
):
    import scanner
    from test_smoke import _write_minimal_gguf

    monkeypatch.setattr(scanner, "_DRAFT_MAX_SIZE_BYTES", 32)
    merged = tmp_path / "Qwen3.8-Flash-Next-MTP-UD-Q2_K_XL.gguf"
    trunk = tmp_path / "Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf"
    head = tmp_path / "mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"
    for path in (merged, trunk, head):
        _write_minimal_gguf(path)
        with path.open("ab") as stream:
            stream.write(b"\0" * 64)
    real = scanner.read_gguf_metadata

    def fake(path, *args, **kwargs):
        metadata = dict(real(path, *args, **kwargs) or {})
        if Path(path).name == merged.name:
            metadata.update(
                {
                    "general.architecture": "qwen4exp",
                    "qwen4exp.block_count": 49,
                    "qwen4exp.nextn_predict_layers": 1,
                    "__mtp_scan__": "found",
                }
            )
        return metadata

    monkeypatch.setattr(scanner, "read_gguf_metadata", fake)
    entries = {entry.name: entry for entry in scanner.scan_models(tmp_path)}
    assert set(entries) == {merged.stem, trunk.stem}
    # The merged file drafts with its own block; the head stays selectable.
    assert entries[merged.stem].draft is None
    assert entries[merged.stem].folder_drafts == [head]
    # The trunk-only target keeps its paired head (blocked later by the gate).
    assert entries[trunk.stem].draft == head
