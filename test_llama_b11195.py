"""b11195 audit: unchanged CLI and the MiMo-V2 MTP-only head floor (PR #29294)."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_smoke import _fake_model_md

ROOT = Path(__file__).resolve().parent


def _profiles():
    return load_profiles(ROOT / "settings")


def _manifest(tag: str) -> dict:
    return json.loads(
        (ROOT / f"docs/llama-{tag}-server-flags.json").read_text(encoding="utf-8")
    )


def test_b11195_manifest_keeps_flags_and_help_text():
    current = _manifest("b11195")
    previous = _manifest("b11160")
    assert current["tag"] == "b11195"
    assert current["commit"] == "d834d44e6"
    assert current["flags"] == previous["flags"]
    assert len(current["flags"]) == 415
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for backend, binary in current["binaries"].items():
        # b11160..b11195 adds no server option or help line; the CPU tiled
        # mul_mat (GGML_CPU_TILED_MM) and GGML_CUDA_MMQ_PREC are env-only.
        assert binary["help_sha256"] == previous["binaries"][backend]["help_sha256"]
        assert binary["sha256"] != previous["binaries"][backend]["sha256"]
        assert current["notes"][backend]["stripped_startup_log_lines"] == 1
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"]


def _mimo2(tmp_path, name, size_gb, **extra):
    md = {"general.architecture": "mimo2", "mimo2.block_count": 51, **extra}
    return _fake_model_md(tmp_path, name, size_gb, md)


@pytest.mark.parametrize(
    ("build", "allowed"), [(11160, False), (11194, False), (11195, True), (None, True)]
)
def test_mimo2_mtp_only_head_needs_b11195(tmp_path, monkeypatch, build, allowed):
    head = _mimo2(
        tmp_path,
        "MiMo-V2.6-Flash-MTP-Q8_0",
        4,
        **{"mimo2.nextn_predict_layers": 3, "__mtp_scan__": "found"},
    )
    assert tuner.is_mimo2_mtp_sidecar(head)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    ok, reason, detected = tuner.check_draft_model_build(head, "llama-server")
    assert ok is allowed
    if not allowed:
        assert detected == build
        assert "b11195" in reason and "#29294" in reason and head.path.name in reason


def test_mimo2_gate_ignores_trunks_and_other_architectures(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11160)
    # --no-nextn trunk export: the converter omits the nextn key entirely.
    trunk = _mimo2(tmp_path, "MiMo-V2.6-Flash-trunk", 8, __mtp_scan__="absent")
    # Stale key over a scan-proven trunk-only file is no MTP head either.
    stale = _mimo2(
        tmp_path,
        "MiMo-V2.6-Flash-stripped",
        8,
        **{"mimo2.nextn_predict_layers": 3, "__mtp_scan__": "absent"},
    )
    qwen = _fake_model_md(
        tmp_path,
        "mtp-Qwen3.8-27B",
        1,
        {"general.architecture": "qwen35", "qwen35.nextn_predict_layers": 1},
    )
    for draft in (trunk, stale, qwen):
        assert not tuner.is_mimo2_mtp_sidecar(draft), draft.name
        assert tuner.check_draft_model_build(draft, "llama-server")[0], draft.name
    assert not tuner.is_mimo2_mtp_sidecar(None)


def test_mimo_profile_note_names_the_mtp_head_floor():
    profile = match_profile("MiMo-V2.6-Flash-RL-Q4_K_M", _profiles(), "mimo2")
    assert profile.source_file == "mimo-v2_6.yaml"
    assert profile.min_llama_build == 11102  # the profile floor is unchanged
    assert "b11195" in profile.notes and "#29294" in profile.notes
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        note = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"][
            "mimo-v2_6.yaml"
        ]
        assert "b11195" in note and "#29294" in note and "--mtp" in note, pack.name


# ------------------------------------------------ newest pinned fork wins ----
# v5.5.7 moves the Ternary/Bonsai recipes to prism-b10743 while the recipes
# never delete the older 2b_b10687 trees. Before, a backend-neutral profile
# hint resolved to the lexicographically first (oldest) tree.


def test_newest_first_key_orders_builds_and_backends():
    from auto_tuner import _fork_newest_first_key

    names = [
        "2b_b10687_vulkan_llama.cpp",
        "2b_b10743_hip_llama.cpp",
        "2b_b10687_hip_llama.cpp",
        "2b_b10743_vulkan_llama.cpp",
    ]
    assert sorted(names, key=_fork_newest_first_key)[0] == "2b_b10743_vulkan_llama.cpp"
    by_hip = sorted(names, key=lambda n: _fork_newest_first_key(n, "hip"))
    assert by_hip[:2] == ["2b_b10743_hip_llama.cpp", "2b_b10687_hip_llama.cpp"]
    stable = ["0.4.1_vulkan_llama.cpp", "0.5.0_vulkan_llama.cpp"]
    assert sorted(stable, key=_fork_newest_first_key)[0] == "0.5.0_vulkan_llama.cpp"


def test_resolver_prefers_newest_fork_build_but_keeps_explicit_choice(
    tmp_path, monkeypatch
):
    from auto_tuner import _resolve_server_binary
    from test_smoke import _fake_llama_server_path, _write_fake_server

    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    container = tmp_path / "ai-local"
    servers = {
        name: _fake_llama_server_path(container / name)
        for name in (
            "b11195_hip_llama.cpp",
            "2b_b10687_vulkan_llama.cpp",
            "2b_b10687_hip_llama.cpp",
            "2b_b10743_vulkan_llama.cpp",
            "2b_b10743_hip_llama.cpp",
        )
    }
    for server in servers.values():
        _write_fake_server(server)
    monkeypatch.chdir(auto_dir)

    def resolve(hint):
        return Path(_resolve_server_binary(hint)).resolve()

    # Mainline HIP selected: the neutral hint keeps HIP and takes the new pin.
    monkeypatch.setenv("LLAMA_CPP_DIR", str(container / "b11195_hip_llama.cpp"))
    assert (
        resolve("2b_llama/llama-server") == servers["2b_b10743_hip_llama.cpp"].resolve()
    )
    assert (
        resolve("2b_vulkan_llama/llama-server")
        == servers["2b_b10743_vulkan_llama.cpp"].resolve()
    )
    # Only the container configured: Vulkan default, newest build.
    monkeypatch.setenv("LLAMA_CPP_DIR", str(container))
    assert (
        resolve("2b_llama/llama-server")
        == servers["2b_b10743_vulkan_llama.cpp"].resolve()
    )
    # An explicitly selected older fork is never overridden.
    monkeypatch.setenv("LLAMA_CPP_DIR", str(container / "2b_b10687_vulkan_llama.cpp"))
    assert (
        resolve("2b_llama/llama-server")
        == servers["2b_b10687_vulkan_llama.cpp"].resolve()
    )


def test_gui_auto_select_takes_newest_build_of_required_family():
    import types

    from qt_launcher import MainWindow

    class Combo:
        def __init__(self, items, index):
            self.items, self.index = items, index

        def count(self):
            return len(self.items)

        def itemText(self, index):
            return self.items[index]

        def currentText(self):
            return self.items[self.index]

        def currentIndex(self):
            return self.index

        def setCurrentIndex(self, index):
            self.index = index

        def blockSignals(self, _blocked):
            pass

    items = [
        "b11195_vulkan_llama.cpp",
        "b11195_hip_llama.cpp",
        "2b_b10687_vulkan_llama.cpp",
        "2b_b10687_hip_llama.cpp",
        "2b_b10743_vulkan_llama.cpp",
        "2b_b10743_hip_llama.cpp",
    ]
    profile = types.SimpleNamespace(server_binary="2b_llama/llama-server")
    for start, expected in (
        (0, "2b_b10743_vulkan_llama.cpp"),
        (1, "2b_b10743_hip_llama.cpp"),
    ):
        combo = Combo(list(items), start)
        applied = []
        window = types.SimpleNamespace(
            _fork_combo=combo,
            _fork_manual_override=False,
            _apply_fork=applied.append,
            _log=lambda _message: None,
        )
        MainWindow._auto_select_fork(window, profile)
        assert combo.currentText() == expected
        assert applied == [items.index(expected)]


def test_bonsai_recipes_and_notes_name_the_prism_b10743_pin():
    profile = match_profile("Ternary-Bonsai-2-27B-PQ2_0", _profiles(), "qwen35")
    assert profile.source_file == "bonsai-2-27b.yaml"
    assert profile.min_llama_build == 10687  # floor unchanged
    assert "prism-b10743" in profile.notes
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        note = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"][
            "bonsai-2-27b.yaml"
        ]
        assert "prism-b10743" in note and "prism-b10687" in note, pack.name
    for name in (
        "ternary_bonsai_vulkan_llama_build.ps1",
        "ternary_bonsai_hip_llama_build.ps1",
    ):
        text = (ROOT / "building llama.cpp" / name).read_text(encoding="utf-8")
        assert '"adfffbe41b2cabcd51fff326ab045662265062bb"' in text, name
        assert '-FixedIdentity "b10743"' in text, name
