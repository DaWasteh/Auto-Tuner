"""Tests for the shared GUI/TUI OCR document pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from ocr_workflow import (
    OcrJobOptions,
    OcrJobRunner,
    OcrWorkflowError,
    discover_input_files,
    is_ocr_model,
    ocr_model_preset,
    ocr_projector_warning,
    parse_page_range,
    strip_grounding_markup,
)
from scanner import ModelEntry

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_QT_APP = None


def _model(name: str, arch: str = "") -> ModelEntry:
    return ModelEntry(
        path=Path(name + ".gguf"),
        name=name,
        group=".",
        size_bytes=1024,
        mmproj=Path("mmproj.gguf"),
        metadata={"general.architecture": arch},
    )


def test_ocr_model_detection_and_unlimited_defaults() -> None:
    model = _model("Unlimited-OCR-BF16", "deepseek2-ocr")
    assert is_ocr_model(model)
    preset = ocr_model_preset(model)
    assert preset.family == "unlimited-ocr"
    assert preset.prompt == "document parsing."
    assert preset.min_llama_build == 10287
    assert preset.strip_grounding

    # The architecture is shared; v1/v2 must never inherit Unlimited's prompt.
    for name in ("DeepSeek-OCR", "DeepSeek-OCR-2"):
        deepseek = ocr_model_preset(_model(name, "deepseek2-ocr"))
        assert deepseek.family == "deepseek-ocr"
        assert deepseek.prompt == "Free OCR."
        assert deepseek.min_llama_build == 0
    assert not is_ocr_model(_model("Qwen3-Coder", "qwen3"))


def test_unlimited_projector_warns_when_32_tile_metadata_is_missing(
    monkeypatch,
) -> None:
    import scanner

    model = _model("Unlimited-OCR-BF16", "deepseek2-ocr")
    monkeypatch.setattr(scanner, "read_gguf_metadata", lambda _path: {})
    assert "preproc_max_tiles=32" in ocr_projector_warning(model)
    monkeypatch.setattr(
        scanner,
        "read_gguf_metadata",
        lambda _path: {"clip.vision.preproc_max_tiles": 32},
    )
    assert ocr_projector_warning(model) == ""


def test_ocr_profile_metadata_does_not_override_global_q4_auto_default() -> None:
    from settings_loader import load_profiles, match_profile
    from tuner import _pick_kv_quant

    profiles = load_profiles(Path(__file__).resolve().parent / "settings")
    profile = match_profile("Unlimited-OCR-BF16", profiles, "deepseek2-ocr")
    assert profile.source_file == "unlimited-ocr.yaml"
    deepseek = match_profile("DeepSeek-OCR-2", profiles, "deepseek2-ocr")
    assert deepseek.source_file == "deepseek-ocr.yaml"
    opaque = match_profile("opaque-checkpoint", profiles, "deepseek2-ocr")
    assert opaque.source_file == "deepseek-ocr.yaml"
    assert profile.min_llama_build == 10287
    assert profile.flash_attn is False
    assert profile.recommended_kv_quant == "f16"
    assert _pick_kv_quant(" f16 ", 32768, 0.01, 64.0) == ("q4_0", "q4_0")
    assert _pick_kv_quant("q4_1", 1024, 0.001, 64.0) == ("q4_0", "q4_0")
    assert _pick_kv_quant("", 1024, 0.001, 64.0) == ("q4_0", "q4_0")


def test_profile_loader_accepts_numeric_flash_attention(tmp_path) -> None:
    from settings_loader import load_profiles

    (tmp_path / "off.yaml").write_text(
        'display_name: "off"\npatterns: [off-model]\nflash_attn: 0\n',
        encoding="utf-8",
    )
    (tmp_path / "on.yaml").write_text(
        'display_name: "on"\npatterns: [on-model]\nflash_attn: 1\n',
        encoding="utf-8",
    )
    profiles = {profile.display_name: profile for profile in load_profiles(tmp_path)}
    assert profiles["off"].flash_attn is False
    assert profiles["on"].flash_attn is True


def test_page_range_parser() -> None:
    assert parse_page_range("", 4) == (0, 1, 2, 3)
    assert parse_page_range("1-2,4", 4) == (0, 1, 3)
    assert parse_page_range("3,1,3", 3) == (0, 2)
    with pytest.raises(ValueError, match="exceeds"):
        parse_page_range("4", 3)
    with pytest.raises(ValueError, match="invalid"):
        parse_page_range("3-2", 3)


def test_discover_inputs_filters_and_deduplicates(tmp_path) -> None:
    image = tmp_path / "page.PNG"
    image.write_bytes(b"not decoded in discovery")
    (tmp_path / "ignore.bin").write_bytes(b"x")
    found = discover_input_files([tmp_path, image])
    assert found == [image.resolve()]


def test_strip_grounding_keeps_recognized_text() -> None:
    raw = (
        "<|ref|>Invoice<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n"
        "text [1, 2, 3, 4]Total 42<|endoftext|>"
    )
    assert strip_grounding_markup(raw) == "Invoice\nTotal 42"


def test_http_payload_places_image_before_unlimited_prompt(
    tmp_path, monkeypatch
) -> None:
    import ocr_workflow

    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    captured = {}

    class Response:
        status = 200

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "recognized"}}]}
            ).encode()

        def close(self):
            return None

    class Connection:
        def __init__(self, host, port, timeout):
            captured["host"] = host
            captured["port"] = port
            captured["timeout"] = timeout

        def request(self, method, endpoint, body, headers):
            captured["method"] = method
            captured["endpoint"] = endpoint
            captured["payload"] = json.loads(body)
            captured["headers"] = headers

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(ocr_workflow.http.client, "HTTPConnection", Connection)
    runner = OcrJobRunner(
        "http://127.0.0.1:1234",
        "Unlimited-OCR",
        OcrJobOptions([image], tmp_path / "out", "document parsing."),
    )
    assert runner._request_page(image) == "recognized"
    content = captured["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "document parsing."}


def test_windows_cancel_always_targets_libreoffice_process_tree(monkeypatch) -> None:
    import ocr_workflow

    calls = []

    class Process:
        pid = 4321

        def poll(self):
            return 0  # wrapper already exited; child may still be alive

        def kill(self):
            pytest.fail("taskkill should handle the tree")

    monkeypatch.setattr(ocr_workflow.os, "name", "nt")
    monkeypatch.setattr(
        ocr_workflow.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    ocr_workflow._terminate_process(Process())
    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]


def test_cancel_interrupts_request_before_response_headers(
    tmp_path, monkeypatch
) -> None:
    import threading
    import ocr_workflow

    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    waiting = threading.Event()
    closed = threading.Event()

    class DelayedConnection:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            waiting.set()
            closed.wait(5)
            raise OSError("connection closed")

        def close(self):
            closed.set()

    monkeypatch.setattr(ocr_workflow.http.client, "HTTPConnection", DelayedConnection)
    runner = OcrJobRunner(
        "http://127.0.0.1:1234",
        "ocr",
        OcrJobOptions([image], tmp_path / "out", "OCR"),
    )
    errors = []
    thread = threading.Thread(
        target=lambda: _capture_exception(lambda: runner._request_page(image), errors)
    )
    thread.start()
    assert waiting.wait(1)
    runner.cancel()
    thread.join(1)
    assert not thread.is_alive()
    assert errors and isinstance(errors[0], ocr_workflow.OcrCancelled)


def _capture_exception(call, errors) -> None:
    try:
        call()
    except Exception as exc:
        errors.append(exc)


class _FakeRunner(OcrJobRunner):
    def _request_page(self, image: Path) -> str:
        assert image.is_file()
        return f"Text from {image.stem}"


def test_wait_for_server_rejects_wrong_model_alias(monkeypatch) -> None:
    import ocr_workflow

    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "unrelated-model"}]}).encode()

    monkeypatch.setattr(
        ocr_workflow.urllib.request,
        "urlopen",
        lambda url, timeout: Response(url),
    )
    with pytest.raises(ocr_workflow.OcrWorkflowError, match="not the OCR server"):
        ocr_workflow.wait_for_server(
            "http://127.0.0.1:1234",
            timeout_seconds=1,
            process_is_running=lambda: True,
            expected_model_alias="expected-ocr",
        )


def test_wait_for_server_stops_when_spawned_process_exits(monkeypatch) -> None:
    import ocr_workflow

    monkeypatch.setattr(
        ocr_workflow.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("must not contact occupied endpoint"),
    )
    with pytest.raises(ocr_workflow.OcrWorkflowError, match="exited"):
        ocr_workflow.wait_for_server(
            "http://127.0.0.1:1234",
            timeout_seconds=1,
            process_is_running=lambda: False,
        )


def test_image_and_pdf_job_share_output_and_manifest(tmp_path) -> None:
    import pymupdf

    image = tmp_path / "photo.jpg"
    Image.new("RGB", (160, 100), "white").save(image)

    pdf = tmp_path / "document.pdf"
    document = pymupdf.open()
    for index in range(2):
        page = document.new_page(width=200, height=100)
        page.insert_text((20, 50), f"PDF page {index + 1}")
    document.save(pdf)
    document.close()

    options = OcrJobOptions(
        inputs=[image, pdf],
        output_dir=tmp_path / "output",
        prompt="OCR",
        page_range="2",
        strip_grounding=True,
        keep_rendered=False,
    )
    runner = _FakeRunner(
        "http://127.0.0.1:1234",
        "ocr-model",
        options,
        model_name="Test OCR",
        llama_build=10329,
        server_command=["llama-server", "-m", "model.gguf"],
    )
    result = runner.run()

    assert result.total_pages == 2  # one image + selected PDF page 2
    assert result.completed_pages == 2
    assert result.failed_pages == 0
    assert result.combined_output.is_file()
    text = result.combined_output.read_text(encoding="utf-8")
    assert "# photo.jpg" in text
    assert "# document.pdf" in text
    assert "<|det|>" not in text
    assert not (result.job_dir / "rendered").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["llama_build"] == 10329
    assert len(manifest["pages"]) == 2
    assert all(source["sha256"] for source in manifest["sources"])


def test_render_error_records_page_and_continues(tmp_path) -> None:
    bad = tmp_path / "bad.png"
    good = tmp_path / "good.png"
    Image.new("RGB", (10, 10), "white").save(bad)
    Image.new("RGB", (10, 10), "white").save(good)

    class RenderFailureRunner(_FakeRunner):
        def _normalize_image(self, source, target, frame_index=0):
            if source.name == "bad.png":
                raise OSError("synthetic decode failure")
            return super()._normalize_image(source, target, frame_index)

    result = RenderFailureRunner(
        "http://127.0.0.1:1234",
        "ocr",
        OcrJobOptions([bad, good], tmp_path / "out", "OCR"),
    ).run()
    assert result.total_pages == 2
    assert result.completed_pages == 1
    assert result.failed_pages == 1
    assert "synthetic decode failure" in result.page_results[0].error
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_errors"


def test_changed_source_is_not_processed_under_stale_manifest_hash(tmp_path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    runner = _FakeRunner(
        "http://127.0.0.1:1234",
        "ocr",
        OcrJobOptions([image], tmp_path / "out", "OCR"),
    )
    job_dir, _ = runner.prepare()
    Image.new("RGB", (11, 10), "black").save(image)
    with pytest.raises(OcrWorkflowError, match="changed after OCR preparation"):
        runner.run()
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "changed after OCR preparation" in manifest["error"]


def test_changed_office_source_is_rejected_after_conversion(tmp_path) -> None:
    source = tmp_path / "document.docx"
    source.write_bytes(b"office-v1")

    class OfficeSnapshotRunner(_FakeRunner):
        def _convert_office(self, original, converted_dir):
            converted_dir.mkdir(parents=True, exist_ok=True)
            pdf = converted_dir / "document.pdf"
            pdf.write_bytes(b"converted-v1")
            return pdf

        def _pdf_page_count(self, path):
            return 1

        def _render_pdf_page(self, pdf, page_index, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), "white").save(target)

    runner = OfficeSnapshotRunner(
        "http://127.0.0.1:1234",
        "ocr",
        OcrJobOptions([source], tmp_path / "out", "OCR"),
    )
    job_dir, _ = runner.prepare()
    source.write_bytes(b"office-v2-is-different")
    with pytest.raises(OcrWorkflowError, match="changed after OCR preparation"):
        runner.run()
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "changed after OCR preparation" in manifest["error"]


def test_transparent_palette_image_is_composited_on_white(tmp_path) -> None:
    image = tmp_path / "transparent.gif"
    palette = Image.new("P", (4, 4), 0)
    palette.putpalette([0, 0, 0] * 256)
    palette.info["transparency"] = 0
    palette.save(image, transparency=0)
    runner = _FakeRunner(
        "http://127.0.0.1:1234",
        "ocr",
        OcrJobOptions([image], tmp_path / "out", "OCR", keep_rendered=True),
    )
    result = runner.run()
    rendered = next((result.job_dir / "rendered").glob("*.png"))
    with Image.open(rendered) as normalized:
        assert normalized.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_multiframe_tiff_pages_are_not_silently_truncated(tmp_path) -> None:
    tiff = tmp_path / "scan.tiff"
    first = Image.new("RGB", (20, 20), "white")
    second = Image.new("RGB", (20, 20), "black")
    first.save(tiff, save_all=True, append_images=[second])
    runner = _FakeRunner(
        "http://127.0.0.1:1234",
        "ocr",
        OcrJobOptions(
            [tiff], tmp_path / "out", "OCR", page_range="2", keep_rendered=True
        ),
    )
    result = runner.run()
    assert result.total_pages == 1
    assert result.page_results[0].page == 2
    assert (result.job_dir / "rendered" / "scan-page-0002.png").is_file()


def test_qt_ocr_dialog_uses_shared_model_preset(tmp_path) -> None:
    global _QT_APP

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    _QT_APP = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    model = _model("Unlimited-OCR-BF16", "deepseek2-ocr")
    dialog = qt_launcher._OcrSetupDialog(model)
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    dialog._add_path(str(image))
    dialog.output_edit.setText(str(tmp_path / "out"))
    options = dialog.options()
    assert options.inputs == [image]
    assert options.prompt == "document parsing."
    assert options.strip_grounding
    assert options.stop_server_when_done
    dialog.close()


def test_keyboard_interrupt_records_cancelled_manifest(tmp_path) -> None:
    class InterruptRunner(OcrJobRunner):
        def _request_page(self, image: Path) -> str:
            raise KeyboardInterrupt

    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    runner = InterruptRunner(
        "http://127.0.0.1:1234",
        "ocr-model",
        OcrJobOptions([image], tmp_path / "output", "OCR"),
    )
    result = runner.run()
    assert result.cancelled
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"


def test_cancel_before_work_returns_cancelled_manifest(tmp_path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    runner = _FakeRunner(
        "http://127.0.0.1:1234",
        "ocr-model",
        OcrJobOptions(inputs=[image], output_dir=tmp_path / "output", prompt="OCR"),
    )
    runner.cancel()
    result = runner.run()
    assert result.cancelled
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
