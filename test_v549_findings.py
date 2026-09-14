"""v5.4.9: control-API state on rejected switches, benchmark time budget, probe worker."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import app_settings
from control_api import ControlApiError, ControlApiServer, ControlRequest
from model_benchmark import (
    BenchmarkCandidate,
    BenchmarkFailure,
    BenchmarkLimits,
    BenchmarkRunner,
    BenchmarkSample,
    CandidateResult,
)
from test_control_api import TOKEN, _model
from test_control_api_qt import SETTINGS_DIR

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# control_api: a switch the GUI rejects before touching the backend keeps the
# previously active model


def _api_with_active_model(switch):
    api = ControlApiServer(port=0, token=TOKEN, switch_callback=switch)
    api.update_models([_model(), _model("other-7b")])
    return api


def test_untouched_rejection_restores_the_active_model():
    calls: list[str] = []

    def switch(model_id, _timeout, _options):
        calls.append(model_id)
        if model_id == "other-7b":
            raise ControlApiError(
                "AutoTuner is busy with an exclusive benchmark or OCR workflow.",
                status=409,
                code="autotuner_busy",
                backend_untouched=True,
            )
        return {"backend_url": "http://127.0.0.1:65530", "alias": "qwen"}

    api = _api_with_active_model(switch)
    assert api.ensure_model("qwen-7b")["active_model"] == "qwen-7b"
    with pytest.raises(ControlApiError) as error:
        api.ensure_model("other-7b")
    assert error.value.status == 409 and error.value.backend_untouched
    status = api.status()
    assert status["active_model"] == "qwen-7b"
    assert status["status"] == "ready"
    # A proxy lease without a model id still routes to the surviving backend.
    lease = api.acquire_proxy_lease()
    assert lease.model_id == "qwen-7b" and lease.backend_url.endswith(":65530")
    lease.release()
    assert calls == ["qwen-7b", "other-7b"]


def test_rejection_after_the_backend_was_touched_keeps_the_cleared_state():
    def switch(model_id, _timeout, _options):
        if model_id == "other-7b":
            raise ControlApiError(
                "AutoTuner could not start the requested model.",
                status=409,
                code="launch_failed",
            )
        return {"backend_url": "http://127.0.0.1:65530", "alias": "qwen"}

    api = _api_with_active_model(switch)
    api.ensure_model("qwen-7b")
    with pytest.raises(ControlApiError):
        api.ensure_model("other-7b")
    status = api.status()
    assert not status["active_model"]
    with pytest.raises(ControlApiError, match="No AutoTuner model is active"):
        api.acquire_proxy_lease()


def test_clear_active_during_a_rejected_switch_wins():
    api_holder: dict = {}

    def switch(model_id, _timeout, _options):
        if model_id == "other-7b":
            # The GUI stopped the old server after all (e.g. it crashed).
            api_holder["api"].clear_active()
            raise ControlApiError(
                "busy", status=409, code="autotuner_busy", backend_untouched=True
            )
        return {"backend_url": "http://127.0.0.1:65530", "alias": "qwen"}

    api = _api_with_active_model(switch)
    api_holder["api"] = api
    api.ensure_model("qwen-7b")
    with pytest.raises(ControlApiError):
        api.ensure_model("other-7b")
    assert not api.status()["active_model"]


def test_control_request_fail_carries_the_flag():
    request = ControlRequest("switch", "m")
    request.fail("busy", status=409, code="autotuner_busy", backend_untouched=True)
    with pytest.raises(ControlApiError) as error:
        request.wait()
    assert error.value.backend_untouched is True
    plain = ControlRequest("switch", "m")
    plain.fail("launch failed", status=409, code="launch_failed")
    with pytest.raises(ControlApiError) as error:
        plain.wait()
    assert error.value.backend_untouched is False


# ---------------------------------------------------------------------------
# model_benchmark: the total deadline decides with the measured candidates


def _config():
    return SimpleNamespace(
        ctx=8192,
        ngl=99,
        threads=8,
        batch_threads=8,
        batch=512,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
        sampling={},
        draft_n_max=0,
        full_offload=True,
    )


def _runner(limits: BenchmarkLimits) -> BenchmarkRunner:
    return BenchmarkRunner(
        model=SimpleNamespace(path=Path("model.gguf")),
        profile=SimpleNamespace(),
        base_config=_config(),
        runtime_binary="llama-server",
        physical_cores=8,
        logical_cores=16,
        limits=limits,
    )


def _measured(candidate: BenchmarkCandidate, prompt: float, generation: float):
    sample = BenchmarkSample(
        prompt_tps=prompt,
        generation_tps=generation,
        prompt_tokens=1024,
        generated_tokens=64,
        elapsed_s=1.0,
    )
    return CandidateResult(candidate=candidate, samples=[sample, sample])


def test_deadline_keeps_completed_measurements(monkeypatch):
    limits = BenchmarkLimits(
        max_candidates=18, confirmation_runs=2, total_timeout_s=0.7
    )
    runner = _runner(limits)
    calls: list[str] = []

    def fake_benchmark(candidate: BenchmarkCandidate) -> CandidateResult:
        calls.append(candidate.id)
        time.sleep(0.3)
        generation = 130.0 if candidate.id != "baseline" else 100.0
        return _measured(candidate, 100.0, generation)

    monkeypatch.setattr(runner, "_benchmark_candidate", fake_benchmark)
    monkeypatch.setattr(
        "model_benchmark.probe_binary_build_number", lambda _binary: 10948
    )
    result = runner.run()
    assert result.budget_exhausted is True
    assert "total time limit reached" in result.reason
    # Exploration stopped long before the 18-candidate budget, nothing was
    # discarded, no phantom candidate errors were recorded.
    assert 2 <= len(calls) < 6
    assert all(not item.error for item in result.candidates)
    # The measured non-baseline candidates survive as valid evidence and the
    # normal uncertainty-safe decision still runs on them.
    assert any(
        item.valid for item in result.candidates if item.candidate.id != "baseline"
    )
    assert result.winner.confirmations == 0


def test_deadline_before_the_baseline_is_still_a_failure(monkeypatch):
    runner = _runner(BenchmarkLimits(total_timeout_s=0.2))

    def slow_baseline(candidate: BenchmarkCandidate) -> CandidateResult:
        time.sleep(0.3)
        runner._check_cancelled()
        return _measured(candidate, 100.0, 100.0)

    monkeypatch.setattr(runner, "_benchmark_candidate", slow_baseline)
    with pytest.raises(BenchmarkFailure, match="before the baseline finished"):
        runner.run()


def test_deadline_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("AUTOTUNER_BENCHMARK_DEADLINE_S", "0.5")
    runner = _runner(BenchmarkLimits(total_timeout_s=0.0))

    def fake_benchmark(candidate: BenchmarkCandidate) -> CandidateResult:
        time.sleep(0.2)
        return _measured(candidate, 100.0, 100.0)

    monkeypatch.setattr(runner, "_benchmark_candidate", fake_benchmark)
    monkeypatch.setattr(
        "model_benchmark.probe_binary_build_number", lambda _binary: 10948
    )
    result = runner.run()
    assert result.budget_exhausted is True


# ---------------------------------------------------------------------------
# qt_launcher: server probes run on the worker; the GUI only applies results


def _fake_urlopen(status: int = 200, body: bytes = b"[]"):
    class Response:
        def __init__(self) -> None:
            self.status = status

        def read(self) -> bytes:
            return body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def urlopen(url, timeout=None):
        _fake_urlopen.calls.append((url, timeout))
        return Response()

    return urlopen


_fake_urlopen.calls = []  # type: ignore[attr-defined]


def test_probe_worker_reports_health_and_slots_off_thread(monkeypatch):
    qt_launcher = pytest.importorskip("qt_launcher")
    monkeypatch.setattr(
        qt_launcher.urllib.request, "urlopen", _fake_urlopen(200, b'{"slots": [{}]}')
    )
    results: list[dict] = []
    worker = qt_launcher._ServerProbeWorker(results.append)
    try:
        assert worker.submit({"kind": "health", "token": 7, "url": "http://h"})
        # The same probe is not queued twice while in flight or queued.
        worker.submit({"kind": "health", "token": 7, "url": "http://h"})
        assert worker.submit({"kind": "slots", "token": 7, "url": "http://h"})
        deadline = time.monotonic() + 5
        while len(results) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
    kinds = {item["kind"]: item for item in results}
    assert kinds["health"]["ready"] is True
    assert kinds["slots"]["payload"] == {"slots": [{}]}
    assert not worker.is_alive()


def test_gui_applies_probe_results_without_blocking_calls(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    monkeypatch.setattr(
        app_settings, "_settings_file", lambda: tmp_path / "settings.json"
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("urlopen must not run on the GUI thread")

    monkeypatch.setattr(qt_launcher.urllib.request, "urlopen", forbidden)
    window = qt_launcher.MainWindow(tmp_path, SETTINGS_DIR, start_background=False)
    submitted: list[dict] = []
    monkeypatch.setattr(
        window._probe_worker, "submit", lambda job: submitted.append(job) or True
    )

    class AliveProcess:
        proc = None

        def is_running(self) -> bool:
            return True

        def returncode(self):
            return None

    request = ControlRequest("switch", "model", timeout_s=10)
    record = {
        "id": 1,
        "proc": AliveProcess(),
        "port": 65530,
        "base_url": "http://127.0.0.1:65530",
        "client_base_url": "http://127.0.0.1:65530",
        "ready": False,
        "model": "loading",
        "control_model_id": "model",
        "control_requests": [request],
        "slots_api_enabled": True,
    }
    window._servers = [record]
    window._control_api_record = record
    window._refresh_server_combo()
    window._poll_server()  # queues the health probe, no HTTP on this thread
    assert [job["kind"] for job in submitted] == ["health"]
    token = submitted[0]["token"]
    window._on_probe_result({"kind": "health", "token": token, "ready": False})
    assert record["ready"] is False and not request.done
    window._on_probe_result({"kind": "health", "token": token, "ready": True})
    app.processEvents()
    assert record["ready"] is True and window._server_ready is True
    assert request.wait()["backend_url"] == "http://127.0.0.1:65530"
    window._poll_server()  # ready + slots enabled: a slots probe is queued
    assert [job["kind"] for job in submitted] == ["health", "slots"]
    window._on_probe_result(
        {
            "kind": "slots",
            "token": token,
            "payload": {"slots": [{"is_processing": True}, {"state": "idle"}]},
        }
    )
    assert record["slots_summary"] == "1/2 busy"
    # A result for a server that no longer exists is ignored.
    window._servers = []
    window._on_probe_result({"kind": "health", "token": token, "ready": True})
    window.close()
