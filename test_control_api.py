from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple
from urllib.parse import urlsplit

import pytest

import app_settings
from control_api import ApiModel, ControlApiError, ControlApiServer, ControlRequest


TOKEN = "test-token-with-at-least-sixteen-characters"
ROOT = Path(__file__).resolve().parent


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": json.loads(body.decode("utf-8")),
            }
        )
        if self.path.endswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            time.sleep(0.7)
            self.wfile.write(b"data: second\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        response = json.dumps(
            {"model": self.server.requests[-1]["body"]["model"], "ok": True}  # type: ignore[attr-defined]
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


@pytest.fixture
def upstream() -> Iterator[Tuple[str, list[Dict[str, Any]]]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server.requests  # type: ignore[attr-defined]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    token: str | None = TOKEN,
    payload: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    headers: Dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = b""
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    result_headers = {key.casefold(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, json.loads(raw.decode("utf-8")), result_headers


def _model(model_id: str = "qwen-7b", *, runnable: bool = True) -> ApiModel:
    return ApiModel(
        id=model_id,
        name="Qwen 7B",
        path="C:/models/qwen.gguf",
        context_window=32768,
        max_tokens=8192,
        reasoning=True,
        input_types=("text", "image"),
        runnable=runnable,
        unavailable_reason="single-shot runner" if not runnable else "",
    )


def test_control_api_auth_catalogue_switch_status_and_stop() -> None:
    switches: list[str] = []
    stops: list[bool] = []

    def switch(model_id: str, _timeout: float) -> Dict[str, Any]:
        switches.append(model_id)
        return {
            "backend_url": "http://127.0.0.1:65530",
            "alias": "qwen",
        }

    def stop(_timeout: float) -> Dict[str, Any]:
        stops.append(True)
        return {"status": "stopped"}

    api = ControlApiServer(
        port=0,
        token=TOKEN,
        switch_callback=switch,
        stop_callback=stop,
    )
    api.update_models([_model(), _model("diffusion-cli", runnable=False)])
    api.start()
    try:
        status, body, _headers = _request(api.base_url, "GET", "/health", token=None)
        assert status == 200 and body["service"] == "autotuner-control-api"

        status, body, headers = _request(
            api.base_url, "GET", "/v1/models", token=None
        )
        assert status == 401
        assert body["error"]["code"] == "unauthorised"
        assert headers["www-authenticate"].startswith("Bearer")

        status, body, _headers = _request(api.base_url, "GET", "/v1/models")
        assert status == 200
        assert [item["id"] for item in body["data"]] == ["qwen-7b"]
        assert body["data"][0]["context_window"] == 32768
        assert body["data"][0]["input"] == ["text", "image"]
        assert "path" not in body["data"][0]

        status, body, _headers = _request(api.base_url, "GET", "/api/v1/models")
        assert status == 200
        assert {item["id"] for item in body["models"]} == {
            "qwen-7b",
            "diffusion-cli",
        }
        assert next(
            item for item in body["models"] if item["id"] == "diffusion-cli"
        )["runnable"] is False

        status, body, _headers = _request(
            api.base_url,
            "POST",
            "/api/v1/switch",
            payload={"model_id": "qwen-7b"},
        )
        assert status == 200 and body["status"] == "ready"
        assert body["active_model"] == "qwen-7b"
        assert switches == ["qwen-7b"]

        # Selecting the already-ready model is idempotent.
        status, _body, _headers = _request(
            api.base_url,
            "POST",
            "/api/v1/switch",
            payload={"model": "qwen-7b"},
        )
        assert status == 200 and switches == ["qwen-7b"]

        status, body, _headers = _request(
            api.base_url, "POST", "/api/v1/stop", payload={}
        )
        assert status == 200 and body["status"] == "stopped"
        assert stops == [True]
        assert api.status()["status"] == "idle"

        status, body, _headers = _request(
            api.base_url,
            "POST",
            "/api/v1/switch",
            payload={"model_id": "missing"},
        )
        assert status == 404 and body["error"]["code"] == "model_not_found"
    finally:
        api.stop()
    assert not api.running


def test_openai_proxy_switches_model_and_rewrites_only_backend_identity(upstream) -> None:
    upstream_url, requests = upstream

    def switch(model_id: str, _timeout: float) -> Dict[str, Any]:
        assert model_id == "qwen-7b"
        return {
            "backend_url": upstream_url,
            "alias": "AutoTuner Qwen Alias",
            "backend_api_key": "llama-secret",
        }

    api = ControlApiServer(port=0, token=TOKEN, switch_callback=switch)
    api.update_models([_model()])
    api.start()
    try:
        status, body, _headers = _request(
            api.base_url,
            "POST",
            "/v1/chat/completions",
            payload={
                "model": "qwen-7b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert status == 200 and body == {"model": "AutoTuner Qwen Alias", "ok": True}
        assert requests[0]["path"] == "/v1/chat/completions"
        assert requests[0]["body"]["model"] == "AutoTuner Qwen Alias"
        assert requests[0]["body"]["messages"][0]["content"] == "hello"
        assert requests[0]["headers"]["Authorization"] == "Bearer llama-secret"
        assert TOKEN not in json.dumps(requests[0])
    finally:
        api.stop()


def test_sse_proxy_flushes_first_event_without_buffering(upstream) -> None:
    upstream_url, _requests = upstream
    api = ControlApiServer(
        port=0,
        token=TOKEN,
        switch_callback=lambda _model, _timeout: {
            "backend_url": upstream_url,
            "alias": "stream-alias",
        },
    )
    api.update_models([_model()])
    api.start()
    parsed = urlsplit(api.base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    payload = json.dumps({"model": "qwen-7b", "stream": True}).encode("utf-8")
    started = time.monotonic()
    try:
        connection.request(
            "POST",
            "/v1/stream",
            body=payload,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.readline() == b"data: first\n"
        assert time.monotonic() - started < 0.55
        assert response.readline() == b"\n"
        assert response.readline() == b"data: second\n"
    finally:
        connection.close()
        api.stop()


def test_proxy_lease_blocks_conflicting_switch_without_truncating_stream(
    upstream,
) -> None:
    upstream_url, _requests = upstream
    switched: list[str] = []

    def switch(model_id: str, _timeout: float) -> Dict[str, Any]:
        switched.append(model_id)
        return {"backend_url": upstream_url, "alias": f"alias-{model_id}"}

    api = ControlApiServer(port=0, token=TOKEN, switch_callback=switch)
    api.update_models([_model("one"), _model("two")])
    api.start()
    parsed = urlsplit(api.base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    payload = json.dumps({"model": "one", "stream": True}).encode("utf-8")
    try:
        connection.request(
            "POST",
            "/v1/stream",
            body=payload,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.readline() == b"data: first\n"
        assert api.status()["inflight_requests"] == 1

        status, body, _headers = _request(
            api.base_url,
            "POST",
            "/api/v1/switch",
            payload={"model_id": "two"},
        )
        assert status == 409
        assert body["error"]["code"] == "model_busy"
        assert switched == ["one"]

        assert response.readline() == b"\n"
        assert response.readline() == b"data: second\n"
        assert response.read() == b"\n"
        deadline = time.monotonic() + 1
        while api.status()["inflight_requests"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert api.status()["inflight_requests"] == 0

        status, body, _headers = _request(
            api.base_url,
            "POST",
            "/api/v1/switch",
            payload={"model_id": "two"},
        )
        assert status == 200 and body["active_model"] == "two"
        assert switched == ["one", "two"]
    finally:
        connection.close()
        api.stop()


def test_control_api_settings_roundtrip_and_environment_overrides(
    tmp_path, monkeypatch
) -> None:
    settings_path = tmp_path / "autotuner_settings.json"
    monkeypatch.setattr(app_settings, "_settings_file", lambda: settings_path)
    for name in (
        "AUTOTUNER_CONTROL_API_ENABLED",
        "AUTOTUNER_CONTROL_API_PORT",
        "AUTOTUNER_CONTROL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert app_settings.get_control_api_enabled() is False
    assert app_settings.get_control_api_port() == 1233
    app_settings.set_control_api_enabled(True)
    app_settings.set_control_api_port(4321)
    first = app_settings.ensure_control_api_token()
    assert len(first) >= 32
    assert app_settings.ensure_control_api_token() == first
    second = app_settings.regenerate_control_api_token()
    assert second != first
    assert app_settings.get_control_api_enabled() is True
    assert app_settings.get_control_api_port() == 4321
    assert app_settings.get_control_api_token() == second
    assert second in settings_path.read_text(encoding="utf-8")

    previous = app_settings.load_settings()
    real_save = app_settings.save_settings
    monkeypatch.setattr(app_settings, "save_settings", lambda _settings: False)
    with pytest.raises(OSError, match="configuration"):
        app_settings.set_control_api_config(False, 9876, "replacement-token-long-enough")
    assert app_settings.load_settings() == previous
    monkeypatch.setattr(app_settings, "save_settings", real_save)

    monkeypatch.setenv("AUTOTUNER_CONTROL_API_ENABLED", "off")
    monkeypatch.setenv("AUTOTUNER_CONTROL_API_PORT", "9876")
    monkeypatch.setenv("AUTOTUNER_CONTROL_API_KEY", "environment-token-long-enough")
    assert app_settings.get_control_api_enabled() is False
    assert app_settings.get_control_api_port() == 9876
    assert app_settings.ensure_control_api_token() == "environment-token-long-enough"
    assert app_settings.regenerate_control_api_token() == "environment-token-long-enough"


def test_switches_are_serialized_across_concurrent_http_clients() -> None:
    state_lock = threading.Lock()
    active_callbacks = 0
    max_callbacks = 0
    order: list[str] = []

    def switch(model_id: str, _timeout: float) -> Dict[str, Any]:
        nonlocal active_callbacks, max_callbacks
        with state_lock:
            active_callbacks += 1
            max_callbacks = max(max_callbacks, active_callbacks)
            order.append(model_id)
        time.sleep(0.08)
        with state_lock:
            active_callbacks -= 1
        port = 65001 if model_id == "one" else 65002
        return {"backend_url": f"http://127.0.0.1:{port}", "alias": model_id}

    api = ControlApiServer(port=0, token=TOKEN, switch_callback=switch)
    api.update_models([_model("one"), _model("two")])
    barrier = threading.Barrier(3)
    errors: list[Exception] = []

    def invoke(model_id: str) -> None:
        try:
            barrier.wait()
            api.ensure_model(model_id)
        except Exception as exc:  # pragma: no cover - assertion captures details
            errors.append(exc)

    threads = [
        threading.Thread(target=invoke, args=("one",)),
        threading.Thread(target=invoke, args=("two",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
    assert not errors
    assert max_callbacks == 1
    assert sorted(order) == ["one", "two"]
    assert api.status()["active_model"] == order[-1]


def test_control_request_timeout_failure_and_loopback_guards() -> None:
    request = ControlRequest("switch", "model", timeout_s=0.1)
    with pytest.raises(ControlApiError, match="timeout") as timeout:
        request.wait()
    assert timeout.value.status == 504

    failed = ControlRequest("switch", "model")
    failed.fail("no memory", status=409, code="insufficient_memory")
    with pytest.raises(ControlApiError, match="no memory") as error:
        failed.wait()
    assert error.value.code == "insufficient_memory"

    with pytest.raises(ValueError, match="loopback"):
        ControlApiServer(
            host="0.0.0.0",
            port=1233,
            token=TOKEN,
            switch_callback=lambda _model, _timeout: {},
        )
    with pytest.raises(ValueError, match="16"):
        ControlApiServer(
            port=1233,
            token="short",
            switch_callback=lambda _model, _timeout: {},
        )


def test_pi_extension_runtime_discovery_and_control_port_precedence(tmp_path) -> None:
    pi_command = shutil.which("pi")
    if pi_command is None:
        pytest.skip("Pi is not installed")
    extension = ROOT / "integrations" / "pi" / "autotuner.ts"
    api = ControlApiServer(
        port=0,
        token=TOKEN,
        switch_callback=lambda _model, _timeout: {
            "backend_url": "http://127.0.0.1:65530",
            "alias": "test",
        },
    )
    api.update_models([_model("pi-runtime-model")])
    api.start()
    try:
        (tmp_path / "autotuner_settings.json").write_text(
            json.dumps(
                {
                    "control_api_port": 65529,
                    "control_api_token": TOKEN,
                }
            ),
            encoding="utf-8",
        )
        environment = {
            **os.environ,
            "AUTOTUNER_DATA_DIR": str(tmp_path),
            "AUTOTUNER_CONTROL_API_PORT": str(api.port),
        }
        result = subprocess.run(
            [
                pi_command,
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-context-files",
                "-e",
                str(extension),
                "--list-models",
                "autotuner",
            ],
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "autotuner" in result.stdout
        assert "pi-runtime-model" in result.stdout
        assert "Assertion failed" not in result.stderr
    finally:
        api.stop()


def test_pi_extension_typechecks_when_local_pi_types_are_available() -> None:
    tsc = shutil.which("tsc")
    pi_command = shutil.which("pi")
    if tsc is None or pi_command is None:
        pytest.skip("TypeScript or Pi is not installed")
    package = (
        Path(pi_command).parent
        / "node_modules"
        / "@earendil-works"
        / "pi-coding-agent"
    )
    declarations = package / "dist" / "index.d.ts"
    type_roots = package / "node_modules" / "@types"
    if not declarations.is_file() or not type_roots.is_dir():
        pytest.skip("Installed Pi TypeScript declarations are unavailable")

    with tempfile.TemporaryDirectory(prefix="autotuner-pi-types-") as directory:
        config = {
            "compilerOptions": {
                "target": "ES2022",
                "module": "NodeNext",
                "moduleResolution": "NodeNext",
                "strict": True,
                "noEmit": True,
                "skipLibCheck": True,
                "baseUrl": directory,
                "paths": {
                    "@earendil-works/pi-coding-agent": [str(declarations)]
                },
                "typeRoots": [str(type_roots)],
            },
            "files": [str(ROOT / "integrations" / "pi" / "autotuner.ts")],
        }
        config_path = Path(directory) / "tsconfig.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = subprocess.run(
            [tsc, "-p", str(config_path)],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
