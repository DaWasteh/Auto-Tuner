"""Authenticated loopback model-control and OpenAI proxy API.

The HTTP server owns no Qt objects.  Request threads communicate with the GUI
through callbacks; the Qt launcher implements those callbacks with queued
signals and completion events.  This keeps every widget/process-list mutation on
the GUI thread while allowing ordinary OpenAI clients to block until a requested
model has loaded.
"""

from __future__ import annotations

import hmac
import http.client
import ipaddress
import json
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import SplitResult, urlsplit

from autotuner_version import VERSION

_CONTROL_BODY_LIMIT = 1024 * 1024
_MIN_SWITCH_TIMEOUT_S = 1.0
_MAX_SWITCH_TIMEOUT_S = 24 * 60 * 60.0
DISCOVERY_SCHEMA = 1
DISCOVERY_FILE_NAME = "control_api.json"
#: Canonical execution-backend identifiers advertised by ``/api/v1/runtimes``.
RUNTIME_BACKENDS = (
    "vulkan",
    "hip",
    "cuda",
    "sycl",
    "metal",
    "opencl",
    "cpu",
    "unknown",
)
_PROXY_BODY_LIMIT = 128 * 1024 * 1024
_COPY_BUFFER = 64 * 1024
_UPSTREAM_TIMEOUT_S = 900.0
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ControlApiError(RuntimeError):
    """An expected HTTP/control-plane failure with a stable status code."""

    def __init__(self, message: str, *, status: int = 500, code: str = "api_error"):
        super().__init__(message)
        self.status = int(status)
        self.code = code


@dataclass(frozen=True)
class ApiModel:
    """A stable external model ID mapped to one scanned AutoTuner model."""

    id: str
    name: str
    path: str
    context_window: int
    max_tokens: int = 16_384
    reasoning: bool = False
    input_types: Tuple[str, ...] = ("text",)
    runnable: bool = True
    unavailable_reason: str = ""
    # Optional descriptive fields for control clients (benchmark campaigns,
    # model switchers). Unknown values are advertised as ``null``.
    size_bytes: int = 0
    quant: str = ""
    params_b: float = 0.0
    family: str = ""
    architecture: str = ""
    default_runtime_id: str = ""

    def openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "created": 0,
            "owned_by": "autotuner",
            "name": self.name,
            "context_window": max(1, int(self.context_window)),
            "max_tokens": max(1, int(self.max_tokens)),
            "reasoning": bool(self.reasoning),
            "input": list(self.input_types or ("text",)),
        }

    def control_dict(self) -> Dict[str, Any]:
        return {
            **self.openai_dict(),
            "path": self.path,
            "runnable": bool(self.runnable),
            "unavailable_reason": self.unavailable_reason,
            "size_bytes": int(self.size_bytes) if self.size_bytes > 0 else None,
            "quant": self.quant or None,
            "params_b": float(self.params_b) if self.params_b > 0 else None,
            "family": self.family or None,
            "architecture": self.architecture or None,
            "default_runtime_id": self.default_runtime_id or None,
        }


@dataclass(frozen=True)
class ApiRuntime:
    """One discovered llama-server build selectable through ``/api/v1/switch``."""

    id: str
    label: str
    server_binary: str
    backend: str = "unknown"
    build: Optional[int] = None
    build_info: str = ""
    is_default: bool = False
    available: bool = True
    unavailable_reason: str = ""

    def runtime_dict(self) -> Dict[str, Any]:
        backend = str(self.backend or "unknown").strip().lower()
        if backend not in RUNTIME_BACKENDS:
            backend = "unknown"
        return {
            "id": self.id,
            "label": self.label,
            "server_binary": self.server_binary,
            "backend": backend,
            "build": f"b{int(self.build)}" if self.build else None,
            "build_number": int(self.build) if self.build else None,
            "build_info": self.build_info or None,
            "is_default": bool(self.is_default),
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class ControlRequest:
    """One request handed from an HTTP thread to the Qt GUI thread."""

    action: str
    model_id: str = ""
    timeout_s: float = 300.0
    runtime_id: str = ""
    created_at: float = field(default_factory=time.monotonic, init=False)
    _event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _result: Optional[Dict[str, Any]] = field(default=None, init=False)
    _error: Optional[ControlApiError] = field(default=None, init=False)

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def deadline(self) -> float:
        return self.created_at + max(0.1, float(self.timeout_s))

    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._result = dict(result or {})
            self._event.set()

    def fail(
        self,
        message: str,
        *,
        status: int = 500,
        code: str = "switch_failed",
    ) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._error = ControlApiError(message, status=status, code=code)
            self._event.set()

    def wait(self) -> Dict[str, Any]:
        if not self._event.wait(max(0.1, float(self.timeout_s))):
            timeout = ControlApiError(
                "AutoTuner did not finish the model operation before the timeout.",
                status=504,
                code="switch_timeout",
            )
            timed_out = False
            with self._lock:
                if not self._event.is_set():
                    self._error = timeout
                    self._event.set()
                    timed_out = True
            if timed_out:
                raise timeout
        if self._error is not None:
            raise self._error
        return dict(self._result or {})


#: ``switch_callback(model_id, timeout_s, options)``. ``options`` carries
#: ``runtime_id`` (empty keeps the GUI's selected build) and ``timeout_s``.
#: The result must contain ``backend_url`` and may contain ``alias``,
#: ``backend_api_key``, and a JSON-serialisable ``detail`` snapshot that is
#: merged into ``/api/v1/status`` while the model stays active.
SwitchCallback = Callable[[str, float, Dict[str, Any]], Dict[str, Any]]
StopCallback = Callable[[float], Dict[str, Any]]
LogCallback = Callable[[str], None]
RuntimeProbe = Callable[[ApiRuntime], ApiRuntime]


@dataclass
class ProxyLease:
    """Immutable backend snapshot that prevents model switches while in use."""

    backend_url: str
    alias: str
    backend_api_key: Optional[str]
    model_id: str
    _api: "ControlApiServer" = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._api._release_proxy_lease()


class _ControlHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], api: "ControlApiServer") -> None:
        self.api = api
        super().__init__(address, _ControlRequestHandler)


class ControlApiServer:
    """A small dependency-free, bearer-authenticated loopback gateway."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 1233,
        token: str,
        switch_callback: SwitchCallback,
        stop_callback: Optional[StopCallback] = None,
        log_callback: Optional[LogCallback] = None,
        switch_timeout_s: float = 300.0,
        runtime_probe: Optional[RuntimeProbe] = None,
    ) -> None:
        self.host = _validated_loopback_host(host)
        self.port = int(port)
        if self.port < 0 or self.port > 65_535:
            raise ValueError("API port must be between 0 and 65535")
        if not isinstance(token, str) or len(token) < 16:
            raise ValueError("API token must contain at least 16 characters")
        self._token = token
        self._switch_callback = switch_callback
        self._stop_callback = stop_callback
        self._log_callback = log_callback
        self._runtime_probe = runtime_probe
        self.switch_timeout_s = max(1.0, float(switch_timeout_s))
        self._catalogue_lock = threading.RLock()
        self._models: Dict[str, ApiModel] = {}
        self._runtimes: Dict[str, ApiRuntime] = {}
        self._default_runtime_id = ""
        self._state_lock = threading.RLock()
        self._active_model_id = ""
        self._loading_model_id = ""
        self._active_runtime_id = ""
        self._backend_url = ""
        self._backend_alias = ""
        self._backend_api_key: Optional[str] = None
        self._active_detail: Dict[str, Any] = {}
        self._active_since = 0.0
        self._switch_lock = threading.Lock()
        self._inflight_requests = 0
        self._httpd: Optional[_ControlHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def token(self) -> str:
        return self._token

    @property
    def base_url(self) -> str:
        port = self.port
        if self._httpd is not None:
            port = int(self._httpd.server_address[1])
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{port}"

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> str:
        if self.running:
            return self.base_url
        httpd = _ControlHTTPServer((self.host, self.port), self)
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        thread = threading.Thread(
            target=httpd.serve_forever,
            name="AutoTunerControlAPI",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        self._log(f"listening on {self.base_url}")
        return self.base_url

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd = None
        self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._log("stopped")

    def update_models(self, models: Iterable[ApiModel]) -> None:
        catalogue: Dict[str, ApiModel] = {}
        for model in models:
            if not isinstance(model, ApiModel):
                raise TypeError("catalogue entries must be ApiModel instances")
            if not model.id or model.id in catalogue:
                raise ValueError(f"duplicate or empty API model ID: {model.id!r}")
            catalogue[model.id] = model
        with self._catalogue_lock:
            self._models = catalogue

    def update_runtimes(
        self, runtimes: Iterable[ApiRuntime], default_runtime_id: str = ""
    ) -> None:
        catalogue: Dict[str, ApiRuntime] = {}
        for runtime in runtimes:
            if not isinstance(runtime, ApiRuntime):
                raise TypeError("runtime entries must be ApiRuntime instances")
            if not runtime.id or runtime.id in catalogue:
                raise ValueError(f"duplicate or empty runtime ID: {runtime.id!r}")
            catalogue[runtime.id] = runtime
        default_id = str(default_runtime_id or "")
        if default_id not in catalogue:
            default_id = next(
                (item.id for item in catalogue.values() if item.is_default), ""
            )
        with self._catalogue_lock:
            self._runtimes = catalogue
            self._default_runtime_id = default_id

    def runtimes(self, *, probe: bool = False) -> List[ApiRuntime]:
        with self._catalogue_lock:
            values = list(self._runtimes.values())
        if probe and self._runtime_probe is not None:
            probed: List[ApiRuntime] = []
            for runtime in values:
                try:
                    probed.append(self._runtime_probe(runtime))
                except Exception as exc:  # pragma: no cover - defensive
                    self._log(f"runtime probe failed for {runtime.id}: {exc}")
                    probed.append(runtime)
            values = probed
        return values

    def runtime(self, runtime_id: str) -> Optional[ApiRuntime]:
        with self._catalogue_lock:
            return self._runtimes.get(runtime_id)

    @property
    def default_runtime_id(self) -> str:
        with self._catalogue_lock:
            return self._default_runtime_id

    def models(self, *, include_unrunnable: bool = False) -> List[ApiModel]:
        with self._catalogue_lock:
            values = list(self._models.values())
        if not include_unrunnable:
            values = [model for model in values if model.runnable]
        return sorted(values, key=lambda model: (model.name.casefold(), model.id))

    def model(self, model_id: str) -> Optional[ApiModel]:
        with self._catalogue_lock:
            return self._models.get(model_id)

    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            ready = bool(
                self._active_model_id
                and self._backend_url
                and not self._loading_model_id
            )
            state = "loading" if self._loading_model_id else "ready" if ready else "idle"
            detail = dict(self._active_detail) if ready else {}
            payload: Dict[str, Any] = {
                "status": state,
                "active_model": self._active_model_id or None,
                "loading_model": self._loading_model_id or None,
                "active_since": self._active_since or None,
                "inflight_requests": self._inflight_requests,
                "endpoint": self.base_url,
                "ready": ready,
                "backend_url": (self._backend_url or None) if ready else None,
                "alias": (self._backend_alias or None) if ready else None,
                "backend_api_key": self._backend_api_key if ready else None,
                "active_runtime": (self._active_runtime_id or None) if ready else None,
                "default_runtime_id": self.default_runtime_id or None,
                "pid": None,
                "log_path": None,
                "runtime": None,
                "model": None,
                "launch": None,
                "devices": None,
                "env": None,
                "command_line": None,
            }
            for key in (
                "pid",
                "log_path",
                "runtime",
                "model",
                "launch",
                "devices",
                "env",
                "command_line",
            ):
                if key in detail:
                    payload[key] = detail[key]
            return payload

    def clear_active(self, model_id: Optional[str] = None) -> None:
        with self._state_lock:
            if model_id:
                owned_ids = {
                    value
                    for value in (self._active_model_id, self._loading_model_id)
                    if value
                }
                if owned_ids and model_id not in owned_ids:
                    return
            self._active_model_id = ""
            self._loading_model_id = ""
            self._active_runtime_id = ""
            self._backend_url = ""
            self._backend_alias = ""
            self._backend_api_key = None
            self._active_detail = {}
            self._active_since = 0.0

    def _runnable_model(self, model_id: str) -> ApiModel:
        model = self.model(model_id)
        if model is None:
            raise ControlApiError(
                f"Unknown AutoTuner model ID: {model_id}",
                status=404,
                code="model_not_found",
            )
        if not model.runnable:
            raise ControlApiError(
                model.unavailable_reason or f"Model {model_id} cannot run as a server.",
                status=409,
                code="model_not_runnable",
            )
        return model

    def _validated_runtime_id(self, runtime_id: Any) -> str:
        if runtime_id is None:
            return ""
        if not isinstance(runtime_id, str):
            raise ControlApiError(
                "runtime_id must be a string.", status=400, code="invalid_request"
            )
        value = runtime_id.strip()
        if not value:
            return ""
        with self._catalogue_lock:
            known = self._runtimes.get(value)
            catalogue_published = bool(self._runtimes)
        if catalogue_published and known is None:
            raise ControlApiError(
                f"Unknown llama-server runtime ID: {value}. "
                "Read GET /api/v1/runtimes for the selectable builds.",
                status=409,
                code="runtime_unavailable",
            )
        if known is not None and not known.available:
            raise ControlApiError(
                known.unavailable_reason or f"Runtime {value} is not available.",
                status=409,
                code="runtime_unavailable",
            )
        return value

    @staticmethod
    def _validated_timeout(timeout_s: Any, default: float) -> float:
        if timeout_s is None:
            return float(default)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ControlApiError(
                "timeout_s must be a number of seconds.",
                status=400,
                code="invalid_request",
            )
        value = float(timeout_s)
        if value != value or value <= 0:  # NaN or non-positive
            raise ControlApiError(
                "timeout_s must be a positive number of seconds.",
                status=400,
                code="invalid_request",
            )
        return min(_MAX_SWITCH_TIMEOUT_S, max(_MIN_SWITCH_TIMEOUT_S, value))

    def ensure_model(
        self,
        model_id: str,
        *,
        runtime_id: Any = None,
        timeout_s: Any = None,
    ) -> Dict[str, Any]:
        model = self._runnable_model(model_id)
        runtime = self._validated_runtime_id(runtime_id)
        timeout = self._validated_timeout(timeout_s, self.switch_timeout_s)
        with self._switch_lock:
            return self._ensure_model_locked(
                model_id, model, runtime_id=runtime, timeout_s=timeout
            )

    def _ensure_model_locked(
        self,
        model_id: str,
        model: ApiModel,
        *,
        runtime_id: str = "",
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        timeout = float(timeout_s) if timeout_s is not None else self.switch_timeout_s
        with self._state_lock:
            if (
                self._active_model_id == model_id
                and self._backend_url
                and not self._loading_model_id
                and (not runtime_id or runtime_id == self._active_runtime_id)
            ):
                return self.status()
            if self._inflight_requests:
                raise ControlApiError(
                    "The active model has in-flight requests; retry the model switch "
                    "after they finish.",
                    status=409,
                    code="model_busy",
                )
            self._active_model_id = ""
            self._active_runtime_id = ""
            self._backend_url = ""
            self._backend_alias = ""
            self._backend_api_key = None
            self._active_detail = {}
            self._active_since = 0.0
            self._loading_model_id = model_id
        self._log(
            f"switch requested: {model_id}"
            + (f" on runtime {runtime_id}" if runtime_id else "")
        )
        try:
            result = self._switch_callback(
                model_id, timeout, {"runtime_id": runtime_id, "timeout_s": timeout}
            )
            backend_url = _validated_backend_url(str(result.get("backend_url", "")))
            alias = str(result.get("alias", "") or model.name)
            backend_api_key = result.get("backend_api_key")
            if backend_api_key is not None and not isinstance(backend_api_key, str):
                raise ControlApiError(
                    "AutoTuner returned an invalid backend API key.",
                    status=500,
                    code="invalid_backend",
                )
            detail = _json_safe(result.get("detail") or {})
            if not isinstance(detail, dict):
                detail = {}
            runtime_info = detail.get("runtime")
            active_runtime = ""
            if isinstance(runtime_info, dict):
                active_runtime = str(runtime_info.get("id") or "")
            if not active_runtime:
                active_runtime = runtime_id or self.default_runtime_id
            with self._state_lock:
                self._active_model_id = model_id
                self._loading_model_id = ""
                self._active_runtime_id = active_runtime
                self._backend_url = backend_url
                self._backend_alias = alias
                self._backend_api_key = backend_api_key
                self._active_detail = detail
                self._active_since = time.time()
            self._log(f"model ready: {model_id} -> {backend_url}")
            return self.status()
        except Exception:
            with self._state_lock:
                self._loading_model_id = ""
            raise

    def acquire_proxy_lease(self, model_id: str = "") -> ProxyLease:
        """Atomically select/snapshot a backend and block conflicting switches."""
        model = self._runnable_model(model_id) if model_id else None
        with self._switch_lock:
            if model is not None:
                self._ensure_model_locked(model_id, model)
            with self._state_lock:
                if not self._active_model_id or not self._backend_url:
                    raise ControlApiError(
                        "No AutoTuner model is active. Include a model ID in the request "
                        "or call /api/v1/switch first.",
                        status=409,
                        code="no_active_model",
                    )
                self._inflight_requests += 1
                return ProxyLease(
                    backend_url=self._backend_url,
                    alias=self._backend_alias,
                    backend_api_key=self._backend_api_key,
                    model_id=self._active_model_id,
                    _api=self,
                )

    def _release_proxy_lease(self) -> None:
        with self._state_lock:
            self._inflight_requests = max(0, self._inflight_requests - 1)

    def stop_active(self) -> Dict[str, Any]:
        if self._stop_callback is None:
            raise ControlApiError(
                "Stopping through the control API is unavailable.",
                status=501,
                code="stop_unavailable",
            )
        with self._switch_lock:
            with self._state_lock:
                if self._inflight_requests:
                    raise ControlApiError(
                        "The active model has in-flight requests; retry stop after "
                        "they finish.",
                        status=409,
                        code="model_busy",
                    )
            result = self._stop_callback(self.switch_timeout_s)
            self.clear_active()
            return result

    def authorised(self, value: Optional[str], x_api_key: Optional[str]) -> bool:
        candidate = ""
        if isinstance(value, str) and value.lower().startswith("bearer "):
            candidate = value[7:].strip()
        elif isinstance(x_api_key, str):
            candidate = x_api_key.strip()
        return bool(candidate) and hmac.compare_digest(candidate, self._token)

    def _log(self, message: str) -> None:
        if self._log_callback is not None:
            try:
                self._log_callback(f"[Control API] {message}")
            except Exception:
                pass


class _ControlRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"AutoTuner/{VERSION}"
    sys_version = ""

    @property
    def api(self) -> ControlApiServer:
        return self.server.api  # type: ignore[attr-defined,no-any-return]

    def log_message(self, _format: str, *_args: object) -> None:
        # Request content and query strings must never enter the application log.
        return

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        try:
            route = urlsplit(self.path).path
            if route == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "service": "autotuner-control-api",
                        "version": VERSION,
                    },
                )
                return
            if not self.api.authorised(
                self.headers.get("Authorization"), self.headers.get("X-API-Key")
            ):
                self._send_error(
                    401,
                    "A valid AutoTuner bearer token is required.",
                    "unauthorised",
                    authenticate=True,
                )
                return
            if route == "/v1/models" and self.command == "GET":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [model.openai_dict() for model in self.api.models()],
                    },
                )
                return
            if route == "/api/v1/models" and self.command == "GET":
                self._send_json(
                    200,
                    {
                        "models": [
                            model.control_dict()
                            for model in self.api.models(include_unrunnable=True)
                        ],
                        **self.api.status(),
                    },
                )
                return
            if route == "/api/v1/status" and self.command == "GET":
                self._send_json(200, self.api.status())
                return
            if route == "/api/v1/runtimes" and self.command == "GET":
                self._send_json(
                    200,
                    {
                        "runtimes": [
                            runtime.runtime_dict()
                            for runtime in self.api.runtimes(probe=True)
                        ],
                        "default_runtime_id": self.api.default_runtime_id or None,
                        "active_runtime": self.api.status().get("active_runtime"),
                    },
                )
                return
            if route == "/api/v1/switch" and self.command == "POST":
                payload = self._read_json(_CONTROL_BODY_LIMIT)
                model_id = payload.get("model_id", payload.get("model"))
                if not isinstance(model_id, str) or not model_id:
                    raise ControlApiError(
                        "model_id must be a non-empty string.",
                        status=400,
                        code="invalid_request",
                    )
                self._send_json(
                    200,
                    self.api.ensure_model(
                        model_id,
                        runtime_id=payload.get("runtime_id", payload.get("runtime")),
                        timeout_s=payload.get("timeout_s"),
                    ),
                )
                return
            if route == "/api/v1/stop" and self.command == "POST":
                # Consume/validate an optional empty object so accidental large
                # request bodies cannot be left unread on a keep-alive socket.
                if int(self.headers.get("Content-Length", "0") or 0):
                    self._read_json(_CONTROL_BODY_LIMIT)
                self._send_json(200, self.api.stop_active())
                return
            if route.startswith("/v1/"):
                self._proxy_openai_request()
                return
            self._send_error(404, "Endpoint not found.", "not_found")
        except ControlApiError as exc:
            self._send_error(exc.status, str(exc), exc.code)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.api._log(f"request failed: {type(exc).__name__}: {exc}")
            self._send_error(500, "Internal control API error.", "internal_error")

    def _read_body(self, limit: int) -> bytes:
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise ControlApiError(
                "Transfer-Encoding request bodies are not supported.",
                status=400,
                code="invalid_request",
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ControlApiError(
                "Invalid Content-Length.", status=400, code="invalid_request"
            ) from exc
        if length < 0 or length > limit:
            raise ControlApiError(
                "Request body is too large.", status=413, code="request_too_large"
            )
        return self.rfile.read(length)

    def _read_json(self, limit: int) -> Dict[str, Any]:
        body = self._read_body(limit)
        try:
            payload = json.loads(body.decode("utf-8") if body else "{}")
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ControlApiError(
                "Request body must be valid UTF-8 JSON.",
                status=400,
                code="invalid_json",
            ) from exc
        if not isinstance(payload, dict):
            raise ControlApiError(
                "JSON request body must be an object.",
                status=400,
                code="invalid_request",
            )
        return payload

    def _proxy_openai_request(self) -> None:
        body = self._read_body(_PROXY_BODY_LIMIT)
        content_type = self.headers.get("Content-Type", "")
        requested_model = ""
        payload: Optional[Dict[str, Any]] = None
        if body and "json" in content_type.casefold():
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ControlApiError(
                    "OpenAI request body must be valid UTF-8 JSON.",
                    status=400,
                    code="invalid_json",
                ) from exc
            if not isinstance(parsed, dict):
                raise ControlApiError(
                    "OpenAI request body must be a JSON object.",
                    status=400,
                    code="invalid_request",
                )
            payload = parsed
            value = payload.get("model")
            if value is not None and not isinstance(value, str):
                raise ControlApiError(
                    "model must be a string.", status=400, code="invalid_request"
                )
            requested_model = value or ""

        lease = self.api.acquire_proxy_lease(requested_model)
        connection: Optional[http.client.HTTPConnection] = None
        try:
            if payload is not None:
                # llama-server knows the launch alias, while external clients use
                # AutoTuner's stable catalogue ID. Never forward a path-like ID.
                payload["model"] = lease.alias
                body = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")

            target = urlsplit(lease.backend_url)
            connection = http.client.HTTPConnection(
                target.hostname,
                target.port,
                timeout=_UPSTREAM_TIMEOUT_S,
            )
            upstream_headers: Dict[str, str] = {}
            for key, value in self.headers.items():
                lower = key.casefold()
                if lower in _HOP_BY_HOP_HEADERS or lower in {
                    "host",
                    "content-length",
                    "authorization",
                    "x-api-key",
                }:
                    continue
                upstream_headers[key] = value
            upstream_headers["Content-Length"] = str(len(body))
            upstream_headers["Connection"] = "close"
            if lease.backend_api_key:
                upstream_headers["Authorization"] = (
                    f"Bearer {lease.backend_api_key}"
                )

            upstream_path = self.path
            if target.path and target.path != "/":
                upstream_path = (
                    target.path.rstrip("/") + "/" + self.path.lstrip("/")
                )
            response_started = False
            try:
                connection.request(
                    self.command, upstream_path, body=body, headers=upstream_headers
                )
                response = connection.getresponse()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    lower = key.casefold()
                    if lower in _HOP_BY_HOP_HEADERS or lower in {
                        "content-length",
                        "server",
                        "date",
                    }:
                        continue
                    self.send_header(key, value)
                # Upstream may be SSE/chunked or ordinary JSON. Closing the
                # client connection after streaming makes EOF the delimiter.
                self.send_header("Connection", "close")
                self.end_headers()
                response_started = True
                self.close_connection = True
                while True:
                    # ``read(amt)`` may wait for *amt* bytes across many HTTP
                    # chunks. ``read1`` forwards each available SSE chunk.
                    block = response.read1(_COPY_BUFFER)
                    if not block:
                        break
                    self.wfile.write(block)
                    self.wfile.flush()
            except (
                socket.timeout,
                ConnectionRefusedError,
                http.client.HTTPException,
            ) as exc:
                if response_started:
                    self.close_connection = True
                    return
                raise ControlApiError(
                    f"The active llama server is unavailable: {exc}",
                    status=502,
                    code="backend_unavailable",
                ) from exc
        finally:
            if connection is not None:
                connection.close()
            lease.release()

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_error(
        self,
        status: int,
        message: str,
        code: str,
        *,
        authenticate: bool = False,
    ) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "autotuner_control_error",
                    "code": code,
                }
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        if authenticate:
            self.send_header("WWW-Authenticate", 'Bearer realm="AutoTuner"')
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _validated_loopback_host(host: str) -> str:
    value = str(host or "").strip()
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        if value.casefold() == "localhost":
            return "127.0.0.1"
        raise ValueError("control API host must be a loopback address") from None
    if not address.is_loopback:
        raise ValueError("control API host must be a loopback address")
    return address.compressed


def _validated_backend_url(value: str) -> str:
    parsed: SplitResult = urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or parsed.query or parsed.fragment:
        raise ControlApiError(
            "AutoTuner returned an invalid llama-server URL.",
            status=500,
            code="invalid_backend",
        )
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.casefold() != "localhost":
            raise ControlApiError(
                "The llama-server proxy target must be loopback-only.",
                status=500,
                code="invalid_backend",
            ) from None
    else:
        if not address.is_loopback:
            raise ControlApiError(
                "The llama-server proxy target must be loopback-only.",
                status=500,
                code="invalid_backend",
            )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ControlApiError(
            "AutoTuner returned an invalid llama-server port.",
            status=500,
            code="invalid_backend",
        ) from exc
    if port is None:
        raise ControlApiError(
            "AutoTuner returned a llama-server URL without a port.",
            status=500,
            code="invalid_backend",
        )
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"http://{host}:{port}{parsed.path.rstrip('/')}"


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Coerce a GUI-provided snapshot into plain JSON-serialisable values."""
    if _depth > 8:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, _depth + 1) for item in value]
    return str(value)


def discovery_payload(
    *,
    enabled: bool,
    port: int,
    token: str = "",
    version: str = VERSION,
    pid: Optional[int] = None,
    started_at: str = "",
    host: str = "127.0.0.1",
) -> Dict[str, Any]:
    """Build the small client-discovery document written next to the settings.

    External clients (Pi, the Supercalc benchmark) read this file instead of
    parsing the potentially very large ``autotuner_settings.json``. The token
    is only present while the gateway is enabled.
    """
    port_value = int(port)
    payload: Dict[str, Any] = {
        "schema": DISCOVERY_SCHEMA,
        "enabled": bool(enabled),
        "base_url": f"http://{host}:{port_value}",
        "port": port_value,
        "version": str(version),
        "pid": int(pid) if pid else None,
        "started_at": str(started_at or ""),
    }
    if enabled and token:
        payload["token"] = str(token)
    return payload


def write_discovery_file(path: Path, payload: Dict[str, Any]) -> bool:
    """Atomically write *payload* to *path* with owner-only permissions."""
    target = Path(path)
    tmp: Optional[Path] = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            # Windows has no POSIX mode bits; the per-user profile directory
            # already restricts access to the owner.
            pass
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, target)
        tmp = None
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def read_discovery_file(path: Path) -> Dict[str, Any]:
    """Return the discovery document, or an empty dict when absent/invalid."""
    try:
        target = Path(path)
        if target.stat().st_size > 64 * 1024:
            return {}
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}
