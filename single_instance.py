"""One running AutoTuner GUI per user and data folder.

Why
---
Windows 11 parks new notification-area icons in the hidden overflow menu.
With *Hide on close* enabled, clicking X therefore leaves an AutoTuner that
is running but invisible, and the next double-click on ``AutoTuner.exe``
started a **second** copy: two control-API servers fought over one port, two
processes appended to the same console log, and the user saw "a leftover
instance" in Task Manager. A second launch must instead bring the running
window back to the front and exit.

How
---
:class:`SingleInstanceGuard` listens on a :class:`QLocalServer` whose name is
derived from the user and the AutoTuner data folder, so two users on one
machine, or a portable ``AUTOTUNER_DATA_DIR`` override, never block each
other. A later launch fails to acquire the name, sends a short *activate*
message to the listener and quits; the primary instance emits
:attr:`SingleInstanceGuard.activate_requested` on its GUI thread.

* Windows uses a named pipe (``\\\\.\\pipe\\<name>``). It disappears with the
  owning process, so a crash never leaves a stale lock behind.
* Unix uses ``$TMPDIR/<name>``. A socket file left over from a crash is
  removed before listening — but only after nobody answered on it.

``AUTOTUNER_ALLOW_MULTIPLE_INSTANCES=1`` disables the guard for developers
who deliberately run a source checkout next to the frozen build.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import time
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QCoreApplication, QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

#: Message a secondary launch sends to the primary instance.
ACTIVATE_COMMAND = b"activate"

#: Environment switch that turns the guard off (any non-empty value except 0).
ALLOW_MULTIPLE_ENV = "AUTOTUNER_ALLOW_MULTIPLE_INSTANCES"

_CONNECT_TIMEOUT_MS = 1500
_CONNECT_ATTEMPT_MS = 200
_WRITE_TIMEOUT_MS = 1500
_DISCONNECT_TIMEOUT_MS = 500


def multiple_instances_allowed() -> bool:
    """Whether the developer override disables the single-instance guard."""
    value = os.environ.get(ALLOW_MULTIPLE_ENV, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def instance_key(data_dir: Path) -> str:
    """Return the local-server name for one user + data folder combination.

    The name must be a valid pipe/socket file name on every platform, so the
    identifying parts are hashed instead of embedded.
    """
    try:
        user = getpass.getuser()
    except Exception:
        user = str(getattr(os, "getuid", lambda: "user")())
    try:
        folder = str(Path(data_dir).expanduser().resolve())
    except OSError:
        folder = str(Path(data_dir).expanduser())
    if os.name == "nt":
        folder = os.path.normcase(folder)
    digest = hashlib.sha256(
        f"{user}\0{folder}".encode("utf-8", "surrogateescape")
    ).hexdigest()
    return f"AutoTuner-{digest[:20]}"


class SingleInstanceGuard(QObject):
    """Hold the per-user instance lock and receive *activate* requests."""

    #: Emitted on the primary instance when another launch asked it to show.
    activate_requested = pyqtSignal()

    def __init__(self, name: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._name = name
        self._server: Optional[QLocalServer] = None
        self._clients: List[QLocalSocket] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_primary(self) -> bool:
        return self._server is not None and self._server.isListening()

    # -- primary side ---------------------------------------------------
    def try_acquire(self) -> bool:
        """Become the primary instance; False when another one is alive."""
        if self.is_primary:
            return True
        if self._peer_answers():
            return False
        # Nobody answered, so a socket file that still exists (Unix, after a
        # crash) is stale. Windows named pipes vanish with their process and
        # removeServer() is a no-op there.
        QLocalServer.removeServer(self._name)
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        server.newConnection.connect(self._on_new_connection)
        if not server.listen(self._name):
            # Lost a race against a simultaneous launch; that one is primary.
            server.deleteLater()
            return False
        self._server = server
        return True

    def release(self) -> None:
        """Stop listening so the next launch can become primary."""
        server = self._server
        self._server = None
        for client in self._clients:
            try:
                client.abort()
                client.deleteLater()
            except RuntimeError:
                pass
        self._clients = []
        if server is not None:
            try:
                server.close()
                server.deleteLater()
            except RuntimeError:
                pass
        # QLocalServer.close() unlinks the Unix socket itself; removing again
        # is harmless and covers a server that never finished listening.
        QLocalServer.removeServer(self._name)

    def _on_new_connection(self) -> None:
        server = self._server
        if server is None:
            return
        while server.hasPendingConnections():
            client = server.nextPendingConnection()
            if client is None:
                break
            self._clients.append(client)
            client.readyRead.connect(lambda c=client: self._consume(c))
            client.disconnected.connect(lambda c=client: self._forget(c))
            # Data may already be buffered when the signal was connected.
            self._consume(client)

    def _consume(self, client: QLocalSocket) -> None:
        try:
            payload = bytes(client.readAll())
        except RuntimeError:
            return
        if not payload:
            return
        for line in payload.split(b"\n"):
            if line.strip().lower() == ACTIVATE_COMMAND:
                self.activate_requested.emit()
                break

    def _forget(self, client: QLocalSocket) -> None:
        # Drain whatever arrived together with the hang-up.
        self._consume(client)
        if client in self._clients:
            self._clients.remove(client)
        try:
            client.deleteLater()
        except RuntimeError:
            pass

    # -- secondary side -------------------------------------------------
    def _connect(self, socket: QLocalSocket) -> bool:
        """Connect with short retries.

        A Windows named-pipe server offers one pending instance at a time and
        re-arms the next one from its event loop. A client that arrives in
        between sees ``ERROR_PIPE_BUSY``; QLocalSocket does not retry that on
        its own, so a single ``waitForConnected`` could report a live primary
        as absent. Retrying for the full timeout closes that gap.
        """
        deadline = time.monotonic() + _CONNECT_TIMEOUT_MS / 1000.0
        while True:
            socket.connectToServer(self._name)
            if socket.waitForConnected(_CONNECT_ATTEMPT_MS):
                return True
            socket.abort()
            if time.monotonic() >= deadline:
                return False
            app = QCoreApplication.instance()
            if app is not None:
                app.processEvents()
            time.sleep(0.05)

    def _peer_answers(self) -> bool:
        socket = QLocalSocket()
        answered = self._connect(socket)
        if answered:
            socket.disconnectFromServer()
            if socket.state() != QLocalSocket.LocalSocketState.UnconnectedState:
                socket.waitForDisconnected(_DISCONNECT_TIMEOUT_MS)
        socket.abort()
        return answered

    def notify_running_instance(self) -> bool:
        """Ask the primary instance to show itself. True when it was reached."""
        if os.name == "nt":
            # A freshly started process may pass its right to take the
            # foreground on to another process; without this Windows only
            # flashes the taskbar entry of the already running AutoTuner.
            try:
                import ctypes

                ASFW_ANY = ctypes.c_int(-1)
                ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY)
            except Exception:
                pass
        socket = QLocalSocket()
        if not self._connect(socket):
            socket.abort()
            return False
        socket.write(ACTIVATE_COMMAND + b"\n")
        # Windows completes pipe writes through the event loop, so a bare
        # waitForBytesWritten() can report failure although the bytes are on
        # their way. Poll the outgoing buffer instead, pumping events between
        # attempts.
        deadline = time.monotonic() + _WRITE_TIMEOUT_MS / 1000.0
        while socket.bytesToWrite() > 0 and time.monotonic() < deadline:
            if socket.waitForBytesWritten(_CONNECT_ATTEMPT_MS):
                break
            app = QCoreApplication.instance()
            if app is not None:
                app.processEvents()
        delivered = socket.bytesToWrite() == 0
        socket.disconnectFromServer()
        if socket.state() != QLocalSocket.LocalSocketState.UnconnectedState:
            socket.waitForDisconnected(_DISCONNECT_TIMEOUT_MS)
        socket.abort()
        return delivered
