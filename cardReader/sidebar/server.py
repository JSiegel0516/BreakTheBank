"""A small local web server for the browser dashboard, run from a daemon thread.

    server = SidebarServer("127.0.0.1", 8765)   # serves straight away; OSError if the port is taken
    print(server.url)
    server.publish(snapshot)                     # once per frame, from the reader loop
    for action in server.drain_actions(): ...    # the buttons pressed on the page since the last call
    server.close()

Routes:
    GET /          page.html
    GET /events    a Server-Sent Events stream: each event is one "data:" line of snapshot JSON
    POST /action   {"action": "<name>"} as JSON queues the name for drain_actions; 204 when queued.
                   Only a JSON body is taken: a browser will not send one from another site
                   without asking first, which this server never allows, so other web pages
                   cannot press buttons.

publish sends a snapshot to every open page only when it differs from the last one
sent, leaving "timing" out of the comparison, plus a heartbeat every HEARTBEAT_SECONDS
so the fps stays live. An idle table sends about two events a second. A page that
opens gets the last snapshot sent straight away. A page more than QUEUE_SIZE events
behind skips the newest until it catches up; the next heartbeat brings it up to date.
"""
from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("page.html")
HEARTBEAT_SECONDS = 0.5  # an unchanged snapshot is sent again this often
KEEPALIVE_SECONDS = 1.0  # a stream with nothing to send writes a comment this often, to notice a closed page
QUEUE_SIZE = 64          # events waiting per page, and actions waiting for the reader
MAX_ACTION_BYTES = 1024  # a longer POST /action body is refused


class SidebarServer:
    def __init__(self, host: str, port: int):
        """Start serving on host:port (port 0 picks a free one). Raises OSError if the port is taken."""
        self._clients: list[queue.Queue] = []
        self._actions: queue.Queue = queue.Queue(QUEUE_SIZE)
        self._lock = threading.Lock()
        self._last_state: str | None = None  # the last snapshot sent, without "timing"
        self._last_message: str | None = None
        self._last_sent = 0.0
        self._closing = threading.Event()
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.sidebar = self
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        """The page's address on this computer."""
        host, port = self._httpd.server_address[:2]
        if host == "0.0.0.0":
            host = "localhost"
        return f"http://{host}:{port}/"

    def publish(self, snapshot: dict):
        """Send snapshot to every open page if it changed, or if the heartbeat is due."""
        state = json.dumps({key: value for key, value in snapshot.items() if key != "timing"})
        now = time.monotonic()
        if state == self._last_state and now - self._last_sent < HEARTBEAT_SECONDS:
            return
        message = json.dumps(snapshot)
        with self._lock:
            self._last_state, self._last_message, self._last_sent = state, message, now
            for client in self._clients:
                with contextlib.suppress(queue.Full):
                    client.put_nowait(message)

    def drain_actions(self) -> list[str]:
        """The action names posted since the last call, oldest first. The names are not checked."""
        actions = []
        while True:
            try:
                actions.append(self._actions.get_nowait())
            except queue.Empty:
                return actions

    def close(self):
        """Stop serving and end every open stream."""
        self._closing.set()
        self._httpd.shutdown()
        self._httpd.server_close()

    def _subscribe(self) -> queue.Queue:
        """A new page's queue of messages, starting with the last snapshot sent."""
        client = queue.Queue(QUEUE_SIZE)
        with self._lock:
            if self._last_message is not None:
                client.put_nowait(self._last_message)
            self._clients.append(client)
        return client

    def _unsubscribe(self, client: queue.Queue):
        with self._lock:
            self._clients.remove(client)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (the name http.server calls)
        path = self.path.split("?")[0]
        if path == "/":
            self._page()
        elif path == "/events":
            self._events()
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802 (the name http.server calls)
        if self.path.split("?")[0] != "/action":
            self.send_error(404)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_error(415, "send the action as JSON")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_ACTION_BYTES:
            self.send_error(400, f"the body must be 1 to {MAX_ACTION_BYTES} bytes")
            return
        try:
            action = json.loads(self.rfile.read(length))["action"]
        except (ValueError, KeyError, TypeError):
            action = None
        if not isinstance(action, str):
            self.send_error(400, 'send {"action": "<name>"}')
            return
        try:
            self.server.sidebar._actions.put_nowait(action)
        except queue.Full:
            self.send_error(503, "the reader is not taking actions")
            return
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass  # the terminal is kept for card events

    def _page(self):
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _events(self):
        sidebar: SidebarServer = self.server.sidebar
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        client = sidebar._subscribe()
        try:
            while not sidebar._closing.is_set():
                try:
                    chunk = f"data: {client.get(timeout=KEEPALIVE_SECONDS)}\n\n"
                except queue.Empty:
                    chunk = ": keepalive\n\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page was closed or reloaded
        finally:
            sidebar._unsubscribe(client)
