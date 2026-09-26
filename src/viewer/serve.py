"""Serve a viewer folder over HTTP (browsers refuse to load scene.glb from a file:// page).

Windows maps ``.js`` to text/plain on some machines, and a module script served with the
wrong type is refused by the browser; the types are therefore set here, not taken from the
registry.
"""

from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
         ".json": "application/json", ".glb": "model/gltf-binary", ".png": "image/png", ".jpg": "image/jpeg",
         ".css": "text/css", "": "application/octet-stream"}

_running: dict[str, tuple[ThreadingHTTPServer, str]] = {}


class _Handler(SimpleHTTPRequestHandler):
    extensions_map = TYPES

    def log_message(self, format, *args):  # noqa: A002 - quiet: the CLI prints the URL once
        pass


def make_server(folder: Path, port: int = 0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    folder = Path(folder).resolve()
    if not (folder / "index.html").is_file():
        raise FileNotFoundError(f"{folder} has no index.html (build the viewer first)")
    return ThreadingHTTPServer((host, int(port)), functools.partial(_Handler, directory=str(folder)))


def url_of(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/"


def serve_background(folder: Path, port: int = 0) -> str:
    """Start (or reuse) a daemon-thread server for ``folder``; returns its URL."""
    key = str(Path(folder).resolve())
    if key in _running:
        return _running[key][1]
    try:
        server = make_server(folder, port)
    except OSError:                       # the configured port is taken: any free one
        server = make_server(folder, 0)
    threading.Thread(target=server.serve_forever, daemon=True, name=f"viewer:{Path(folder).name}").start()
    _running[key] = (server, url_of(server))
    return _running[key][1]


def stop_all() -> None:
    for server, _ in _running.values():
        server.shutdown()
        server.server_close()
    _running.clear()
