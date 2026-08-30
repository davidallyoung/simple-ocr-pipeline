"""Embedded EasyOCR HTTP server implementing the LiteParse OCR API spec."""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

from PIL import Image
from werkzeug.serving import make_server


def build_app(engine: Any) -> Any:
    """Build a Flask app exposing ``POST /ocr`` around an EasyOCR engine.

    ``engine`` only needs a ``recognize(image) -> list[(box, text, conf)]``
    method (see app.engine.OcrEngine), so tests can inject a stub.
    """
    from flask import Flask, jsonify, request

    logging.getLogger("werkzeug").disabled = True
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.post("/ocr")
    def ocr() -> Any:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "missing 'file' multipart field"}), 400
        language = request.form.get("language")
        try:
            image = Image.open(file.stream).convert("RGB")
            raw = engine.recognize(image, language=language)
        except Exception as exc:  # surface engine failures to liteparse
            return jsonify({"error": str(exc)}), 500

        results = []
        for box, text, confidence in raw:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            results.append(
                {
                    "text": text,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": confidence,
                }
            )
        return jsonify({"results": results})

    return app


class EasyOcrHttpServer:
    """Runs the Flask /ocr endpoint on an ephemeral port in a daemon thread."""

    def __init__(self, engine: Any) -> None:
        self._app = build_app(engine)
        self._server = make_server("127.0.0.1", 0, self._app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._port = self._server.server_port
        self._thread.start()
        self._wait_ready()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/ocr"

    def _wait_ready(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=1):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("EasyOCR HTTP server failed to become ready")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


_server_holder: dict[str, EasyOcrHttpServer | None] = {"server": None}


def ensure_ocr_server(engine: Any) -> EasyOcrHttpServer:
    """Return a session-wide EasyOCR HTTP server singleton (lazily started)."""
    holder = _server_holder
    server = holder["server"]
    if server is None:
        server = EasyOcrHttpServer(engine)
        holder["server"] = server
    return server


def close_ocr_server() -> None:
    holder = _server_holder
    server = holder["server"]
    if server is not None:
        server.close()
        holder["server"] = None
