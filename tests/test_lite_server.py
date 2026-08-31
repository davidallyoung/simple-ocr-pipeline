from __future__ import annotations

import io

from PIL import Image

from app import lite_server
from app.geometry import Quad


class StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Image.Image, str | None]] = []

    def recognize(
        self, image: Image.Image, language: str | None = None
    ) -> list[tuple[Quad, str, float]]:
        self.calls.append((image, language))
        return [
            (
                Quad.from_quad([[10.0, 20.0], [110.0, 20.0], [110.0, 40.0], [10.0, 40.0]]),
                "hello",
                0.95,
            )
        ]


def _image_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _post_ocr(client, image: Image.Image, language: str = "en"):
    data = {
        "file": (io.BytesIO(_image_bytes(image)), "page.png"),
        "language": language,
    }
    return client.post("/ocr", data=data, content_type="multipart/form-data")


def test_ocr_endpoint_shape() -> None:
    engine = StubEngine()
    app = lite_server.build_app(engine)
    client = app.test_client()

    img = Image.new("RGB", (50, 30), "white")
    resp = _post_ocr(client, img, language="en")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["results"][0]["text"] == "hello"
    assert payload["results"][0]["confidence"] == 0.95
    assert payload["results"][0]["bbox"] == [10.0, 20.0, 110.0, 40.0]
    assert engine.calls and engine.calls[0][1] == "en"


def test_ocr_endpoint_missing_file() -> None:
    app = lite_server.build_app(StubEngine())
    resp = app.test_client().post("/ocr", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_server_url_and_close() -> None:
    engine = StubEngine()
    server = lite_server.EasyOcrHttpServer(engine)
    try:
        assert server.url.startswith("http://127.0.0.1:")
        assert "/ocr" in server.url
    finally:
        server.close()


def test_ensure_server_singleton() -> None:
    engine = StubEngine()
    lite_server.close_ocr_server()
    try:
        s1 = lite_server.ensure_ocr_server(engine)
        s2 = lite_server.ensure_ocr_server(engine)
        assert s1 is s2
    finally:
        lite_server.close_ocr_server()
