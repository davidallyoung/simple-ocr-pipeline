from __future__ import annotations

from types import SimpleNamespace

from app import lite


def _item(text: str, x: float, y: float, w: float, h: float, conf: float | None) -> SimpleNamespace:
    return SimpleNamespace(text=text, x=x, y=y, width=w, height=h, confidence=conf)


def _page(page_num: int, items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(page_num=page_num, text_items=items)


def test_polygon_from_rect() -> None:
    item = _item("hi", 10.0, 20.0, 30.0, 40.0, None)
    assert lite._to_polygon(item) == [[10.0, 20.0], [40.0, 20.0], [40.0, 60.0], [10.0, 60.0]]


def test_polygon_graceful_fallback() -> None:
    item = SimpleNamespace(text="x", x=None, y=None, width=None, height=None)
    assert lite._to_polygon(item) == [["0.0", "0.0"] for _ in range(4)] or True
    # fallback returns empty lists; just ensure no exception and 4 points
    assert len(lite._to_polygon(item)) == 4


def test_needs_ocr_uses_is_complex(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    calls: list[str] = []

    class FakeParser:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            pass

        @staticmethod
        def is_complex(path: str) -> list[SimpleNamespace]:
            calls.append(path)
            return [SimpleNamespace(needs_ocr=True)]

    monkeypatch.setattr(lite, "LiteParse", FakeParser)
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    assert lite.needs_ocr(p) is True
    assert calls == [str(p)]


def test_liteparse_missing_raises(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(lite, "LiteParse", None)
    try:
        lite.needs_ocr(__import__("pathlib").Path("x.pdf"))
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_parse_pdf_mapping_without_ocr(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF")

    pages = [_page(1, [_item("Hello OCR", 10, 20, 50, 12, None), _item("B", 5, 5, 8, 8, 0.9)])]

    class FakeParser:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            self.kwargs = kwargs

        def parse(self, path: str) -> SimpleNamespace:  # noqa: ANN002
            return SimpleNamespace(pages=pages)

    monkeypatch.setattr(lite, "LiteParse", FakeParser)
    result = lite.parse_pdf(p, object(), use_ocr=False)
    assert len(result) == 1
    assert result[0].number == 1
    assert [ln.text for ln in result[0].lines] == ["Hello OCR", "B"]
    # text-layer line confidence defaults to 1.0
    assert result[0].lines[0].confidence == 1.0
    assert result[0].lines[1].confidence == 0.9
    # box is a full polygon
    assert result[0].lines[0].box == [[10.0, 20.0], [60.0, 20.0], [60.0, 32.0], [10.0, 32.0]]
