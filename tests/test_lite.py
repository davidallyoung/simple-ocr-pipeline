from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import lite


def _item(text: str, x: float, y: float, w: float, h: float, conf: float | None) -> SimpleNamespace:
    return SimpleNamespace(text=text, x=x, y=y, width=w, height=h, confidence=conf)


def _page(page_num: int, items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(page_num=page_num, text_items=items)


def test_quad_from_rect() -> None:
    item = _item("hi", 10.0, 20.0, 30.0, 40.0, None)
    quad = lite._to_quad(item)
    assert quad.to_list() == [[10.0, 20.0], [40.0, 20.0], [40.0, 60.0], [10.0, 60.0]]


def test_quad_graceful_fallback() -> None:
    item = SimpleNamespace(text="x", x=None, y=None, width=None, height=None)
    quad = lite._to_quad(item)
    # fallback returns a zero quad; just ensure no exception and 4 points
    assert quad.to_list() == [[0.0, 0.0] for _ in range(4)]


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
    # box is a full quad in canonical point space
    assert result[0].lines[0].box.to_list() == [
        [10.0, 20.0],
        [60.0, 20.0],
        [60.0, 32.0],
        [10.0, 32.0],
    ]
    # LiteParse page dimensions carry through
    assert result[0].width is None and result[0].height is None


def test_parse_pdf_carries_page_dimensions(monkeypatch, tmp_path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF")

    page = _page(1, [_item("Hello", 10, 20, 50, 12, None)])
    page.width = 612.0
    page.height = 792.0

    class FakeParser:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            pass

        def parse(self, path: str) -> SimpleNamespace:  # noqa: ANN002
            return SimpleNamespace(pages=[page])

    monkeypatch.setattr(lite, "LiteParse", FakeParser)
    result = lite.parse_pdf(p, object(), use_ocr=False)
    assert result[0].width == 612.0
    assert result[0].height == 792.0


def _block(
    kind: str, x: float, y: float, w: float, h: float, text: str | None
) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind, text=text, bbox=SimpleNamespace(x=x, y=y, width=w, height=h)
    )


def test_parse_pdf_maps_layout_blocks_to_paragraphs(monkeypatch, tmp_path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF")

    page = _page(
        1,
        [_item("Title line", 10, 10, 100, 12, None), _item("Body", 10, 40, 200, 12, None)],
    )
    page.blocks = [
        _block("heading", 10, 10, 100, 12, "Title line"),
        _block("paragraph", 10, 40, 200, 12, "Body text"),
        _block("table", 0, 0, 50, 50, None),  # excluded kind
    ]

    class FakeParser:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            assert kwargs.get("extract_blocks") is True

        def parse(self, path: str) -> SimpleNamespace:  # noqa: ANN002
            return SimpleNamespace(pages=[page])

    monkeypatch.setattr(lite, "LiteParse", FakeParser)
    result = lite.parse_pdf(p, object(), use_ocr=False)
    paras = result[0].paragraphs
    assert paras is not None
    assert [pp.kind for pp in paras] == ["heading", "paragraph"]
    assert paras[0].text == "Title line"
    assert paras[0].box.xyxy == (10.0, 10.0, 110.0, 22.0)
    # confidence derives from member lines inside the block bbox
    assert paras[0].confidence == pytest.approx(1.0)


def test_parse_pdf_splits_oversized_block_at_paragraph_gaps(
    monkeypatch, tmp_path
) -> None:  # noqa: ANN001
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF")

    # Two flowing rows, then a blank-line-sized gap before the third row:
    # the provider block spans two true paragraphs and must be split.
    items = [
        _item("first row", 10, 10, 100, 12, None),
        _item("second row", 10, 22.2, 100, 12, None),
        _item("next para", 10, 46.0, 100, 12, None),
    ]
    page = _page(1, items)
    page.blocks = [_block("paragraph", 10, 10, 200, 48, "first row second row next para")]

    class FakeParser:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            pass

        def parse(self, path: str) -> SimpleNamespace:  # noqa: ANN002
            return SimpleNamespace(pages=[page])

    monkeypatch.setattr(lite, "LiteParse", FakeParser)
    result = lite.parse_pdf(p, object(), use_ocr=False)
    paras = result[0].paragraphs
    assert paras is not None
    assert [pp.text for pp in paras] == ["first row second row", "next para"]
    assert paras[0].box.xyxy == (10.0, 10.0, 210.0, 34.2)  # block x-span kept
    assert paras[1].box.xyxy == (10.0, 46.0, 210.0, 58.0)
    assert [pp.kind for pp in paras] == ["paragraph", "paragraph"]
    assert all(pp.confidence == pytest.approx(1.0) for pp in paras)


def test_block_with_flowing_lines_stays_whole(monkeypatch, tmp_path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF")

    # Rows without a paragraph-sized gap keep the provider's own text.
    items = [
        _item("row one", 10, 10, 100, 12, None),
        _item("row two", 10, 22.2, 100, 12, None),
    ]
    page = _page(1, items)
    page.blocks = [_block("paragraph", 10, 10, 200, 24.2, "provider text verbatim")]

    class FakeParser:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            pass

        def parse(self, path: str) -> SimpleNamespace:  # noqa: ANN002
            return SimpleNamespace(pages=[page])

    monkeypatch.setattr(lite, "LiteParse", FakeParser)
    result = lite.parse_pdf(p, object(), use_ocr=False)
    paras = result[0].paragraphs
    assert paras is not None
    assert len(paras) == 1
    assert paras[0].text == "provider text verbatim"
    assert paras[0].box.xyxy == (10.0, 10.0, 210.0, 34.2)


def test_parse_pdf_falls_back_to_geometric_paragraphs(monkeypatch, tmp_path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF")

    page = _page(
        1,
        [_item("one", 10, 10, 100, 12, None), _item("two", 10, 23, 100, 12, None)],
    )
    page.blocks = []  # no blocks -> geometric grouping

    class FakeParser:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            pass

        def parse(self, path: str) -> SimpleNamespace:  # noqa: ANN002
            return SimpleNamespace(pages=[page])

    monkeypatch.setattr(lite, "LiteParse", FakeParser)
    result = lite.parse_pdf(p, object(), use_ocr=False)
    paras = result[0].paragraphs
    assert paras is not None
    assert len(paras) == 1
    assert paras[0].text == "one two"
    assert paras[0].kind == "paragraph"
