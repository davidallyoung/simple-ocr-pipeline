from __future__ import annotations

import pytest
from textual.content import Content

from app import canvas, inspector, output
from app.geometry import Quad
from app.inspector import OcrInspector, PageCanvas, format_box_details


def _line(text: str, x: float, y: float, w: float, h: float, conf: float) -> output.Line:
    return output.Line(text=text, box=Quad.from_xywh(x, y, w, h), confidence=conf)


def _doc(
    pages: list[list[output.Line]], width: float = 612.0, height: float = 792.0
) -> dict:
    return {
        "source": "sample.pdf",
        "pages": [
            {
                "page": number + 1,
                "width": width,
                "height": height,
                "lines": [
                    {
                        "box": line.box.to_list(),
                        "text": line.text,
                        "confidence": line.confidence,
                    }
                    for line in lines
                ],
            }
            for number, lines in enumerate(pages)
        ],
    }


SAMPLE_DOC = _doc(
    [
        [_line("first box", 36.0, 48.0, 300.0, 20.0, conf=1.0)],
        [_line("second page", 36.0, 96.0, 300.0, 20.0, conf=0.5)],
    ]
)


@pytest.fixture
def inspector_app() -> OcrInspector:
    return OcrInspector(SAMPLE_DOC)


async def test_canvas_shows_page_content(inspector_app: OcrInspector) -> None:
    async with inspector_app.run_test() as pilot:
        canvas_widget = inspector_app.query_one("#canvas", PageCanvas)
        assert "first box" in str(canvas_widget.render())
        assert inspector_app.page_index == 0
        await pilot.pause()


async def test_status_line_shows_page_info(inspector_app: OcrInspector) -> None:
    async with inspector_app.run_test() as pilot:
        await pilot.pause()
        status = str(inspector_app.query_one("#status").render())
        assert "page 1/2" in status
        assert "hover a box" in status


async def test_tab_cycles_boxes_and_updates_status(
    inspector_app: OcrInspector,
) -> None:
    async with inspector_app.run_test() as pilot:
        await pilot.press("tab")
        await pilot.pause()
        assert inspector_app.hover_index == 0
        status = str(inspector_app.query_one("#status").render())
        assert "first box" in status
        assert "conf 1.00" in status
        assert "(36.0,48.0)" in status
        assert "(336.0,68.0)" in status

        await pilot.press("tab")
        await pilot.pause()
        assert inspector_app.hover_index == 0  # single box wraps to itself


async def test_hovered_box_brightens_in_canvas(inspector_app: OcrInspector) -> None:
    from textual.color import Color

    async with inspector_app.run_test() as pilot:
        canvas_widget = inspector_app.query_one("#canvas", PageCanvas)

        def has_bright_bg() -> bool:
            content = canvas_widget.render()
            assert isinstance(content, Content)
            return any(
                getattr(span.style, "background", None) == Color(46, 160, 66)
                for span in content.spans
            )

        await pilot.pause()
        assert not has_bright_bg()  # dark tint only while nothing is hovered
        await pilot.press("tab")
        await pilot.pause()
        assert has_bright_bg()  # hovered box swaps to full-brightness bg


async def test_page_navigation(inspector_app: OcrInspector) -> None:
    async with inspector_app.run_test() as pilot:
        await pilot.press("n")
        await pilot.pause()
        assert inspector_app.page_index == 1
        canvas_widget = inspector_app.query_one("#canvas", PageCanvas)
        assert "second page" in str(canvas_widget.render())

        await pilot.press("p")
        await pilot.pause()
        assert inspector_app.page_index == 0


async def test_next_page_wraps_at_end(inspector_app: OcrInspector) -> None:
    async with inspector_app.run_test() as pilot:
        await pilot.press("p")  # before the first page: no-op
        await pilot.pause()
        assert inspector_app.page_index == 0
        await pilot.press("n")
        await pilot.press("n")  # past the last page: no-op
        await pilot.pause()
        assert inspector_app.page_index == 1


async def test_quit_binding_exits(inspector_app: OcrInspector) -> None:
    async with inspector_app.run_test() as pilot:
        await pilot.press("q")
        await pilot.pause()
        assert not inspector_app.is_running


async def test_canvas_hit_testing_returns_box_index() -> None:
    page = output.Page(
        number=1,
        lines=[_line("sample", 100.0, 100.0, 200.0, 20.0, conf=0.8)],
        width=612.0,
        height=792.0,
    )
    rects, _rows, _cols = canvas.layout_page(page, cols=72)
    assert len(rects) == 1
    _, x0, y0, x1, y1 = rects[0]
    assert inspector.hit_index(rects, x0, y0) == 0  # top-left corner inside
    assert inspector.hit_index(rects, x1, y1) == 0  # bottom-right corner inside
    assert inspector.hit_index(rects, 0, 0) is None  # page top-left is empty


def test_format_box_details_rounds_coordinates() -> None:
    line = _line("hello world", 36.0, 48.0, 300.0, 20.0, conf=0.87)
    text = format_box_details(line)
    assert "hello world" in text
    assert "conf 0.87" in text
    assert "(36.0,48.0)(336.0,68.0) pt" in text


def test_format_box_details_prefers_non_paragraph_kinds() -> None:
    para = output.Paragraph(
        text="A heading", box=Quad.from_xywh(36.0, 48.0, 300.0, 20.0),
        confidence=1.0, kind="heading",
    )
    text = format_box_details(para)
    assert "[heading] A heading" in text
    plain = format_box_details(
        output.Paragraph(text="x", box=para.box, confidence=1.0, kind="paragraph")
    )
    assert "[" not in plain  # default kind is not annotated


async def test_hover_shows_paragraph_details() -> None:
    doc = _doc([[]])  # one page, no lines
    doc["pages"][0]["paragraphs"] = [
        {
            "box": [[36.0, 48.0], [336.0, 48.0], [336.0, 68.0], [36.0, 68.0]],
            "text": "merged paragraph",
            "confidence": 0.95,
            "kind": "paragraph",
        }
    ]
    async with OcrInspector(doc).run_test() as pilot:
        await pilot.press("tab")
        await pilot.pause()
        status = str(pilot.app.query_one("#status").render())
        assert "merged paragraph" in status
        assert "conf 0.95" in status
        assert "(36.0,48.0)" in status


def test_format_box_details_handles_empty_text() -> None:
    line = _line("", 36.0, 48.0, 300.0, 20.0, conf=1.0)
    assert "(empty)" in format_box_details(line)
