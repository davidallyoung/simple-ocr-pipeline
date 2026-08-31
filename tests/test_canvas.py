from __future__ import annotations

from app import canvas, output
from app.geometry import Quad


def _line(text: str, x: float, y: float, w: float, h: float, conf: float) -> output.Line:
    return output.Line(text=text, box=Quad.from_xywh(x, y, w, h), confidence=conf)


def _page(
    lines: list[output.Line], width: float | None = 612.0, height: float | None = 792.0
) -> output.Page:
    return output.Page(number=1, lines=lines, width=width, height=height)


def test_conf_rich_style_thresholds() -> None:
    assert canvas.conf_rich_style(0.5) == "bright_red"
    assert canvas.conf_rich_style(0.8) == "bright_yellow"
    assert canvas.conf_rich_style(0.95) == "green"


def test_short_box_renders_compact_border_at_position() -> None:
    # 10pt-tall box on a letter page at 72 cols maps to 2 grid rows -> compact.
    page = _page([_line("Sample", 190.0, 48.0, 138.0, 10.0, conf=1.0)])
    text = canvas.render_page(page, cols=72)
    rows = str(text).split("\n")

    sx = 72 / 612.0
    sy = len(rows) / 792.0
    top = int(48.0 * sy)
    left = int(190.0 * sx)
    right = int(328.0 * sx)
    assert rows[top][left] == "┌" and rows[top][right] == "┐"
    assert rows[top + 1][left] == "│" and rows[top + 1][right] == "│"
    assert rows[top + 1][left + 1:].startswith("Sample")
    styles = {span.style for span in text.spans}
    assert "bold green" in styles


def test_narrow_box_falls_back_to_underlined_text() -> None:
    # A 1pt-wide box is too narrow for a border: colored underline only.
    page = _page([_line("tiny", 36.0, 48.0, 1.0, 20.0, conf=1.0)])
    text = canvas.render_page(page, cols=72)
    styles = {span.style for span in text.spans}
    assert "bold underline green" in styles
    assert "t" in text.plain  # first character still drawn


def test_subrow_box_still_gets_compact_border() -> None:
    # ~16pt box floors to a single grid row; it must still render boxed.
    page = _page([_line("I digress", 36.0, 236.99, 540.0, 16.0, conf=1.0)])
    rows = canvas.render_page(page, cols=74).plain.split("\n")
    text_row = next(i for i, r in enumerate(rows) if "I digress" in r)
    assert rows[text_row - 1].lstrip().startswith("┌")
    assert rows[text_row].lstrip().startswith("│")


def test_body_line_heights_render_uniformly() -> None:
    # 19pt lines land on 2 or 3 grid rows depending on position; both must
    # render the same compact way (no alternating closed/open boxes).
    page = _page(
        [
            _line("one", 36.0, 48.0, 300.0, 19.4, conf=1.0),
            _line("two", 36.0, 68.0, 300.0, 19.4, conf=1.0),
            _line("three", 36.0, 88.0, 300.0, 19.4, conf=1.0),
        ]
    )
    rows = canvas.render_page(page, cols=72).plain.split("\n")
    text_rows = [i for i, r in enumerate(rows) if "one" in r or "two" in r or "three" in r]
    assert len(text_rows) == 3
    for i in text_rows:
        assert rows[i - 1].lstrip().startswith("┌")  # top border directly above
        assert rows[i].lstrip().startswith("│")  # text between verticals
        assert "└" not in rows[i + 1]  # no partial bottom border under some


def test_tall_box_gets_closed_box_drawing_border() -> None:
    # 43pt-tall box at 72 cols spans >= 3 grid rows -> full closed rectangle.
    page = _page([_line("Sample", 190.0, 48.0, 138.0, 43.0, conf=1.0)])
    plain = canvas.render_page(page, cols=72).plain

    rows = plain.split("\n")
    top = next(i for i, r in enumerate(rows) if "┌" in r)
    bottom = next(i for i, r in enumerate(rows) if "└" in r)
    assert bottom > top
    assert "┐" in rows[top] and "┘" in rows[bottom]
    text_rows = [i for i, r in enumerate(rows) if "Sample" in r]
    assert text_rows == [top + 1]  # text sits on the first interior row


def test_low_confidence_text_gets_red_style() -> None:
    page = _page([_line("shaky", 36.0, 240.0, 200.0, 19.0, conf=0.5)])
    styles = {span.style for span in canvas.render_page(page, cols=72).spans}
    assert "bold bright_red" in styles


def test_long_text_clipped_to_box_width() -> None:
    page = _page([_line("x" * 300, 36.0, 100.0, 100.0, 10.0, conf=1.0)])
    plain = canvas.render_page(page, cols=72).plain
    text_row = next(r for r in plain.split("\n") if "x" in r)
    x1_col = int((36.0 + 100.0) * (72 / 612.0))
    assert len(text_row.rstrip()) <= x1_col + 1
    assert "…" in text_row


def test_estimated_page_size_from_line_extents() -> None:
    page = output.Page(
        number=1,
        lines=[_line("a", 10.0, 20.0, 100.0, 10.0, conf=1.0)],
        width=None,
        height=None,
    )
    width, height = canvas.estimate_page_size(page)
    assert (width, height) == (112.0, 32.0)  # max x1/y1 + 2pt padding


def test_estimated_size_falls_back_to_letter() -> None:
    assert canvas.estimate_page_size(output.Page(number=1, lines=[])) == (612.0, 792.0)


def test_overlapping_box_is_pushed_below_not_overwritten() -> None:
    page = _page(
        [
            _line("first", 10.0, 10.0, 200.0, 10.0, conf=1.0),
            _line("second", 150.0, 10.0, 100.0, 10.0, conf=1.0),  # overlaps first
        ]
    )
    rows = canvas.render_page(page, cols=80).plain.split("\n")
    first_row = next(i for i, r in enumerate(rows) if "first" in r)
    second_row = next(i for i, r in enumerate(rows) if "second" in r)
    assert first_row < second_row  # pushed below, both fully legible


def test_adjacent_boxes_sharing_one_column_stay_on_the_same_row() -> None:
    # Word boxes that touch at exactly one grid column must not stack.
    page = _page(
        [
            _line("alpha", 10.0, 10.0, 200.0, 10.0, conf=1.0),
            _line("beta", 210.0, 10.0, 100.0, 10.0, conf=1.0),
        ]
    )
    rows = canvas.render_page(page, cols=80).plain.split("\n")
    text_rows = [i for i, r in enumerate(rows) if "alpha" in r or "beta" in r]
    assert len(set(text_rows)) == 1


def test_no_line_is_lost_on_a_dense_page() -> None:
    lines = [
        _line(f"line{i}", 36.0, 100.0 + i * 5.0, 300.0, 19.0, conf=1.0)
        for i in range(8)
    ]
    plain = canvas.render_page(_page(lines), cols=80).plain
    for i in range(8):
        assert f"line{i}" in plain  # every box survives de-overlap


def test_two_lines_render_at_their_own_rows() -> None:
    page = _page(
        [
            _line("top line", 36.0, 50.0, 300.0, 10.0, conf=1.0),
            _line("bottom line", 36.0, 300.0, 300.0, 10.0, conf=1.0),
        ]
    )
    rows = canvas.render_page(page, cols=80).plain.split("\n")
    first_row = next(i for i, r in enumerate(rows) if "top" in r)
    second_row = next(i for i, r in enumerate(rows) if "bottom" in r)
    assert 0 <= first_row < second_row < len(rows)


def test_page_without_lines_renders_blank() -> None:
    plain = canvas.render_page(_page([]), cols=40).plain
    assert plain.strip() == ""
