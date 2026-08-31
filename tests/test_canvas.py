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


def test_conf_rgb_matches_annotate_palette() -> None:
    assert canvas.conf_rgb(0.5) == canvas.RED
    assert canvas.conf_rgb(0.8) == canvas.AMBER
    assert canvas.conf_rgb(0.95) == canvas.GREEN


def test_tint_hex_is_dark_scaled_confidence_color() -> None:
    # 25% brightness of GREEN (46, 160, 66) -> (11, 40, 16).
    assert canvas.tint_hex(1.0) == "#0b2810"
    assert canvas.tint_hex(0.5) == "#380e0e"  # 25% of RED (224, 56, 56)
    assert canvas.tint_hex(0.8) == "#3a2806"  # 25% of AMBER (232, 160, 24)


def test_box_shades_background_with_bright_text() -> None:
    page = _page([_line("Sample", 190.0, 48.0, 138.0, 20.0, conf=1.0)])
    text = canvas.render_page(page, cols=72)

    styles = {span.style for span in text.spans}
    assert "bold green on #0b2810" in styles  # text cells
    assert "on #0b2810" in styles  # background-only cells
    assert "Sample" in text.plain


def test_shade_covers_full_rectangle_not_just_text() -> None:
    # A wide box with short text must shade the whole rect width.
    page = _page([_line("hi", 36.0, 48.0, 400.0, 20.0, conf=1.0)])
    text = canvas.render_page(page, cols=72)
    bg_spans = [s for s in text.spans if s.style == "on #0b2810"]
    assert sum(s.end - s.start for s in bg_spans) >= int(436.0 * (72 / 612.0)) - 2


def test_low_confidence_gets_red_tint() -> None:
    page = _page([_line("shaky", 36.0, 240.0, 200.0, 19.0, conf=0.5)])
    styles = {span.style for span in canvas.render_page(page, cols=72).spans}
    assert "bold bright_red on #380e0e" in styles


def test_subrow_box_still_shades_one_row() -> None:
    # ~16pt box floors to a single grid row; it must still shade + show text.
    page = _page([_line("I digress", 36.0, 236.99, 540.0, 16.0, conf=1.0)])
    text = canvas.render_page(page, cols=74)
    assert "I digress" in text.plain
    styles = {span.style for span in text.spans}
    assert "bold green on #0b2810" in styles


def test_narrow_box_shades_its_column_and_clips_text() -> None:
    page = _page([_line("tiny", 36.0, 48.0, 1.0, 20.0, conf=1.0)])
    text = canvas.render_page(page, cols=72)
    assert "t" in text.plain  # first character fits the 1-column shade
    styles = {span.style for span in text.spans}
    assert "bold green on #0b2810" in styles


def test_long_text_wraps_within_box_width_and_clips_at_the_end() -> None:
    page = _page([_line("x" * 300, 36.0, 100.0, 100.0, 10.0, conf=1.0)])
    rows = canvas.render_page(page, cols=72).plain.split("\n")
    text_rows = [r for r in rows if "x" in r]
    assert len(text_rows) >= 2  # the run wraps onto several shaded rows
    x1_col = int((36.0 + 100.0) * (72 / 612.0))
    for row in text_rows:
        assert len(row.rstrip()) <= x1_col + 1
    assert "…" not in text_rows[0]
    assert "…" in text_rows[-1]


def test_wrap_text_splits_long_words_and_marks_overflow() -> None:
    assert canvas._wrap_text("hello world", 10, 3) == ["hello", "world"]
    assert canvas._wrap_text("x" * 25, 10, 2) == ["x" * 10, "x" * 9 + "…"]
    assert canvas._wrap_text("", 10, 3) == ["—"]
    assert canvas._wrap_text("a b c", 10, 1) == ["a b c"]  # fits: no ellipsis
    assert canvas._wrap_text("alpha beta", 5, 1) == ["alph…"]


def test_paragraph_text_wraps_across_box_rows_without_ellipsis() -> None:
    text = " ".join(f"word{i}" for i in range(60))
    page = _page([_line(text, 36.0, 100.0, 540.0, 200.0, conf=1.0)])
    rows = canvas.render_page(page, cols=72).plain.split("\n")
    text_rows = [r for r in rows if "word" in r]
    assert len(text_rows) >= 2  # text flows down the box, not one clipped row
    assert all("…" not in row for row in text_rows)
    assert "word0" in text_rows[0]
    assert "word59" in text_rows[-1]


def test_unfittable_text_keeps_ellipsis_on_last_visible_row() -> None:
    # A degenerate one-row box: wrap collapses back to a single clipped row.
    page = _page(
        [_line("alpha beta gamma delta", 36.0, 100.0, 100.0, 1.0, conf=1.0)]
    )
    rows = canvas.render_page(page, cols=72).plain.split("\n")
    text_rows = [r for r in rows if "alpha" in r]
    assert len(text_rows) == 1
    assert text_rows[0].rstrip().endswith("…")


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


def test_overlapping_shades_keep_both_texts() -> None:
    # Shaded rects may blend, but neither line's text may be lost.
    page = _page(
        [
            _line("alpha", 10.0, 10.0, 200.0, 20.0, conf=1.0),
            _line("beta", 30.0, 12.0, 100.0, 20.0, conf=0.5),
        ]
    )
    plain = canvas.render_page(page, cols=80).plain
    assert "alpha" in plain
    assert "beta" in plain


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
