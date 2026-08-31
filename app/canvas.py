"""Text-mode page canvas: OCR boxes laid out on a character grid.

The page (in canonical 72-DPI point space) maps onto a grid of terminal
cells — ``cols`` wide, with rows derived from the page aspect ratio and a
character-cell correction (terminal glyphs are roughly twice as tall as
wide). Each line's quad becomes a grid rectangle:

- boxes at least 3 rows tall get a box-drawing border in the confidence
  color with the line text placed inside;
- flatter boxes (the common single-line case) render their text in the
  confidence color, bold and underlined.

Later lines never overwrite already-drawn cells, so overlapping boxes stay
legible. The result is a rich ``Text`` the viewer can drop into a Panel.
"""

from __future__ import annotations

from rich.text import Text

from app import output

BORDER_H = "─"
BORDER_V = "│"
CORNER_TL, CORNER_TR = "┌", "┐"
CORNER_BL, CORNER_BR = "└", "┘"

#: Grid rows per point of page height, relative to columns per point of
#: width — terminal glyphs are roughly twice as tall as they are wide.
CELL_ASPECT = 0.5
DEFAULT_PAGE_SIZE = (612.0, 792.0)
MIN_COLS, MAX_COLS = 20, 160


def conf_rich_style(conf: float) -> str:
    """Rich color style for a confidence (mirrors app.viewer thresholds)."""
    if conf < 0.7:
        return "bright_red"
    if conf < 0.9:
        return "bright_yellow"
    return "green"


def estimate_page_size(page: output.Page) -> tuple[float, float]:
    """Page size in points; falls back to the line-box extents when unknown."""
    if page.width and page.height:
        return page.width, page.height
    rects = [line.box.xyxy for line in page.lines]
    if not rects:
        return DEFAULT_PAGE_SIZE
    width = max(x1 for _, _, x1, _ in rects) + 2.0
    height = max(y1 for _, _, _, y1 in rects) + 2.0
    return (max(width, 1.0), max(height, 1.0))


def _clip_text(text: str, width: int) -> str:
    text = " ".join(str(text).split()) or "—"
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


class _Grid:
    """Character buffer with per-cell rich styles."""

    def __init__(self, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows
        self.chars: list[list[str]] = [[" "] * cols for _ in range(rows)]
        self.styles: list[list[str | None]] = [[None] * cols for _ in range(rows)]

    def put(self, col: int, row: int, char: str, style: str | None) -> None:
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.chars[row][col] = char
            self.styles[row][col] = style

    def try_text(self, col: int, row: int, text: str, style: str) -> bool:
        """Write text into still-empty cells (never overwriting); False if nothing fit."""
        if not 0 <= row < self.rows or not text:
            return False
        wrote = 0
        for i, char in enumerate(text):
            target = col + i
            if target >= self.cols:
                break
            if self.styles[row][target] is None:
                self.put(target, row, char, style)
                wrote += 1
        return wrote > 0

    def to_text(self) -> Text:
        text = Text()
        for row in range(self.rows):
            if row:
                text.append("\n")
            run: list[str] = []
            run_style: str | None = None
            for col in range(self.cols):
                cell_style = self.styles[row][col]
                if run and cell_style != run_style:
                    text.append("".join(run), style=run_style)
                    run = []
                run.append(self.chars[row][col])
                run_style = cell_style
            if run:
                text.append("".join(run), style=run_style)
        return text


def render_page(page: output.Page, cols: int) -> Text:
    """Render one page's line boxes onto a character grid (rich Text).

    Boxes are placed in reading order; a box that would share grid rows with
    a column-overlapping predecessor is pushed below it, so text and borders
    never overwrite each other. The grid grows past its natural height when
    dense pages need the extra room.
    """
    cols = max(MIN_COLS, min(cols, MAX_COLS))
    width, height = estimate_page_size(page)
    natural_rows = max(3, round(cols * height / width * CELL_ASPECT))
    sx, sy = cols / width, natural_rows / height

    ordered = sorted(page.lines, key=lambda ln: (ln.box.xyxy[1], ln.box.xyxy[0]))
    placed: list[tuple[output.Line, int, int, int, int]] = []
    for line in ordered:
        x0, y0, x1, y1 = line.box.xyxy
        cx0 = max(0, int(x0 * sx))
        cx1 = min(int(x1 * sx), cols - 1)
        cy0, cy1 = int(y0 * sy), int(y1 * sy)
        if cx0 > cols - 1 or cx1 < cx0 or cy1 < cy0:
            continue
        cy0 = max(0, cy0)
        # Normalize: sub-row boxes still get a compact 2-row box so identical
        # lines never alternate between boxed and bare underlined text.
        if cx1 - cx0 + 1 >= 3:
            render_rows = 2 if cy1 - cy0 + 1 <= 3 else cy1 - cy0 + 1
        else:
            render_rows = 1
        floor = cy0
        for _, px0, _, px1, py1 in placed:
            if py1 < cy0:
                continue
            if min(cx1, px1) - max(cx0, px0) + 1 >= 2:  # real overlap, not a
                floor = max(floor, py1 + 1)  # shared 1-col boundary wall
        if floor > cy0:
            cy0 = floor
        cy1 = cy0 + render_rows - 1
        placed.append((line, cx0, cy0, cx1, cy1))

    deepest = max((p[4] for p in placed), default=-1)
    grid = _Grid(cols, max(natural_rows, deepest + 1))
    for line, bx0, by0, bx1, by1 in placed:
        _draw_box(grid, line, bx0, by0, bx1, by1)
    return grid.to_text()


def _draw_box(
    grid: _Grid, line: output.Line, x0: int, y0: int, x1: int, y1: int
) -> None:
    """Draw one line's box; rendering scales with the box's grid height.

    Boxes up to 3 grid rows tall all render the same compact way (top border
    + text between verticals) so identical line heights never alternate
    between styles due to int rounding; only genuinely tall boxes get a full
    closed rectangle.
    """
    style = conf_rich_style(line.confidence)
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    if width < 3 or height <= 1:
        # Too small for a box: colored, underlined text only.
        grid.try_text(x0, y0, _clip_text(line.text, width), f"bold underline {style}")
        return
    if height <= 3:
        # Compact box: top border + text between verticals.
        _draw_border(grid, x0, y0, x1, y0, style)
        grid.put(x0, y0 + 1, BORDER_V, style)
        grid.put(x1, y0 + 1, BORDER_V, style)
        grid.try_text(x0 + 1, y0 + 1, _clip_text(line.text, width - 2), f"bold {style}")
        return
    _draw_border(grid, x0, y0, x1, y1, style)
    grid.try_text(x0 + 1, y0 + 1, _clip_text(line.text, width - 2), f"bold {style}")


def _draw_border(grid: _Grid, x0: int, y0: int, x1: int, y1: int, style: str) -> None:
    for col in range(x0, x1 + 1):
        grid.put(col, y0, BORDER_H, style)
        if y1 > y0:
            grid.put(col, y1, BORDER_H, style)
    for row in range(y0 + 1, y1):
        grid.put(x0, row, BORDER_V, style)
        grid.put(x1, row, BORDER_V, style)
    grid.put(x0, y0, CORNER_TL, style)
    grid.put(x1, y0, CORNER_TR, style)
    if y1 > y0:
        grid.put(x0, y1, CORNER_BL, style)
        grid.put(x1, y1, CORNER_BR, style)
