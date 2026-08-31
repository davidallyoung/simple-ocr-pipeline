"""Text-mode page canvas: OCR regions shaded onto a character grid.

The page (in canonical 72-DPI point space) maps onto a grid of terminal
cells — ``cols`` wide, with rows derived from the page aspect ratio and a
character-cell correction (terminal glyphs are roughly twice as tall as
wide). Each line's quad shades its grid rectangle like a highlighter pen:
the background is a dark tint of the confidence color and the line text
renders bright in the same color family, so red/amber/green reads at a
glance.

Boxes are placed in reading order; a box that would share grid rows with a
column-overlapping predecessor is pushed below it, so text never collides.
Shaded backgrounds tolerate touching (a one-column boundary is fine), and
text flows into free cells. The result is a rich ``Text`` the viewer can
drop into a Panel.
"""

from __future__ import annotations

from rich.text import Text

from app import output

RED = (224, 56, 56)
AMBER = (232, 160, 24)
GREEN = (46, 160, 66)

#: Background tint = confidence color scaled to this fraction (a dark wash).
TINT_SCALE = 0.25

#: Grid rows per point of page height, relative to columns per point of
#: width — terminal glyphs are roughly twice as tall as they are wide.
CELL_ASPECT = 0.5
DEFAULT_PAGE_SIZE = (612.0, 792.0)
MIN_COLS, MAX_COLS = 20, 160


def conf_rich_style(conf: float) -> str:
    """Rich foreground color for a confidence (mirrors app.viewer thresholds)."""
    if conf < 0.7:
        return "bright_red"
    if conf < 0.9:
        return "bright_yellow"
    return "green"


def conf_rgb(conf: float) -> tuple[int, int, int]:
    """RGB triple for a confidence (mirrors app.annotate)."""
    if conf < 0.7:
        return RED
    if conf < 0.9:
        return AMBER
    return GREEN


def tint_hex(conf: float) -> str:
    """Dark background tint (hex) for a confidence, for rich ``on #...`` styles."""
    r, g, b = (int(c * TINT_SCALE) for c in conf_rgb(conf))
    return f"#{r:02x}{g:02x}{b:02x}"


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

    def fill(self, x0: int, y0: int, x1: int, y1: int, style: str) -> None:
        """Shade a rectangle's background; cells holding text are left alone."""
        for row in range(max(0, y0), min(y1, self.rows - 1) + 1):
            for col in range(max(0, x0), min(x1, self.cols - 1) + 1):
                if self.chars[row][col] == " ":
                    self.put(col, row, " ", style)

    def try_text(self, col: int, row: int, text: str, style: str) -> bool:
        """Write text into free (space) cells only; False if nothing fit."""
        if not 0 <= row < self.rows or not text:
            return False
        wrote = 0
        for i, char in enumerate(text):
            target = col + i
            if target >= self.cols:
                break
            if self.chars[row][target] == " ":
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
    """Render one page's shaded line regions onto a character grid.

    Boxes are placed in reading order; a box that would share grid rows with
    a column-overlapping predecessor is pushed below it, so text never
    collides. The grid grows past its natural height when dense pages need
    the extra room.
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
        cy1 = max(cy1, cy0)  # degenerate heights collapse to one shaded row
        floor = cy0
        for _, px0, _, px1, py1 in placed:
            if py1 < cy0:
                continue
            if min(cx1, px1) - max(cx0, px0) + 1 >= 2:  # real overlap, not a
                floor = max(floor, py1 + 1)  # shared 1-col boundary
        if floor > cy0:
            cy1 += floor - cy0
            cy0 = floor
        placed.append((line, cx0, cy0, cx1, cy1))

    deepest = max((p[4] for p in placed), default=-1)
    grid = _Grid(cols, max(natural_rows, deepest + 1))
    for line, bx0, by0, bx1, by1 in placed:
        _shade_box(grid, line, bx0, by0, bx1, by1)
    return grid.to_text()


def _shade_box(
    grid: _Grid, line: output.Line, x0: int, y0: int, x1: int, y1: int
) -> None:
    """Shade one line's rectangle and write its text in the bright color."""
    bg = tint_hex(line.confidence)
    grid.fill(x0, y0, x1, y1, f"on {bg}")
    fg = conf_rich_style(line.confidence)
    grid.try_text(x0, y0, _clip_text(line.text, x1 - x0 + 1), f"bold {fg} on {bg}")
