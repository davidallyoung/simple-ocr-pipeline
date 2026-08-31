"""Interactive full-screen document inspector with hover box details.

A small Textual app showing one OCR page as the shaded character-grid
canvas (``app.canvas``). Hovering (or Tab-cycling) a box brightens it and
shows its details — text, confidence, and point-space coordinates — in a
status bar; ``n``/``p`` navigate pages, click pins a detail, ``q``/``esc``
quits.
"""

from __future__ import annotations

from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from app import annotate, canvas, output


def format_box_details(region: output.Line | output.Paragraph) -> str:
    """One-line detail string for a region: text, confidence, point coords."""
    x0, y0, x1, y1 = region.box.xyxy
    text = " ".join(region.text.split()) or "(empty)"
    kind = getattr(region, "kind", "")
    prefix = f"[{kind}] " if kind and kind != "paragraph" else ""
    return (
        f"{prefix}{text} · conf {region.confidence:.2f} · "
        f"({x0:.1f},{y0:.1f})({x1:.1f},{y1:.1f}) pt"
    )


def hit_index(
    rects: list[tuple[canvas.Region, int, int, int, int]], x: int, y: int
) -> int | None:
    """Index of the placed rect containing the cell (x, y), if any."""
    for index, (_, x0, y0, x1, y1) in enumerate(rects):
        if x0 <= x <= x1 and y0 <= y <= y1:
            return index
    return None


class PageCanvas(Static):
    """One OCR page rendered as the shaded canvas, with box hit-testing."""

    def __init__(self, page: output.Page, cols: int) -> None:
        super().__init__()
        self.page = page
        self.rects, self.rows, self.page_cols = canvas.layout_page(page, cols)
        self.update(canvas.render_page(page, cols))

    def hit(self, x: int, y: int) -> int | None:
        """Index of the placed region containing the cell (x, y), if any."""
        return hit_index(self.rects, x, y)


class InspectorScreen(Screen[None]):
    """Layout + interaction; hover state lives on the owning app."""

    CSS = """
    #canvas-scroll {
        height: 1fr;
    }
    #status {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    # Tab/shift+tab shadow the screen's default focus bindings so the box
    # cycle works for keyboard-only hover; the rest bind on the app.
    BINDINGS = [
        ("tab", "app.next_box", "Next box"),
        ("shift+tab", "app.prev_box", "Previous box"),
    ]

    def compose(self) -> ComposeResult:
        app = self.app
        assert isinstance(app, OcrInspector)
        yield Header(show_clock=False)
        with VerticalScroll(id="canvas-scroll"):
            yield app._build_canvas()
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        app = self.app
        assert isinstance(app, OcrInspector)
        app.title = str(Path(str(app._doc.get("source", "document"))).name)
        app.sub_title = f"page {app.page_index + 1}/{len(app.pages)}"
        app._update_status()

    # -- hover handling ------------------------------------------------------

    def on_mouse_move(self, event: events.MouseMove) -> None:
        app = self.app
        assert isinstance(app, OcrInspector)
        hit = app._canvas().hit(event.x, event.y)
        if hit != app.hover_index:
            app.hover_index = hit
            app._refresh_canvas()
        app._update_status()

    def on_mouse_leave(self, event: events.Leave) -> None:
        app = self.app
        assert isinstance(app, OcrInspector)
        if app.hover_index is not None:
            app.hover_index = None
            app._refresh_canvas()
        app._update_status()

    def on_click(self, event: events.Click) -> None:
        app = self.app
        assert isinstance(app, OcrInspector)
        hit = app._canvas().hit(event.x, event.y)
        app.pinned = None if hit == app.pinned else hit
        app.hover_index = hit
        app._refresh_canvas()
        app._update_status()


class OcrInspector(App[None]):
    """Hover-driven page inspector for one OCR document."""

    BINDINGS = [
        ("n", "next_page", "Next page"),
        ("p", "prev_page", "Previous page"),
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self, doc: dict) -> None:
        super().__init__()
        self._doc = doc
        self.pages: list[output.Page] = annotate.pages_from_document(doc)
        self.page_index = 0
        self.hover_index: int | None = None
        self.pinned: int | None = None

    def get_default_screen(self) -> InspectorScreen:
        return InspectorScreen()

    # -- construction -----------------------------------------------------

    def _canvas(self) -> PageCanvas:
        return self.query_one("#canvas", PageCanvas)

    # -- keyboard parity ------------------------------------------------------

    def action_next_box(self) -> None:
        self._cycle_box(1)

    def action_prev_box(self) -> None:
        self._cycle_box(-1)

    def _cycle_box(self, step: int) -> None:
        count = len(self.pages[self.page_index].regions)
        if not count:
            self.hover_index = None
        elif self.hover_index is None:
            self.hover_index = 0 if step > 0 else count - 1
        else:
            self.hover_index = (self.hover_index + step) % count
        self.pinned = None
        if self.hover_index is not None:
            self._scroll_box_into_view(self.hover_index)
        self._refresh_canvas()
        self._update_status()

    def _scroll_box_into_view(self, index: int) -> None:
        canvas_widget = self._canvas()
        if index >= len(canvas_widget.rects):
            return
        _, _, y0, _, y1 = canvas_widget.rects[index]
        top = canvas_widget.scroll_offset.y
        visible_rows = canvas_widget.scrollable_content_region.height
        if y0 < top or y1 > top + visible_rows - 1:
            canvas_widget.scroll_to(y=max(0, y0 - 1), animate=False)

    # -- page navigation -------------------------------------------------------

    def action_next_page(self) -> None:
        self._switch_page(self.page_index + 1)

    def action_prev_page(self) -> None:
        self._switch_page(self.page_index - 1)

    def _switch_page(self, index: int) -> None:
        if not 0 <= index < len(self.pages):
            return
        self.page_index = index
        self.hover_index = None
        self.pinned = None
        widget = self._canvas()
        widget.page = self.pages[index]
        widget.rects, widget.rows, widget.page_cols = canvas.layout_page(
            widget.page, widget.page_cols
        )
        widget.scroll_to(y=0, animate=False)
        self.sub_title = f"page {self.page_index + 1}/{len(self.pages)}"
        self._refresh_canvas()
        self._update_status()

    # -- rendering ---------------------------------------------------------------

    def _build_canvas(self) -> PageCanvas:
        page = self.pages[self.page_index]
        widget = PageCanvas(page, cols=max(canvas.MIN_COLS, self.size.width - 4))
        widget.id = "canvas"
        return widget

    def _refresh_canvas(self) -> None:
        widget = self._canvas()
        widget.update(
            canvas.render_page(widget.page, widget.page_cols, hover=self.hover_index)
        )

    def _update_status(self) -> None:
        status = self.query_one("#status", Static)
        page = self.pages[self.page_index]
        index = self.pinned if self.pinned is not None else self.hover_index
        regions = page.regions
        if index is None or index >= len(regions):
            details = "hover a box (or Tab) for coordinates"
            marker = ""
        else:
            details = format_box_details(regions[index])
            marker = "[pin] " if index == self.pinned else ""
        status.update(
            f" page {self.page_index + 1}/{len(self.pages)}  |  {marker}{details}"
        )
