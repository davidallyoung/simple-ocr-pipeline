"""LiteParse-backed PDF parsing, mapped into the shared output schema."""

from __future__ import annotations

from pathlib import Path

from app import output, paragraphs
from app.geometry import Quad

#: LayoutBlock kinds worth surfacing as visual regions. Tables are excluded:
#: their block bbox spans the whole grid, which reads poorly as a highlight.
BLOCK_KINDS = ("paragraph", "heading", "list_item", "code")

try:
    from liteparse import LiteParse  # type: ignore
except ImportError:  # pragma: no cover - liteparse is an optional dependency
    LiteParse = None  # type: ignore


def liteparse_available() -> bool:
    return LiteParse is not None


def needs_ocr(path: Path) -> bool:
    """Return True if any page of the PDF needs OCR (cheap text-layer pass)."""
    if LiteParse is None:
        raise RuntimeError("liteparse is not installed")
    pages = LiteParse(quiet=True).is_complex(str(path))
    return any(getattr(p, "needs_ocr", False) for p in pages)


def parse_pdf(
    path: Path,
    engine: object,
    dpi: int = 200,
    use_ocr: bool = False,
    language: str = "en",
) -> list[output.Page]:
    """Parse a PDF with LiteParse and map the result into ``output.Page``s.

    When ``use_ocr`` is True, LiteParse sends scanned pages to the embedded
    EasyOCR HTTP server (spawned via ``lite_server``). Otherwise it extracts
    the PDF text layer directly with OCR disabled. Layout blocks
    (``extract_blocks=True``) provide naturally occurring paragraph regions;
    pages without usable blocks fall back to geometric line grouping.
    """
    if LiteParse is None:
        raise RuntimeError("liteparse is not installed")

    if use_ocr:
        from app import lite_server

        server = lite_server.ensure_ocr_server(engine)
        parser = LiteParse(
            quiet=True,
            ocr_server_url=server.url,
            ocr_language=language,
            dpi=dpi,
            continue_on_page_error=True,
            extract_blocks=True,
        )
    else:
        parser = LiteParse(
            quiet=True,
            ocr_enabled=False,
            dpi=dpi,
            continue_on_page_error=True,
            extract_blocks=True,
        )

    result = parser.parse(str(path))
    pages: list[output.Page] = []
    for page in result.pages:
        lines: list[output.Line] = []
        for item in getattr(page, "text_items", []) or []:
            text = str(getattr(item, "text", ""))
            if not text:
                continue
            box = _to_quad(item)
            conf = float(item.confidence or 1.0)
            lines.append(output.Line(text=text, box=box, confidence=conf))
        pages.append(
            output.Page(
                number=int(getattr(page, "page_num", 1)),
                lines=lines,
                width=_opt_float(getattr(page, "width", None)),
                height=_opt_float(getattr(page, "height", None)),
                paragraphs=_paragraphs_for_page(page, lines),
            )
        )
    return pages


def _paragraphs_for_page(
    page: object, lines: list[output.Line]
) -> list[output.Paragraph] | None:
    """Map provider layout blocks to paragraphs; geometric fallback otherwise."""
    blocks = getattr(page, "blocks", None) or []
    mapped: list[output.Paragraph] = []
    for block in blocks:
        kind = str(getattr(block, "kind", "") or "")
        if kind not in BLOCK_KINDS:
            continue
        box = _block_bbox_quad(block)
        if box is None:
            continue
        text = str(getattr(block, "text", "") or "")
        mapped.append(
            output.Paragraph(
                text=text,
                box=box,
                confidence=_block_confidence(block, lines),
                kind=kind or "paragraph",
            )
        )
    return mapped if mapped else paragraphs.group_paragraphs(lines)


def _block_bbox_quad(block: object) -> Quad | None:
    rect = getattr(block, "bbox", None)
    if rect is None:
        return None
    try:
        x = float(rect.x)
        y = float(rect.y)
        w = float(rect.width)
        h = float(rect.height)
    except (TypeError, ValueError, AttributeError):
        return None
    if w <= 0 or h <= 0:
        return None
    return Quad.from_xywh(x, y, w, h)


def _block_confidence(block: object, lines: list[output.Line]) -> float:
    """Mean confidence of the lines whose centers fall inside the block."""
    rect = getattr(block, "bbox", None)
    if rect is None or not lines:
        return 1.0
    x0, y0, x1, y1 = (
        float(rect.x),
        float(rect.y),
        float(rect.x + rect.width),
        float(rect.y + rect.height),
    )
    members = [
        ln
        for ln in lines
        if x0 <= (ln.box.xyxy[0] + ln.box.xyxy[2]) / 2 <= x1
        and y0 <= (ln.box.xyxy[1] + ln.box.xyxy[3]) / 2 <= y1
    ]
    if not members:
        return 1.0
    return sum(ln.confidence for ln in members) / len(members)


def _to_quad(item: object) -> Quad:
    """Convert a TextItem's (x, y, width, height) rect into a Quad.

    LiteParse coordinates are already in the canonical 72-DPI point space,
    so no scaling is needed. Malformed items fall back to a zero quad.
    """
    try:
        return Quad.from_xywh(
            float(getattr(item, "x", 0.0)),
            float(getattr(item, "y", 0.0)),
            float(getattr(item, "width", 0.0)),
            float(getattr(item, "height", 0.0)),
        )
    except (TypeError, ValueError):
        return Quad.zero()


def _opt_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
