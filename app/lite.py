"""LiteParse-backed PDF parsing, mapped into the shared output schema."""

from __future__ import annotations

from pathlib import Path
from statistics import median

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
    (``extract_blocks=True``) provide naturally occurring paragraph regions,
    split at paragraph-sized vertical gaps; pages without usable blocks fall
    back to geometric line grouping.
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
        mapped.extend(_paragraphs_from_block(block, box, lines))
    return mapped if mapped else paragraphs.group_paragraphs(lines)


def _paragraphs_from_block(
    block: object, box: Quad, lines: list[output.Line]
) -> list[output.Paragraph]:
    """Paragraph(s) for one provider block, split at paragraph-sized gaps.

    Providers sometimes emit a single block spanning several true paragraphs
    (separated by roughly a blank line of vertical space). Member lines whose
    vertical gap to the previous line exceeds the paragraph-break threshold
    (``paragraphs.MAX_GAP_FACTOR`` of the typical line height) start a new
    segment. Segments keep the block's x-span — so they tile the original
    box — and carry per-segment text and confidence from their member lines.
    A block without member lines, or whose lines flow without a paragraph
    gap, stays a single paragraph using the provider's own text.
    """
    kind = str(getattr(block, "kind", "") or "") or "paragraph"
    members = _lines_in_block(block, lines)
    if len(members) < 2:
        return [
            output.Paragraph(
                text=str(getattr(block, "text", "") or ""),
                box=box,
                confidence=_block_confidence(block, lines),
                kind=kind,
            )
        ]
    heights = [m.box.xyxy[3] - m.box.xyxy[1] for m in members]
    positive = [h for h in heights if h > 0]
    typical = median(positive) if positive else 1.0
    threshold = max(typical * paragraphs.MAX_GAP_FACTOR, 0.0)

    segments: list[list[output.Line]] = [[members[0]]]
    for line in members[1:]:
        prev_y1 = segments[-1][-1].box.xyxy[3]
        if line.box.xyxy[1] - prev_y1 > threshold:
            segments.append([line])
        else:
            segments[-1].append(line)
    if len(segments) == 1:
        return [
            output.Paragraph(
                text=str(getattr(block, "text", "") or ""),
                box=box,
                confidence=_block_confidence(block, lines),
                kind=kind,
            )
        ]
    x0, _, x1, _ = box.xyxy
    return [
        output.Paragraph(
            text=" ".join(" ".join(ln.text.split()) for ln in segment).strip(),
            box=Quad.from_xywh(
                x0,
                segment[0].box.xyxy[1],
                x1 - x0,
                segment[-1].box.xyxy[3] - segment[0].box.xyxy[1],
            ),
            confidence=sum(ln.confidence for ln in segment) / len(segment),
            kind=kind,
        )
        for segment in segments
    ]


def _lines_in_block(block: object, lines: list[output.Line]) -> list[output.Line]:
    """Lines whose centers fall inside the block bbox, in reading order."""
    rect = getattr(block, "bbox", None)
    if rect is None or not lines:
        return []
    x0, y0 = float(rect.x), float(rect.y)
    x1, y1 = x0 + float(rect.width), y0 + float(rect.height)
    members = [
        ln
        for ln in lines
        if x0 <= (ln.box.xyxy[0] + ln.box.xyxy[2]) / 2 <= x1
        and y0 <= (ln.box.xyxy[1] + ln.box.xyxy[3]) / 2 <= y1
    ]
    return sorted(members, key=lambda ln: (ln.box.xyxy[1], ln.box.xyxy[0]))


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
    members = _lines_in_block(block, lines)
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
