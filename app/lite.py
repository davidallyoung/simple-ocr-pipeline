"""LiteParse-backed PDF parsing, mapped into the shared output schema."""

from __future__ import annotations

from pathlib import Path

from app import output

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
    the PDF text layer directly with OCR disabled.
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
        )
    else:
        parser = LiteParse(
            quiet=True, ocr_enabled=False, dpi=dpi, continue_on_page_error=True
        )

    result = parser.parse(str(path))
    pages: list[output.Page] = []
    for page in result.pages:
        lines: list[output.Line] = []
        for item in getattr(page, "text_items", []) or []:
            text = str(getattr(item, "text", ""))
            if not text:
                continue
            box = _to_polygon(item)
            conf = float(item.confidence or 1.0)
            lines.append(output.Line(text=text, box=box, confidence=conf))
        pages.append(output.Page(number=int(getattr(page, "page_num", 1)), lines=lines))
    return pages


def _to_polygon(item: object) -> list[list[float]]:
    """Convert a TextItem's (x, y, width, height) into a 4-point box polygon."""
    try:
        x = float(getattr(item, "x", 0.0))
        y = float(getattr(item, "y", 0.0))
        w = float(getattr(item, "width", 0.0))
        h = float(getattr(item, "height", 0.0))
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    except (TypeError, ValueError):
        return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
