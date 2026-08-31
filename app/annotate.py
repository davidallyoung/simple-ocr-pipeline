"""Render page images with OCR bounding boxes and confidence colors overlaid.

All boxes are in the canonical 72-DPI point space (see ``app.geometry``).
Each page is rendered at a chosen DPI and boxes are scaled by ``dpi / 72``;
image sources are treated as their own 72-DPI viewport (1px == 1pt).

Confidence maps to a traffic-light color matching the TUI/viewer thresholds:
red < 0.7, amber 0.7-0.9, green >= 0.9.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from app import output
from app.geometry import Quad

RED = (224, 56, 56)
AMBER = (232, 160, 24)
GREEN = (46, 160, 66)

FILL_ALPHA = 38  # ~15% translucent fill
OUTLINE_WIDTH = 2
LABEL_BACKING = (25, 25, 25, 210)


def conf_color(conf: float) -> tuple[int, int, int]:
    """Map a 0-1 confidence to a traffic-light RGB (mirrors app.viewer)."""
    if conf < 0.7:
        return RED
    if conf < 0.9:
        return AMBER
    return GREEN


def pages_dir_for(json_path: Path) -> Path:
    """Annotated-pages folder for a JSON: ``report.pdf.json`` -> ``report.pdf.pages``."""
    return json_path.parent / f"{json_path.stem}.pages"


def page_file_for(pages_dir: Path, page_number: int) -> Path:
    return pages_dir / f"page-{page_number:03d}.png"


def annotate_page(
    image: Image.Image,
    lines: list[output.Line],
    scale: float = 1.0,
    label: bool = True,
) -> Image.Image:
    """Draw line boxes onto a copy of ``image`` and return a fresh RGB image.

    ``scale`` converts canonical point space to the image's pixel space
    (``dpi / 72`` for rendered PDF pages, 1.0 for native-size images).
    """
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for line in lines:
        pts = [(x * scale, y * scale) for x, y in line.box.to_list()]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
            continue  # degenerate (e.g. zero-fallback) box
        color = conf_color(line.confidence)
        draw.polygon(pts, fill=color + (FILL_ALPHA,))
        draw.line([*pts, pts[0]], fill=color + (255,), width=OUTLINE_WIDTH, joint="curve")
        if label:
            _draw_label(draw, pts, line.confidence)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _draw_label(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], conf: float) -> None:
    x0 = min(p[0] for p in pts)
    y0 = min(p[1] for p in pts)
    height = max(p[1] for p in pts) - y0
    font = _font(max(10, min(int(height * 0.6), 24)))
    text = f"{conf * 100:.0f}%"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    ty = y0 - th - 4 if y0 - th - 4 >= 0 else y0 + 2
    draw.rectangle([x0, ty - 1, x0 + tw + 5, ty + th + 2], fill=LABEL_BACKING)
    draw.text((x0 + 2, ty), text, fill=(255, 255, 255, 255), font=font)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size parameter
        return ImageFont.load_default()


def annotate_document(
    source: Path,
    pages: list[output.Page],
    dpi: int,
    out_dir: Path,
) -> list[Path]:
    """Write annotated PNGs for each page; boxes are in canonical point space.

    PDFs render at ``dpi`` (boxes scale by ``dpi / 72``); image sources are
    used at native size. Pages stream one at a time so a whole document is
    never held in memory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if source.suffix.lower() == ".pdf":
        with pymupdf.open(source) as doc:
            for page in pages:
                rendered = _render_pdf_page(doc, page.number, dpi)
                annotated = annotate_page(rendered, page.lines, scale=dpi / 72.0)
                out = page_file_for(out_dir, page.number)
                annotated.save(out)
                written.append(out)
    else:
        with Image.open(source) as img:
            rendered = img.convert("RGB")
        for page in pages:
            annotated = annotate_page(rendered, page.lines, scale=1.0)
            out = page_file_for(out_dir, page.number)
            annotated.save(out)
            written.append(out)
    return written


def pages_from_document(doc: dict) -> list[output.Page]:
    """Rebuild canonical ``output.Page``s from a written output JSON."""
    pages: list[output.Page] = []
    for raw in doc.get("pages", []):
        lines = [
            output.Line(
                text=str(line.get("text", "")),
                box=_quad_from_json(line.get("box", [])),
                confidence=float(line.get("confidence", 0.0)),
            )
            for line in raw.get("lines", [])
        ]
        pages.append(
            output.Page(
                number=int(raw.get("page", len(pages) + 1)),
                lines=lines,
                width=_opt_float(raw.get("width")),
                height=_opt_float(raw.get("height")),
            )
        )
    return pages


def _quad_from_json(box: object) -> Quad:
    if not isinstance(box, list):
        return Quad.zero()
    try:
        pts = [[float(x), float(y)] for x, y in box]
    except (TypeError, ValueError):
        return Quad.zero()
    if len(pts) != 4:
        return Quad.zero()
    return Quad.from_quad(pts)


def _opt_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def annotate_json(doc: dict, out_dir: Path, dpi: int = 200) -> list[Path]:
    """Annotate a parsed output JSON (retro path, e.g. from the viewer).

    New-format JSONs carry canonical point-space boxes and page dimensions,
    so annotation is exact. Legacy JSONs (written before the geometry model)
    are handled best-effort: point-space boxes are detected by fitting inside
    the rendered page, while legacy pixel-space boxes (old ``--ocr-only``
    runs) cannot be rescaled without a stored DPI and raise ``ValueError``.
    """
    source = Path(str(doc.get("source", "")))
    if not source.exists():
        raise FileNotFoundError(f"source document not found: {source}")
    pages = pages_from_document(doc)
    if not pages:
        return []
    if source.suffix.lower() == ".pdf" and not _has_page_dims(doc):
        _require_point_space(source, pages)
    return annotate_document(source, pages, dpi=dpi, out_dir=out_dir)


def _has_page_dims(doc: dict) -> bool:
    return bool(doc.get("pages")) and "width" in doc["pages"][0]


def _require_point_space(source: Path, pages: list[output.Page]) -> None:
    """Reject legacy PDF JSONs whose boxes overflow the rendered page."""
    with pymupdf.open(source) as doc:
        for page in pages:
            idx = page.number - 1
            if not 0 <= idx < doc.page_count:
                continue
            rect = doc[idx].rect
            for line in page.lines:
                _, _, x1, y1 = line.box.xyxy
                if x1 > rect.width + 1 or y1 > rect.height + 1:
                    raise ValueError(
                        f"{source.name}: boxes exceed the page's point space "
                        "(legacy pixel-space JSON without stored dpi); "
                        "re-run OCR to annotate this document"
                    )


def _render_pdf_page(doc: pymupdf.Document, number: int, dpi: int) -> Image.Image:
    pix = doc[number - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def open_folder(path: Path) -> None:
    """Open a folder in the platform file manager (Windows-first repo)."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
