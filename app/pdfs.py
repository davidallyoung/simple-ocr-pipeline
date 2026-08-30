"""Rasterize PDF pages into RGB PIL images for OCR."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pymupdf
from PIL import Image


def pdf_page_count(path: Path) -> int:
    with pymupdf.open(path) as doc:
        return int(doc.page_count)


def iter_pdf_pages(path: Path, dpi: int = 200) -> Iterator[Image.Image]:
    """Yield each PDF page as an RGB PIL image at the given DPI.

    Streams one page at a time so an entire document is never held in memory
    at once.
    """
    with pymupdf.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
