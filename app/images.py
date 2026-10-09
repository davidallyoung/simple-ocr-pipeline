"""Find the images in a document: embedded PDF images and standalone image files.

Extraction is independent of the text paths (LiteParse and EasyOCR), which do not
report images. Boxes use the canonical 72-DPI top-left point space of
:class:`app.geometry.Quad`.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf
from PIL import Image as PILImage

from app.geometry import Quad

SkipReason = Literal["cap", "too_small"]

PIXMAP_DPI = 144
SEND_AS_IS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}
PDF_NATIVE_EXTENSIONS = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}


@dataclass(frozen=True)
class ImageOccurrence:
    """One placement of an image on a page, with the bytes to send to a model.

    ``width`` and ``height`` are the pixel size of ``data``. ``sha256`` is the
    digest of ``data``, so equal images on different pages match.
    """

    page: int
    box: Quad
    data: bytes
    mime: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class Selection:
    """Decision for one occurrence. ``skip_reason`` is ``None`` when it is kept.

    A kept occurrence whose ``sha256`` was already kept earlier in the document
    reuses that description and does not use another cap slot.
    """

    occurrence: ImageOccurrence
    skip_reason: SkipReason | None


def _occurrence(
    page: int, box: Quad, data: bytes, mime: str, width: int, height: int
) -> ImageOccurrence:
    return ImageOccurrence(
        page=page,
        box=box,
        data=data,
        mime=mime,
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
    )


def _box(rect: pymupdf.Rect) -> Quad:
    return Quad.from_xywh(rect.x0, rect.y0, rect.width, rect.height)


def _png_from_pixmap(pix: pymupdf.Pixmap) -> bytes:
    if pix.colorspace is not None and pix.colorspace.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    encoded: bytes = pix.tobytes("png")
    return encoded


def _pdf_occurrence(
    doc: pymupdf.Document, page: pymupdf.Page, number: int, info: dict
) -> ImageOccurrence:
    box = _box(pymupdf.Rect(info["bbox"]))
    xref = int(info["xref"])
    if xref == 0:
        pix = page.get_pixmap(clip=pymupdf.Rect(info["bbox"]), dpi=PIXMAP_DPI)
        return _occurrence(number, box, _png_from_pixmap(pix), "image/png", pix.width, pix.height)
    raw = doc.extract_image(xref)
    mime = PDF_NATIVE_EXTENSIONS.get(raw["ext"])
    if mime is None:
        pix = pymupdf.Pixmap(doc, xref)
        return _occurrence(number, box, _png_from_pixmap(pix), "image/png", pix.width, pix.height)
    return _occurrence(number, box, raw["image"], mime, int(info["width"]), int(info["height"]))


def iter_pdf_images(path: Path) -> Iterator[ImageOccurrence]:
    """Yield every image placement in a PDF in page order, then in drawing order."""
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            for info in page.get_image_info(xrefs=True):
                yield _pdf_occurrence(doc, page, number, info)


def read_image_file(path: Path) -> ImageOccurrence:
    """Read a standalone image as one whole-page occurrence.

    Formats a model accepts are sent unchanged. Anything else is re-encoded as PNG.
    """
    raw = path.read_bytes()
    with PILImage.open(io.BytesIO(raw)) as img:
        width, height = img.size
        mime = SEND_AS_IS.get(img.format or "")
        if mime is not None:
            data = raw
        else:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            data = buffer.getvalue()
            mime = "image/png"
    return _occurrence(1, Quad.from_xywh(0, 0, width, height), data, mime, width, height)


def select(occurrences: Sequence[ImageOccurrence], *, min_px: int, cap: int) -> list[Selection]:
    """Decide which occurrences to describe, in input order.

    An occurrence is ``too_small`` when its shorter side is below ``min_px``.
    Otherwise it is kept if its image is already kept, or while fewer than
    ``cap`` distinct images are kept. Anything else is ``cap``.
    """
    kept: set[str] = set()
    selections: list[Selection] = []
    for occurrence in occurrences:
        if min(occurrence.width, occurrence.height) < min_px:
            selections.append(Selection(occurrence, "too_small"))
        elif occurrence.sha256 in kept or len(kept) < cap:
            kept.add(occurrence.sha256)
            selections.append(Selection(occurrence, None))
        else:
            selections.append(Selection(occurrence, "cap"))
    return selections
