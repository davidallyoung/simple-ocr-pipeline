from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image as PILImage

from app import images
from app.geometry import Quad


def _solid_png(width: int, height: int, gray: int) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 0)
    pix.clear_with(gray)
    encoded: bytes = pix.tobytes("png")
    return encoded


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    big = _solid_png(400, 300, 120)
    tiny = _solid_png(8, 8, 200)
    jpeg_pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), 0)
    jpeg_pix.clear_with(60)
    jpeg = jpeg_pix.tobytes("jpeg")

    first = doc.new_page(width=612, height=792)
    first.insert_image(pymupdf.Rect(72, 100, 372, 325), stream=big)
    first.insert_image(pymupdf.Rect(400, 100, 420, 120), stream=tiny)

    second = doc.new_page(width=612, height=792)
    second.insert_image(pymupdf.Rect(72, 100, 372, 325), stream=big)
    second.insert_image(pymupdf.Rect(72, 400, 172, 500), stream=jpeg)

    doc.save(path)
    doc.close()
    return path


def _by_page(occurrences: list[images.ImageOccurrence]) -> list[tuple[int, int, int]]:
    return [(occ.page, occ.width, occ.height) for occ in occurrences]


def test_pdf_images_report_page_box_size_and_mime(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "images.pdf")

    found = list(images.iter_pdf_images(pdf))

    assert _by_page(found) == [(1, 400, 300), (1, 8, 8), (2, 400, 300), (2, 200, 200)]
    assert found[0].box == Quad.from_xywh(72, 100, 300, 225)
    assert found[0].mime == "image/png"
    assert found[3].mime == "image/jpeg"


def test_same_image_on_two_pages_shares_a_sha256(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "images.pdf")

    found = list(images.iter_pdf_images(pdf))

    assert found[0].sha256 == found[2].sha256
    assert found[0].sha256 != found[3].sha256


def test_select_skips_tiny_images_and_reuses_duplicates(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "images.pdf")
    found = list(images.iter_pdf_images(pdf))

    selections = images.select(found, min_px=32, cap=5)

    assert [s.skip_reason for s in selections] == [None, "too_small", None, None]


def test_select_caps_distinct_images_and_records_the_rest(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "images.pdf")
    found = list(images.iter_pdf_images(pdf))

    selections = images.select(found, min_px=32, cap=1)

    assert [s.skip_reason for s in selections] == [None, "too_small", None, "cap"]


def test_select_counts_duplicate_of_a_kept_image_against_no_cap(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "images.pdf")
    found = list(images.iter_pdf_images(pdf))

    selections = images.select(found, min_px=32, cap=1)

    assert selections[2].occurrence.sha256 == selections[0].occurrence.sha256
    assert selections[2].skip_reason is None


def test_inline_image_is_extracted_as_a_rendered_region(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    stream = (
        b"q 200 0 0 100 72 600 cm BI /W 2 /H 2 /BPC 8 /CS /RGB /F /AHx ID "
        b"ff000000ff000000ff00ffffff> EI Q"
    )
    contents = doc.get_new_xref()
    doc.update_object(contents, "<<>>")
    doc.update_stream(contents, stream)
    page.set_contents(contents)
    path = tmp_path / "inline.pdf"
    doc.save(path)
    doc.close()

    found = list(images.iter_pdf_images(path))

    assert len(found) == 1
    assert found[0].box == Quad.from_xywh(72, 92, 200, 100)
    assert found[0].data.startswith(b"\x89PNG")


def test_standalone_png_is_sent_unchanged_as_one_whole_page_image(tmp_path: Path) -> None:
    src = tmp_path / "photo.png"
    src.write_bytes(_solid_png(400, 300, 90))

    occurrence = images.read_image_file(src)

    assert occurrence.data == src.read_bytes()
    assert occurrence.mime == "image/png"
    assert occurrence.page == 1
    assert occurrence.box == Quad.from_xywh(0, 0, 400, 300)


def test_standalone_bmp_is_converted_to_png(tmp_path: Path) -> None:
    src = tmp_path / "scan.bmp"
    PILImage.new("RGB", (64, 48), "white").save(src)

    occurrence = images.read_image_file(src)

    assert occurrence.data.startswith(b"\x89PNG")
    assert occurrence.mime == "image/png"
    assert (occurrence.width, occurrence.height) == (64, 48)
