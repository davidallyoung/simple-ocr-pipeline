from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from app import annotate, output
from app.geometry import Quad


def _line(x: float, y: float, w: float, h: float, conf: float, text: str = "hi") -> output.Line:
    return output.Line(text=text, box=Quad.from_xywh(x, y, w, h), confidence=conf)


def _pdf(tmp_path: Path, width: float = 612.0, height: float = 792.0, pages: int = 1) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=width, height=height)
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return path


def test_conf_color_tiers_match_viewer_thresholds() -> None:
    assert annotate.conf_color(0.0) == annotate.RED
    assert annotate.conf_color(0.69) == annotate.RED
    assert annotate.conf_color(0.7) == annotate.AMBER
    assert annotate.conf_color(0.89) == annotate.AMBER
    assert annotate.conf_color(0.9) == annotate.GREEN
    assert annotate.conf_color(1.0) == annotate.GREEN


def test_annotate_page_draws_colored_border_and_label() -> None:
    image = Image.new("RGB", (120, 60), "white")
    line = _line(10, 40, 60, 20, conf=0.5)
    out = annotate.annotate_page(image, [line])
    assert out.size == image.size
    # Left edge of the box carries the exact confidence color...
    assert out.getpixel((10, 55)) == annotate.RED
    # ...and pixels outside the box stay untouched.
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_annotate_page_skips_degenerate_boxes() -> None:
    image = Image.new("RGB", (80, 80), "white")
    line = output.Line(text="x", box=Quad.zero(), confidence=1.0)
    out = annotate.annotate_page(image, [line])
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_annotate_page_can_suppress_labels() -> None:
    image = Image.new("RGB", (120, 60), "white")
    line = _line(10, 40, 60, 20, conf=0.5)
    out = annotate.annotate_page(image, [line], label=False)
    # No dark label backing above the box.
    assert out.getpixel((12, 30)) == (255, 255, 255)


def test_annotate_page_preserves_polygon_skew() -> None:
    image = Image.new("RGB", (100, 100), "white")
    quad = Quad.from_points([(20, 60), (60, 14), (64, 64), (24, 60)])
    line = output.Line(text="skew", box=quad, confidence=0.95)
    out = annotate.annotate_page(image, [line])
    assert out.getpixel((60, 14)) == annotate.GREEN


def test_conf_color_is_rgb_tuple() -> None:
    for conf in (0.0, 0.8, 1.0):
        color = annotate.conf_color(conf)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


def test_annotate_document_pdf_scales_points_to_dpi(tmp_path: Path) -> None:
    # US Letter (612x792pt) at 144 DPI -> 2x scale -> 1224x1584px PNG.
    pdf = _pdf(tmp_path)
    page = output.Page(
        number=1,
        lines=[_line(100, 100, 200, 40, conf=0.95)],
        width=612.0,
        height=792.0,
    )
    out_dir = tmp_path / "doc.pdf.pages"
    paths = annotate.annotate_document(pdf, [page], dpi=144, out_dir=out_dir)

    assert [p.name for p in paths] == ["page-001.png"]
    img = Image.open(paths[0])
    assert img.size == (1224, 1584)
    # Box top-left at 100pt * 2 == 200px must carry the >= 0.9 color.
    assert img.getpixel((200, 200)) == annotate.GREEN


def test_annotate_document_image_source_uses_native_size(tmp_path: Path) -> None:
    src = tmp_path / "scan.png"
    Image.new("RGB", (200, 100), "white").save(src)
    page = output.Page(number=1, lines=[_line(20, 20, 80, 20, conf=0.75)])
    paths = annotate.annotate_document(src, [page], dpi=200, out_dir=tmp_path / "p")

    assert [p.name for p in paths] == ["page-001.png"]
    img = Image.open(paths[0])
    assert img.size == (200, 100)
    assert img.getpixel((20, 40)) == annotate.AMBER


def test_annotate_document_renders_every_page(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path, pages=2)
    pages = [
        output.Page(number=1, lines=[_line(10, 10, 50, 12, conf=0.99)]),
        output.Page(number=2, lines=[]),
    ]
    paths = annotate.annotate_document(pdf, pages, dpi=72, out_dir=tmp_path / "p")
    assert [p.name for p in paths] == ["page-001.png", "page-002.png"]
    assert all(p.exists() for p in paths)


def test_pages_from_document_round_trips(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    page = output.Page(
        number=1, lines=[_line(5, 6, 7, 8, conf=0.9)], width=612.0, height=792.0
    )
    doc = output.build_document(pdf, [page], "easyocr", ["en"], dpi=144)
    parsed = json.loads(json.dumps(doc))

    pages = annotate.pages_from_document(parsed)
    assert pages[0].number == 1
    assert pages[0].width == 612.0
    assert pages[0].lines[0].box.to_list() == [
        [5.0, 6.0],
        [12.0, 6.0],
        [12.0, 14.0],
        [5.0, 14.0],
    ]

    paths = annotate.annotate_json(parsed, tmp_path / "pages")
    assert paths and paths[0].exists()
    assert Image.open(paths[0]).size == (1700, 2200)  # 612x792pt at 200dpi


def test_annotate_json_rejects_missing_source(tmp_path: Path) -> None:
    doc = {"source": str(tmp_path / "missing.pdf"), "pages": []}
    with pytest.raises(FileNotFoundError):
        annotate.annotate_json(doc, tmp_path / "out")


def test_annotate_json_rejects_legacy_pixel_space_boxes(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    # Legacy shape: no page dims; boxes in pixels at some DPI overflow the 612pt page.
    doc = {
        "source": str(pdf),
        "pages": [
            {
                "page": 1,
                "lines": [
                    {
                        "box": [[0, 0], [1200, 20], [1200, 40], [0, 40]],
                        "text": "x",
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="pixel-space"):
        annotate.annotate_json(doc, tmp_path / "out")


def test_annotate_json_accepts_legacy_point_space_boxes(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    # Legacy LiteParse shape: no stored dims, but boxes fit the 612pt page.
    doc = {
        "source": str(pdf),
        "pages": [
            {
                "page": 1,
                "lines": [
                    {
                        "box": [[10, 10], [60, 10], [60, 22], [10, 22]],
                        "text": "x",
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }
    paths = annotate.annotate_json(doc, tmp_path / "out")
    assert paths and paths[0].exists()


def test_pages_dir_and_file_naming(tmp_path: Path) -> None:
    json_path = Path("output/sub/report.pdf.json")
    assert annotate.pages_dir_for(json_path).name == "report.pdf.pages"
    assert annotate.page_file_for(tmp_path, 7).name == "page-007.png"


def test_open_folder_is_noop_off_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(annotate.sys, "platform", "linux")
    annotate.open_folder(tmp_path)  # must not raise
