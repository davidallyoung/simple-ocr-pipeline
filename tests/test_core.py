from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from rich.console import Console

from app import ingest, output, pdfs, viewer
from app.geometry import Quad


def test_ingest_filters_supported_files(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "b.png").write_bytes(b"img")
    (tmp_path / "c.txt").write_bytes(b"not ocr")
    (tmp_path / "sub" / "d.JPG").write_bytes(b"img")

    anchor, files = ingest.resolve_targets(str(tmp_path))
    assert anchor == tmp_path
    assert [f.name for f in files] == ["a.pdf", "b.png", "d.JPG"]
    assert "c.txt" not in [f.name for f in files]


def test_ingest_raises_for_unsupported_file(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    try:
        ingest.resolve_targets(str(bad))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_output_path_mapping(tmp_path: Path) -> None:
    anchor = tmp_path / "scans"
    (anchor / "letters").mkdir(parents=True)
    src = anchor / "letters" / "one.pdf"

    out = output.output_path_for(src, anchor, tmp_path / "out")
    assert out == tmp_path / "out" / "letters" / "one.pdf.json"


def test_rasterize_pdf(tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    pdf_path = tmp_path / "two.pdf"
    doc.save(pdf_path)
    doc.close()

    images = list(pdfs.iter_pdf_pages(pdf_path, dpi=100))
    assert len(images) == 2
    assert images[0].mode == "RGB"


def test_output_document_shape(tmp_path: Path) -> None:
    src = tmp_path / "single.png"
    Image.new("RGB", (4, 4), "white").save(src)

    page = output.Page(
        number=1,
        lines=[output.Line(text="hello", box=Quad.from_xywh(0, 0, 1, 1), confidence=0.9)],
    )
    doc = output.build_document(src, [page], "easyocr", ["en"], dpi=200)
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(src, src.parent, out_dir)
    output.write_document(doc, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["pages"][0]["text"] == "hello"
    assert data["pages"][0]["lines"][0]["confidence"] == 0.9


def test_viewer_renders_document(tmp_path: Path) -> None:
    src = tmp_path / "report.pdf"
    pages = [
        output.Page(
            number=1,
            lines=[output.Line("hello world", Quad.from_xywh(0, 0, 1, 1), 0.99)],
        ),
        output.Page(number=2, lines=[]),
    ]
    doc = output.build_document(src, pages, "easyocr", ["en"])
    out = tmp_path / "out" / "report.pdf.json"
    output.write_document(doc, out)

    console = Console(record=True)
    viewer.render_document(console, json.loads(out.read_text(encoding="utf-8")))
    text = console.export_text()
    assert "hello world" in text
    assert "Page 1" in text
    assert "no text detected" in text
    assert "0.99" in text


def test_choose_file_skips_when_no_done(tmp_path: Path) -> None:
    job = viewer.tui.FileJob(name="x.png", status="failed")
    console = Console(record=True)
    viewer.choose_file(console, [(job, tmp_path / "x.png.json")])
    assert "Inspect" not in console.export_text()


def test_choose_file_bare_v_targets_single_file(tmp_path: Path, monkeypatch) -> None:
    job = viewer.tui.FileJob(name="x.pdf", status="done", pages_done=1, pages_total=1)
    json_path = tmp_path / "x.pdf.json"
    json_path.write_text("{}", encoding="utf-8")

    prompted: list[str] = []

    def fake_ask(*_args, **_kwargs) -> str:
        return "v" if not prompted else ""

    monkeypatch.setattr(viewer.Prompt, "ask", fake_ask)
    monkeypatch.setattr(
        viewer, "_annotate_document", lambda c, d, p: prompted.append(p)
    )
    console = Console(record=True)
    viewer.choose_file(console, [(job, json_path)])
    assert prompted == [json_path]


def test_choose_file_bare_v_ambiguous_with_multiple_files(
    tmp_path: Path, monkeypatch
) -> None:
    jobs = [
        (viewer.tui.FileJob(name="a.pdf", status="done"), tmp_path / "a.pdf.json"),
        (viewer.tui.FileJob(name="b.pdf", status="done"), tmp_path / "b.pdf.json"),
    ]
    answers = iter(["v", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    annotated: list[Path] = []
    monkeypatch.setattr(
        viewer, "_annotate_document", lambda c, d, p: annotated.append(p)
    )
    console = Console(record=True)
    viewer.choose_file(console, jobs)
    assert annotated == []  # ambiguous: must ask for a number instead
