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
        paragraphs=[
            output.Paragraph(text="hello", box=Quad.from_xywh(0, 0, 1, 1), confidence=0.9)
        ],
    )
    doc = output.build_document(src, [page], "easyocr", ["en"], dpi=200)
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(src, src.parent, out_dir)
    output.write_document(doc, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["pages"][0]["text"] == "hello"
    assert data["pages"][0]["lines"][0]["confidence"] == 0.9
    assert data["pages"][0]["paragraphs"][0]["kind"] == "paragraph"


def _sample_doc(src: Path, *, dpi: int = 200, engine: str = "easyocr") -> dict:
    pages = [
        output.Page(
            number=1,
            lines=[output.Line("hello", Quad.from_xywh(0, 0, 1, 1), 0.9)],
        ),
        output.Page(number=2, lines=[]),
    ]
    return output.build_document(src, pages, engine, ["en"], dpi=dpi)


def test_existing_document_accepts_fresh_doc(tmp_path: Path) -> None:
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF")
    out = tmp_path / "a.pdf.json"
    output.write_document(_sample_doc(src), out)

    doc = output.existing_document(out, src, dpi=200, languages=["en"])
    assert doc is not None
    assert doc["page_count"] == 2


def test_existing_document_rejects_bad_json_and_empty(tmp_path: Path) -> None:
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF")
    out = tmp_path / "a.pdf.json"

    assert output.existing_document(out, src, dpi=200, languages=["en"]) is None

    out.write_text("", encoding="utf-8")
    assert output.existing_document(out, src, dpi=200, languages=["en"]) is None

    out.write_text("{not json", encoding="utf-8")
    assert output.existing_document(out, src, dpi=200, languages=["en"]) is None

    out.write_text('["a list"]', encoding="utf-8")
    assert output.existing_document(out, src, dpi=200, languages=["en"]) is None


def test_existing_document_rejects_mismatches(tmp_path: Path) -> None:
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF")
    out = tmp_path / "a.pdf.json"
    output.write_document(_sample_doc(src), out)

    # page_count no longer matches the pages list
    broken = json.loads(out.read_text(encoding="utf-8"))
    broken["page_count"] = 5
    out.write_text(json.dumps(broken), encoding="utf-8")
    assert output.existing_document(out, src, dpi=200, languages=["en"]) is None

    # dpi mismatch
    output.write_document(_sample_doc(src, dpi=200), out)
    assert output.existing_document(out, src, dpi=300, languages=["en"]) is None

    # language mismatch
    assert output.existing_document(out, src, dpi=200, languages=["fr"]) is None

    # source mismatch
    other = tmp_path / "b.pdf"
    other.write_bytes(b"%PDF")
    assert output.existing_document(out, other, dpi=200, languages=["en"]) is None


def test_existing_document_require_easyocr(tmp_path: Path) -> None:
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF")
    out = tmp_path / "a.pdf.json"
    output.write_document(_sample_doc(src, engine="liteparse"), out)

    assert output.existing_document(out, src, dpi=200, languages=["en"]) is not None
    assert (
        output.existing_document(
            out, src, dpi=200, languages=["en"], require_easyocr=True
        )
        is None
    )


def test_write_document_is_atomic_and_leaves_no_temp(tmp_path: Path) -> None:
    src = tmp_path / "a.png"
    out = tmp_path / "out" / "a.png.json"
    output.write_document(_sample_doc(src), out)

    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["page_count"] == 2
    leftovers = list(out.parent.glob(f".{out.name}.*.tmp"))
    assert leftovers == []


def test_write_document_cleans_up_on_replace_failure(
    tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "out" / "a.png.json"

    def boom(_src, _dst) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(output.os, "replace", boom)
    try:
        output.write_document({"pages": []}, out)
        raise AssertionError("expected OSError")
    except OSError:
        pass
    assert not out.exists()
    assert list(out.parent.glob(f".{out.name}.*.tmp")) == []


def test_document_stats() -> None:
    doc = {
        "pages": [
            {"text_char_count": 10, "mean_confidence": 0.8},
            {"text_char_count": 20, "mean_confidence": 0.6},
        ]
    }
    assert output.document_stats(doc) == (2, 30, 0.7)
    assert output.document_stats({"pages": []}) == (0, 0, 0.0)


def test_page_regions_prefer_paragraphs() -> None:
    lines = [output.Line(text="a", box=Quad.from_xywh(0, 0, 1, 1), confidence=1.0)]
    paras = [output.Paragraph(text="a b", box=Quad.from_xywh(0, 0, 2, 2), confidence=0.9)]
    assert output.Page(number=1, lines=lines, paragraphs=paras).regions == paras
    assert output.Page(number=1, lines=lines).regions == lines  # fallback


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
    entry = viewer.ViewEntry(name="x.png", path=tmp_path / "x.png.json", status="failed")
    console = Console(record=True)
    viewer.choose_file(console, [entry])
    assert "Inspect" not in console.export_text()


def test_choose_file_bare_v_targets_single_file(tmp_path: Path, monkeypatch) -> None:
    json_path = tmp_path / "x.pdf.json"
    json_path.write_text("{}", encoding="utf-8")
    entry = viewer.ViewEntry(name="x.pdf", path=json_path, pages=1)

    prompted: list[Path] = []

    def fake_ask(*_args, **_kwargs) -> str:
        return "v" if not prompted else ""

    def fake_annotate(_console, _doc, json_path, *, reveal=True) -> bool:
        prompted.append(json_path)
        return True

    monkeypatch.setattr(viewer.Prompt, "ask", fake_ask)
    monkeypatch.setattr(viewer, "_annotate_document", fake_annotate)
    console = Console(record=True)
    viewer.choose_file(console, [entry])
    assert prompted == [json_path]


def test_choose_file_bare_v_ambiguous_with_multiple_files(
    tmp_path: Path, monkeypatch
) -> None:
    entries = [
        viewer.ViewEntry(name="a.pdf", path=tmp_path / "a.pdf.json"),
        viewer.ViewEntry(name="b.pdf", path=tmp_path / "b.pdf.json"),
    ]
    answers = iter(["v", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    annotated: list[Path] = []

    def fake_annotate(_console, _doc, json_path, *, reveal=True) -> bool:
        annotated.append(json_path)
        return True

    monkeypatch.setattr(viewer, "_annotate_document", fake_annotate)
    console = Console(record=True)
    viewer.choose_file(console, entries)
    assert annotated == []  # ambiguous: must ask for a number instead
