from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from app import export, output, tui, viewer
from app.geometry import Quad
from main import parse_args


def _line(text: str, conf: float = 0.9) -> output.Line:
    return output.Line(text=text, box=Quad.from_xywh(0, 0, 10, 10), confidence=conf)


def _para(
    text: str, kind: str = "paragraph", conf: float = 0.9
) -> output.Paragraph:
    return output.Paragraph(
        text=text, box=Quad.from_xywh(0, 0, 10, 10), confidence=conf, kind=kind
    )


# --------------------------------------------------------------------- render_text


def test_render_text_matches_build_document(tmp_path: Path) -> None:
    pages = [
        output.Page(number=1, lines=[_line("one")]),
        output.Page(number=2, lines=[_line("two")]),
    ]
    doc = output.build_document(tmp_path / "x.pdf", pages, "easyocr", ["en"])
    assert export.render_text(pages) == doc["text"]
    assert export.render_text(pages) == "one\n\ntwo"


def test_render_text_empty() -> None:
    assert export.render_text([]) == ""


def test_render_text_page_separator() -> None:
    pages = [
        output.Page(number=1, lines=[_line("a"), _line("b")]),
        output.Page(number=2, lines=[]),
        output.Page(number=3, lines=[_line("c")]),
    ]
    assert export.render_text(pages) == "a\nb\n\n\n\nc"


# ------------------------------------------------------------------ render_markdown


def test_render_markdown_title_and_page_headings() -> None:
    pages = [
        output.Page(number=1, lines=[_line("hello")]),
        output.Page(number=2, lines=[]),
    ]
    md = export.render_markdown(pages, title="doc.pdf")
    assert md.startswith("# doc.pdf")
    assert "## Page 1" in md
    assert "## Page 2" in md
    assert "_No text detected._" in md


def test_render_markdown_no_title() -> None:
    md = export.render_markdown([output.Page(number=1, lines=[_line("hi")])])
    assert md.startswith("## Page 1")
    assert "hi" in md


def test_render_markdown_level_offsets() -> None:
    page = output.Page(
        number=1, lines=[_line("raw")], paragraphs=[_para("Section", kind="heading")]
    )
    md = export.render_markdown([page], title="T", level=2)
    assert md.startswith("## T")
    assert "### Page 1" in md
    assert "#### Section" in md


def test_render_markdown_kind_mapping() -> None:
    page = output.Page(
        number=1,
        lines=[_line("raw")],
        paragraphs=[
            _para("A paragraph   with\nnewline"),
            _para("Big Title", kind="heading"),
            _para("- first item", kind="list_item"),
            _para("x = 1\n  y = 2", kind="code"),
        ],
    )
    md = export.render_markdown([page])
    assert "### A paragraph with newline" not in md  # paragraph, not a heading
    assert "A paragraph with newline" in md
    assert "### Big Title" in md
    assert "- first item" in md
    assert "```text\nx = 1\n  y = 2\n```" in md


def test_render_markdown_list_item_strips_various_bullets() -> None:
    page = output.Page(
        number=1,
        lines=[],
        paragraphs=[
            _para("• bullet", kind="list_item"),
            _para("* star", kind="list_item"),
            _para("plain item", kind="list_item"),
        ],
    )
    md = export.render_markdown([page])
    assert "- bullet" in md
    assert "- star" in md
    assert "- plain item" in md
    assert "•" not in md


def test_render_markdown_unknown_kind_is_paragraph() -> None:
    page = output.Page(
        number=1, lines=[], paragraphs=[_para("weird  spacing", kind="figure")]
    )
    assert "weird spacing" in export.render_markdown([page])


def test_render_markdown_code_fence_widens_for_backticks() -> None:
    page = output.Page(
        number=1, lines=[], paragraphs=[_para("```\ncode\n```", kind="code")]
    )
    md = export.render_markdown([page])
    assert "````text" in md
    assert md.rstrip().endswith("````")


def test_render_markdown_line_fallback_without_paragraphs() -> None:
    page = output.Page(number=1, lines=[_line("line one"), _line("line two")])
    md = export.render_markdown([page])
    assert "line one" in md
    assert "line two" in md


def test_render_markdown_no_text_detected() -> None:
    md = export.render_markdown([output.Page(number=1, lines=[])])
    assert "_No text detected._" in md


def test_render_markdown_empty_heading_falls_back() -> None:
    page = output.Page(number=1, lines=[], paragraphs=[_para("   ", kind="heading")])
    assert "_No text detected._" in export.render_markdown([page])


# ------------------------------------------------------------------- parse_formats


def test_parse_formats_valid() -> None:
    assert export.parse_formats("json,txt") == ["json", "txt"]
    assert export.parse_formats("md") == ["md"]


def test_parse_formats_strips_lowercases_and_dedupes() -> None:
    assert export.parse_formats(" JSON , md ,txt") == ["json", "md", "txt"]
    assert export.parse_formats("txt,json,txt") == ["txt", "json"]
    assert export.parse_formats(",txt,,md,") == ["txt", "md"]


def test_parse_formats_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        export.parse_formats("xml")


def test_parse_formats_rejects_empty() -> None:
    with pytest.raises(ValueError):
        export.parse_formats("")
    with pytest.raises(ValueError):
        export.parse_formats(" , ,")


# --------------------------------------------------------------------- path helpers


def test_text_and_markdown_paths() -> None:
    json_path = Path("out/report.pdf.json")
    assert export.text_path_for(json_path) == Path("out/report.pdf.txt")
    assert export.markdown_path_for(json_path) == Path("out/report.pdf.md")


def test_path_helpers_without_json_suffix() -> None:
    assert export.text_path_for(Path("out/report")) == Path("out/report.txt")


def test_combine_path_single_file() -> None:
    anchor = Path("/scans")
    files = [Path("/scans/invoice.pdf")]
    assert export.combine_path_for(anchor, files, Path("out"), ".txt") == Path(
        "out/invoice.txt"
    )


def test_combine_path_folder() -> None:
    anchor = Path("/scans/2026")
    files = [Path("/scans/2026/a.pdf"), Path("/scans/2026/b.pdf")]
    assert export.combine_path_for(anchor, files, Path("out"), ".md") == Path(
        "out/2026.md"
    )


def test_batch_label_single_vs_folder() -> None:
    assert export.batch_label(Path("/scans"), [Path("/scans/a.pdf")]) == "a"
    assert (
        export.batch_label(Path("/scans"), [Path("/scans/a.pdf"), Path("/scans/b.pdf")])
        == "scans"
    )
    assert export.batch_label(Path("/"), [Path("/a"), Path("/b")]) == "batch"


def test_relative_label_uses_posix() -> None:
    anchor = Path("/scans")
    assert export.relative_label(Path("/scans/sub/a.pdf"), anchor) == "sub/a.pdf"


# ----------------------------------------------------------------------- write_text


def test_write_text_normalizes_newlines(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "x.txt"
    export.write_text("a\r\nb\rc\n", out)
    assert out.read_bytes() == b"a\nb\nc\n"


# ------------------------------------------------------------------- write_combined


def test_write_combined_selected_formats(tmp_path: Path) -> None:
    entries = [("a.pdf", [output.Page(number=1, lines=[_line("hello")])])]
    files = [tmp_path / "a.pdf"]
    out_dir = tmp_path / "out"
    txt_paths = export.write_combined(
        entries, anchor=tmp_path, files=files, output_dir=out_dir, formats=["txt"]
    )
    assert txt_paths == [out_dir / "a.txt"]
    assert "hello" in txt_paths[0].read_text(encoding="utf-8")

    md_paths = export.write_combined(
        entries, anchor=tmp_path, files=files, output_dir=out_dir, formats=["md"]
    )
    assert md_paths == [out_dir / "a.md"]
    assert "# a" in md_paths[0].read_text(encoding="utf-8")


def test_write_combined_empty_entries(tmp_path: Path) -> None:
    paths = export.write_combined(
        [], anchor=tmp_path, files=[], output_dir=tmp_path / "out", formats=["txt", "md"]
    )
    assert paths == []


# --------------------------------------------------------------- combined rendering


def test_render_combined_text_separators_and_order() -> None:
    entries = [
        ("dir/a.pdf", [output.Page(number=1, lines=[_line("first")])]),
        ("dir/b.pdf", [output.Page(number=1, lines=[_line("second")])]),
    ]
    text = export.render_combined_text(entries)
    assert text.startswith(
        f"{export.COMBINE_RULE}\ndir/a.pdf\n{export.COMBINE_RULE}\n\nfirst"
    )
    assert text.index("first") < text.index("second")
    assert text.count(export.COMBINE_RULE) == 4


def test_render_combined_markdown_nesting() -> None:
    entries = [("dir/a.pdf", [output.Page(number=1, lines=[_line("first")])])]
    md = export.render_combined_markdown(entries, title="batch")
    assert md.startswith("# batch")
    assert "## dir/a.pdf" in md
    assert "### Page 1" in md


# ----------------------------------------------------------------------------- CLI


def test_parse_args_formats_default() -> None:
    args = parse_args([])
    assert args.formats == ["json", "txt"]
    assert args.combine is False


def test_parse_args_formats_alias() -> None:
    assert parse_args(["--format", "json,md"]).formats == ["json", "md"]
    assert parse_args(["--formats", "md"]).formats == ["md"]
    assert parse_args(["--combine"]).combine is True


def test_parse_args_rejects_bad_format() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--format", "bogus"])


# -------------------------------------------------------------- completion render


def test_render_completion_reports_formats_and_combined() -> None:
    job = tui.FileJob(
        name="a.pdf", status="done", pages_done=1, pages_total=1, chars=5, conf=0.9
    )
    console = Console(record=True)
    tui.render_completion(
        [job],
        "/out",
        formats=["json", "txt"],
        combined=[Path("/out/a.txt")],
        console=console,
    )
    text = console.export_text()
    assert "json, txt" in text
    assert "a.txt" in text


# -------------------------------------------------------------------------- viewer


def test_viewer_lists_companion_names(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    json_path = tmp_path / "x.pdf.json"
    json_path.write_text("{}", encoding="utf-8")
    (tmp_path / "x.pdf.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "x.pdf.md").write_text("# hello", encoding="utf-8")
    entry = viewer.ViewEntry(
        name="x.pdf", path=json_path, pages=1, chars=5, conf=0.9, status="done"
    )

    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: "")
    console = Console(record=True)
    viewer.choose_file(console, [entry])
    text = console.export_text()
    assert "x.pdf.txt" in text
    assert "x.pdf.md" in text


def test_viewer_p_prints_text_sibling(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    json_path = tmp_path / "x.pdf.json"
    json_path.write_text("{}", encoding="utf-8")
    (tmp_path / "x.pdf.txt").write_text("printed text", encoding="utf-8")
    entry = viewer.ViewEntry(name="x.pdf", path=json_path, status="done")

    answers = iter(["p", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    console = Console(record=True)
    viewer.choose_file(console, [entry])
    assert "printed text" in console.export_text()


def test_viewer_p_falls_back_to_json(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    pages = [output.Page(number=1, lines=[_line("from json")])]
    doc = output.build_document(tmp_path / "x.pdf", pages, "easyocr", ["en"])
    json_path = tmp_path / "x.pdf.json"
    json_path.write_text(json.dumps(doc), encoding="utf-8")
    entry = viewer.ViewEntry(name="x.pdf", path=json_path, status="done")

    answers = iter(["p", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    console = Console(record=True)
    viewer.choose_file(console, [entry])
    assert "from json" in console.export_text()


def test_viewer_p_errors_when_no_text(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    json_path = tmp_path / "x.pdf.json"  # neither JSON nor .txt exists
    entry = viewer.ViewEntry(name="x.pdf", path=json_path, status="done")

    answers = iter(["p", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    console = Console(record=True)
    viewer.choose_file(console, [entry])
    assert "No text output found" in console.export_text()


def test_viewer_p_prints_markup_like_text_verbatim(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """Bracket tokens in OCR text must not be parsed as Rich markup.

    Regression: printing the raw string let Rich swallow tokens like ``[a]``
    as style tags and raise ``MarkupError`` on an unbalanced ``[/]``.
    """
    json_path = tmp_path / "x.pdf.json"
    (tmp_path / "x.pdf.txt").write_text(
        "Section [a] covers intro\nand [/] closes nothing", encoding="utf-8"
    )
    entry = viewer.ViewEntry(name="x.pdf", path=json_path, status="done")

    answers = iter(["p", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    console = Console(record=True)
    viewer.choose_file(console, [entry])  # must not raise
    text = console.export_text()
    assert "Section [a] covers intro" in text
    assert "and [/] closes nothing" in text


def test_viewer_p_falls_back_to_json_markup_verbatim(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """Same guarantee for the JSON-rebuilt fallback path."""
    lines = [_line("citation [12] and tag [ipsum]"), _line("lone closer [/]")]
    pages = [output.Page(number=1, lines=lines)]
    doc = output.build_document(tmp_path / "x.pdf", pages, "easyocr", ["en"])
    json_path = tmp_path / "x.pdf.json"
    json_path.write_text(json.dumps(doc), encoding="utf-8")
    entry = viewer.ViewEntry(name="x.pdf", path=json_path, status="done")

    answers = iter(["p", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    console = Console(record=True)
    viewer.choose_file(console, [entry])  # must not raise
    text = console.export_text()
    assert "citation [12] and tag [ipsum]" in text
    assert "lone closer [/]" in text
