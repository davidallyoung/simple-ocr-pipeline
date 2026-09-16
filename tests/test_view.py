from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

import main
from app import viewer


def _page(chars: int = 10, conf: float = 0.9) -> dict:
    return {
        "page": 1,
        "text_char_count": chars,
        "mean_confidence": conf,
        "lines": [],
    }


def _write_doc(
    path: Path,
    source: str | None = "doc.pdf",
    pages: list[dict] | None = None,
) -> Path:
    doc: dict = {"pages": pages if pages is not None else [_page()]}
    if source is not None:
        doc["source"] = source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# -- ViewEntry -----------------------------------------------------------------


def test_view_entry_from_json_uses_source_name(tmp_path: Path) -> None:
    path = _write_doc(tmp_path / "out" / "report.pdf.json", source="/some/dir/report.pdf")
    entry = viewer.ViewEntry.from_json(path)
    assert entry.name == "report.pdf"
    assert entry.path == path
    assert entry.pages == 1
    assert entry.chars == 10
    assert entry.conf == pytest.approx(0.9)
    assert entry.status == "done"


def test_view_entry_from_json_falls_back_to_path_name(tmp_path: Path) -> None:
    path = _write_doc(tmp_path / "report.pdf.json", source=None)
    entry = viewer.ViewEntry.from_json(path)
    assert entry.name == "report.pdf.json"


def test_document_stats_math() -> None:
    doc = {"pages": [_page(chars=10, conf=0.9), _page(chars=20, conf=0.5)]}
    pages, chars, conf = viewer.document_stats(doc)
    assert pages == 2
    assert chars == 30
    assert conf == pytest.approx(0.7)


def test_document_stats_defaults_for_empty_document() -> None:
    assert viewer.document_stats({}) == (0, 0, 0.0)


# -- discover_targets ----------------------------------------------------------


def test_discover_targets_dedupes_preserving_order(tmp_path: Path) -> None:
    first = _write_doc(tmp_path / "a.json")
    second = _write_doc(tmp_path / "b.json")
    files, errors = viewer.discover_targets([first, second, first])
    assert files == [first, second]
    assert errors == []


def test_discover_targets_directory_search(tmp_path: Path) -> None:
    _write_doc(tmp_path / "sub" / "one.json")
    _write_doc(tmp_path / "two.json")
    files, errors = viewer.discover_targets([tmp_path])
    assert [f.name for f in files] == ["one.json", "two.json"]
    assert errors == []


def test_discover_targets_reports_missing(tmp_path: Path) -> None:
    files, errors = viewer.discover_targets([tmp_path / "nope.json"])
    assert files == []
    assert len(errors) == 1
    assert "nope.json" in errors[0]


def test_discover_targets_reports_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    files, errors = viewer.discover_targets([tmp_path / "empty"])
    assert files == []
    assert errors and "No JSON" in errors[0]


def test_discover_targets_rejects_non_json_file(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("hi", encoding="utf-8")
    files, errors = viewer.discover_targets([other])
    assert files == []
    assert errors and "notes.txt" in errors[0]


# -- choose_file ---------------------------------------------------------------


def test_choose_file_accepts_view_entry(tmp_path: Path, monkeypatch) -> None:
    entry = viewer.ViewEntry.from_json(_write_doc(tmp_path / "x.pdf.json"))
    answers = iter(["1j", ""])
    monkeypatch.setattr(viewer.Prompt, "ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(viewer, "_annotate_document", lambda *a, **k: True)
    console = Console(record=True)
    viewer.choose_file(console, [entry])
    text = console.export_text()
    assert "Inspect generated results" in text
    assert '"source"' in text


# -- arg parsing ---------------------------------------------------------------


def test_parse_view_collects_paths() -> None:
    args = main.parse_args(["--view", "a.json", "b.json"])
    assert args.view == ["a.json", "b.json"]


def test_parse_bare_view_is_empty_list() -> None:
    args = main.parse_args(["--view"])
    assert args.view == []


def test_parse_view_absent_is_none() -> None:
    args = main.parse_args([])
    assert args.view is None


@pytest.mark.parametrize("mode", ["preview", "json", "images", "inspector"])
def test_parse_view_mode_choices(mode: str) -> None:
    args = main.parse_args(["--view", "--view-mode", mode])
    assert args.view_mode == mode


def test_parse_view_mode_requires_view() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--view-mode", "json"])


def test_parse_view_mode_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--view", "--view-mode", "bogus"])


@pytest.mark.parametrize(
    "argv",
    [
        ["some.pdf", "--view", "a.json"],
        ["--view", "--list"],
        ["--view", "--ocr-only"],
        ["--view", "--annotate"],
    ],
)
def test_parse_view_conflicts_exit(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        main.parse_args(argv)


# -- run_view ------------------------------------------------------------------


def test_run_view_preview_non_interactive(tmp_path: Path) -> None:
    path = _write_doc(tmp_path / "report.pdf.json")
    console = Console(record=True)
    assert viewer.run_view(console, [path], "preview") == 0
    text = console.export_text()
    assert "Pages" in text
    assert "doc.pdf" in text


def test_run_view_json_mode(tmp_path: Path) -> None:
    path = _write_doc(tmp_path / "report.pdf.json")
    console = Console(record=True)
    assert viewer.run_view(console, [path], "json") == 0
    assert '"source"' in console.export_text()


def test_run_view_discovers_directory(tmp_path: Path) -> None:
    _write_doc(tmp_path / "sub" / "one.json")
    _write_doc(tmp_path / "two.json")
    console = Console(record=True)
    assert viewer.run_view(console, [tmp_path], "json") == 0


def test_run_view_missing_target_returns_one(tmp_path: Path) -> None:
    console = Console(record=True)
    assert viewer.run_view(console, [tmp_path / "nope.json"], None) == 1
    assert "Not found" in console.export_text()


def test_run_view_corrupt_json_returns_one(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    console = Console(record=True)
    assert viewer.run_view(console, [bad], "preview") == 1
    assert "Could not read" in console.export_text()


def test_run_view_zero_page_document(tmp_path: Path) -> None:
    path = _write_doc(tmp_path / "empty.pdf.json", pages=[])
    console = Console(record=True)
    assert viewer.run_view(console, [path], "preview") == 0
    assert "0" in console.export_text()


def test_run_view_images_delegates(tmp_path: Path, monkeypatch) -> None:
    path = _write_doc(tmp_path / "report.pdf.json")
    calls: list[Path] = []

    def fake_annotate(console, doc, json_path, *, reveal=True) -> bool:
        calls.append(json_path)
        return True

    monkeypatch.setattr(viewer, "_annotate_document", fake_annotate)
    console = Console(record=True)
    assert viewer.run_view(console, [path], "images") == 0
    assert calls == [path]


def test_inspector_refuses_non_tty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(viewer.sys, "stdin", io.StringIO())
    console = Console(record=True)
    assert viewer._run_inspector(console, {}) is False
    assert "needs a terminal" in console.export_text()


def test_run_view_inspector_non_tty_returns_one(tmp_path: Path, monkeypatch) -> None:
    path = _write_doc(tmp_path / "report.pdf.json")
    monkeypatch.setattr(viewer.sys, "stdin", io.StringIO())
    console = Console(record=True)
    assert viewer.run_view(console, [path], "inspector") == 1


# -- end-to-end proof: view mode never touches the OCR pipeline ----------------


def test_main_view_mode_avoids_engine_output_and_server(
    tmp_path: Path, monkeypatch
) -> None:
    path = _write_doc(tmp_path / "report.pdf.json")
    out_dir = tmp_path / "unused-output"

    class BoomEngine:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("OcrEngine must not be constructed in view mode")

    monkeypatch.setattr(main, "OcrEngine", BoomEngine)

    closed: list[int] = []
    monkeypatch.setattr(main.lite_server, "close_ocr_server", lambda: closed.append(1))

    code = main.main(
        ["--view", str(path), "--view-mode", "preview", "--output", str(out_dir)]
    )
    assert code == 0
    assert not out_dir.exists()
    assert closed == []
