from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from PIL import Image
from rich.console import Console

import main
from app import annotate, output
from app.engine import OcrEngine
from app.geometry import Quad


class FakeEngine:
    """Offline stand-in for OcrEngine: counts recognize() calls."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages if languages is not None else ["en"]
        self.calls = 0

    def recognize(self, _image: Image.Image) -> list:
        self.calls += 1
        return []


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "dpi": 200,
        "lang": "en",
        "cpu": True,
        "list": False,
        "ocr_only": False,
        "annotate": False,
        "force": False,
        "formats": ["json", "txt"],
        "combine": False,
        "output": Path("output"),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _png(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    Image.new("RGB", (8, 8), "white").save(path)
    return path


def _valid_doc(src: Path, *, engine: str = "easyocr", dpi: int = 200) -> dict:
    page = output.Page(
        number=1,
        lines=[output.Line("hi", Quad.from_xywh(0, 0, 1, 1), 0.9)],
    )
    return output.build_document(src, [page], engine, ["en"], dpi=dpi)


def _silence(monkeypatch) -> list:  # noqa: ANN001
    captured: list = []
    monkeypatch.setattr(main.tui, "run_live", lambda *a, **k: None)
    monkeypatch.setattr(main.tui, "render_completion", lambda *a, **k: None)
    monkeypatch.setattr(
        main.viewer, "choose_file", lambda console, entries: captured.extend(entries)
    )
    return captured


def test_plan_batch_partitions_completed_and_pending(tmp_path: Path) -> None:
    a = _png(tmp_path, "a.png")
    b = _png(tmp_path, "b.png")
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(a, tmp_path, out_dir)
    output.write_document(_valid_doc(a), out_path)

    plan = main.plan_batch([a, b], tmp_path, out_dir, dpi=200, languages=["en"])

    assert [(i, f.name) for i, f, _ in plan.queued] == [(1, "b.png")]
    assert [(i, f.name) for i, f, _, _ in plan.skipped] == [(0, "a.png")]
    assert plan.skipped[0][3]["page_count"] == 1


def test_run_batch_reuses_existing_doc(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    a = _png(tmp_path, "a.png")
    b = _png(tmp_path, "b.png")
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(a, tmp_path, out_dir)
    output.write_document(_valid_doc(a), out_path)
    before = out_path.read_text(encoding="utf-8")

    captured = _silence(monkeypatch)
    engine = FakeEngine()
    main.run_batch(
        [a, b], tmp_path, out_dir, cast(OcrEngine, engine), _args(), Console(record=True)
    )

    assert engine.calls == 1  # only b was processed
    assert out_path.read_text(encoding="utf-8") == before  # not clobbered
    statuses = {entry.name: entry.status for entry in captured}
    assert statuses == {"a.png": "skipped", "b.png": "done"}


def test_run_batch_force_reprocesses_all(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    a = _png(tmp_path, "a.png")
    b = _png(tmp_path, "b.png")
    out_dir = tmp_path / "out"
    output.write_document(_valid_doc(a), output.output_path_for(a, tmp_path, out_dir))

    _silence(monkeypatch)
    engine = FakeEngine()
    main.run_batch(
        [a, b], tmp_path, out_dir, cast(OcrEngine, engine), _args(force=True), Console(record=True)
    )
    assert engine.calls == 2


def test_run_batch_reprocesses_corrupt_json(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    a = _png(tmp_path, "a.png")
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(a, tmp_path, out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("{not json", encoding="utf-8")

    _silence(monkeypatch)
    engine = FakeEngine()
    main.run_batch(
        [a], tmp_path, out_dir, cast(OcrEngine, engine), _args(), Console(record=True)
    )
    assert engine.calls == 1
    assert json.loads(out_path.read_text(encoding="utf-8"))["engine"] == "easyocr"


def test_ocr_only_rejects_liteparse_json(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    a = _png(tmp_path, "a.png")
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(a, tmp_path, out_dir)
    output.write_document(_valid_doc(a, engine="liteparse"), out_path)

    _silence(monkeypatch)
    engine = FakeEngine()
    main.run_batch(
        [a], tmp_path, out_dir, cast(OcrEngine, engine), _args(ocr_only=True), Console(record=True)
    )
    assert engine.calls == 1  # liteparse output is not reusable under --ocr-only
    assert json.loads(out_path.read_text(encoding="utf-8"))["engine"] == "easyocr"


def test_ensure_annotations_only_regenerates_missing(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    src = _png(tmp_path, "a.png")
    page = output.Page(
        number=1, lines=[output.Line("hi", Quad.from_xywh(0, 0, 1, 1), 0.9)]
    )
    doc = output.build_document(src, [page], "easyocr", ["en"], dpi=200)
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(src, tmp_path, out_dir)
    output.write_document(doc, out_path)

    calls: list[object] = []

    def fake_annotate(*args: object, **_kwargs: object) -> list:
        calls.append(args)
        return []

    monkeypatch.setattr(main.annotate, "annotate_document", fake_annotate)
    console = Console(record=True)

    main._ensure_annotations(out_path, doc, 200, console)
    assert len(calls) == 1

    # The page PNG now exists, so a second pass must skip rendering entirely.
    pages_dir = annotate.pages_dir_for(out_path)
    pages_dir.mkdir(parents=True, exist_ok=True)
    annotate.page_file_for(pages_dir, 1).write_bytes(b"png")
    main._ensure_annotations(out_path, doc, 200, console)
    assert len(calls) == 1
