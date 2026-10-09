from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import pytest
from PIL import Image as PILImage
from rich.console import Console

import main
from app import describe as image_describe
from app import output
from app.engine import OcrEngine
from app.geometry import Quad

MODEL = "anthropic/claude-haiku-5.5"


class FakeEngine:
    def __init__(self) -> None:
        self.languages = ["en"]
        self.calls = 0

    def recognize(self, _image: PILImage.Image) -> list:
        self.calls += 1
        return []


class FakeDescriber:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def describe(self, image: bytes, mime: str) -> image_describe.Description:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return image_describe.Description(
            text="a red square",
            usage=None,
            latency_seconds=0.0,
            described_at="2026-01-01T00:00:00+00:00",
        )


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "dpi": 200,
        "lang": "en",
        "cpu": True,
        "list": False,
        "ocr_only": False,
        "annotate": False,
        "force": False,
        "formats": ["json"],
        "combine": False,
        "output": Path("output"),
        "describe_images": True,
        "describe_model": MODEL,
        "max_images_per_doc": 5,
        "min_image_px": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _png(tmp_path: Path, name: str = "a.png") -> Path:
    path = tmp_path / name
    PILImage.new("RGB", (8, 8), "white").save(path)
    return path


def _pre_describe_doc(src: Path) -> dict:
    page = output.Page(number=1, lines=[output.Line("hi", Quad.from_xywh(0, 0, 1, 1), 0.9)])
    return output.build_document(src, [page], "easyocr", ["en"], dpi=200)


@pytest.fixture
def describer(monkeypatch: pytest.MonkeyPatch) -> FakeDescriber:
    fake = FakeDescriber()
    monkeypatch.setenv(main.OPENROUTER_KEY_ENV, "test-key")
    monkeypatch.setattr(
        main.image_describe, "OpenRouterDescriber", lambda api_key, model: fake
    )
    return fake


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> list:
    entries: list = []
    monkeypatch.setattr(main.tui, "run_live", lambda *a, **k: None)
    monkeypatch.setattr(main.tui, "render_completion", lambda *a, **k: None)
    monkeypatch.setattr(
        main.viewer, "choose_file", lambda console, found: entries.extend(found)
    )
    return entries


def _run(
    src: Path, out_dir: Path, engine: FakeEngine, args: argparse.Namespace, root: Path
) -> None:
    main.run_batch(
        [src], root, out_dir, cast(OcrEngine, engine), args, Console(record=True)
    )


def test_topup_adds_image_descriptions_without_ocr(
    tmp_path: Path, describer: FakeDescriber, run: list
) -> None:
    src = _png(tmp_path)
    out_dir = tmp_path / "out"
    out_path = output.output_path_for(src, tmp_path, out_dir)
    output.write_document(_pre_describe_doc(src), out_path)
    engine = FakeEngine()

    _run(src, out_dir, engine, _args(), tmp_path)

    stored = json.loads(out_path.read_text(encoding="utf-8"))
    page = stored["pages"][0]
    assert engine.calls == 0
    assert describer.calls == 1
    assert page["text"] == "hi"
    assert page["images"][0]["status"] == "described"
    assert page["images"][0]["description"] == "a red square"
    assert stored["describe"] == {
        "model": MODEL,
        "max_images_per_doc": 5,
        "prompt_version": output.DESCRIBE_PROMPT_VERSION,
    }
    assert [(entry.status, entry.pages) for entry in run] == [("done", 1)]


def test_up_to_date_doc_is_skipped_with_no_describer_calls(
    tmp_path: Path, describer: FakeDescriber, run: list
) -> None:
    src = _png(tmp_path)
    out_dir = tmp_path / "out"
    _run(src, out_dir, FakeEngine(), _args(), tmp_path)
    assert describer.calls == 1

    engine = FakeEngine()
    _run(src, out_dir, engine, _args(), tmp_path)

    assert describer.calls == 1
    assert engine.calls == 0
    assert [entry.status for entry in run[-1:]] == ["skipped"]


def test_changed_model_redescribes_without_ocr(
    tmp_path: Path, describer: FakeDescriber, run: list
) -> None:
    src = _png(tmp_path)
    out_dir = tmp_path / "out"
    _run(src, out_dir, FakeEngine(), _args(), tmp_path)

    engine = FakeEngine()
    _run(src, out_dir, engine, _args(describe_model="other/model"), tmp_path)

    stored = json.loads(
        output.output_path_for(src, tmp_path, out_dir).read_text(encoding="utf-8")
    )
    assert engine.calls == 0
    assert describer.calls == 2
    assert stored["describe"]["model"] == "other/model"


def test_failed_image_is_recorded_and_file_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: list
) -> None:
    src = _png(tmp_path)
    out_dir = tmp_path / "out"
    failing = FakeDescriber(error=image_describe.DescribeError("model timed out"))
    monkeypatch.setenv(main.OPENROUTER_KEY_ENV, "test-key")
    monkeypatch.setattr(
        main.image_describe, "OpenRouterDescriber", lambda api_key, model: failing
    )

    _run(src, out_dir, FakeEngine(), _args(), tmp_path)

    stored = json.loads(
        output.output_path_for(src, tmp_path, out_dir).read_text(encoding="utf-8")
    )
    image = stored["pages"][0]["images"][0]
    assert image["status"] == "error"
    assert image["error"] == "model timed out"
    assert [entry.status for entry in run] == ["done"]
