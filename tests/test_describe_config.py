from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from rich.console import Console

import main
from app import output
from app.geometry import Quad

MODEL = "anthropic/claude-haiku-5.5"
KEY_ENV = {"OPENROUTER_API_KEY": "test-key"}


def _png(tmp_path: Path) -> Path:
    path = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return path


def _doc(src: Path, describe: output.DescribeSettings | None) -> dict:
    page = output.Page(number=1, lines=[output.Line("hi", Quad.from_xywh(0, 0, 1, 1), 0.9)])
    return output.build_document(src, [page], "easyocr", ["en"], dpi=200, describe=describe)


def test_describe_defaults_on_when_key_is_set() -> None:
    args = main.parse_args([])
    settings = main.describe_settings(args, Console(record=True), env=KEY_ENV)
    assert settings == output.DescribeSettings(model=MODEL, max_images_per_doc=5)


def test_describe_defaults_off_without_key_and_stays_quiet() -> None:
    console = Console(record=True)
    assert main.describe_settings(main.parse_args([]), console, env={}) is None
    assert console.export_text() == ""


def test_explicit_describe_without_key_warns_and_continues() -> None:
    console = Console(record=True)
    args = main.parse_args(["--describe-images"])
    assert main.describe_settings(args, console, env={}) is None
    assert "OPENROUTER_API_KEY is not set" in console.export_text()


def test_no_describe_images_wins_over_key() -> None:
    args = main.parse_args(["--no-describe-images"])
    assert main.describe_settings(args, Console(record=True), env=KEY_ENV) is None


def test_model_and_cap_flags_reach_settings() -> None:
    args = main.parse_args(["--describe-model", "other/model", "--max-images-per-doc", "2"])
    settings = main.describe_settings(args, Console(record=True), env=KEY_ENV)
    assert settings == output.DescribeSettings(model="other/model", max_images_per_doc=2)


def test_negative_cap_and_min_px_are_rejected() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--max-images-per-doc", "-1"])
    with pytest.raises(SystemExit):
        main.parse_args(["--min-image-px", "abc"])


def test_build_document_records_describe_settings_only_when_set(tmp_path: Path) -> None:
    src = _png(tmp_path)
    settings = output.DescribeSettings(model=MODEL, max_images_per_doc=5)
    assert "describe" not in _doc(src, None)
    assert _doc(src, settings)["describe"] == {
        "model": MODEL,
        "max_images_per_doc": 5,
        "prompt_version": output.DESCRIBE_PROMPT_VERSION,
    }


def test_existing_document_reuse_follows_describe_settings(tmp_path: Path) -> None:
    src = _png(tmp_path)
    out = tmp_path / "a.png.json"
    described = output.DescribeSettings(model=MODEL, max_images_per_doc=5)
    output.write_document(_doc(src, described), out)

    def reuse(describe: output.DescribeSettings | None) -> dict | None:
        return output.existing_document(out, src, dpi=200, languages=["en"], describe=describe)

    assert reuse(described) is not None
    assert reuse(None) is not None
    assert reuse(output.DescribeSettings(model=MODEL, max_images_per_doc=3)) is None
    assert reuse(output.DescribeSettings(model="other/model", max_images_per_doc=5)) is None


def test_existing_document_without_describe_is_stale_when_descriptions_on(
    tmp_path: Path,
) -> None:
    src = _png(tmp_path)
    out = tmp_path / "a.png.json"
    output.write_document(_doc(src, None), out)

    described = output.DescribeSettings(model=MODEL, max_images_per_doc=5)
    assert output.existing_document(out, src, dpi=200, languages=["en"], describe=described) is None
    assert output.existing_document(out, src, dpi=200, languages=["en"]) is not None


def test_plan_batch_reuses_doc_only_for_matching_describe(tmp_path: Path) -> None:
    src = _png(tmp_path)
    out_dir = tmp_path / "out"
    described = output.DescribeSettings(model=MODEL, max_images_per_doc=5)
    output.write_document(_doc(src, described), output.output_path_for(src, tmp_path, out_dir))

    same = main.plan_batch([src], tmp_path, out_dir, dpi=200, languages=["en"], describe=described)
    assert [f.name for _, f, _, _ in same.skipped] == ["a.png"]

    changed = main.plan_batch(
        [src],
        tmp_path,
        out_dir,
        dpi=200,
        languages=["en"],
        describe=output.DescribeSettings(model=MODEL, max_images_per_doc=1),
    )
    assert changed.queued == []
    assert [f.name for _, f, _, _ in changed.topup] == ["a.png"]
