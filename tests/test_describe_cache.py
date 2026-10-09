from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app import describe, images, output
from app.geometry import Quad

MODEL = "anthropic/claude-haiku-5.5"
PROMPT = "1"


class FakeDescriber:
    """Offline describer: the text is the image bytes, and some images fail."""

    def __init__(self, fail_for: frozenset[bytes] = frozenset()) -> None:
        self.calls: list[bytes] = []
        self._fail_for = fail_for

    def describe(self, image: bytes, mime: str) -> describe.Description:
        self.calls.append(image)
        if image in self._fail_for:
            raise describe.DescribeError("model refused")
        return describe.Description(
            text=f"picture of {image.decode()}",
            usage=None,
            latency_seconds=0.5,
            described_at="2026-10-09T12:00:00+00:00",
        )


def _occ(data: bytes, page: int = 1) -> images.ImageOccurrence:
    return images.ImageOccurrence(
        page=page,
        box=Quad.from_xywh(0, 0, 64, 64),
        data=data,
        mime="image/png",
        sha256=hashlib.sha256(data).hexdigest(),
        width=64,
        height=64,
    )


def _saved_document(records: list[output.Image], tmp_path: Path) -> dict:
    """Round-trip a document through JSON, as a second run would read it from disk."""
    pages = [output.Page(number=1, lines=[], images=records)]
    doc = output.build_document(tmp_path / "scan.pdf", pages, "easyocr", ["en"], dpi=200)
    saved: dict = json.loads(json.dumps(doc))
    return saved


def test_second_run_makes_no_client_calls_for_unchanged_images(tmp_path: Path) -> None:
    selections = images.select([_occ(b"cat"), _occ(b"dog", page=2)], min_px=32, cap=5)
    first = FakeDescriber()
    records = describe.describe_images(
        selections, first, model=MODEL, prompt_version=PROMPT, cache={}
    )
    assert first.calls == [b"cat", b"dog"]

    saved = _saved_document(records, tmp_path)
    cache = describe.cached_descriptions(saved, model=MODEL, prompt_version=PROMPT)
    second = FakeDescriber()
    again = describe.describe_images(
        selections, second, model=MODEL, prompt_version=PROMPT, cache=cache
    )

    assert second.calls == []
    assert [r.description for r in again] == ["picture of cat", "picture of dog"]


def test_changed_model_or_prompt_invalidates_cached_descriptions(tmp_path: Path) -> None:
    selections = images.select([_occ(b"cat")], min_px=32, cap=5)
    records = describe.describe_images(
        selections, FakeDescriber(), model=MODEL, prompt_version=PROMPT, cache={}
    )
    saved = _saved_document(records, tmp_path)

    assert describe.cached_descriptions(saved, model="other/model", prompt_version=PROMPT) == {}
    assert describe.cached_descriptions(saved, model=MODEL, prompt_version="2") == {}
    assert describe.cached_descriptions(saved, model=MODEL, prompt_version=PROMPT) == {
        hashlib.sha256(b"cat").hexdigest(): "picture of cat"
    }


def test_duplicate_image_is_described_once_per_run() -> None:
    selections = images.select([_occ(b"cat"), _occ(b"cat", page=2)], min_px=32, cap=5)
    describer = FakeDescriber()

    records = describe.describe_images(
        selections, describer, model=MODEL, prompt_version=PROMPT, cache={}
    )

    assert describer.calls == [b"cat"]
    assert [r.status for r in records] == ["described", "described"]
    assert [r.description for r in records] == ["picture of cat", "picture of cat"]


def test_failed_image_is_recorded_on_every_occurrence_and_not_retried() -> None:
    selections = images.select(
        [_occ(b"bad"), _occ(b"good", page=2), _occ(b"bad", page=3)], min_px=32, cap=5
    )
    describer = FakeDescriber(fail_for=frozenset({b"bad"}))

    records = describe.describe_images(
        selections, describer, model=MODEL, prompt_version=PROMPT, cache={}
    )

    assert describer.calls == [b"bad", b"good"]
    assert [r.status for r in records] == ["error", "described", "error"]
    assert [r.error for r in records] == ["model refused", None, "model refused"]


def test_cap_skipped_image_makes_no_call() -> None:
    selections = images.select([_occ(b"cat"), _occ(b"dog", page=2)], min_px=32, cap=1)
    describer = FakeDescriber()

    records = describe.describe_images(
        selections, describer, model=MODEL, prompt_version=PROMPT, cache={}
    )

    assert describer.calls == [b"cat"]
    assert (records[1].status, records[1].skip_reason) == ("skipped", "cap")


def test_malformed_prior_output_yields_an_empty_cache() -> None:
    assert describe.cached_descriptions({"pages": "nope"}, model=MODEL, prompt_version=PROMPT) == {}

    doc = {
        "pages": [
            {
                "images": [
                    "not an image",
                    {
                        "status": "described",
                        "model": MODEL,
                        "prompt_version": PROMPT,
                        "sha256": "s1",
                        "description": "",
                    },
                    {
                        "status": "error",
                        "model": MODEL,
                        "prompt_version": PROMPT,
                        "sha256": "s2",
                        "description": "text",
                    },
                ]
            }
        ]
    }
    assert describe.cached_descriptions(doc, model=MODEL, prompt_version=PROMPT) == {}
