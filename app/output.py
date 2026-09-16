"""Assemble OCR results into JSON documents and write them to the output dir."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.geometry import Quad


@dataclass
class Line:
    text: str
    box: Quad
    confidence: float


@dataclass
class Paragraph:
    """A semantic text region (provider block or geometric line grouping).

    Shares the ``text``/``box``/``confidence`` field names with :class:`Line`
    so renderers can treat both interchangeably; ``kind`` discriminates the
    provider block type (``paragraph``, ``heading``, ``list_item``, ...).
    """

    text: str
    box: Quad
    confidence: float
    kind: str = "paragraph"


@dataclass
class Page:
    number: int
    lines: list[Line]
    width: float | None = None
    height: float | None = None
    paragraphs: list[Paragraph] | None = None

    @property
    def regions(self) -> list[Line | Paragraph]:
        """Boxes to visualize: paragraphs when present, else raw lines."""
        if self.paragraphs:
            return list(self.paragraphs)
        return list(self.lines)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def text_char_count(self) -> int:
        return sum(len(line.text) for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 1.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


def build_document(
    source: Path,
    pages: list[Page],
    engine_name: str,
    languages: list[str],
    dpi: int | None = None,
) -> dict:
    def page_dict(page: Page) -> dict:
        data = {
            "page": page.number,
            "text": page.text,
            "text_char_count": page.text_char_count,
            "mean_confidence": page.mean_confidence,
            "lines": [
                {
                    "box": line.box.to_list(),
                    "text": line.text,
                    "confidence": line.confidence,
                }
                for line in page.lines
            ],
        }
        if page.paragraphs:
            data["paragraphs"] = [
                {
                    "box": para.box.to_list(),
                    "text": para.text,
                    "confidence": para.confidence,
                    "kind": para.kind,
                }
                for para in page.paragraphs
            ]
        if page.width is not None:
            data["width"] = page.width
        if page.height is not None:
            data["height"] = page.height
        return data

    return {
        "source": str(source.resolve()),
        "engine": engine_name,
        "language": languages,
        "processed_at": datetime.now(UTC).isoformat(),
        "text": "\n\n".join(page.text for page in pages),
        "page_count": len(pages),
        "dpi": dpi,
        "pages": [page_dict(page) for page in pages],
    }


def relative_source_path(source: Path, anchor: Path) -> Path:
    """The path of ``source`` relative to ``anchor`` (falling back to the name).

    Mirrors the input folder structure under the output dir; a source outside
    the anchor collapses to its bare name.
    """
    try:
        return source.resolve().relative_to(anchor.resolve())
    except ValueError:
        return Path(source.name)


def output_path_for(source: Path, anchor: Path, output_dir: Path) -> Path:
    """Map a source file to its JSON location under the central output dir.

    Mirrors the path relative to the anchor (the input folder, or the parent
    for a single file). The original extension is kept to avoid collisions
    (``report.pdf`` -> ``report.pdf.json``).
    """
    return output_dir / f"{relative_source_path(source, anchor)}.json"


def write_document(document: dict, out_path: Path) -> None:
    """Write ``document`` to ``out_path`` atomically.

    The JSON is serialized to a temporary file in the destination directory
    (dotted name ending in ``.tmp`` so ``ingest`` never picks it up) and then
    moved into place with :func:`os.replace`, so a crash mid-write can never
    leave a truncated/partial JSON behind.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=f".{out_path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, out_path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _same_source(stored: object, expected: str) -> bool:
    """Compare stored/expected sources case-insensitively (Windows-friendly)."""
    try:
        return os.path.normcase(str(stored)) == os.path.normcase(expected)
    except (TypeError, ValueError):
        return False


def existing_document(
    path: Path,
    source: Path,
    *,
    dpi: int,
    languages: list[str],
    require_easyocr: bool = False,
) -> dict | None:
    """Return a prior output document if it is complete and config-compatible.

    Reuse is keyed on the output JSON plus the source path, ``dpi`` and
    language list, so re-running a folder can skip work that is already done
    without silently mixing results from a different configuration. Returns
    ``None`` when ``path`` is missing/empty, not valid JSON, structurally
    incomplete, or was produced with a different configuration.
    """
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    pages = loaded.get("pages")
    if not isinstance(pages, list):
        return None
    page_count = loaded.get("page_count")
    if page_count is not None and page_count != len(pages):
        return None
    if loaded.get("source") is not None and not _same_source(
        loaded.get("source"), str(source.resolve())
    ):
        return None
    stored_dpi = loaded.get("dpi")
    if stored_dpi is not None and stored_dpi != dpi:
        return None
    stored_languages = loaded.get("language")
    if stored_languages is not None and stored_languages != languages:
        return None
    if require_easyocr and loaded.get("engine") != "easyocr":
        return None
    return loaded


def document_stats(doc: dict) -> tuple[int, int, float]:
    """Return ``(page_count, total_chars, mean_confidence)`` for a document."""
    pages = doc.get("pages", [])
    if not isinstance(pages, list):
        return 0, 0, 0.0
    chars = 0
    conf_sum = 0.0
    for page in pages:
        if not isinstance(page, dict):
            continue
        chars += int(page.get("text_char_count", 0) or 0)
        conf_sum += float(page.get("mean_confidence", 0.0) or 0.0)
    mean_conf = conf_sum / len(pages) if pages else 0.0
    return len(pages), chars, mean_conf
