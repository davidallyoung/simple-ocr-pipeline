"""Assemble OCR results into JSON documents and write them to the output dir."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Line:
    text: str
    box: list[list[float]]
    confidence: float


@dataclass
class Page:
    number: int
    lines: list[Line]

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
) -> dict:
    return {
        "source": str(source.resolve()),
        "engine": engine_name,
        "language": languages,
        "processed_at": datetime.now(UTC).isoformat(),
        "text": "\n\n".join(page.text for page in pages),
        "page_count": len(pages),
        "pages": [
            {
                "page": page.number,
                "text": page.text,
                "text_char_count": page.text_char_count,
                "mean_confidence": page.mean_confidence,
                "lines": [
                    {"box": line.box, "text": line.text, "confidence": line.confidence}
                    for line in page.lines
                ],
            }
            for page in pages
        ],
    }


def output_path_for(source: Path, anchor: Path, output_dir: Path) -> Path:
    """Map a source file to its JSON location under the central output dir.

    Mirrors the path relative to the anchor (the input folder, or the parent
    for a single file). The original extension is kept to avoid collisions
    (``report.pdf`` -> ``report.pdf.json``).
    """
    try:
        rel = source.resolve().relative_to(anchor.resolve())
    except ValueError:
        rel = Path(source.name)
    return output_dir / f"{rel}.json"


def write_document(document: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
