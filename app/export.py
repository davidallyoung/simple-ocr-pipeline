"""Render OCR results as human-readable plain text and Markdown.

The JSON document written by :mod:`app.output` is the canonical, lossless
artefact; this module derives the friendly siblings (``<source>.txt`` /
``<source>.md``) and optional whole-batch combined files from the same
``output.Page`` objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app import output

#: Formats the CLI understands, in preferred order.
FORMATS: tuple[str, ...] = ("json", "txt", "md")

#: 78-column rule used to delimit files in a combined plain-text export.
COMBINE_RULE = "=" * 78

#: Characters treated as an existing list bullet when normalizing ``list_item``.
_BULLET_CHARS = "-*•·‣◦–—"


def parse_formats(value: str) -> list[str]:
    """Parse a comma-separated format list, validating against :data:`FORMATS`.

    Empty entries are dropped, values are lower-cased, and duplicates are
    removed while preserving first-seen order. Raises ``ValueError`` when the
    list is empty or contains an unknown format.
    """
    parsed: list[str] = []
    for raw in value.split(","):
        fmt = raw.strip().lower()
        if not fmt:
            continue
        if fmt not in FORMATS:
            raise ValueError(
                f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}"
            )
        if fmt not in parsed:
            parsed.append(fmt)
    if not parsed:
        raise ValueError("no formats selected; choose from json, txt, md")
    return parsed


def render_text(pages: list[output.Page]) -> str:
    """Plain text for a document; matches ``build_document(...)[\"text\"]``."""
    return "\n\n".join(page.text for page in pages)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _strip_bullet(text: str) -> str:
    stripped = text.lstrip()
    if stripped[:1] in _BULLET_CHARS:
        stripped = stripped[1:].lstrip()
    return stripped


def _fence_for(text: str) -> str:
    """A backtick fence longer than any backtick run inside ``text`` (min 3)."""
    longest = 0
    run = 0
    for char in text:
        if char == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


def _render_region(region: output.Line | output.Paragraph, level: int) -> str:
    kind = getattr(region, "kind", "paragraph") or "paragraph"
    if kind == "heading":
        text = _normalize(region.text)
        return f"{'#' * level} {text}" if text else ""
    if kind == "list_item":
        text = _normalize(_strip_bullet(region.text))
        return f"- {text}" if text else ""
    if kind == "code":
        fence = _fence_for(region.text)
        body = region.text.strip("\n")
        return f"{fence}text\n{body}\n{fence}"
    return _normalize(region.text)


def render_markdown(
    pages: list[output.Page],
    *,
    title: str | None = None,
    level: int = 1,
) -> str:
    """Markdown for a document: optional title, ``Page N`` headings, regions.

    Region rendering is driven by ``region.kind`` (``heading``, ``list_item``,
    ``code``, otherwise a plain paragraph); blocks are separated by blank lines.
    """
    blocks: list[str] = []
    if title:
        blocks.append(f"{'#' * level} {title}")

    page_level = level + 1
    region_level = level + 2
    for page in pages:
        page_blocks: list[str] = [f"{'#' * page_level} Page {page.number}"]
        for region in page.regions:
            rendered = _render_region(region, region_level)
            if rendered:
                page_blocks.append(rendered)
        if len(page_blocks) == 1:
            page_blocks.append("_No text detected._")
        blocks.append("\n\n".join(page_blocks))
    return "\n\n".join(blocks)


def render_combined_text(entries: list[tuple[str, list[output.Page]]]) -> str:
    """One plain-text file: rule/name/rule headers joined by blank lines."""
    blocks = [
        f"{COMBINE_RULE}\n{name}\n{COMBINE_RULE}\n\n{render_text(pages)}"
        for name, pages in entries
    ]
    return "\n\n".join(blocks)


def render_combined_markdown(
    entries: list[tuple[str, list[output.Page]]], *, title: str
) -> str:
    """One Markdown file: a top title plus each entry nested at level 2."""
    blocks = [f"# {title}"]
    blocks.extend(
        render_markdown(pages, title=name, level=2) for name, pages in entries
    )
    return "\n\n".join(blocks)


def _sibling(json_path: Path, suffix: str) -> Path:
    name = json_path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return json_path.parent / f"{name}{suffix}"


def text_path_for(json_path: Path) -> Path:
    """Map ``report.pdf.json`` to ``report.pdf.txt``."""
    return _sibling(json_path, ".txt")


def markdown_path_for(json_path: Path) -> Path:
    """Map ``report.pdf.json`` to ``report.pdf.md``."""
    return _sibling(json_path, ".md")


def batch_label(anchor: Path, files: list[Path]) -> str:
    """A friendly name for a combined batch file."""
    if len(files) == 1:
        return files[0].stem or files[0].name
    return anchor.name or "batch"


def combine_path_for(
    anchor: Path, files: list[Path], output_dir: Path, suffix: str
) -> Path:
    """Location of a combined file: ``<output_dir>/<batch_label><suffix>``."""
    return output_dir / f"{batch_label(anchor, files)}{suffix}"


def relative_label(source: Path, anchor: Path) -> str:
    """A POSIX-style label for ``source`` relative to ``anchor``."""
    return output.relative_source_path(source, anchor).as_posix()


def write_text(text: str, out_path: Path) -> None:
    """Write ``text`` as UTF-8 with normalized (LF-only) newlines."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    out_path.write_text(normalized, encoding="utf-8", newline="\n")


def write_combined(
    entries: list[tuple[str, list[output.Page]]],
    *,
    anchor: Path,
    files: list[Path],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Write the selected combined export formats; return the paths written."""
    if not entries:
        return []
    written: list[Path] = []
    if "txt" in formats:
        out = combine_path_for(anchor, files, output_dir, ".txt")
        write_text(render_combined_text(entries), out)
        written.append(out)
    if "md" in formats:
        out = combine_path_for(anchor, files, output_dir, ".md")
        write_text(render_combined_markdown(entries, title=batch_label(anchor, files)), out)
        written.append(out)
    return written
