"""Paragraph construction from OCR line boxes.

Two sources feed the pipeline's paragraph-level view:

- **Provider blocks** (LiteParse ``LayoutBlock``) are the preferred source —
  naturally occurring regions mapped in :mod:`app.lite`.
- This module provides the geometric fallback for paths without block data
  (the direct EasyOCR path): lines are merged into paragraphs when
  consecutive boxes are vertically close and horizontally related, similar
  in spirit to EasyOCR's own ``get_paragraph`` heuristic but preserving
  per-line confidences.

All boxes are canonical 72-DPI point-space Quads (see ``app.geometry``).
"""

from __future__ import annotations

from statistics import median

from app import output
from app.geometry import Quad

#: Vertical gap between lines (as a fraction of line height) below which two
#: lines are considered part of the same paragraph.
MAX_GAP_FACTOR = 0.6

#: Horizontal overlap (as a fraction of the narrower box) required to merge
#: two consecutive lines; keeps table columns and side-by-side blocks apart.
MIN_OVERLAP_FACTOR = 0.25


def _merge_quads(quads: list[Quad]) -> Quad:
    x0 = min(q.xyxy[0] for q in quads)
    y0 = min(q.xyxy[1] for q in quads)
    x1 = max(q.xyxy[2] for q in quads)
    y1 = max(q.xyxy[3] for q in quads)
    return Quad.from_xywh(x0, y0, x1 - x0, y1 - y0)


def _horizontal_overlap(a: Quad, b: Quad) -> float:
    """Overlap of two x-spans as a fraction of the narrower span."""
    ax0, _, ax1, _ = a.xyxy
    bx0, _, bx1, _ = b.xyxy
    overlap = min(ax1, bx1) - max(ax0, bx0)
    if overlap <= 0:
        return 0.0
    narrower = min(ax1 - ax0, bx1 - bx0)
    if narrower <= 0:
        return 0.0
    return overlap / narrower


def group_paragraphs(lines: list[output.Line]) -> list[output.Paragraph]:
    """Merge OCR lines into paragraph regions geometrically.

    Lines already in reading order are walked top to bottom; a line joins
    the current paragraph when its vertical gap to the previous line is
    small relative to line height and their x-spans overlap. A vertical gap
    larger than the line height (typical paragraph break) or a disjoint
    x-span (new column, table cell) starts a new paragraph.
    """
    if not lines:
        return []
    ordered = sorted(lines, key=lambda ln: (ln.box.xyxy[1], ln.box.xyxy[0]))
    heights = [ln.box.xyxy[3] - ln.box.xyxy[1] for ln in ordered]
    typical_height = median(h for h in heights if h > 0) if any(h > 0 for h in heights) else 1.0

    groups: list[list[output.Line]] = [[ordered[0]]]
    for line in ordered[1:]:
        prev = groups[-1][-1]
        prev_y1 = prev.box.xyxy[3]
        y0 = line.box.xyxy[1]
        gap = y0 - prev_y1
        close = gap <= max(typical_height * MAX_GAP_FACTOR, 0.0)
        aligned = _horizontal_overlap(prev.box, line.box) >= MIN_OVERLAP_FACTOR
        if close and aligned:
            groups[-1].append(line)
        else:
            groups.append([line])

    paragraphs: list[output.Paragraph] = []
    for group in groups:
        text = " ".join(" ".join(ln.text.split()) for ln in group).strip()
        confidence = sum(ln.confidence for ln in group) / len(group)
        paragraphs.append(
            output.Paragraph(
                text=text,
                box=_merge_quads([ln.box for ln in group]),
                confidence=confidence,
                kind="paragraph",
            )
        )
    return paragraphs
