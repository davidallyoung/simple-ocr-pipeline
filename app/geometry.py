"""Canonical geometry model for OCR results.

All bounding boxes live in one canonical space: **72-DPI points (PDF points),
top-left origin, page-relative**. Producers map into it:

- EasyOCR boxes arrive as pixel polygons of the rasterized page; the caller
  converts them with :meth:`Quad.scaled` (``72 / dpi``) where the DPI context
  is known.
- LiteParse text items already come in 72-DPI viewport points and map in via
  :meth:`Quad.from_xywh` unchanged.
- Standalone image files are treated as their own 72-DPI viewport, so one
  pixel equals one point and no scaling applies.

Consumers (annotation, output, the OCR HTTP endpoint) read the model without
caring which engine produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Quad:
    """Four corner points, clockwise from top-left, in 72-DPI point space."""

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    @classmethod
    def from_quad(cls, pts: Sequence[Sequence[float]]) -> Quad:
        """Normalize a raw 4-point polygon (as EasyOCR emits) into a Quad."""
        if len(pts) != 4:
            raise ValueError(f"expected 4 points, got {len(pts)}")
        points = [Point(float(x), float(y)) for x, y in pts]
        return cls(*points)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> Quad:
        """Build a Quad from a LiteParse TextItem rect (x, y, width, height)."""
        x, y, w, h = float(x), float(y), float(w), float(h)
        return cls(
            top_left=Point(x, y),
            top_right=Point(x + w, y),
            bottom_right=Point(x + w, y + h),
            bottom_left=Point(x, y + h),
        )

    @classmethod
    def zero(cls) -> Quad:
        """Degenerate all-zero quad used as a graceful fallback."""
        return cls(Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0))

    @classmethod
    def from_points(cls, pts: Sequence[tuple[float, float]]) -> Quad:
        if len(pts) != 4:
            raise ValueError(f"expected 4 points, got {len(pts)}")
        return cls(*(Point(float(x), float(y)) for x, y in pts))

    @property
    def points(self) -> tuple[Point, Point, Point, Point]:
        return (self.top_left, self.top_right, self.bottom_right, self.bottom_left)

    def scaled(self, factor: float) -> Quad:
        """Return a Quad scaled by ``factor`` around the page origin."""
        return Quad.from_points([(p.x * factor, p.y * factor) for p in self.points])

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding rect as (x0, y0, x1, y1)."""
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_list(self) -> list[list[float]]:
        """Serialize to the JSON polygon shape ``[[x, y], ...]``."""
        return [[p.x, p.y] for p in self.points]
