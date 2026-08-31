from __future__ import annotations

import pytest

from app.geometry import Point, Quad


def test_from_quad_normalizes_to_floats() -> None:
    quad = Quad.from_quad([[10, 20], [110, 20], [110, 40], [10, 40]])
    assert quad.top_left == Point(10.0, 20.0)
    assert quad.bottom_right == Point(110.0, 40.0)
    assert quad.to_list() == [[10.0, 20.0], [110.0, 20.0], [110.0, 40.0], [10.0, 40.0]]


def test_from_quad_rejects_wrong_point_count() -> None:
    with pytest.raises(ValueError):
        Quad.from_quad([[0, 0], [1, 1], [2, 2]])


def test_from_xywh_builds_clockwise_quad() -> None:
    quad = Quad.from_xywh(10.0, 20.0, 30.0, 40.0)
    assert quad.to_list() == [[10.0, 20.0], [40.0, 20.0], [40.0, 60.0], [10.0, 60.0]]


def test_zero_quad() -> None:
    assert Quad.zero().to_list() == [[0.0, 0.0] for _ in range(4)]


def test_scaled_converts_pixel_space_to_points() -> None:
    # A 200 DPI raster: 100px == 36pt.
    quad = Quad.from_xywh(100, 200, 50, 25)
    scaled = quad.scaled(72.0 / 200)
    assert scaled.xyxy == (36.0, 72.0, 54.0, 81.0)


def test_scaled_is_immutable() -> None:
    quad = Quad.from_xywh(10, 10, 10, 10)
    quad.scaled(2.0)
    assert quad.top_left == Point(10.0, 10.0)


def test_xyxy_is_axis_aligned_bounds() -> None:
    # A skewed quad's xyxy rect must span all four corners.
    quad = Quad.from_points([(10, 12), (50, 8), (54, 40), (14, 44)])
    assert quad.xyxy == (10.0, 8.0, 54.0, 44.0)


def test_xyxy_of_axis_aligned_quad() -> None:
    quad = Quad.from_xywh(2, 3, 7, 5)
    assert quad.xyxy == (2.0, 3.0, 9.0, 8.0)
