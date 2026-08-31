from __future__ import annotations

import pytest

from app import output, paragraphs
from app.geometry import Quad


def _line(text: str, x: float, y: float, w: float, h: float, conf: float) -> output.Line:
    return output.Line(text=text, box=Quad.from_xywh(x, y, w, h), confidence=conf)


def test_consecutive_close_lines_merge_into_one_paragraph() -> None:
    lines = [
        _line("one", 36.0, 48.0, 300.0, 19.4, conf=1.0),
        _line("two", 36.0, 68.0, 300.0, 19.4, conf=0.8),
    ]
    paras = paragraphs.group_paragraphs(lines)
    assert len(paras) == 1
    assert paras[0].text == "one two"
    assert paras[0].confidence == pytest.approx(0.9)
    assert paras[0].box.xyxy == (36.0, 48.0, 336.0, 87.4)


def test_vertical_gap_starts_new_paragraph() -> None:
    lines = [
        _line("first para", 36.0, 48.0, 300.0, 19.4, conf=1.0),
        _line("second para", 36.0, 100.0, 300.0, 19.4, conf=1.0),  # 32pt gap
    ]
    paras = paragraphs.group_paragraphs(lines)
    assert len(paras) == 2


def test_disjoint_x_spans_stay_separate_columns() -> None:
    lines = [
        _line("left col", 36.0, 48.0, 200.0, 19.4, conf=1.0),
        _line("right col", 400.0, 48.0, 150.0, 19.4, conf=1.0),
    ]
    paras = paragraphs.group_paragraphs(lines)
    assert len(paras) == 2
    assert paras[0].text == "left col"
    assert paras[1].text == "right col"


def test_unsorted_input_is_sorted_by_position() -> None:
    lines = [
        _line("second", 36.0, 68.0, 300.0, 19.4, conf=1.0),
        _line("first", 36.0, 48.0, 300.0, 19.4, conf=1.0),
    ]
    paras = paragraphs.group_paragraphs(lines)
    assert len(paras) == 1
    assert paras[0].text == "first second"


def test_empty_lines_return_no_paragraphs() -> None:
    assert paragraphs.group_paragraphs([]) == []


def test_box_is_union_of_member_lines() -> None:
    lines = [
        _line("wide", 10.0, 10.0, 400.0, 19.0, conf=1.0),
        _line("narrow", 10.0, 30.0, 100.0, 19.0, conf=1.0),
    ]
    paras = paragraphs.group_paragraphs(lines)
    assert paras[0].box.xyxy == (10.0, 10.0, 410.0, 49.0)


def test_all_zero_boxes_do_not_crash() -> None:
    lines = [output.Line(text="x", box=Quad.zero(), confidence=1.0)]
    paras = paragraphs.group_paragraphs(lines)
    assert len(paras) == 1
