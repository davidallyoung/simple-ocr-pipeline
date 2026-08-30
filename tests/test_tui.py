from __future__ import annotations

import time

from app import tui


def test_run_live_redraws_only_on_state_change(monkeypatch) -> None:  # noqa: ANN001
    t = tui.Tui()
    t.begin(["a.png", "b.png"], pages_total=2)

    render_calls = {"n": 0}
    original_render = t.render

    def counting_render():
        render_calls["n"] += 1
        return original_render()

    monkeypatch.setattr(t, "render", counting_render)

    started = time.monotonic()

    def worker_done() -> bool:
        return time.monotonic() - started > 0.4

    tui.run_live(t, worker_done, interval=0.05)

    # Initial Live() render + one final update (+1 internal teardown render).
    # The key property: the count stays tiny and bounded even though the loop
    # spins for 0.4s with no state change. The old timer-driven version would
    # have rendered ~8+ times in that window, flickering on large batches.
    assert 2 <= render_calls["n"] <= 3


def test_revision_increments_on_state_change() -> None:
    t = tui.Tui()
    t.begin(["a.png"], pages_total=1)
    rev = t.revision()
    t.start_file(0, 1)
    assert t.revision() == rev + 1
    t.page_done(0, 5, 0.9)
    assert t.revision() == rev + 2
    t.finish_file(0)
    assert t.revision() == rev + 3
