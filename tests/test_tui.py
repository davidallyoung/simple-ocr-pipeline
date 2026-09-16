from __future__ import annotations

import time

from rich.console import Console

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


def test_mark_skipped_sets_status_and_stats() -> None:
    t = tui.Tui()
    t.begin(["a.png"], pages_total=1)
    rev = t.revision()
    t.mark_skipped(0, pages_total=3, chars=42, conf=0.88)
    assert t.revision() == rev + 1
    job = t.snapshot()[0]
    assert job.status == "skipped"
    assert (job.pages_total, job.pages_done) == (3, 3)
    assert (job.chars, job.conf) == (42, 0.88)
    assert job.elapsed == 0.0


def test_summary_accounts_for_skipped() -> None:
    t = tui.Tui()
    t.begin(["a.png", "b.png", "c.png"], pages_total=2)
    t.finish_file(0)
    t.mark_skipped(1, pages_total=1, chars=5, conf=0.9)
    summary = str(t._summary(t.snapshot()))
    assert "skipped" in summary
    assert "1/3 processed" in summary  # only done+failed count as processed


def test_summary_all_skipped_has_no_eta() -> None:
    t = tui.Tui()
    t.begin(["a.png", "b.png"], pages_total=0)
    t.mark_skipped(0, pages_total=1, chars=5, conf=0.9)
    t.mark_skipped(1, pages_total=1, chars=5, conf=0.9)
    summary = str(t._summary(t.snapshot()))
    assert "ETA" not in summary
    assert "2 skipped" in summary


def test_skipped_table_row_has_frozen_elapsed() -> None:
    t = tui.Tui()
    t.begin(["a.png"], pages_total=1)
    t.mark_skipped(0, pages_total=1, chars=5, conf=0.9)
    time.sleep(0.05)
    console = Console(record=True, width=120)
    console.print(t._table(t.snapshot()))
    text = console.export_text()
    assert "skipped" in text
    assert "0.0s" in text
    assert "1/1" in text


def test_render_completion_includes_skipped(monkeypatch) -> None:  # noqa: ANN001
    console = Console(record=True, width=120)
    monkeypatch.setattr(tui, "Console", lambda *a, **k: console)
    jobs = [
        tui.FileJob(
            name="a.png",
            status="skipped",
            pages_total=1,
            pages_done=1,
            chars=5,
            conf=0.9,
        )
    ]
    tui.render_completion(jobs, "/out")
    text = console.export_text()
    assert "skipped (existing output reused)" in text
