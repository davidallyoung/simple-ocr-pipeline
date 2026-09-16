"""Rich-based TUI: global progress bar, per-file status table, current-file line."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

STATUS_LABEL = {
    "pending": "pending",
    "running": "running",
    "done": "done",
    "failed": "failed",
    "skipped": "skipped",
}

STATUS_STYLE = {
    "pending": "dim",
    "running": "bright_cyan",
    "done": "green",
    "failed": "red",
    "skipped": "yellow",
}


@dataclass
class FileJob:
    name: str
    pages_total: int = 0
    pages_done: int = 0
    status: str = "pending"
    started_at: float = 0.0
    elapsed: float = 0.0
    chars: int = 0
    conf: float = 0.0
    error: str = ""


class Tui:
    """Renders an always-up-to-date status view driven by a worker thread."""

    def __init__(self) -> None:
        self.console = Console()
        self.lock = Lock()
        self.jobs: list[FileJob] = []
        self.batch_started_at = 0.0
        self.pages_total = 0
        self.pages_done = 0
        self._revision = 0

        self.global_progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.header = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/bold]"),
            console=self.console,
        )
        self._header_id: TaskID | None = None

    def _bump(self) -> None:
        self._revision += 1

    def revision(self) -> int:
        with self.lock:
            return self._revision

    # ------------------------------------------------------------------ state
    def begin(self, names: list[str], pages_total: int) -> None:
        with self.lock:
            self.jobs = [FileJob(name=name) for name in names]
            self.pages_total = pages_total
            self.pages_done = 0
            self.batch_started_at = time.monotonic()
            self._header_id = self.header.add_task("")
            self.global_progress.add_task("Files", total=len(names))
            self._bump()

    def start_file(self, index: int, pages_total: int) -> None:
        with self.lock:
            job = self.jobs[index]
            job.status = "running"
            job.pages_total = pages_total
            job.started_at = time.monotonic()
            self._bump()

    def page_done(self, index: int, chars: int, conf: float) -> None:
        with self.lock:
            job = self.jobs[index]
            job.pages_done += 1
            job.chars += chars
            job.conf = (job.conf * (job.pages_done - 1) + conf) / job.pages_done
            self.pages_done += 1
            self._bump()

    def finish_file(self, index: int) -> None:
        with self.lock:
            job = self.jobs[index]
            job.status = "done"
            job.elapsed = time.monotonic() - job.started_at
            self._bump()

    def mark_skipped(
        self, index: int, pages_total: int = 0, chars: int = 0, conf: float = 0.0
    ) -> None:
        """Mark a file as reused from an existing output (no work performed)."""
        with self.lock:
            job = self.jobs[index]
            job.status = "skipped"
            job.pages_total = pages_total
            job.pages_done = pages_total
            job.chars = chars
            job.conf = conf
            job.elapsed = 0.0
            job.started_at = time.monotonic()
            self._bump()

    def fail_file(self, index: int, error: str) -> None:
        with self.lock:
            job = self.jobs[index]
            job.status = "failed"
            job.error = error
            job.elapsed = time.monotonic() - job.started_at
            self._bump()

    # ---------------------------------------------------------------- render
    def snapshot(self) -> list[FileJob]:
        return self._snapshot()

    def _snapshot(self) -> list[FileJob]:
        with self.lock:
            return [FileJob(**vars(j)) for j in self.jobs]

    def _table(self, jobs: list[FileJob]) -> Table:
        table = Table(expand=True, header_style="bold")
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("File", max_width=40, no_wrap=True)
        table.add_column("Pages", justify="right", width=9)
        table.add_column("Status", width=8)
        table.add_column("Elapsed", justify="right", width=9)
        table.add_column("Chars", justify="right", width=7)
        table.add_column("Conf", justify="right", width=6)

        for i, job in enumerate(jobs, start=1):
            now = time.monotonic()
            elapsed = (
                job.elapsed
                if job.status in ("done", "failed", "skipped")
                else now - job.started_at
            )
            pages = (
                f"{job.pages_done}/{job.pages_total}"
                if job.status in ("running", "done", "failed", "skipped")
                else "—"
            )
            status = Text(STATUS_LABEL[job.status], style=STATUS_STYLE[job.status])
            if job.status == "failed":
                status.append(f"\n{job.error[:48]}", style="red")
            conf = f"{job.conf:.2f}" if job.chars else "—"
            table.add_row(
                str(i),
                job.name,
                pages,
                status,
                f"{elapsed:.1f}s",
                str(job.chars) if job.chars else "—",
                conf,
            )
        return table

    def _summary(self, jobs: list[FileJob]) -> Text:
        done = sum(1 for j in jobs if j.status == "done")
        failed = sum(1 for j in jobs if j.status == "failed")
        skipped = sum(1 for j in jobs if j.status == "skipped")
        total = len(jobs)
        remaining = total - done - failed - skipped

        parts = [f"[bold blue]Files:[/] {total}"]
        if remaining:
            elapsed = time.monotonic() - self.batch_started_at
            avg = elapsed / max(done + failed, 1)
            eta = avg * remaining
            parts.append(f"[yellow]{done + failed}/{total} processed (ETA {eta:.0f}s)[/]")
        else:
            parts.append(f"[green]{done} done[/]")
        if failed:
            parts.append(f"[red]{failed} failed[/]")
        if skipped:
            parts.append(f"[yellow]{skipped} skipped[/]")
        if self.pages_total:
            parts.append(f"[cyan]Pages:[/] {self.pages_done}/{self.pages_total}")
        return Text.from_markup("  ".join(parts))

    def render(self) -> Panel:
        jobs = self._snapshot()
        running = next((j for j in jobs if j.status == "running"), None)
        if running is not None and self._header_id is not None:
            self.header.update(self._header_id, description=f"Processing {running.name}")
        elif self._header_id is not None:
            self.header.update(self._header_id, description="Idle")

        task_id = self.global_progress.tasks[0].id
        done = sum(1 for j in jobs if j.status in ("done", "failed", "skipped"))
        self.global_progress.update(task_id, completed=done)

        body = Group(self._summary(jobs), self.header, self._table(jobs))
        return Panel(body, title="[bold] OCR In Progress [/]", border_style="blue")


def render_completion(
    jobs: list[FileJob],
    output_dir: str,
    formats: Sequence[str] | None = None,
    combined: Sequence[Path] | None = None,
    console: Console | None = None,
) -> None:
    """Static summary printed after the Live view closes."""
    if console is None:
        console = Console()
    done = sum(1 for j in jobs if j.status == "done")
    failed = sum(1 for j in jobs if j.status == "failed")
    skipped = sum(1 for j in jobs if j.status == "skipped")
    table = Table(expand=True, header_style="bold")
    table.add_column("File")
    table.add_column("Status", width=10)
    table.add_column("Chars", justify="right", width=8)
    table.add_column("Conf", justify="right", width=6)
    for job in jobs:
        status = Text(STATUS_LABEL[job.status], style=STATUS_STYLE[job.status])
        if job.error:
            status.append(f"\n{job.error}", style="red")
        table.add_row(
            job.name,
            status,
            str(job.chars) if job.chars else "0",
            f"{job.conf:.2f}" if job.chars else "—",
        )
    console.print(table)
    console.print(f"[green]Processed {done}/{len(jobs)} files[/]")
    if skipped:
        console.print(f"[yellow]{skipped} skipped (existing output reused)[/]")
    if failed:
        console.print(f"[red]{failed} failed[/]")
    console.print(f"Output written to: [bold cyan]{output_dir}[/]")
    if formats:
        console.print(f"Formats: [bold cyan]{', '.join(formats)}[/]")
    for path in combined or ():
        console.print(f"Combined: [bold cyan]{path}[/]")
    console.print(
        f"[dim]Re-open results later: [bold]uv run main.py --view[/bold] {output_dir}[/]"
    )


def run_live(tui: Tui, worker_done: Callable[[], bool], interval: float = 0.03) -> None:
    """Drive the Live view, redrawing only when the underlying state changes.

    ``auto_refresh`` is disabled and updates are gated on ``revision()`` so an
    idle screen (e.g. many pending rows) is not redrawn on a fixed timer -
    that constant full re-render is what caused flicker on large batches.
    """
    last_revision = -1
    with Live(
        tui.render(), console=tui.console, auto_refresh=False, transient=False
    ) as live:
        while not worker_done():
            revision = tui.revision()
            if revision != last_revision:
                live.update(tui.render())
                last_revision = revision
            time.sleep(interval)
        live.update(tui.render())
