"""Interactive post-batch viewer for generated OCR JSON documents."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from app import annotate, canvas, tui


def _conf_style(conf: float) -> str:
    if conf < 0.7:
        return "bright_red"
    if conf < 0.9:
        return "bright_yellow"
    return "green"


def render_document(console: Console, doc: dict) -> None:
    """Render one OCR JSON as a readable document (header + page canvases)."""
    source = doc.get("source", "unknown")
    language = doc.get("language", [])
    pages = doc.get("pages", [])
    total_conf = sum(p.get("mean_confidence", 1.0) for p in pages) / max(len(pages), 1)
    total_chars = sum(p.get("text_char_count", 0) for p in pages)

    header = Table.grid(padding=(0, 1))
    header.add_column(justify="right", style="dim")
    header.add_column()
    header.add_row("Source", source)
    header.add_row("Language", ", ".join(language) if language else "—")
    header.add_row("Pages", str(len(pages)))
    header.add_row("Chars", str(total_chars))
    header.add_row(
        "Mean conf",
        Text(f"{total_conf:.2f}", style=_conf_style(total_conf)),
    )
    console.print(
        Panel(header, title=f"[bold]{Path(source).name}[/]", border_style="blue")
    )

    for page in annotate.pages_from_document(doc):
        if page.lines:
            body: Text | Text = canvas.render_page(page, cols=console.width - 6)
        else:
            body = Text("(no text detected)", style="dim italic")
        console.print(
            Panel(
                body,
                title=f"[bold]Page {page.number}[/]",
                subtitle=f"mean conf {page.mean_confidence:.2f}",
                subtitle_align="right",
                border_style="blue",
            )
        )

    json_path = doc.get("source", "")
    console.print(f"JSON: [cyan]{json_path}[/]")


def choose_file(
    console: Console, entries: list[tuple[tui.FileJob, Path]]
) -> None:
    """Offer the user an interactive picker over the completed output JSONs.

    Entering a number previews that file; a ``j`` suffix dumps the raw JSON.
    Returns when the user hits Enter with no input.
    """
    jobs = [(job, path) for job, path in entries if job.status == "done"]
    if not jobs:
        return

    console.print("[bold]Inspect generated results:[/]")
    for idx, (job, _path) in enumerate(jobs, start=1):
        console.print(
            f"  [cyan]{idx}[/]  {job.name}  "
            f"[dim]({job.pages_done}/{job.pages_total} pg · "
            f"{job.chars} chars · conf {job.conf:.2f})[/]"
        )
    console.print("  [dim]Hint: enter a number to preview, add [bold]j[/bold] for raw "
                  "JSON or [bold]v[/bold] for annotated page images "
                  "(e.g. [bold]1v[/bold]; bare v/j applies to the only file), "
                  "or Enter to continue.[/]")

    while True:
        try:
            raw = Prompt.ask(
                "[bold]Inspect?[/] (number; [dim]j[/dim] for JSON, [dim]v[/dim] for "
                "annotated pages, [dim]Enter[/dim] to skip)"
            )
        except EOFError:
            return
        choice = (raw or "").strip().lower()
        if not choice:
            return
        raw_json = choice.endswith("j")
        view_images = choice.endswith("v")
        if raw_json or view_images:
            choice = choice[:-1]
        if not choice:
            # Bare "j"/"v": only unambiguous when a single file completed.
            if len(jobs) == 1:
                choice = "1"
            else:
                console.print(
                    f"[yellow]{len(jobs)} files done — enter a number "
                    "(e.g. 1v).[/]"
                )
                continue
        if not choice.isdigit():
            continue
        idx = int(choice) - 1
        if not 0 <= idx < len(jobs):
            console.print("[red]No such file.[/]")
            continue

        job, path = jobs[idx]
        if not path.exists():
            console.print(f"[red]Output missing: {path}[/]")
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if raw_json:
            console.print(Syntax(json.dumps(doc, indent=2), "json", word_wrap=True))
        elif view_images:
            _annotate_document(console, doc, path)
        else:
            render_document(console, doc)


def _annotate_document(console: Console, doc: dict, json_path: Path) -> None:
    """Generate annotated page PNGs for a completed JSON and open the folder."""
    console.print("[dim]Rendering annotated pages...[/]")
    try:
        pages_dir = annotate.pages_dir_for(json_path)
        paths = annotate.annotate_json(doc, pages_dir)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        return
    console.print(f"[green]Wrote {len(paths)} annotated page(s) -> {pages_dir}[/]")
    annotate.open_folder(pages_dir)
