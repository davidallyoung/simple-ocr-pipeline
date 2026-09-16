"""Interactive / batch viewer for generated OCR JSON documents.

Standalone from the processing session: :func:`run_view` resolves a set of
targets (files or directories), builds :class:`ViewEntry` summaries, and
either lets the user pick one interactively or applies a fixed mode to every
document.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from app import annotate, canvas, export

VIEW_MODES = ("preview", "json", "images", "inspector")
_MODE_ALIASES = {"j": "json", "v": "images", "t": "inspector"}


@dataclass
class ViewEntry:
    """A single viewable OCR JSON and the stats shown in the picker."""

    name: str
    path: Path
    pages: int = 0
    chars: int = 0
    conf: float = 0.0
    status: str = "done"

    @classmethod
    def from_json(cls, path: Path) -> ViewEntry:
        """Read ``path`` and summarise it for the picker listing."""
        doc = load_document(path)
        source = doc.get("source")
        name = Path(str(source)).name if source else path.name
        pages, chars, conf = document_stats(doc)
        return cls(name=name, path=path, pages=pages, chars=chars, conf=conf)


def _conf_style(conf: float) -> str:
    if conf < 0.7:
        return "bright_red"
    if conf < 0.9:
        return "bright_yellow"
    return "green"


def load_document(path: Path) -> dict:
    """Read one OCR output JSON from disk."""
    data: dict = json.loads(path.read_text(encoding="utf-8"))
    return data


def document_stats(doc: dict) -> tuple[int, int, float]:
    """Return ``(page_count, total_chars, mean_confidence)`` for a document."""
    pages = doc.get("pages", [])
    total_conf = sum(p.get("mean_confidence", 1.0) for p in pages) / max(len(pages), 1)
    total_chars = sum(p.get("text_char_count", 0) for p in pages)
    return len(pages), total_chars, total_conf


def discover_targets(targets: list[Path]) -> tuple[list[Path], list[str]]:
    """Resolve user-supplied paths to JSON files, never raising.

    ``targets`` may contain JSON files, directories (searched recursively for
    ``*.json``), or missing/invalid paths. Returns ``(files, errors)`` where
    ``files`` is de-duplicated preserving first-seen order.
    """
    found: list[Path] = []
    errors: list[str] = []
    seen: set[Path] = set()
    for target in targets:
        if not target.exists():
            errors.append(f"Not found: {target}")
            continue
        if target.is_dir():
            matches = sorted(target.rglob("*.json"))
            if not matches:
                errors.append(f"No JSON files found in: {target}")
                continue
            candidates = matches
        elif target.suffix.lower() == ".json":
            candidates = [target]
        else:
            errors.append(f"Not a JSON file or directory: {target}")
            continue
        for candidate in candidates:
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(candidate)
    return found, errors


def render_document(console: Console, doc: dict) -> None:
    """Render one OCR JSON as a readable document (header + page canvases)."""
    source = doc.get("source", "unknown")
    language = doc.get("language", [])
    page_count, total_chars, total_conf = document_stats(doc)

    header = Table.grid(padding=(0, 1))
    header.add_column(justify="right", style="dim")
    header.add_column()
    header.add_row("Source", source)
    header.add_row("Language", ", ".join(language) if language else "—")
    header.add_row("Pages", str(page_count))
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


def _companion_names(json_path: Path) -> list[str]:
    """Names of existing ``.txt``/``.md`` siblings for a result JSON."""
    names: list[str] = []
    for path in (export.text_path_for(json_path), export.markdown_path_for(json_path)):
        if path.exists():
            names.append(path.name)
    return names


def _print_text(console: Console, json_path: Path) -> None:
    """Print the ``.txt`` sibling, falling back to text rebuilt from the JSON.

    OCR text is untrusted input for Rich's markup parser: bracketed tokens
    (``[a]``, ``[/]`` ...) would otherwise be parsed as style tags and
    silently dropped, and an unbalanced closing tag raises ``MarkupError``.
    Wrapping in ``Text`` renders it verbatim.
    """
    txt_path = export.text_path_for(json_path)
    if txt_path.exists():
        console.print(Text(txt_path.read_text(encoding="utf-8")))
        return
    if json_path.exists():
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        rendered = export.render_text(annotate.pages_from_document(doc))
        console.print(Text(rendered))
        return
    console.print(f"[red]No text output found for {json_path.name}.[/]")


def apply_action(
    console: Console, path: Path, action: str, *, reveal: bool = True
) -> bool:
    """Apply a named view action to one JSON file. Returns success."""
    if not path.exists():
        console.print(f"[red]Output missing: {path}[/]")
        return False
    try:
        doc = load_document(path)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not read {path}: {exc}[/]")
        return False

    if action == "json":
        console.print(Syntax(json.dumps(doc, indent=2), "json", word_wrap=True))
        return True
    if action == "images":
        return _annotate_document(console, doc, path, reveal=reveal)
    if action == "inspector":
        return _run_inspector(console, doc)
    render_document(console, doc)
    return True


def choose_file(console: Console, entries: list[ViewEntry]) -> None:
    """Offer the user an interactive picker over completed output JSONs.

    Entering a number previews that file; a ``j``/``p``/``v``/``t`` suffix
    dumps the raw JSON, prints the plain text, writes annotated page images,
    or launches the inspector. Returns when the user hits Enter with no input.
    """
    files = [entry for entry in entries if entry.status in ("done", "skipped")]
    if not files:
        return

    console.print("[bold]Inspect generated results:[/]")
    for idx, entry in enumerate(files, start=1):
        companions = _companion_names(entry.path)
        extra = f"  [dim]· {', '.join(companions)}[/]" if companions else ""
        console.print(
            f"  [cyan]{idx}[/]  {entry.name}  "
            f"[dim]({entry.pages} pg · {entry.chars} chars · "
            f"conf {entry.conf:.2f})[/]{extra}"
        )
    console.print("  [dim]Hint: enter a number to preview, add [bold]j[/bold] for raw "
                  "JSON, [bold]p[/bold] for plain text, [bold]v[/bold] for annotated "
                  "page images, or [bold]t[/bold] for "
                  "the interactive inspector "
                  "(e.g. [bold]1t[/bold]; bare j/p/v/t applies to the only file), "
                  "or Enter to continue.[/]")

    while True:
        try:
            raw = Prompt.ask(
                "[bold]Inspect?[/] (number; [dim]j[/dim] JSON, [dim]p[/dim] text, "
                "[dim]v[/dim] images, "
                "[dim]t[/dim] inspector, [dim]Enter[/dim] to skip)"
            )
        except EOFError:
            return
        choice = (raw or "").strip().lower()
        if not choice:
            return
        wants_text = choice[-1:] == "p"
        action = _MODE_ALIASES.get(choice[-1:])
        if wants_text or action is not None:
            choice = choice[:-1]
        if not choice:
            # Bare "j"/"p"/"v"/"t": only unambiguous when a single file completed.
            if len(files) == 1:
                choice = "1"
            else:
                console.print(
                    f"[yellow]{len(files)} files done — enter a number "
                    "(e.g. 1t).[/]"
                )
                continue
        if not choice.isdigit():
            continue
        idx = int(choice) - 1
        if not 0 <= idx < len(files):
            console.print("[red]No such file.[/]")
            continue

        if wants_text:
            _print_text(console, files[idx].path)
            continue
        apply_action(console, files[idx].path, action or "preview")


def _annotate_document(
    console: Console, doc: dict, json_path: Path, *, reveal: bool = True
) -> bool:
    """Generate annotated page PNGs for a completed JSON and open the folder."""
    console.print("[dim]Rendering annotated pages...[/]")
    try:
        pages_dir = annotate.pages_dir_for(json_path)
        paths = annotate.annotate_json(doc, pages_dir)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        return False
    console.print(f"[green]Wrote {len(paths)} annotated page(s) -> {pages_dir}[/]")
    if reveal:
        annotate.open_folder(pages_dir)
    return True


def _run_inspector(console: Console, doc: dict) -> bool:
    """Launch the Textual hover inspector (requires an interactive terminal)."""
    if not sys.stdin.isatty():
        console.print(
            "[yellow]The interactive inspector needs a terminal (stdin is "
            "piped); use the numbered preview or run without piping.[/]"
        )
        return False
    from app.inspector import OcrInspector

    OcrInspector(doc).run()
    return True


def run_view(console: Console, targets: list[Path], mode: str | None) -> int:
    """Standalone entrypoint: inspect existing JSON results without OCR.

    ``mode`` is one of :data:`VIEW_MODES`, or ``None`` for the interactive
    picker. Returns 1 when there is nothing to show (or every file failed),
    else 0.
    """
    files, errors = discover_targets(targets)
    for error in errors:
        console.print(f"[red]{error}[/]")

    entries: list[ViewEntry] = []
    for path in files:
        try:
            entries.append(ViewEntry.from_json(path))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            console.print(f"[red]Could not read {path}: {exc}[/]")

    if not entries:
        console.print("[yellow]Nothing to view.[/]")
        return 1

    if mode is None:
        choose_file(console, entries)
        return 0

    reveal = sys.stdout.isatty()
    results = [apply_action(console, entry.path, mode, reveal=reveal) for entry in entries]
    return 0 if any(results) else 1
