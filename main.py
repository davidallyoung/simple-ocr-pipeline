"""On-premise OCR pipeline with an interactive rich TUI."""

from __future__ import annotations

import argparse
import io
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.prompt import Prompt

from app import (
    annotate,
    export,
    ingest,
    lite,
    lite_server,
    output,
    paragraphs,
    pdfs,
    tui,
    viewer,
)
from app.engine import OcrEngine, cuda_available

DEFAULT_OUTPUT = Path("output")


def _formats_arg(value: str) -> list[str]:
    try:
        return export.parse_formats(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ocr",
        description="OCR PDFs and images on-premise, with a live status TUI.",
    )
    parser.add_argument(
        "path", nargs="?", help="File or folder to OCR (omit for interactive loop)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Central directory for JSON results (default: ./output)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="PDF rasterization DPI (default: 200)"
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Comma-separated EasyOCR language codes (default: en)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Dry run: list files and page counts without OCR",
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="PDFs: rasterize every page and OCR with EasyOCR directly "
        "(bypass LiteParse; no text-layer extraction)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Write per-page PNGs with OCR boxes overlaid, colored by "
        "confidence (output/<file>.pages/page-NNN.png)",
    )
    parser.add_argument(
        "--format",
        "--formats",
        dest="formats",
        type=_formats_arg,
        default=export.parse_formats("json,txt"),
        help="Comma-separated output formats: json,txt,md (default: json,txt)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also write one combined text/markdown file for the whole batch",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if their output JSON already exists "
        "(default: skip already-processed files)",
    )
    parser.add_argument(
        "--view",
        nargs="*",
        metavar="PATH",
        default=None,
        help="Inspect existing result JSONs without OCR: a JSON file or a "
        "directory searched recursively for *.json (default: --output)",
    )
    parser.add_argument(
        "--view-mode",
        choices=viewer.VIEW_MODES,
        default=None,
        help="Apply one view to every target: preview, json, images, inspector "
        "(default: interactive picker)",
    )
    args = parser.parse_args(argv)
    if args.view is not None:
        if args.path:
            parser.error("--view cannot be combined with a positional path")
        if args.list:
            parser.error("--view cannot be combined with --list")
        if args.ocr_only:
            parser.error("--view cannot be combined with --ocr-only")
        if args.annotate:
            parser.error("--view cannot be combined with --annotate")
    if args.view_mode is not None and args.view is None:
        parser.error("--view-mode requires --view")
    return args


def count_pages(file: Path, dpi: int) -> int:
    if file.suffix.lower() == ".pdf":
        try:
            return pdfs.pdf_page_count(file)
        except Exception:
            return 0
    return 1


def iter_pages(file: Path, dpi: int) -> Iterator[Image.Image]:
    if file.suffix.lower() == ".pdf":
        yield from pdfs.iter_pdf_pages(file, dpi)
    else:
        yield Image.open(file).convert("RGB")


def pdf_via_liteparse(
    file: Path, engine: OcrEngine, args: argparse.Namespace
) -> list[output.Page]:
    """Parse a PDF with LiteParse (text layer + selective OCR) unless --ocr-only."""
    use_ocr = False
    if args.ocr_only or not lite.liteparse_available():
        raise RuntimeError("LiteParse path unavailable; use the EasyOCR-only path")
    use_ocr = lite.needs_ocr(file)
    language = engine.languages[0] if engine.languages else "en"
    return lite.parse_pdf(file, engine, dpi=args.dpi, use_ocr=use_ocr, language=language)


@dataclass
class BatchPlan:
    """Partition of a batch into work to run and outputs to reuse.

    ``queued`` holds ``(index, file, page_count)`` for files that will be
    processed; ``skipped`` holds ``(index, file, out_path, doc)`` for files
    whose complete, config-compatible output already exists.
    """

    queued: list[tuple[int, Path, int]]
    skipped: list[tuple[int, Path, Path, dict]]


def plan_batch(
    files: list[Path],
    anchor: Path,
    output_dir: Path,
    *,
    dpi: int,
    languages: list[str],
    force: bool = False,
    require_easyocr: bool = False,
) -> BatchPlan:
    """Pre-pass deciding which files to process and which to reuse.

    Runs before the TUI/worker start so skipped PDFs are never opened: the
    page count for reused files comes from the cached JSON, not the source.
    """
    queued: list[tuple[int, Path, int]] = []
    skipped: list[tuple[int, Path, Path, dict]] = []
    for index, file in enumerate(files):
        out_path = output.output_path_for(file, anchor, output_dir)
        if not force:
            doc = output.existing_document(
                out_path,
                file,
                dpi=dpi,
                languages=languages,
                require_easyocr=require_easyocr,
            )
            if doc is not None:
                skipped.append((index, file, out_path, doc))
                continue
        queued.append((index, file, count_pages(file, dpi)))
    return BatchPlan(queued=queued, skipped=skipped)


def _ensure_annotations(
    out_path: Path, doc: dict, dpi: int, console: Console
) -> None:
    """Regenerate annotated PNGs for a reused document, missing pages only.

    Best-effort: annotation failures must never fail an otherwise resumed run.
    """
    try:
        pages_dir = annotate.pages_dir_for(out_path)
        pages = annotate.pages_from_document(doc)
        missing = [
            page
            for page in pages
            if not annotate.page_file_for(pages_dir, page.number).exists()
        ]
        if not missing:
            return
        source = Path(str(doc.get("source", "")))
        annotate.annotate_document(source, missing, dpi=dpi, out_dir=pages_dir)
    except Exception as exc:  # annotation is best-effort
        console.print(f"[yellow]Annotation failed for {out_path.name}: {exc!r}[/]")


def run_batch(
    files: list[Path],
    anchor: Path,
    output_dir: Path,
    engine: OcrEngine,
    args: argparse.Namespace,
    console: Console,
) -> None:
    formats = list(args.formats)
    plan = plan_batch(
        files,
        anchor,
        output_dir,
        dpi=args.dpi,
        languages=engine.languages,
        force=args.force,
        require_easyocr=args.ocr_only,
    )
    pages_total = sum(pages for _, _, pages in plan.queued)
    t = tui.Tui()
    t.begin([f.name for f in files], pages_total)
    combine_entries: list[tuple[str, list[output.Page]]] = []

    for index, _file, out_path, doc in plan.skipped:
        t.mark_skipped(index, *output.document_stats(doc))
        if args.annotate:
            _ensure_annotations(out_path, doc, args.dpi, console)

    def worker() -> None:
        for index, file, page_count in plan.queued:
            try:
                t.start_file(index, pages_total=page_count)
                doc_pages: list[output.Page] = []
                used_liteparse = False
                if file.suffix.lower() == ".pdf" and not args.ocr_only:
                    if lite.liteparse_available():
                        doc_pages = pdf_via_liteparse(file, engine, args)
                        used_liteparse = bool(doc_pages)
                    else:
                        console.print(
                            f"[yellow]LiteParse not available for {file.name}; "
                            "rasterizing with EasyOCR.[/]"
                        )
                if not doc_pages:
                    pt_scale = 72.0 / args.dpi
                    for page_num, image in enumerate(
                        iter_pages(file, args.dpi), start=1
                    ):
                        raw = engine.recognize(image)
                        box_scale = pt_scale if file.suffix.lower() == ".pdf" else 1.0
                        lines = [
                            output.Line(
                                text=text, box=quad.scaled(box_scale), confidence=conf
                            )
                            for quad, text, conf in raw
                        ]
                        doc_pages.append(
                            output.Page(
                                number=page_num,
                                lines=lines,
                                width=image.width * box_scale,
                                height=image.height * box_scale,
                                paragraphs=paragraphs.group_paragraphs(lines),
                            )
                        )
                for page in doc_pages:
                    t.page_done(index, page.text_char_count, page.mean_confidence)
                engine_name = "liteparse" if used_liteparse else "easyocr"
                doc = output.build_document(
                    file,
                    doc_pages,
                    engine_name=engine_name,
                    languages=engine.languages,
                    dpi=args.dpi,
                )
                out_path = output.output_path_for(file, anchor, output_dir)
                if "json" in formats:
                    output.write_document(doc, out_path)
                if "txt" in formats:
                    export.write_text(
                        export.render_text(doc_pages), export.text_path_for(out_path)
                    )
                if "md" in formats:
                    export.write_text(
                        export.render_markdown(doc_pages, title=file.name),
                        export.markdown_path_for(out_path),
                    )
                if args.combine and ("txt" in formats or "md" in formats):
                    combine_entries.append(
                        (export.relative_label(file, anchor), doc_pages)
                    )
                t.finish_file(index)
                if args.annotate and doc_pages:
                    try:
                        pages_dir = annotate.pages_dir_for(out_path)
                        annotate.annotate_document(file, doc_pages, dpi=args.dpi, out_dir=pages_dir)
                    except Exception as exc:  # annotation is best-effort
                        console.print(f"[yellow]Annotation failed for {file.name}: {exc!r}[/]")
            except Exception as exc:  # per-file isolation; keep batch going
                t.fail_file(index, repr(exc))

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    def worker_done() -> bool:
        return not worker_thread.is_alive()

    tui.run_live(t, worker_done)
    worker_thread.join()

    combined: list[Path] = []
    if args.combine:
        if "txt" in formats or "md" in formats:
            combined = export.write_combined(
                combine_entries,
                anchor=anchor,
                files=files,
                output_dir=output_dir,
                formats=formats,
            )
            for path in combined:
                console.print(f"[green]Combined[/] {path}")
        else:
            console.print(
                "[yellow]--combine ignored: select a text format "
                "(txt and/or md) to combine.[/]"
            )

    tui.render_completion(
        t.snapshot(),
        str(output_dir.resolve()),
        formats=formats,
        combined=combined,
    )

    done_entries = (
        [
            viewer.ViewEntry(
                name=job.name,
                path=output.output_path_for(file, anchor, output_dir),
                pages=job.pages_done,
                chars=job.chars,
                conf=job.conf,
                status=job.status,
            )
            for job, file in zip(t.snapshot(), files, strict=True)
            if job.status in ("done", "skipped")
        ]
        if "json" in formats
        else []
    )
    viewer.choose_file(console, done_entries)
    viewer.choose_file(console, done_entries)


def handle_batch_input(
    raw_input: str | None,
    console: Console,
    args: argparse.Namespace,
    engine: OcrEngine,
) -> bool:
    """Resolve and process one user-provided path. Returns False if input was empty."""
    prompt = "[bold]Enter a file or folder path to OCR[/] (or press Enter to quit)"
    try:
        user_path = raw_input if raw_input else Prompt.ask(prompt)
    except EOFError:
        return False
    user_path = (user_path or "").strip()
    if not user_path:
        return False
    try:
        anchor, files = ingest.resolve_targets(user_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        return True

    if args.list:
        plan = plan_batch(
            files,
            anchor,
            args.output,
            dpi=args.dpi,
            languages=engine.languages,
            force=args.force,
            require_easyocr=args.ocr_only,
        )
        console.print(f"{len(files)} file(s):")
        total_pages = 0
        skipped = 0
        entries: list[tuple[int, Path, int, dict | None]] = [
            (index, file, pages, None) for index, file, pages in plan.queued
        ]
        entries += [
            (index, file, output.document_stats(doc)[0], doc)
            for index, file, _out_path, doc in plan.skipped
        ]
        for _index, file, pages, doc in sorted(entries, key=lambda entry: entry[0]):
            if doc is None:
                total_pages += pages
                console.print(
                    f"  [cyan]{file}[/]  ({pages} {'page' if pages == 1 else 'pages'})"
                )
            else:
                skipped += 1
                console.print(
                    f"  [yellow]{file}[/]  (already processed, "
                    f"{pages} {'page' if pages == 1 else 'pages'})"
                )
        if skipped:
            console.print(
                f"Total: {total_pages} pages to process "
                f"({skipped} already processed)"
            )
        else:
            console.print(f"Total: {total_pages} pages")
        return True

    run_batch(files, anchor, args.output, engine, args, console)
    return True


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    console = Console()
    console.print("[bold]On-prem OCR (EasyOCR)[/]")

    if args.view is not None:
        targets = [Path(p).expanduser() for p in args.view] or [args.output]
        return viewer.run_view(console, targets, args.view_mode)

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    languages = [lang.strip() for lang in args.lang.split(",") if lang.strip()] or ["en"]
    engine = OcrEngine(languages, gpu=False if args.cpu else None)
    if args.cpu:
        console.print("[yellow]CPU forced via --cpu.[/]")
    elif not cuda_available():
        console.print("[yellow]CUDA not available — falling back to CPU.[/]")

    try:
        first = True
        while True:
            if first and args.path:
                cont = handle_batch_input(args.path, console, args, engine)
                if args.list:
                    break
            else:
                cont = handle_batch_input(None, console, args, engine)
            if not cont:
                break
            first = False
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/]")
    finally:
        lite_server.close_ocr_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
