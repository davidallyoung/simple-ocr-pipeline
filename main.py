"""On-premise OCR pipeline with an interactive rich TUI."""

from __future__ import annotations

import argparse
import io
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.prompt import Prompt

from app import ingest, lite, lite_server, output, pdfs, tui, viewer
from app.engine import OcrEngine, cuda_available

DEFAULT_OUTPUT = Path("output")


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
    return parser.parse_args(argv)


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


def run_batch(
    files: list[Path],
    anchor: Path,
    output_dir: Path,
    engine: OcrEngine,
    args: argparse.Namespace,
    console: Console,
) -> None:
    pages_total = sum(count_pages(f, args.dpi) for f in files)
    t = tui.Tui()
    t.begin([f.name for f in files], pages_total)

    def worker() -> None:
        for index, file in enumerate(files):
            try:
                t.start_file(index, pages_total=count_pages(file, args.dpi))
                doc_pages: list[output.Page] = []
                if file.suffix.lower() == ".pdf" and not args.ocr_only:
                    if lite.liteparse_available():
                        doc_pages = pdf_via_liteparse(file, engine, args)
                    else:
                        console.print(
                            f"[yellow]LiteParse not available for {file.name}; "
                            "rasterizing with EasyOCR.[/]"
                        )
                if not doc_pages:
                    for page_num, image in enumerate(
                        iter_pages(file, args.dpi), start=1
                    ):
                        raw = engine.recognize(image)
                        lines = [
                            output.Line(text=text, box=box, confidence=conf)
                            for box, text, conf in raw
                        ]
                        doc_pages.append(output.Page(number=page_num, lines=lines))
                for page in doc_pages:
                    t.page_done(index, page.text_char_count, page.mean_confidence)
                doc = output.build_document(
                    file, doc_pages, engine_name="easyocr", languages=engine.languages
                )
                output.write_document(doc, output.output_path_for(file, anchor, output_dir))
                t.finish_file(index)
            except Exception as exc:  # per-file isolation; keep batch going
                t.fail_file(index, repr(exc))

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    def worker_done() -> bool:
        return not worker_thread.is_alive()

    tui.run_live(t, worker_done)
    worker_thread.join()

    tui.render_completion(t.snapshot(), str(output_dir.resolve()))

    done_entries = [
        (job, output.output_path_for(file, anchor, output_dir))
        for job, file in zip(t.snapshot(), files, strict=True)
        if job.status == "done"
    ]
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
        console.print(f"{len(files)} file(s):")
        total_pages = 0
        for file in files:
            pages = count_pages(file, args.dpi)
            total_pages += pages
            console.print(f"  [cyan]{file}[/]  ({pages} {'page' if pages == 1 else 'pages'})")
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
