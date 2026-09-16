# Simple OCR Pipeline

> [!NOTE]
> This is a passive project for trying out emerging LLM models — it's a playground, not a production tool.

On-premise OCR for PDFs and images, written in Python with a live TUI. PDFs
are parsed with **LiteParse** (LlamaIndex's fast Rust/PDFium parser): the
embedded text layer is extracted exactly where present, and only pages flagged
as needing OCR (scans, images, sparse text) are sent to a session-wide
EasyOCR GPU server. Standalone images use EasyOCR directly.

## Features

- Input is a **file or a folder** (folders are scanned recursively).
- **PDFs via LiteParse**: `is_complex()` cheaply flags pages needing OCR;
  native text-layer pages are extracted losslessly, flagged pages go through
  an embedded EasyOCR HTTP server (LiteParse OCR API) that reuses the shared
  GPU Reader. Pass `--ocr-only` to rasterize every page with EasyOCR instead.
- **Images** (`png, jpg, jpeg, bmp, tif, tiff, webp`): direct EasyOCR GPU.
- **Live TUI** while processing: per-file status table, running page counts,
  per-page character/confidence stats (moving average), overall progress and
  ETA.
- Results are written per source file to a central output directory,
  mirroring the input folder structure: lossless **JSON** plus human-readable
  **plain text** (`.txt`) and **Markdown** (`.md`) siblings. Pick formats with
  `--format`, and add `--combine` for a single whole-batch text/Markdown file.
- Interactive loop: after a batch finishes, you're told where results landed
  and prompted for the next path — or quit.
- Per-file error isolation: a bad file is marked `failed` and the batch
  continues.
- `--list` dry-run to enumerate files/page counts without running OCR.

## Requirements

- Windows (this guide is Windows-first; Linux/macOS need their own torch index)
- NVIDIA GPU with a recent driver, or CPU-only fallback
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Install

```powershell
uv sync
```

That creates a `.venv` and installs everything, including a **CUDA 12.4 build
of torch/torchvision** pulled from the PyTorch index via `[tool.uv.sources]`
(see `pyproject.toml`). On first OCR run, EasyOCR downloads its detection and
recognition model weights (~100 MB) to `~/.EasyOCR`.

## Usage

Run interactively (prompts for a path, then loops):

```powershell
uv run main.py
```

Run once on a file or folder:

```powershell
uv run main.py "C:\some\invoice.pdf"
uv run main.py "C:\some\scans"
```

Options:

```text
--output DIR     Central output dir (default: ./output)
--dpi N          PDF rasterization DPI (default: 200)
--lang en,fr,... EasyOCR language codes (default: en)
--cpu            Force CPU inference
--list           Dry run: list files and page counts, no OCR
--ocr-only       PDFs: rasterize every page and OCR with EasyOCR directly
                 (bypass LiteParse text-layer extraction)
--annotate       Write per-page PNGs with OCR boxes overlaid, colored by
                 confidence (output/<file>.pages/page-NNN.png)
--format LIST    Comma-separated output formats: json,txt,md
                 (default: json,txt; alias: --formats)
--combine        Also write one combined text/markdown file for the whole batch
```

Example dry run:

```powershell
uv run main.py C:\some\scans --list
```

## Output

For an input folder `C:\some\scans`, results land under:

```
output\C:\some\scans\...   ->   output\<relative-path>.json
```

Wait — paths on Windows are rooted, so mirroring is relative to the anchor.
For a **single file** results are written flat as `output\<name>.<ext>.json`.
For a **folder**, the path *relative to that folder* is preserved under
`output\`.

By default each JSON is accompanied by `.txt` and `.md` siblings named by
stripping the `.json` suffix (`report.pdf.json` -> `report.pdf.txt` /
`report.pdf.md`). Text files are written UTF-8 with LF-only newlines (any
`\r\n`/`\r` is normalized), so they diff cleanly and open anywhere.
`--format txt,md` (or `--format json`) narrows what is written. With
`--combine`, the whole batch gets one file named after the input
(`output\<name-or-folder>.txt` and/or `.md`), with per-file section headers
for text and nested `##` headings for Markdown. Selecting a format list
without `json` also disables the JSON-based inspection actions below (the `j`
dump and the `t` inspector), since there is no JSON to read.

After each batch, the TUI offers to **inspect the generated results**:
enter a file's number to see a rendered view (per-page text, per-line
confidence and bounding box), add a `j` suffix (e.g. `2j`) to dump the raw
JSON, add a `p` suffix (e.g. `2p`) to print the plain-text sibling, or press
Enter to continue to the next-path prompt.

Each JSON looks like:

```json
{
  "source": "C:\\\\some\\\\scans\\\\page1.png",
  "engine": "easyocr",
  "language": ["en"],
  "processed_at": "2026-08-25T...Z",
  "text": "Line one\\nLine two",
  "page_count": 1,
  "pages": [
    {
      "page": 1,
      "text": "Line one\\nLine two",
      "text_char_count": 18,
      "mean_confidence": 0.97,
      "lines": [
        {"box": [[x, y], ...], "text": "Line one", "confidence": 0.99}
      ]
    }
  ]
}
```

## Development

```powershell
uv run pytest          # smoke tests
uv run ruff check .
uv run mypy .
```

## Notes / trade-offs

- EasyOCR loads all requested language models into **[memory]**, so stick to a
  few languages.
- First `readtext` call triggers model load inside the working thread; the TUI
  keeps rendering (spinner) meanwhile.
- Speed knobs worth knowing: `--dpi` controls raster quality vs. speed; the
  moving-average confidence/char counts live-update in the table.
- LiteParse's `is_complex()` heuristic can flag clean but sparse digital pages
  as needing OCR (e.g. `sparse-text`). If you see scans being OCR'd
  unnecessarily, compare against `--ocr-only` and consider raising `--dpi`.
- The embedded EasyOCR server binds `127.0.0.1` on an ephemeral port for the
  session and is closed automatically on exit.