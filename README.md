# Simple OCR Pipeline

Turn PDFs and scanned images into text you can search, Markdown you can read, and JSON that records where every line sits on the page. Point it at a file or a folder, watch the batch run in a live terminal view, then open the results and see what the OCR got right and where it was unsure.

Text recognition runs on your machine. A PDF with a real text layer is read directly, and only pages that need OCR go through [EasyOCR](https://github.com/JaidedAI/EasyOCR). The one network feature is optional image description through OpenRouter, and it stays off unless you set an API key. See [Describe images](#describe-images).

> [!NOTE]
> This is a playground for trying out new LLM models, not a production tool.

## Try it

You need [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```sh
git clone https://github.com/davidallyoung/simple-ocr-pipeline.git
cd simple-ocr-pipeline
uv sync
uv run main.py samples/sample-local-pdf.pdf
```

A status table appears and the sample's 3 pages finish in a fraction of a second. The PDF has a text layer, so nothing needs recognizing and no models load. When the run ends, you see a prompt:

```text
Inspect? (number; j JSON, p text, v images, t inspector, Enter to skip)
```

Type `1` to preview the document as a page layout in your terminal. Type `1p` to print its plain text. Press Enter to leave the picker, then Enter again at the path prompt to quit.

Your results are in `output/`:

```text
output/sample-local-pdf.pdf.json
output/sample-local-pdf.pdf.txt
```

## What you can do

Each section below builds on the one before it.

### Process a file or a folder

```sh
uv run main.py "path/to/invoice.pdf"
uv run main.py "path/to/scans"
```

A folder is searched recursively. The supported types are `.pdf`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, and `.webp`.

Run `uv run main.py` with no path to get an interactive prompt. It asks for a path, processes it, and asks again until you press Enter on an empty line.

If one file fails, the batch continues. The failed file shows `failed` in the table with the error.

To see what a run would do without running OCR, add `--list`:

```sh
uv run main.py path/to/scans --list
```

It prints each file with its page count, and marks files that already have results.

### Choose your output formats

Every source file gets its own results next to a mirrored folder structure under `output/`. JSON and plain text are written by default. Pick formats with `--format`:

```sh
uv run main.py path/to/scans --format json,txt,md
uv run main.py path/to/scans --format md --combine
```

- `json` is the full record: text, boxes, confidence, and image descriptions.
- `txt` is the page text, UTF-8 with LF newlines, so it diffs cleanly.
- `md` is Markdown built from the document's structure: headings become `#` headings, list items become `-` bullets, and code blocks are fenced. Each page gets a `Page N` heading.

`--combine` also writes one text or Markdown file for the whole batch, with a section per source file. It needs `txt`, `md`, or both in `--format`.

### Check how well it did

Every line and paragraph carries a confidence score. After a batch, the picker lists the finished files so you can look at them straight away. Enter a file's number, optionally followed by a letter:

| Input | What you get |
|---|---|
| `2` | A terminal preview of each page as a character grid, with boxes colored by confidence |
| `2j` | The raw JSON |
| `2p` | The plain text |
| `2v` | Page PNGs with boxes drawn on them, written to `<name>.pages/page-NNN.png` and opened in your file manager |
| `2t` | The full-screen inspector |

A bare `j`, `p`, `v`, or `t` works when only one file finished.

Colors mean the same thing everywhere. Red is below 0.70, amber is 0.70 to 0.90, and green is 0.90 or higher.

The inspector is a full-screen Textual app. Hover a box, or press Tab and Shift+Tab to cycle through them, and the status bar shows its text, confidence, and coordinates in PDF points. Click a box to pin its details. Press `n` and `p` to change pages and `q` or Esc to quit. It needs a real terminal, so it will not start when input is piped.

To write the annotated PNGs during processing instead of after, add `--annotate`.

### Pick up where you left off

Run the same command again and finished files are marked `skipped` instead of reprocessed. You pay only for the work that is missing, so you can stop a large folder halfway and restart it.

Pass `--force` to redo everything. A file is reused only when its output JSON exists and matches the source path, `--dpi`, `--lang`, and image-description settings of the current run. Change any of those and the file is processed again. `--ocr-only` also refuses to reuse a JSON that LiteParse produced.

Two limits to know about:

- Editing a source file does not trigger reprocessing, because no content hash is stored. Use `--force`.
- Reuse reads the JSON, so a run that leaves `json` out of `--format` has nothing to reuse.

Output JSON is written to a temporary file and moved into place, so an interrupted run never leaves a half-written result. With `--annotate`, a reused file regenerates only the page images that are missing.

### Describe images

PDFs and image files often hold pictures that OCR cannot read: charts, photos, diagrams. If you set `OPENROUTER_API_KEY`, the pipeline sends each picture to a vision model and stores a one or two sentence description, plus a transcription of any text visible in it.

```sh
export OPENROUTER_API_KEY=sk-or-...        # PowerShell: $env:OPENROUTER_API_KEY = "sk-or-..."
uv run main.py path/to/scans
```

Descriptions turn on by default when the key is set. Pass `--no-describe-images` to turn them off. Passing `--describe-images` without a key prints a warning and the run continues without descriptions.

Descriptions send image data to OpenRouter. If your documents must not leave the machine, leave the key unset.

How images are chosen and described:

- Embedded images from a PDF and standalone image files are both described. A standalone image counts as one image covering the whole page.
- Images whose shorter side is under `--min-image-px` (default 32) are skipped as `too_small`.
- At most `--max-images-per-doc` distinct images (default 5) are described. The rest are recorded as skipped with the reason `cap`.
- Identical images, matched by SHA-256, are described once per document.
- A failed call is recorded as an `error` on that image and does not fail the file. The client retries rate limits, server errors, and network errors twice with a growing delay.
- The model defaults to `anthropic/claude-haiku-5.5`. Choose another with `--describe-model`.

Each image record in the JSON stores the description, the model, the prompt version, latency, token counts, and the cost OpenRouter reports. Descriptions appear only in the JSON, not in the `txt` or `md` files. When the run ends, they are also printed under an "Image descriptions" heading so you can spot-check them.

Descriptions also work on files you already processed. If the OCR output is reusable but its descriptions are missing or were made with different settings, the pipeline keeps the OCR results and describes only the images. Images already described by the same model and prompt version are not sent again. Changing `--describe-model` or `--max-images-per-doc` describes them again.

### Re-open results later

`--view` inspects existing result JSONs without running OCR. It takes a file, a folder searched recursively, or several paths, and defaults to `output/`.

```sh
uv run main.py --view
uv run main.py --view output/some/folder
uv run main.py --view output/report.pdf.json --view-mode json
```

Without `--view-mode`, you get the same numbered picker as after a batch. `--view-mode` applies one view to every file. Its values are `preview`, `json`, `images`, and `inspector`. The `images` view redraws the pages, so it needs the original source file to still be at the path stored in the JSON.

`--view` cannot be combined with a path, `--list`, `--ocr-only`, or `--annotate`. Other processing flags are ignored.

## How it works

PDFs take one of two paths.

1. LiteParse, the default. [LiteParse](https://github.com/run-llama/liteparse) reads the embedded text layer exactly. It also flags pages that need OCR, such as scans, image-only pages, and sparse text. When any page is flagged, LiteParse sends the pages that need it to a local EasyOCR server that the pipeline starts on `127.0.0.1` and shuts down when you quit. The server shares one EasyOCR reader with everything else.
2. EasyOCR on every page. `--ocr-only` renders each page at `--dpi` and recognizes all of it. The pipeline also falls back to this path when LiteParse is not installed or returns no pages.

Image files always go straight to EasyOCR.

The two paths give very different costs. On the 3-page sample, a CPU-only Mac took about 0.1 seconds through LiteParse and about 82 seconds with `--ocr-only`. Text-layer text reports confidence 1.00 because nothing was guessed. The OCR run on the same sample scored 0.81, which is the better test of the OCR itself.

The engine loads EasyOCR models on first use, so `--list` and fully digital PDFs never pay the load cost. It uses a CUDA GPU when one is available and otherwise prints a notice and runs on CPU. `--cpu` forces CPU.

All boxes share one coordinate model. Each box is four corners, clockwise from the top left, measured in 72-DPI points with the origin at the top left of the page. EasyOCR pixel boxes are scaled into that space, LiteParse boxes already use it, and a standalone image is treated as one pixel per point. This is why the viewer, the annotated PNGs, and the JSON agree no matter which path made the text.

Results are grouped into paragraph regions as well as raw lines. On the LiteParse path the regions come from LiteParse's layout blocks, labeled `paragraph`, `heading`, `list_item`, or `code`. On the EasyOCR path, nearby lines are grouped by geometry. The viewer, the annotated PNGs, and the Markdown export all use paragraphs, and fall back to lines for pages that have none.

## Command reference

```text
uv run main.py [PATH] [options]
```

| Option | Default | Description |
|---|---|---|
| `PATH` | none | File or folder to process. Omit it for the interactive prompt. |
| `--output DIR` | `./output` | Directory for results. |
| `--dpi N` | `200` | Rendering resolution for `--ocr-only`, for pages LiteParse sends to OCR, and for annotated PNGs. Part of the reuse key. |
| `--lang en,fr,...` | `en` | EasyOCR language codes. Part of the reuse key. |
| `--cpu` | off | Force CPU inference. |
| `--list` | off | Dry run. List files and page counts without OCR. |
| `--ocr-only` | off | Render and OCR every PDF page with EasyOCR, bypassing LiteParse. |
| `--annotate` | off | Write page PNGs with boxes colored by confidence. |
| `--format LIST` | `json,txt` | Comma-separated formats from `json`, `txt`, `md`. `--formats` is an alias. |
| `--combine` | off | Also write one combined text or Markdown file for the batch. |
| `--force` | off | Reprocess files that already have results. |
| `--view [PATH...]` | off | Inspect existing result JSONs instead of running OCR. |
| `--view-mode MODE` | picker | With `--view`, apply `preview`, `json`, `images`, or `inspector` to every file. |
| `--describe-images` / `--no-describe-images` | on if `OPENROUTER_API_KEY` is set | Turn image descriptions on or off. |
| `--describe-model ID` | `anthropic/claude-haiku-5.5` | OpenRouter model for descriptions. |
| `--max-images-per-doc N` | `5` | Distinct images described per document. |
| `--min-image-px N` | `32` | Skip images whose shorter side is smaller. |

## Output reference

Results mirror the input structure under the output directory, and the source file keeps its extension, so `report.pdf` becomes `report.pdf.json`.

- For a single file, results are written flat: `output/report.pdf.json`.
- For a folder, the path relative to that folder is kept. The folder's own name is not part of the path.

```text
scans/                         output/
  a.pdf                          a.pdf.json
  2024/                          2024/
    b.png                          b.png.json
```

The `txt` and `md` files sit beside the JSON and replace `.json` with their own extension: `report.pdf.json`, `report.pdf.txt`, `report.pdf.md`. Annotated pages go in `report.pdf.pages/page-001.png`. With `--combine`, the combined file is named after the input: `output/<folder-or-file-name>.txt` and `.md`.

A JSON file looks like this, trimmed to one page:

```json
{
  "source": "/home/me/scans/report.pdf",
  "engine": "liteparse",
  "language": ["en"],
  "processed_at": "2026-10-09T20:39:43+00:00",
  "text": "Sample PDF\nCreated for testing PDFObject\n...",
  "page_count": 3,
  "dpi": 200,
  "pages": [
    {
      "page": 1,
      "text": "Sample PDF\nCreated for testing PDFObject\n...",
      "text_char_count": 2913,
      "mean_confidence": 1.0,
      "lines": [
        {
          "box": [[190.3, 48.6], [328.3, 48.6], [328.3, 91.9], [190.3, 91.9]],
          "text": "Sample",
          "confidence": 1.0
        }
      ],
      "paragraphs": [
        {
          "box": [[190.3, 48.6], [421.7, 48.6], [421.7, 91.6], [190.3, 91.6]],
          "text": "Sample PDF",
          "confidence": 1.0,
          "kind": "heading"
        }
      ],
      "images": [
        {
          "box": [[72.0, 120.0], [300.0, 120.0], [300.0, 260.0], [72.0, 260.0]],
          "sha256": "9f2c...",
          "width": 640,
          "height": 360,
          "status": "described",
          "description": "A bar chart of quarterly revenue...",
          "model": "anthropic/claude-haiku-5.5",
          "prompt_version": "2",
          "latency_seconds": 1.8,
          "usage": {"prompt_tokens": 410, "completion_tokens": 62, "cost_usd": 0.0004}
        }
      ],
      "width": 612.0,
      "height": 792.0
    }
  ],
  "describe": {"model": "anthropic/claude-haiku-5.5", "max_images_per_doc": 5, "prompt_version": "2"}
}
```

`engine` is `liteparse` or `easyocr`. `paragraphs`, `images`, and `describe` appear only when there is something to put in them. The `images` record and the `describe` block above are illustrative. The rest comes from a real run on the sample PDF.

An image record's `status` is `described`, `skipped` (with `skip_reason` of `cap` or `too_small`), or `error` (with an `error` message).

## Notes and trade-offs

- EasyOCR loads every requested language model into memory, so keep `--lang` short. Models download once, about 100 MB for English, to `~/.EasyOCR`.
- The first recognition call loads models inside the worker thread. The terminal view keeps rendering with a spinner while it waits.
- `--dpi` trades recognition quality for speed. Higher is slower and often more accurate on small print.
- LiteParse's OCR flag can fire on clean but sparse digital pages, for example with a `sparse-text` reason. If scans are going through OCR unnecessarily, compare against `--ocr-only` and consider raising `--dpi`.
- The local OCR server binds `127.0.0.1` on a free port for the session and closes on exit.

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU is optional. Without one, everything runs on CPU, slowly for scans.
- On Windows, `uv sync` installs the CUDA 12.4 build of torch and torchvision from the PyTorch index, set in `[tool.uv.sources]` in `pyproject.toml`. Removing that setting makes Windows resolve a CPU-only build. On other platforms uv takes torch from PyPI. The pipeline runs on macOS with CPU inference.

## Development

```sh
uv run pytest
uv run ruff check .
uv run mypy .
```

The tests run offline in about a second and need no GPU. They stub EasyOCR and LiteParse instead of loading real models. Ruff uses a line length of 100, and mypy checks `main.py` and `app/`.

Where things live:

| Path | Role |
|---|---|
| `main.py` | The only entry point. Parses arguments, plans the batch, and runs one worker thread per batch. |
| `app/ingest.py` | Resolves a path into a list of supported files. |
| `app/lite.py`, `app/lite_server.py` | LiteParse parsing and the local EasyOCR server it calls. |
| `app/engine.py`, `app/pdfs.py` | EasyOCR wrapper and PDF rasterizing for `--ocr-only`. |
| `app/geometry.py`, `app/paragraphs.py` | The shared box model and paragraph grouping. |
| `app/output.py`, `app/export.py` | JSON documents and reuse checks, plus `txt` and `md` rendering. |
| `app/images.py`, `app/describe.py` | Finding images and describing them through OpenRouter. |
| `app/tui.py` | The live progress view. |
| `app/viewer.py`, `app/canvas.py`, `app/annotate.py`, `app/inspector.py` | The picker, character-grid pages, annotated PNGs, and the Textual inspector. |
