# AGENTS.md

## Commands

All Python execution goes through uv (the `.venv` is uv-managed; don't invoke system Python):

```powershell
uv run pytest            # full suite (~1s, offline, no GPU needed)
uv run pytest tests/test_lite.py::test_polygon_from_rect   # single test
uv run ruff check .
uv run mypy .
```

No CI or pre-commit exists; run all three before pushing.

## Toolchain quirks

- torch/torchvision are pinned to the PyTorch **cu124 index** via `[tool.uv.sources]` in `pyproject.toml` — removing that makes Windows resolve CPU-only torch builds.
- All tool config (pytest/ruff/mypy) lives in `pyproject.toml`; mypy only checks `main.py` and `app/`. Ruff line-length is 100.

## Architecture

- `main.py` is the only entrypoint; it runs an interactive prompt loop and drives one worker thread per batch (TUI renders from a separate thread — state changes go through `app/tui.py`'s lock/revision mechanism).
- PDFs have two paths: LiteParse text-layer extraction with selective OCR for flagged pages (`app/lite.py`), or full rasterization + EasyOCR (`app/pdfs.py`, forced by `--ocr-only`).
- `app/lite_server.py` hosts a Flask `/ocr` endpoint on `127.0.0.1` as a session singleton; `main()`'s `finally` closes it. Don't spawn it in tests.
- `app/engine.py:OcrEngine` lazy-loads EasyOCR models on first `recognize()`. Tests must stay offline: stub the engine/LiteParse instead of loading real models (existing tests do this via fakes in `tests/`).
- Output JSONs land in `output/` (gitignored), mirroring input structure relative to the anchor; the original extension is kept: `report.pdf` -> `report.pdf.json`.

## Conventions

- Windows-first repo (expect git CRLF warnings; paths are backslash-rooted).
- Every module uses `from __future__ import annotations` with full type hints; keep that style.
