# Feature Map: Image Descriptions (epic `simple-ocr-pipeline-vrd`)

Goal: describe embedded PDF images and standalone image files with
`anthropic/claude-haiku-5.5` via OpenRouter, store the descriptions additively in
the output JSON, and surface them in exports and the viewer.

Source of truth for task state is Beads (`bd show <id>`). This file is a
planning map only.

## 1. Dependency graph

```
vrd.1 Spike (P1, ready)
  ├──> vrd.3 Image extraction (P2)  ──┐
  └──> vrd.4 OpenRouter client (P2) ──┤
                                      │
vrd.2 Schema (P2, ready) ──┬──> vrd.6 Cache (P2) ──┤
                           │                       │
vrd.5 Config (P2, ready) ──┼───────────────────────┤
                           │                       ▼
                           └────────────> vrd.7 Pipeline in run_batch (P1)
                                                   │
                           vrd.2 ──────────┐       ▼
                                           └──> vrd.8 Exports + viewer (P3)
                                                   │
                                                   ▼
                                          vrd.9 Docs + E2E verification (P3)
```

## 2. Delivery phases

| Phase | Work | Can run in parallel | Unblocks |
|-------|------|---------------------|----------|
| 0 | `vrd.1` spike: pymupdf image bboxes/bytes, OpenRouter data-URI call shape | — | vrd.3, vrd.4 |
| 1 | `vrd.2` schema, `vrd.5` CLI/config | yes (with phase 0) | vrd.6, vrd.7 |
| 2 | `vrd.3` extraction, `vrd.4` client, `vrd.6` cache | yes | vrd.7 |
| 3 | `vrd.7` wire into `run_batch` | — | vrd.8 |
| 4 | `vrd.8` txt/md exports, viewer, annotate boxes | — | vrd.9 |
| 5 | `vrd.9` README, gates, TUI run on `samples/sample-local-pdf.pdf` | — | epic close |

## 3. Current state (2026-10-09)

All 10 issues are open; none in progress. `bd ready` reports 4 ready items:
`vrd` (epic), `vrd.1`, `vrd.2`, `vrd.5`. No dependency cycles.

Code touchpoints (verified against the repo):

| Area | File | Relevance |
|------|------|-----------|
| Output model / JSON | `app/output.py` | `Page`, `build_document`, `existing_document` (signature has no describe settings) |
| Batch worker | `main.py` `run_batch` | two OCR branches: LiteParse (`pdf_via_liteparse`) and EasyOCR raster; images must attach to both |
| Reuse planning | `main.py` `plan_batch` | a complete JSON is skipped outright; no "top up images only" path exists |
| PDF rasterizing | `app/pdfs.py` | pymupdf already a dependency |
| Exports | `app/export.py` | txt/md renderers; needs image blocks |
| Viewer / inspector | `app/viewer.py`, `app/inspector.py`, `app/canvas.py` | needs image boxes and description in status bar |
| Annotate | `app/annotate.py` | optional image-box overlay |
| HTTP | none | no `requests`/`httpx` in `pyproject.toml` |

## 4. Gaps found in the current backlog

Each gap is either a missing decision or missing work that no current bead owns.

| # | Gap | Where it bites | Suggested fix |
|---|-----|----------------|---------------|
| G1 | `vrd.2` Image fields have no status/skip reason, but `vrd.3` and the epic need `skipped (reason: cap)` and min-px skips | schema | Add `status` (`described`/`skipped`/`error`) and `skip_reason` (`cap`/`too_small`/`duplicate`) to `vrd.2` |
| G2 | Reuse path: `plan_batch` skips any file with a complete JSON, so `vrd.6`'s "describe only missing images on re-run" has no code path; `existing_document` must also take describe settings (`vrd.5`), and the owner of that change is unclear | `main.py`, `app/output.py` | New bead: "Reuse: top-up images on existing JSON without re-OCR", owned by `vrd.6`, depends on `vrd.5` |
| G3 | No HTTP client dependency. `vrd.4` must choose `urllib` (stdlib, no new dep) or `httpx` (new dep in `pyproject.toml`) | `pyproject.toml`, `app/describe.py` | Decide in `vrd.1` spike and record on epic |
| G4 | Undefined semantics: which images get the cap (first N in page order?), whether `duplicate` sha256 occurrences are recorded per page or dropped, and whether too-small images are recorded at all | `vrd.3`, `vrd.2` | Decide in `vrd.1` spike and write into the epic DESIGN |
| G5 | LiteParse branch: `vrd.7` says "after OCR/LiteParse page output is built", but LiteParse does not surface images, so the extraction call must run in both branches | `main.py` `run_batch` | Clarify in `vrd.7` description |
| G6 | No dry-run cost preview. `--list` should show how many images would be described and the cap impact before spending money | `main.py` `handle_batch_input` | New bead, P3, depends on `vrd.5` |
| G7 | No per-image usage record (model, prompt version, tokens, timestamp, latency). Needed to audit spend and debug cache misses | `vrd.4`, `vrd.2` | New bead, P3, or fold into `vrd.4` |
| G8 | Priority inversion: `vrd.7` (P1) depends on P2 items, and the epic is P2 while its P1 spike gates everything | backlog | Raise `vrd.2`, `vrd.4`, `vrd.5`, `vrd.3`, `vrd.6` to P1 (or lower `vrd.7` to P2). Pick one |
| G9 | Test strategy for PDFs with embedded images is not specified beyond "generated PDFs/PNGs". Need a fixture that has a known embedded image, plus a fixture with a duplicate | `vrd.3` tests | Add to `vrd.3` description |
| G10 | Exports for combined output (`--combine`) do not yet say whether descriptions appear there | `vrd.8` | Add a line to `vrd.8` |

## 5. Filed beads

1. `vrd.12` Reuse top-up path for images (P2): depends on `vrd.5` and `vrd.2`, blocks `vrd.7`. Covers G2.
2. `vrd.10` Dry-run image preview in `--list` (P3): depends on `vrd.5`, blocks `vrd.9`. Covers G6.
3. `vrd.11` Per-image usage record (P3): blocks `vrd.9`. Covers G7.

Gap notes were appended to `vrd.1`, `vrd.2`, `vrd.3`, `vrd.7`, and `vrd.8`. Priorities (G8) were not changed; that choice is still open.

Decisions to record on the epic during `vrd.1` (no new bead needed): HTTP client (G3), cap ordering and dedupe semantics (G4), and priority alignment (G8).

## 6. Recommended order

1. Decide G3, G4, G8 (quick, no code).
2. Run `vrd.1` spike and `vrd.2` schema in parallel. Fold G1 into `vrd.2`.
3. `vrd.5` config, then file the reuse top-up bead (G2).
4. `vrd.3`, `vrd.4`, `vrd.6` in parallel.
5. `vrd.7` pipeline, then `vrd.8`, then `vrd.9`.

## 7. Unrelated observations in `main.py`

- Lines 379-380: `viewer.choose_file(console, done_entries)` is called twice in a row. Probably a copy-paste error; the viewer opens twice.
- Line 152-153 in `pdf_via_liteparse`: `use_ocr = False` is assigned and then overwritten on the next line. Dead assignment.
