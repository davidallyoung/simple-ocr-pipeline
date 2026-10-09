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
- All boxes share one canonical coordinate model (`app/geometry.py:Quad`, 72-DPI points, top-left origin). EasyOCR pixel boxes are scaled by `72/dpi` at the call site in `main.py`; LiteParse maps in unchanged. Annotation (`app/annotate.py`, `--annotate` or viewer `v` suffix) renders pages and scales boxes by `dpi/72`; the TUI viewer (`1` at the inspect prompt) shows each page as a character-grid canvas (`app/canvas.py`) with confidence-colored boxes.
- Visualizations render **paragraph regions, not raw lines** (`Page.regions`): LiteParse path uses provider `LayoutBlock`s (`extract_blocks=True`, mapped in `app/lite.py`); the EasyOCR path geometrically groups lines (`app/paragraphs.py`). Paragraphs share `text`/`box`/`confidence` with `Line` plus a `kind` (`paragraph`/`heading`/...), and are stored additively in the JSON `paragraphs` array. Pages without paragraphs fall back to lines everywhere.
- The Textual inspector (`t` at the inspect prompt, `app/inspector.py`) is a full-screen hover-driven page viewer: hovering/Tab-cycling a box brightens it and shows text/confidence/point coords in a status bar. Textual apps are tested headlessly via `App.run_test()` + `Pilot` (`tests/test_inspector.py`); don't launch Textual apps under piped stdin (guarded by `sys.stdin.isatty()`).
- `app/lite_server.py` hosts a Flask `/ocr` endpoint on `127.0.0.1` as a session singleton; `main()`'s `finally` closes it. Don't spawn it in tests.
- `app/engine.py:OcrEngine` lazy-loads EasyOCR models on first `recognize()`. Tests must stay offline: stub the engine/LiteParse instead of loading real models (existing tests do this via fakes in `tests/`).
- Output JSONs land in `output/` (gitignored), mirroring input structure relative to the anchor; the original extension is kept: `report.pdf` -> `report.pdf.json`.

## Conventions

- Windows-first repo (expect git CRLF warnings; paths are backslash-rooted).
- Every module uses `from __future__ import annotations` with full type hints; keep that style.
- Anytime you complete a new task, feature, or bug fix, exercise it through the application end-to-end by running the TUI yourself against the sample document in `samples/sample-local-pdf.pdf` (e.g. `'' | uv run python main.py samples\sample-local-pdf.pdf --annotate`; pipe input like `"1"` or `"v"` to drive the interactive prompts).

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:46cd31e7 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->

## Beads Sync (repo policy)

- After creating, updating, claiming, or closing beads, run `bd sync` from the repo root to pull, recompute blocked state, and push the Dolt data to `origin`. Do this without asking; it is the standing authority for beads sync in this repo.
- `bd sync` does not touch git commits. Git commit and push still need explicit user instruction.
- If `bd sync` exits 2 (unresolvable conflict) or 4 (stuck working set), stop, do not force or auto-resolve, and report the exact command and output.

## Git Delivery (repo policy)

- Standing authority: once work is verified complete (tests, ruff, mypy, and the TUI run from the Conventions section where applicable), commit and push it without asking.
- New features go on a feature branch and open as a pull request (`gh pr create`). Push the branch once the work is verified; do not push new features straight to `master`.
- Bug fixes and docs-only changes may follow the same branch-and-PR flow; use judgment, and state which path you took.
- Do not force-push, rewrite published history, or merge PRs without explicit user instruction.
- If a push or PR creation is blocked, stop and report the exact command and error.

## Parallel Work (repo policy)

- Work on independent beads in separate git worktrees, one branch per worktree. Git refuses to check out the same branch in two worktrees.
- Claim the bead before starting (`bd update <id> --claim`) so two agents don't take the same work.
- Create worktrees next to the repo, from `origin/master`: `git worktree add ../simple-ocr-pipeline.worktrees/<name> -b <branch> origin/master`.
- Run `uv sync` in each new worktree. `.venv` is not shared.
- Check `git worktree list` before creating one. Do not reuse a branch that is checked out elsewhere.
- After the PR merges, remove the worktree with `git worktree remove <path>` and delete the branch with `git branch -d <branch>`.

## Skill Activity Map (repo policy)

Load the listed global skills (`~/.config/opencode/skills/<name>/SKILL.md`) when the matching activity happens. Skills marked **always** apply to every instance of the activity; the rest apply when their trigger fits.

| Activity in this repo | Skills |
|---|---|
| Any written output: docs, README, PR descriptions, commit messages, reports to the user | **always** `unslop`; `technical-writing` for docs, README, PRs, commits |
| Explaining code or a subsystem, onboarding | `how` (mechanics), `why` (rationale), `teach` (plain explanation) |
| Plain-language summaries for the user | `bro` |
| Designing a new feature or module before coding | `architect`, `principle-foundational-thinking`, `principle-redesign-from-first-principles` |
| Competing designs or UI/TUI choices with no precedent | `principle-exhaust-the-design-space`, `arena`, `principle-experience-first` |
| Scope and product tradeoffs in the TUI or CLI | `principle-experience-first` |
| Bug fixes and debugging | `principle-fix-root-causes`; `principle-attack-the-premise` after two failed fixes sharing one premise; `tdd` only when a cheap local test exists or the user asks |
| Writing or changing tests | `principle-test-behavior-not-implementation` |
| Typed Python: signatures, new modules, mypy failures | `principle-type-system-discipline`, `principle-boundary-discipline` |
| Validating external input (CLI args, env vars, OpenRouter responses) | `principle-boundary-discipline` |
| Reuse, resume, cache, or re-run logic (`plan_batch`, `existing_document`, describe cache) | `principle-make-operations-idempotent` |
| Shared state between the worker thread and the TUI (`app/tui.py`) | `principle-separate-before-serializing-shared-state` |
| Refactors, diff review, deleting code | `principle-laziness-protocol`, `principle-subtract-before-you-add`, `principle-minimize-reader-load` |
| Replacing an internal API or output schema | `principle-migrate-callers-then-delete-legacy-apis`, `principle-outcome-oriented-execution` |
| Multi-step work and stacking PRs | `principle-sequence-verifiable-units`, `principle-build-the-lever` |
| Long or unattended multi-phase runs | `show-me-your-work`, `figure-it-out` |
| Parallel research or sweeps | `swarm`, `principle-guard-the-context-window` |
| Reviewing a change before it ships | `blast-radius`, `interrogate` (for high-risk changes) |
| Before declaring any task done | **always** `principle-prove-it-works`; `create-verification-skill` if the repo lacks a scripted way to prove the feature |
| Keeping a verification skill current | `maintain-verification-skill` |
| Reporting a perf number or speedup | `principle-explain-the-number`, `benchmark-checklist` |
| A repeated mistake or correction | `correct`, `principle-encode-lessons-in-structure` (turn it into a ruff rule or test) |
| Tradeoffs on reversible work | `principle-never-block-on-the-human` |
| Code comments | `no-comments` (match the existing density, per the comment guidance above) |

Not applicable to this repo: `typescript-best-practices`. `report` is for OpenCode bugs only, not this project.
