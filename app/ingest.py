"""Resolve a user-supplied path into a batch of OCR targets."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | {".pdf"}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def resolve_targets(user_path: str) -> tuple[Path, list[Path]]:
    """Resolve a file/folder path into an anchor plus an ordered list of files.

    Returns ``(anchor, files)`` where ``anchor`` is used to mirror output paths:
    a single file gets the anchor of its parent directory, a folder gets the
    folder itself.
    """
    root = Path(user_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if root.is_file():
        if not is_supported(root):
            raise ValueError(
                f"Unsupported file type: {root.suffix} "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTS))})"
            )
        return root.parent, [root]

    files = sorted(f for f in root.rglob("*") if f.is_file() and is_supported(f))
    if not files:
        raise ValueError(f"No supported files found under: {root}")
    return root, files
