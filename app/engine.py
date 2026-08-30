"""EasyOCR inference wrapper: lazy-loaded, GPU-aware, reused across pages."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PIL import Image


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class OcrEngine:
    """Thin wrapper around EasyOCR's Reader.

    The Reader is created lazily on first use so that listing mode and
    pre-flight checks never pay the (large) model-load cost.
    """

    def __init__(self, languages: list[str], gpu: bool | None = None) -> None:
        self.languages = languages
        self.gpu = gpu
        self._reader: Any = None
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def using_gpu(self) -> bool:
        use_gpu = cuda_available() if self.gpu is None else self.gpu
        return use_gpu

    def _ensure_reader(self) -> Any:
        with self._lock:
            if not self._loaded:
                import easyocr

                self._reader = easyocr.Reader(self.languages, gpu=self.using_gpu)
                self._loaded = True
        return self._reader

    def recognize(
        self, image: Image.Image, language: str | None = None
    ) -> list[tuple[list[list[float]], str, float]]:
        """Run OCR on one image.

        Returns a list of ``(box, text, confidence)`` tuples as produced by
        EasyOCR's ``readtext`` with ``detail=1``. ``language`` is accepted for
        the LiteParse OCR endpoint and ignored (the Reader is constructed with
        all configured languages).
        """
        reader = self._ensure_reader()
        raw = reader.readtext(np.asarray(image), detail=1, paragraph=False)
        results: list[tuple[list[list[float]], str, float]] = []
        for box, text, conf in raw:
            box = [[float(x), float(y)] for x, y in box]
            results.append((box, str(text), float(conf)))
        return results
