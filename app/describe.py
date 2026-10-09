"""Describe one image per call with a vision model via OpenRouter chat completions."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from app.images import Selection
from app.output import Image, ImageUsage

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
PROMPT_VERSION = "1"
PROMPT = (
    "Describe this image for a searchable document index. Say what it shows in one "
    "or two sentences, and transcribe any text visible in it."
)
MAX_OUTPUT_TOKENS = 300
BACKOFF_SECONDS = 1.0
EXCERPT_CHARS = 200

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


class DescribeError(Exception):
    """An image could not be described. Messages never include the API key."""


@dataclass(frozen=True)
class Description:
    """A successful description and what its call cost.

    ``latency_seconds`` covers the whole call, retries included. ``usage`` is
    ``None`` when the response carried no usage block.
    """

    text: str
    usage: ImageUsage | None
    latency_seconds: float
    described_at: str


def _post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    if not url.startswith("https://"):
        raise ValueError("OpenRouter requests must use https")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _opt_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _opt_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _parse_usage(payload: dict) -> ImageUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    return ImageUsage(
        prompt_tokens=_opt_int(usage.get("prompt_tokens")),
        completion_tokens=_opt_int(usage.get("completion_tokens")),
        cost_usd=_opt_float(usage.get("cost")),
    )


def _parse_response(raw: bytes) -> tuple[str, ImageUsage | None]:
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise DescribeError("response body is not JSON") from exc
    if not isinstance(payload, dict):
        raise DescribeError("response body is not a JSON object")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DescribeError("response has no message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise DescribeError("response description is empty")
    return content.strip(), _parse_usage(payload)


def _excerpt(raw: bytes) -> str:
    return raw[:EXCERPT_CHARS].decode("utf-8", errors="replace")


class OpenRouterDescriber:
    """Sends one image per request and returns its description with usage.

    Retries 429, 5xx and network errors with exponential backoff. Any other
    non-200 response fails immediately. ``transport``, ``sleep`` and ``clock``
    are injectable so tests run offline.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: Transport = _post,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport
        self._sleep = sleep
        self._clock = clock

    def describe(self, image: bytes, mime: str) -> Description:
        body = json.dumps(self._payload(image, mime)).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started = self._clock()
        attempts = self._max_retries + 1
        last_failure = ""
        for attempt in range(attempts):
            try:
                status, raw = self._transport(API_URL, headers, body, self._timeout)
            except OSError as exc:
                last_failure = f"network error: {type(exc).__name__}"
            else:
                if status == 200:
                    text, usage = _parse_response(raw)
                    return Description(
                        text=text,
                        usage=usage,
                        latency_seconds=self._clock() - started,
                        described_at=datetime.now(UTC).isoformat(),
                    )
                last_failure = f"HTTP {status}: {_excerpt(raw)}"
                if status != 429 and status < 500:
                    raise DescribeError(last_failure)
            if attempt + 1 < attempts:
                self._sleep(BACKOFF_SECONDS * 2**attempt)
        raise DescribeError(f"gave up after {attempts} attempts; last failure: {last_failure}")

    def _payload(self, image: bytes, mime: str) -> dict[str, object]:
        encoded = base64.b64encode(image).decode("ascii")
        return {
            "model": self._model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        }


class ImageDescriber(Protocol):
    def describe(self, image: bytes, mime: str) -> Description: ...


def _items(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def cached_descriptions(doc: dict, *, model: str, prompt_version: str) -> dict[str, str]:
    """Descriptions from a prior output document, keyed by image sha256.

    Only images described with the same model and prompt version qualify, so
    changing either one describes every image again.
    """
    cache: dict[str, str] = {}
    for page in _items(doc.get("pages")):
        for image in _items(page.get("images")):
            if image.get("status") != "described":
                continue
            if image.get("model") != model or image.get("prompt_version") != prompt_version:
                continue
            sha256 = image.get("sha256")
            text = image.get("description")
            if isinstance(sha256, str) and isinstance(text, str) and text:
                cache[sha256] = text
    return cache


def describe_images(
    selections: Sequence[Selection],
    describer: ImageDescriber,
    *,
    model: str,
    prompt_version: str,
    cache: Mapping[str, str],
) -> list[Image]:
    """Turn selections into output records, calling the describer only for new images.

    An image that is already in ``cache`` or was described earlier in this run
    is not sent again. A failed image is recorded as an error on every
    occurrence and is not retried, and the other images still run.
    """
    texts: dict[str, str] = dict(cache)
    failures: dict[str, str] = {}
    records: list[Image] = []
    for selection in selections:
        occurrence = selection.occurrence
        if selection.skip_reason is not None:
            records.append(
                Image(
                    box=occurrence.box,
                    sha256=occurrence.sha256,
                    width=occurrence.width,
                    height=occurrence.height,
                    status="skipped",
                    skip_reason=selection.skip_reason,
                )
            )
        elif occurrence.sha256 in failures:
            records.append(
                Image(
                    box=occurrence.box,
                    sha256=occurrence.sha256,
                    width=occurrence.width,
                    height=occurrence.height,
                    status="error",
                    error=failures[occurrence.sha256],
                    model=model,
                    prompt_version=prompt_version,
                )
            )
        elif occurrence.sha256 in texts:
            records.append(
                Image(
                    box=occurrence.box,
                    sha256=occurrence.sha256,
                    width=occurrence.width,
                    height=occurrence.height,
                    status="described",
                    description=texts[occurrence.sha256],
                    model=model,
                    prompt_version=prompt_version,
                )
            )
        else:
            try:
                result = describer.describe(occurrence.data, occurrence.mime)
            except DescribeError as exc:
                failures[occurrence.sha256] = str(exc)
                records.append(
                    Image(
                        box=occurrence.box,
                        sha256=occurrence.sha256,
                        width=occurrence.width,
                        height=occurrence.height,
                        status="error",
                        error=str(exc),
                        model=model,
                        prompt_version=prompt_version,
                    )
                )
            else:
                texts[occurrence.sha256] = result.text
                records.append(
                    Image(
                        box=occurrence.box,
                        sha256=occurrence.sha256,
                        width=occurrence.width,
                        height=occurrence.height,
                        status="described",
                        description=result.text,
                        model=model,
                        prompt_version=prompt_version,
                        described_at=result.described_at,
                        latency_seconds=result.latency_seconds,
                        usage=result.usage,
                    )
                )
    return records
