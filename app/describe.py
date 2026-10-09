"""Describe one image per call with a vision model via OpenRouter chat completions."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

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


def _post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    if not url.startswith("https://"):
        raise ValueError("OpenRouter requests must use https")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _parse_content(raw: bytes) -> str:
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise DescribeError("response body is not JSON") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DescribeError("response has no message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise DescribeError("response description is empty")
    return content.strip()


def _excerpt(raw: bytes) -> str:
    return raw[:EXCERPT_CHARS].decode("utf-8", errors="replace")


class OpenRouterDescriber:
    """Sends one image per request and returns its text description.

    Retries 429, 5xx and network errors with exponential backoff. Any other
    non-200 response fails immediately. ``transport`` and ``sleep`` are
    injectable so tests run offline.
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
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport
        self._sleep = sleep

    def describe(self, image: bytes, mime: str) -> str:
        body = json.dumps(self._payload(image, mime)).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempts = self._max_retries + 1
        last_failure = ""
        for attempt in range(attempts):
            try:
                status, raw = self._transport(API_URL, headers, body, self._timeout)
            except OSError as exc:
                last_failure = f"network error: {type(exc).__name__}"
            else:
                if status == 200:
                    return _parse_content(raw)
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
