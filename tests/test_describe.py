from __future__ import annotations

import json
from datetime import datetime

import pytest

from app import describe
from app.output import ImageUsage

KEY = "sk-or-test-SECRET"

SPIKE_USAGE = {
    "prompt_tokens": 187,
    "completion_tokens": 30,
    "total_tokens": 217,
    "cost": 3.37e-05,
    "is_byok": False,
}


class ScriptedTransport:
    """Offline stand-in for the HTTP call: replays scripted outcomes in order."""

    def __init__(self, outcomes: list[tuple[int, bytes] | OSError]) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[tuple[str, dict[str, str], dict]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        self.requests.append((url, headers, json.loads(body)))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, OSError):
            raise outcome
        return outcome


def _ok(text: str, usage: dict | None = None) -> tuple[int, bytes]:
    body: dict = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return 200, json.dumps(body).encode()


def _describer(
    transport: ScriptedTransport,
    sleeps: list[float],
    *,
    max_retries: int = 2,
    clock: list[float] | None = None,
) -> describe.OpenRouterDescriber:
    ticks = iter(clock if clock is not None else [0.0, 0.0])
    return describe.OpenRouterDescriber(
        KEY,
        model="anthropic/claude-haiku-5.5",
        max_retries=max_retries,
        transport=transport,
        sleep=sleeps.append,
        clock=lambda: next(ticks),
    )


def test_sends_model_prompt_and_image_as_data_uri() -> None:
    transport = ScriptedTransport([_ok("  A bar chart.  ")])
    result = _describer(transport, []).describe(b"abc", "image/png")

    assert result.text == "A bar chart."
    url, headers, body = transport.requests[0]
    assert url == describe.API_URL
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert body["model"] == "anthropic/claude-haiku-5.5"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": describe.PROMPT}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,YWJj"},
    }


def test_returns_usage_from_response() -> None:
    transport = ScriptedTransport([_ok("A chart.", SPIKE_USAGE)])
    result = _describer(transport, []).describe(b"abc", "image/png")

    assert result.usage == ImageUsage(prompt_tokens=187, completion_tokens=30, cost_usd=3.37e-05)


def test_usage_is_none_when_response_has_no_usage_block() -> None:
    transport = ScriptedTransport([_ok("A chart.")])
    result = _describer(transport, []).describe(b"abc", "image/png")

    assert result.usage is None


def test_latency_covers_the_whole_call() -> None:
    transport = ScriptedTransport([_ok("A chart.")])
    result = _describer(transport, [], clock=[10.0, 10.25]).describe(b"abc", "image/png")

    assert result.latency_seconds == 0.25


def test_described_at_is_an_iso_timestamp() -> None:
    transport = ScriptedTransport([_ok("A chart.")])
    result = _describer(transport, []).describe(b"abc", "image/png")

    assert datetime.fromisoformat(result.described_at).tzinfo is not None


def test_retries_rate_limit_then_returns_description() -> None:
    transport = ScriptedTransport([(429, b"slow down"), _ok("Recovered.")])
    sleeps: list[float] = []

    result = _describer(transport, sleeps).describe(b"abc", "image/png")

    assert result.text == "Recovered."
    assert len(transport.requests) == 2
    assert sleeps == [describe.BACKOFF_SECONDS]


def test_retries_network_error_then_returns_description() -> None:
    transport = ScriptedTransport([TimeoutError("slow"), _ok("Back online.")])

    assert _describer(transport, []).describe(b"abc", "image/png").text == "Back online."


def test_gives_up_after_retry_budget_on_server_error() -> None:
    transport = ScriptedTransport([(503, b"busy")] * 3)
    sleeps: list[float] = []

    with pytest.raises(describe.DescribeError, match="gave up after 3 attempts"):
        _describer(transport, sleeps, max_retries=2).describe(b"abc", "image/png")
    assert len(transport.requests) == 3
    assert sleeps == [describe.BACKOFF_SECONDS, 2 * describe.BACKOFF_SECONDS]


def test_client_error_is_not_retried() -> None:
    transport = ScriptedTransport([(400, b"bad image")])

    with pytest.raises(describe.DescribeError, match="HTTP 400: bad image"):
        _describer(transport, []).describe(b"abc", "image/png")
    assert len(transport.requests) == 1


def test_error_message_never_contains_the_key() -> None:
    transport = ScriptedTransport([(401, b"invalid API key")] * 3)

    with pytest.raises(describe.DescribeError) as info:
        _describer(transport, [], max_retries=0).describe(b"abc", "image/png")
    assert KEY not in str(info.value)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"<html>", "not JSON"),
        (b"[]", "not a JSON object"),
        (b'{"choices": []}', "no message content"),
        (_ok("   ")[1], "empty"),
    ],
)
def test_malformed_success_body_raises(raw: bytes, message: str) -> None:
    transport = ScriptedTransport([(200, raw)])

    with pytest.raises(describe.DescribeError, match=message):
        _describer(transport, []).describe(b"abc", "image/png")
