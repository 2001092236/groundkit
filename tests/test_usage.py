import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from groundkit import llm
from groundkit.llm import LLMResponse, NotConfigured, RateLimited, complete_with_fallback
from groundkit.usage import UsageLedger, get_ledger, parse_ratelimit_headers

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def test_parse_groq_style_headers():
    parsed = parse_ratelimit_headers({
        "llm_provider-x-ratelimit-limit-requests": "1000", "llm_provider-x-ratelimit-remaining-requests": "998",
        "llm_provider-x-ratelimit-reset-requests": "2m59.56s", "x-ratelimit-remaining-tokens": "11990",
        "x-ratelimit-reset-tokens": "7.66s", "content-type": "application/json"}, NOW)
    assert parsed["limit_requests"] == 1000 and parsed["remaining_requests"] == 998
    assert parsed["reset_requests_at"] == (NOW + timedelta(minutes=2, seconds=59.56)).isoformat()
    assert parsed["remaining_tokens"] == 11990 and "content-type" not in parsed["raw"]


def test_parse_cerebras_openrouter_mistral_and_retry_after():
    parsed = parse_ratelimit_headers({
        "x-ratelimit-limit-requests-day": "14400", "x-ratelimit-remaining-requests-day": "14399",
        "x-ratelimit-reset-requests-day": "43200", "x-ratelimit-remaining-tokens-minute": "59000",
        "x-ratelimit-reset": str(int((NOW + timedelta(hours=1)).timestamp() * 1000)),
        "x-ratelimitbysize-remaining-minute": "0", "retry-after": "30"}, NOW)
    assert parsed["remaining_requests_day"] == 14399 and parsed["limit_requests_day"] == 14400
    assert parsed["reset_requests_day_at"] == (NOW + timedelta(hours=12)).isoformat()
    assert parsed["remaining_tokens_minute"] == 59000
    assert parsed["reset_requests_at"] == (NOW + timedelta(hours=1)).isoformat()
    assert parsed["remaining_requests_minute"] == 0
    assert parsed["retry_at"] == (NOW + timedelta(seconds=30)).isoformat()
    assert parse_ratelimit_headers(None) == {"raw": {}}


def test_ledger_records_and_summarises(tmp_path):
    ledger = UsageLedger(tmp_path / "u.json")
    ledger.record("groq/x", ok=True, input_tokens=10, output_tokens=5, latency_s=1.5,
                  headers={"x-ratelimit-remaining-requests": "990", "x-ratelimit-limit-requests": "1000",
                           "x-ratelimit-reset-requests": "10m"})
    ledger.record("groq/x", ok=False, error="boom")
    rows = {r["model"]: r for r in UsageLedger(tmp_path / "u.json").summary([{"model": "groq/x", "label": "Groq", "rpd": 1000}])}
    row = rows["groq/x"]
    assert row["used_today"] == 2 and row["ok_today"] == 1 and row["errors_today"] == 1
    assert row["tokens_today"] == 15 and row["remaining"] == 990 and row["remaining_source"] == "provider"
    assert row["last_error"] == "boom" and row["blocked_until"] is None
    assert row["resets_at"].startswith(datetime.now(UTC).date().isoformat()) or row["resets_at"] > datetime.now(UTC).isoformat()


def test_ledger_estimates_remaining_and_blocks_after_429(tmp_path):
    ledger = UsageLedger(tmp_path / "u.json")
    for _ in range(3):
        ledger.record("m", ok=True)
    row = ledger.summary([{"model": "m", "label": "M", "rpd": 50}])[0]
    assert row["remaining"] == 47 and row["remaining_source"] == "estimate"
    ledger.record("m", ok=False, rate_limited=True, error="429", headers={"retry-after": "120"})
    until = ledger.blocked_until("m")
    assert until and timedelta(seconds=100) < until - datetime.now(UTC) <= timedelta(seconds=120)
    ledger.record("m", ok=False, rate_limited=True, error="429")  # без заголовков — дефолтный откат
    assert ledger.blocked_until("m") - datetime.now(UTC) > timedelta(minutes=9)
    ledger.record("m", ok=True)  # успешный вызов снимает блокировку
    assert ledger.blocked_until("m") is None


def test_ledger_survives_broken_file(tmp_path):
    path = tmp_path / "u.json"
    path.write_text("{not json")
    ledger = UsageLedger(path)
    ledger.record("m", ok=True)
    assert UsageLedger(path).today("m")["requests"] == 1


class Stub:
    def __init__(self, name, text=None, error=None):
        self.name, self.text, self.error = name, text, error
        self.calls = 0

    def complete(self, messages, temperature=0.2, max_tokens=None):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(text=self.text, model=self.name, provider="stub")


def test_fallback_skips_models_blocked_by_ledger():
    ledger = get_ledger()
    ledger.record("a", ok=False, rate_limited=True, error="429")
    a, b = Stub("a", text="from a"), Stub("b", text="from b")
    out = complete_with_fallback([{"role": "user", "content": "q"}], [a, b])
    assert out.text == "from b" and a.calls == 0
    assert out.attempts[0]["skipped"] and "лимит" in out.attempts[0]["error"]


def test_fallback_still_tries_when_everything_is_blocked():
    ledger = get_ledger()
    ledger.record("a", ok=False, rate_limited=True, error="429")
    a = Stub("a", text="from a")
    assert complete_with_fallback([{"role": "user", "content": "q"}], [a]).text == "from a" and a.calls == 1


def test_litellm_provider_records_usage(monkeypatch):
    import litellm as real

    class Msg:
        content = "answer"

    class Choice:
        message = Msg()

    class Usage:
        prompt_tokens, completion_tokens = 10, 3

    class Resp:
        choices = [Choice()]
        usage = Usage()
        _hidden_params = {"additional_headers": {"llm_provider-x-ratelimit-remaining-requests": "5"}}

    monkeypatch.setattr(real, "completion", lambda **kw: Resp())
    monkeypatch.setattr(real, "completion_cost", lambda completion_response: 0.0)
    llm.LiteLLM("gemini/flash").complete([{"role": "user", "content": "hi"}])
    row = get_ledger().summary([{"model": "gemini/flash", "label": "x"}])[0]
    assert row["used_today"] == 1 and row["tokens_today"] == 13 and row["remaining"] == 5

    class RateLimitError(Exception):
        headers = {"retry-after": "60"}

    def boom(**kw):
        raise RateLimitError("429")

    monkeypatch.setattr(real, "completion", boom)
    try:
        llm.LiteLLM("gemini/flash").complete([{"role": "user", "content": "hi"}])
    except RateLimited:
        pass
    assert get_ledger().blocked_until("gemini/flash") is not None


@respx.mock
def test_openai_compat_records_provider_headers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(return_value=httpx.Response(
        200, json={"choices": [{"message": {"content": "Париж"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
        headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "997",
                 "x-ratelimit-reset-requests": "4m10s", "x-ratelimit-remaining-tokens": "7900"}))
    out = llm.OpenAICompat("groq/qwen/qwen3.8-27b").complete([{"role": "user", "content": "hi"}])
    assert out.text == "Париж" and out.provider == "groq" and out.input_tokens == 12
    sent = json.loads(respx.calls.last.request.content)
    assert sent["model"] == "qwen/qwen3.8-27b" and respx.calls.last.request.headers["Authorization"] == "Bearer k"
    row = get_ledger().summary([{"model": "groq/qwen/qwen3.8-27b", "label": "g", "rpd": 1000}])[0]
    assert row["remaining"] == 997 and row["remaining_source"] == "provider" and row["tokens_today"] == 14
    assert row["resets_at"] > datetime.now(UTC).isoformat()


@respx.mock
def test_openai_compat_classifies_http_errors(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    route = respx.post("https://openrouter.ai/api/v1/chat/completions")
    route.mock(return_value=httpx.Response(429, json={"error": {"message": "rate-limited upstream"}}, headers={"retry-after": "5"}))
    with pytest.raises(RateLimited):
        llm.OpenAICompat("openrouter/google/gemma-4-26b-a4b-it:free").complete([{"role": "user", "content": "hi"}])
    assert get_ledger().blocked_until("openrouter/google/gemma-4-26b-a4b-it:free") is not None
    route.mock(return_value=httpx.Response(402, text="payment required"))
    with pytest.raises(NotConfigured):
        llm.OpenAICompat("openrouter/x").complete([{"role": "user", "content": "hi"}])
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(NotConfigured):
        llm.OpenAICompat("openrouter/x").complete([{"role": "user", "content": "hi"}])
    assert isinstance(llm.build_llm("mistral/mistral-small-latest"), llm.OpenAICompat)
    assert isinstance(llm.build_llm("gemini/x"), llm.LiteLLM)


def test_block_uses_soonest_reset_when_no_retry_after(tmp_path):
    ledger = UsageLedger(tmp_path / "u.json")
    ledger.record("g", ok=False, rate_limited=True, error="429 tokens per minute",
                  headers={"x-ratelimit-reset-requests": "1m26s", "x-ratelimit-reset-tokens": "2s"})
    until = ledger.blocked_until("g")
    assert until and until - datetime.now(UTC) <= timedelta(seconds=2.5)
