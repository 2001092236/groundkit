import json
import subprocess
import time

import httpx
import pytest

from groundkit import llm
from groundkit.llm import (
    ClaudeCLI,
    LiteLLM,
    LLMError,
    LLMResponse,
    NotConfigured,
    RateLimited,
    build_llm,
    complete_with_fallback,
    default_chain,
    list_models,
    model_configured,
)

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def _fake_run(outputs):
    """Подменяет subprocess.run: каждая запись — dict (JSON в stdout) или строка (stderr)."""
    calls = []

    def run(argv, input=None, capture_output=None, text=None, timeout=None, env=None, check=None):
        calls.append({"argv": argv, "input": input, "env": env})
        item = outputs.pop(0)
        if isinstance(item, dict):
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(item), stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=item)

    return run, calls


def test_claude_cli_builds_pure_llm_invocation(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda _: "/usr/bin/claude")
    run, calls = _fake_run([{"is_error": False, "result": "Paris", "total_cost_usd": 0.01,
                             "usage": {"input_tokens": 5, "output_tokens": 1},
                             "modelUsage": {"claude-haiku-4-5": {}, "claude-opus-5": {}}}])
    monkeypatch.setattr(llm.subprocess, "run", run)
    out = ClaudeCLI(model="opus").complete(MSGS)
    assert out.text == "Paris" and out.model == "claude-opus-5" and out.provider == "claude-cli"
    argv = calls[0]["argv"]
    assert argv[:3] == ["claude", "-p", "--max-turns"]
    assert "--tools" in argv and argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--system-prompt") + 1] == "sys"
    assert argv[argv.index("--model") + 1] == "opus"
    assert "--bare" in argv and calls[0]["input"] == "hi"
    assert "CLAUDECODE" not in calls[0]["env"]


def test_claude_cli_retries_without_bare_when_not_logged_in(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda _: "/usr/bin/claude")
    run, calls = _fake_run([{"is_error": True, "result": "Not logged in · Please run /login"},
                            {"is_error": False, "result": "ok", "modelUsage": {}}])
    monkeypatch.setattr(llm.subprocess, "run", run)
    assert ClaudeCLI().complete(MSGS).text == "ok"
    assert "--bare" in calls[0]["argv"] and "--bare" not in calls[1]["argv"]


def test_claude_cli_errors_are_classified(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda _: "/usr/bin/claude")
    run, _ = _fake_run([{"is_error": True, "result": "Not logged in"}, {"is_error": True, "result": "Not logged in"}])
    monkeypatch.setattr(llm.subprocess, "run", run)
    with pytest.raises(NotConfigured):
        ClaudeCLI().complete(MSGS)
    run, _ = _fake_run(["boom"])
    monkeypatch.setattr(llm.subprocess, "run", run)
    with pytest.raises(LLMError):
        ClaudeCLI(bare=False).complete(MSGS)


def test_claude_cli_missing_binary(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda _: None)
    with pytest.raises(NotConfigured):
        ClaudeCLI().complete(MSGS)


def test_litellm_provider_classifies_errors(monkeypatch):
    import litellm as real

    class RateLimitError(Exception):
        pass

    def boom(**kw):
        raise RateLimitError("429 too many")

    monkeypatch.setattr(real, "completion", boom)
    with pytest.raises(RateLimited):
        LiteLLM("gemini/x").complete(MSGS)

    class AuthenticationError(Exception):
        pass

    def no_key(**kw):
        raise AuthenticationError("missing api key")

    monkeypatch.setattr(real, "completion", no_key)
    with pytest.raises(NotConfigured):
        LiteLLM("groq/x").complete(MSGS)


def test_litellm_provider_success(monkeypatch):
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

    seen = {}

    def completion(**kw):
        seen.update(kw)
        return Resp()

    monkeypatch.setattr(real, "completion", completion)
    monkeypatch.setattr(real, "completion_cost", lambda completion_response: 0.0)
    out = LiteLLM("groq/llama").complete(MSGS, temperature=0.5)
    assert out.text == "answer" and out.input_tokens == 10 and out.cost_usd == 0.0
    assert seen["model"] == "groq/llama" and seen["temperature"] == 0.5


class Stub:
    def __init__(self, name, text=None, error=None):
        self.name, self.text, self.error = name, text, error

    def complete(self, messages, temperature=0.2, max_tokens=None):
        if self.error:
            raise self.error
        return LLMResponse(text=self.text, model=self.name, provider="stub")


def test_fallback_chain_skips_failures_and_records_attempts():
    out = complete_with_fallback(MSGS, [Stub("a", error=RateLimited("429")), Stub("b", text="ok"), Stub("c", text="no")])
    assert out.text == "ok" and out.model == "b"
    assert [a["ok"] for a in out.attempts] == [False, True]


def test_fallback_chain_raises_when_all_fail():
    with pytest.raises(LLMError, match="a — .*b — "):
        complete_with_fallback(MSGS, [Stub("a", error=LLMError("x")), Stub("b", error=NotConfigured("y"))])
    with pytest.raises(NotConfigured):
        complete_with_fallback(MSGS, [])


def test_build_llm_specs():
    assert isinstance(build_llm("claude-cli"), ClaudeCLI)
    assert build_llm("claude-cli/sonnet").model == "sonnet"
    assert build_llm("gemini/gemini-flash-latest").model == "gemini/gemini-flash-latest"


def test_model_detection_from_env(monkeypatch):
    assert default_chain() == []
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert default_chain() == ["groq/qwen/qwen3.8-27b", "groq/openai/gpt-oss-120b"]
    assert model_configured("groq/any-other") and not model_configured("gemini/x")
    monkeypatch.setenv("GROUNDKIT_CLAUDE_CLI", "1")
    monkeypatch.setattr(llm.shutil, "which", lambda _: "/usr/bin/claude")
    assert default_chain()[-1] == "claude-cli"
    assert {m["model"] for m in list_models() if m["configured"]} == {"groq/qwen/qwen3.8-27b", "groq/openai/gpt-oss-120b", "claude-cli"}


def test_gigachat_oauth_caches_token_and_calls_chat(monkeypatch):
    """Токен берётся один раз на два вызова, запрос уходит с Bearer и моделью."""
    import respx

    from groundkit.llm import GIGACHAT_API_URL, GIGACHAT_OAUTH_URL, GigaChat, _gigachat_token

    _gigachat_token.update(value="", expires_at=0.0)
    with respx.mock:
        oauth = respx.post(GIGACHAT_OAUTH_URL).mock(return_value=httpx.Response(
            200, json={"access_token": "tok-1", "expires_at": (time.time() + 1800) * 1000}))
        chat = respx.post(f"{GIGACHAT_API_URL}/chat/completions").mock(return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "Париж"}}],
                       "usage": {"prompt_tokens": 17, "completion_tokens": 4}}))
        provider = GigaChat(auth_key="base64key", model="GigaChat-2")
        first = provider.complete(MSGS)
        provider.complete(MSGS)

    assert first.text == "Париж" and first.model == "gigachat/GigaChat-2" and first.provider == "gigachat"
    assert first.input_tokens == 17 and first.output_tokens == 4
    assert oauth.call_count == 1 and chat.call_count == 2  # токен переиспользован
    assert oauth.calls[0].request.headers["authorization"] == "Basic base64key"
    assert "RqUID" in oauth.calls[0].request.headers
    assert b"scope=GIGACHAT_API_PERS" in oauth.calls[0].request.content
    assert chat.calls[0].request.headers["authorization"] == "Bearer tok-1"
    assert json.loads(chat.calls[0].request.content)["model"] == "GigaChat-2"


def test_gigachat_errors_are_classified(monkeypatch):
    import respx

    from groundkit.llm import GIGACHAT_API_URL, GIGACHAT_OAUTH_URL, GigaChat, _gigachat_token

    with pytest.raises(NotConfigured):
        _gigachat_token.update(value="", expires_at=0.0)
        GigaChat(auth_key="").complete(MSGS)

    _gigachat_token.update(value="", expires_at=0.0)
    with respx.mock:
        respx.post(GIGACHAT_OAUTH_URL).mock(return_value=httpx.Response(401, text="bad key"))
        with pytest.raises(NotConfigured):
            GigaChat(auth_key="x").complete(MSGS)

    _gigachat_token.update(value="tok", expires_at=time.time() + 900)
    with respx.mock:
        respx.post(f"{GIGACHAT_API_URL}/chat/completions").mock(return_value=httpx.Response(429, text="too many"))
        with pytest.raises(RateLimited):
            GigaChat(auth_key="x").complete(MSGS)

    _gigachat_token.update(value="tok", expires_at=time.time() + 900)
    with respx.mock:
        respx.post(f"{GIGACHAT_API_URL}/chat/completions").mock(return_value=httpx.Response(402, text="no tokens"))
        with pytest.raises(NotConfigured):
            GigaChat(auth_key="x").complete(MSGS)
    assert _gigachat_token["value"] == ""  # протухший токен сброшен


def test_gigachat_tls_bundle_from_env(monkeypatch, tmp_path):
    from groundkit.llm import _gigachat_verify

    monkeypatch.delenv("GIGACHAT_CA_BUNDLE", raising=False)
    assert _gigachat_verify() is False
    import ssl

    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", str(certifi_path()))
    assert isinstance(_gigachat_verify(), ssl.SSLContext)


def certifi_path():
    """Любой валидный CA-бандл: проверяем, что путь превращается в SSL-контекст."""
    import ssl as _ssl

    return _ssl.get_default_verify_paths().cafile


def test_gigachat_build_and_detect(monkeypatch):
    from groundkit.llm import GigaChat

    provider = build_llm("gigachat/GigaChat-2-Pro")
    assert isinstance(provider, GigaChat) and provider.model == "GigaChat-2-Pro"
    assert not model_configured("gigachat/GigaChat-2")
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "k")
    assert model_configured("gigachat/GigaChat-2")
    assert "gigachat/GigaChat-2" in default_chain()
