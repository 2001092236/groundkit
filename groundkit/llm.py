"""LLM-провайдеры за одним интерфейсом: LiteLLM (Gemini, Groq, OpenRouter, Anthropic…)
и Claude Code CLI как «чистая» модель для локальных экспериментов.

Все провайдеры реализуют ``complete(messages) -> LLMResponse``. Цепочка
``complete_with_fallback`` перебирает модели, пока одна не ответит.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger("groundkit.llm")

Message = dict[str, str]


class LLMError(RuntimeError):
    """Модель не ответила."""


class RateLimited(LLMError):
    """Исчерпан лимит запросов — стоит переключиться на следующую модель."""


class NotConfigured(LLMError):
    """Нет ключа / не авторизовано."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    attempts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "latency_s": round(self.latency_s, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "attempts": self.attempts,
        }


class LLMProvider(Protocol):
    name: str

    def complete(
        self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int | None = None
    ) -> LLMResponse: ...


def _classify(exc: Exception) -> LLMError:
    name = type(exc).__name__
    text = str(exc).lower()
    if "ratelimit" in name.lower() or "429" in text or "quota" in text or "rate limit" in text:
        return RateLimited(f"{name}: {exc}")
    if "authentication" in name.lower() or "api key" in text or "api_key" in text or "401" in text:
        return NotConfigured(f"{name}: {exc}")
    return LLMError(f"{name}: {exc}")


@dataclass
class LiteLLM:
    """Любая модель, которую знает LiteLLM: ``gemini/…``, ``groq/…``, ``openrouter/…``, ``anthropic/…``."""

    model: str
    timeout: float = 90.0

    @property
    def name(self) -> str:
        return self.model

    def complete(
        self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int | None = None
    ) -> LLMResponse:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover
            raise NotConfigured("Установите `pip install litellm`") from exc

        litellm.suppress_debug_info = True
        started = time.monotonic()
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 — классифицируем и пробрасываем
            raise _classify(exc) from exc

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        cost: float | None
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:  # noqa: BLE001 — для бесплатных моделей цены может не быть
            cost = None
        return LLMResponse(
            text=text,
            model=self.model,
            provider="litellm",
            latency_s=time.monotonic() - started,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cost_usd=cost,
        )


@dataclass
class ClaudeCLI:
    """Claude Code в неинтерактивном режиме как чистая LLM.

    Запускает ``claude -p --max-turns 1 --tools "" --output-format json``: без
    инструментов, без файловых операций, с системным промптом через флаг.
    ``bare=True`` отключает CLAUDE.md, хуки, скиллы и MCP; если в таком режиме
    CLI не видит авторизацию, автоматически повторяет без ``--bare``.

    ⚠️ Только для собственных экспериментов. Использовать подписку как бэкенд
    приложения для чужих пользователей запрещено условиями Anthropic.
    """

    model: str | None = None  # "sonnet", "opus", "haiku" или полный id
    bare: bool = True
    executable: str = "claude"
    timeout: float = 240.0
    extra_args: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"claude-cli/{self.model}" if self.model else "claude-cli"

    def _argv(self, system: str, bare: bool) -> list[str]:
        argv = [self.executable, "-p", "--max-turns", "1", "--output-format", "json", "--tools", ""]
        if system:
            argv += ["--system-prompt", system]
        if self.model:
            argv += ["--model", self.model]
        if bare:
            argv.append("--bare")
        return argv + list(self.extra_args)

    @staticmethod
    def _flatten(messages: list[Message]) -> tuple[str, str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(m["content"])
            elif m["role"] == "assistant":
                parts.append(f"(предыдущий ответ ассистента)\n{m['content']}")
        return system, "\n\n".join(parts)

    def _run(self, argv: list[str], prompt: str) -> dict:
        if shutil.which(self.executable) is None:
            raise NotConfigured(f"`{self.executable}` не найден в PATH")
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        try:
            proc = subprocess.run(
                argv, input=prompt, capture_output=True, text=True, timeout=self.timeout, env=env, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude CLI не ответил за {self.timeout:.0f} с") from exc
        stdout = proc.stdout.strip()
        if not stdout:
            raise LLMError(f"claude CLI пустой вывод (код {proc.returncode}): {proc.stderr.strip()[:300]}")
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI вернул не JSON: {stdout[:300]}") from exc

    def complete(
        self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int | None = None
    ) -> LLMResponse:
        del temperature, max_tokens  # CLI не принимает эти параметры
        system, prompt = self._flatten(messages)
        started = time.monotonic()
        data = self._run(self._argv(system, self.bare), prompt)
        result = str(data.get("result") or "")
        if data.get("is_error") and self.bare and "not logged in" in result.lower():
            log.info("claude --bare не видит авторизацию, повторяю без --bare")
            data = self._run(self._argv(system, bare=False), prompt)
            result = str(data.get("result") or "")
        if data.get("is_error"):
            lowered = result.lower()
            if "not logged in" in lowered or "login" in lowered:
                raise NotConfigured(f"claude CLI: {result[:300]}")
            if "rate limit" in lowered or "usage limit" in lowered:
                raise RateLimited(f"claude CLI: {result[:300]}")
            raise LLMError(f"claude CLI: {result[:300]}")

        usage = data.get("usage") or {}
        used_models = [m for m in (data.get("modelUsage") or {}) if "haiku" not in m] or list(
            data.get("modelUsage") or {}
        )
        return LLMResponse(
            text=result,
            model=used_models[0] if used_models else self.name,
            provider="claude-cli",
            latency_s=time.monotonic() - started,
            input_tokens=(usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0) or None,
            output_tokens=usage.get("output_tokens"),
            cost_usd=data.get("total_cost_usd"),
        )


# Известные бесплатные/дешёвые модели: как их включить и на что рассчитывать.
KNOWN_MODELS: list[dict] = [
    {"model": "gemini/gemini-flash-latest", "label": "Google Gemini Flash", "env": "GEMINI_API_KEY",
     "free": "~1500 запросов/день", "signup": "https://aistudio.google.com/apikey"},
    {"model": "groq/llama-3.3-70b-versatile", "label": "Groq · Llama 3.3 70B", "env": "GROQ_API_KEY",
     "free": "14 400 запросов/день", "signup": "https://console.groq.com/keys"},
    {"model": "openrouter/meta-llama/llama-3.3-70b-instruct:free", "label": "OpenRouter · free-модели",
     "env": "OPENROUTER_API_KEY", "free": "50 запросов/день", "signup": "https://openrouter.ai/keys"},
    {"model": "mistral/mistral-small-latest", "label": "Mistral Small", "env": "MISTRAL_API_KEY",
     "free": "1 млрд токенов/мес, 2 RPM", "signup": "https://console.mistral.ai/api-keys"},
    {"model": "cerebras/llama3.1-8b", "label": "Cerebras · Llama 3.1 8B", "env": "CEREBRAS_API_KEY",
     "free": "1M токенов/день", "signup": "https://cloud.cerebras.ai"},
    {"model": "anthropic/claude-haiku-4-5-20251001", "label": "Anthropic · Claude Haiku 4.5 (платно)",
     "env": "ANTHROPIC_API_KEY", "free": "нет, платный API", "signup": "https://console.anthropic.com"},
    {"model": "claude-cli", "label": "Claude Code CLI (локальные эксперименты)", "env": None,
     "free": "в рамках подписки", "signup": "https://docs.anthropic.com/claude-code"},
]

CLAUDE_CLI_ENV = "GROUNDKIT_CLAUDE_CLI"


def claude_cli_enabled() -> bool:
    return os.getenv(CLAUDE_CLI_ENV, "").lower() in {"1", "true", "yes"}


def model_configured(spec: str) -> bool:
    if spec.startswith("claude-cli"):
        return shutil.which("claude") is not None and claude_cli_enabled()
    for known in KNOWN_MODELS:
        if known["model"] == spec and known["env"]:
            return bool(os.getenv(known["env"]))
    prefix = spec.split("/", 1)[0]
    env_by_prefix = {
        "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY", "cerebras": "CEREBRAS_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY", "together_ai": "TOGETHER_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
    }
    env = env_by_prefix.get(prefix)
    return bool(os.getenv(env)) if env else True


def list_models() -> list[dict]:
    """Известные модели с признаком «настроена ли»."""
    return [{**m, "configured": model_configured(m["model"])} for m in KNOWN_MODELS]


def default_chain() -> list[str]:
    """Настроенные модели в порядке предпочтения. Claude CLI — только если явно включён."""
    return [m["model"] for m in KNOWN_MODELS if model_configured(m["model"])]


def build_llm(spec: str | LLMProvider) -> LLMProvider:
    if not isinstance(spec, str):
        return spec
    if spec == "claude-cli" or spec.startswith("claude-cli/"):
        _, _, model = spec.partition("/")
        return ClaudeCLI(model=model or None)
    return LiteLLM(model=spec)


def complete_with_fallback(
    messages: list[Message],
    models: list[str | LLMProvider],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Перебирает модели по очереди. Возвращает первый успешный ответ с историей попыток."""
    if not models:
        raise NotConfigured(
            "Не задано ни одной модели. Положите ключ в .env (GEMINI_API_KEY, GROQ_API_KEY, …) "
            f"или включите Claude CLI: {CLAUDE_CLI_ENV}=1"
        )
    attempts: list[dict] = []
    for spec in models:
        provider = build_llm(spec)
        started = time.monotonic()
        try:
            response = provider.complete(messages, temperature=temperature, max_tokens=max_tokens)
        except LLMError as exc:
            attempts.append({"model": provider.name, "ok": False, "error": str(exc)[:300],
                             "seconds": round(time.monotonic() - started, 2)})
            log.warning("Модель %s не ответила: %s", provider.name, exc)
            continue
        attempts.append({"model": provider.name, "ok": True, "seconds": round(response.latency_s, 2)})
        response.attempts = attempts
        return response
    raise LLMError("Все модели недоступны: " + "; ".join(f"{a['model']} — {a['error']}" for a in attempts))
