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
import ssl
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .usage import get_ledger

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


def _error_headers(exc: Exception) -> dict | None:
    """Заголовки ответа из исключения LiteLLM/OpenAI-совместимого клиента, если они есть."""
    for candidate in (getattr(exc, "headers", None), getattr(getattr(exc, "response", None), "headers", None)):
        if candidate:
            try:
                return dict(candidate)
            except (TypeError, ValueError):
                continue
    return None


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
        litellm.return_response_headers = True  # иначе x-ratelimit-* не попадают в _hidden_params
        started = time.monotonic()
        ledger = get_ledger()
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 — классифицируем и пробрасываем
            err = _classify(exc)
            ledger.record(self.model, ok=False, latency_s=time.monotonic() - started, error=str(err),
                          rate_limited=isinstance(err, RateLimited), headers=_error_headers(exc))
            raise err from exc

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        cost: float | None
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:  # noqa: BLE001 — для бесплатных моделей цены может не быть
            cost = None
        result = LLMResponse(
            text=text,
            model=self.model,
            provider="litellm",
            latency_s=time.monotonic() - started,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cost_usd=cost,
        )
        hidden = getattr(response, "_hidden_params", None) or {}
        ledger.record(self.model, ok=True, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                      cost_usd=cost, latency_s=result.latency_s, headers=hidden.get("additional_headers"))
        return result


# OpenAI-совместимые провайдеры, которые вызываем напрямую: так видны заголовки x-ratelimit-*
# (LiteLLM их не пробрасывает), а код остаётся прозрачным.
OPENAI_COMPAT: dict[str, dict] = {
    "groq": {"base": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY"},
    "cerebras": {"base": "https://api.cerebras.ai/v1", "env": "CEREBRAS_API_KEY"},
    "mistral": {"base": "https://api.mistral.ai/v1", "env": "MISTRAL_API_KEY"},
    "openrouter": {"base": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY"},
    # Z.ai (Zhipu), международный эндпоинт. У GLM размышления включены по умолчанию и
    # съедают в разы больше токенов, поэтому по умолчанию выключаем их явно.
    "zai": {"base": "https://api.z.ai/api/paas/v4", "env": "ZAI_API_KEY",
            "extra_body": {"thinking": {"type": "disabled"}}},
}


@dataclass
class OpenAICompat:
    """Прямой вызов ``/chat/completions`` у Groq, Cerebras, Mistral, OpenRouter.

    ``model`` — в формате LiteLLM: ``groq/qwen/qwen3.8-27b``; префикс выбирает провайдера.
    """

    model: str
    timeout: float = 90.0

    @property
    def name(self) -> str:
        return self.model

    def complete(
        self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int | None = None
    ) -> LLMResponse:
        prefix, _, remote = self.model.partition("/")
        cfg = OPENAI_COMPAT[prefix]
        key = os.getenv(cfg["env"], "")
        ledger = get_ledger()
        if not key:
            raise NotConfigured(f"{cfg['env']} не задан")
        body: dict = {"model": remote, "messages": messages, "temperature": temperature,
                      **cfg.get("extra_body", {})}
        if max_tokens:
            body["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if prefix == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/2001092236/groundkit"
            headers["X-Title"] = "groundkit"
        started = time.monotonic()
        try:
            resp = httpx.post(f"{cfg['base']}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            err = LLMError(f"{type(exc).__name__}: {exc}")
            ledger.record(self.model, ok=False, latency_s=time.monotonic() - started, error=str(err))
            raise err from exc
        latency = time.monotonic() - started
        resp_headers = dict(resp.headers)
        if resp.status_code >= 400:
            detail = resp.text[:300]
            if resp.status_code == 429:
                err = RateLimited(f"{prefix} 429: {detail}")
            elif resp.status_code in (401, 402, 403):
                err = NotConfigured(f"{prefix} {resp.status_code}: {detail}")
            else:
                err = LLMError(f"{prefix} {resp.status_code}: {detail}")
            ledger.record(self.model, ok=False, latency_s=latency, error=str(err),
                          rate_limited=resp.status_code == 429, headers=resp_headers)
            raise err
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        result = LLMResponse(
            text=text, model=self.model, provider=prefix, latency_s=latency,
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"), cost_usd=None,
        )
        ledger.record(self.model, ok=True, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                      latency_s=latency, headers=resp_headers)
        return result


GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1"
# Серверы Сбера подписаны сертификатом «Russian Trusted Root CA» Минцифры, которого
# нет в стандартных бандлах. Путь к скачанному .cer кладётся в GIGACHAT_CA_BUNDLE.
GIGACHAT_CA_URL = "https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer"

_gigachat_token: dict = {"value": "", "expires_at": 0.0}
_gigachat_lock = threading.Lock()
_gigachat_tls_warned = False


def _gigachat_verify() -> ssl.SSLContext | bool:
    """Чем проверять TLS Сбера: бандл из GIGACHAT_CA_BUNDLE, иначе без проверки."""
    global _gigachat_tls_warned
    bundle = os.getenv("GIGACHAT_CA_BUNDLE", "")
    if bundle and os.path.exists(bundle):
        return ssl.create_default_context(cafile=bundle)
    if not _gigachat_tls_warned:
        log.warning("GigaChat: TLS-проверка отключена — нет сертификата Минцифры. "
                    "Скачайте %s и укажите путь в GIGACHAT_CA_BUNDLE.", GIGACHAT_CA_URL)
        _gigachat_tls_warned = True
    return False


@dataclass
class GigaChat:
    """Сбер GigaChat. Ключ авторизации — base64 от ``client_id:client_secret``.

    Access-token живёт 30 минут и обновляется автоматически. Работает и из России,
    и с зарубежного сервера. Модели: ``GigaChat``, ``GigaChat-2``, ``GigaChat-2-Pro``,
    ``GigaChat-2-Max``.
    """

    model: str = "GigaChat-2"
    auth_key: str = field(default_factory=lambda: os.getenv("GIGACHAT_AUTH_KEY", ""))
    scope: str = field(default_factory=lambda: os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"))
    timeout: float = 90.0

    @property
    def name(self) -> str:
        return f"gigachat/{self.model}"

    def _token(self) -> str:
        """Access-token из кэша или новый по ключу авторизации."""
        with _gigachat_lock:
            if _gigachat_token["value"] and time.time() < _gigachat_token["expires_at"] - 60:
                return _gigachat_token["value"]
            if not self.auth_key:
                raise NotConfigured("GIGACHAT_AUTH_KEY не задан (base64 от client_id:client_secret)")
            try:
                resp = httpx.post(
                    GIGACHAT_OAUTH_URL,
                    data={"scope": self.scope},
                    headers={"Authorization": f"Basic {self.auth_key}", "RqUID": str(uuid.uuid4()),
                             "Content-Type": "application/x-www-form-urlencoded"},
                    timeout=self.timeout,
                    verify=_gigachat_verify(),
                )
            except httpx.HTTPError as exc:
                raise LLMError(f"GigaChat OAuth недоступен: {type(exc).__name__}: {exc}") from exc
            if resp.status_code in (401, 403):
                raise NotConfigured(f"GigaChat OAuth {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise LLMError(f"GigaChat OAuth {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            _gigachat_token["value"] = data["access_token"]
            expires = float(data.get("expires_at") or 0) / 1000
            _gigachat_token["expires_at"] = expires or time.time() + 1500
            return _gigachat_token["value"]

    def complete(
        self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int | None = None
    ) -> LLMResponse:
        ledger = get_ledger()
        started = time.monotonic()
        try:
            token = self._token()
        except LLMError as exc:
            ledger.record(self.name, ok=False, latency_s=time.monotonic() - started, error=str(exc))
            raise
        body: dict = {"model": self.model, "messages": messages, "temperature": max(temperature, 0.01)}
        if max_tokens:
            body["max_tokens"] = max_tokens
        try:
            resp = httpx.post(
                f"{GIGACHAT_API_URL}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=self.timeout, verify=_gigachat_verify(),
            )
        except httpx.HTTPError as exc:
            err = LLMError(f"GigaChat: {type(exc).__name__}: {exc}")
            ledger.record(self.name, ok=False, latency_s=time.monotonic() - started, error=str(err))
            raise err from exc
        latency = time.monotonic() - started
        if resp.status_code >= 400:
            detail = resp.text[:300]
            if resp.status_code == 429:
                err = RateLimited(f"GigaChat 429: {detail}")
            elif resp.status_code in (401, 402, 403):
                _gigachat_token["value"] = ""  # токен мог протухнуть — возьмём новый на следующей попытке
                err = NotConfigured(f"GigaChat {resp.status_code}: {detail}")
            else:
                err = LLMError(f"GigaChat {resp.status_code}: {detail}")
            ledger.record(self.name, ok=False, latency_s=latency, error=str(err),
                          rate_limited=resp.status_code == 429, headers=dict(resp.headers))
            raise err
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        result = LLMResponse(
            text=(choice.get("message") or {}).get("content") or "", model=self.name, provider="gigachat",
            latency_s=latency, input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
        )
        ledger.record(self.name, ok=True, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                      latency_s=latency, headers=dict(resp.headers))
        return result


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
                err: LLMError = NotConfigured(f"claude CLI: {result[:300]}")
            elif "rate limit" in lowered or "usage limit" in lowered:
                err = RateLimited(f"claude CLI: {result[:300]}")
            else:
                err = LLMError(f"claude CLI: {result[:300]}")
            get_ledger().record(self.name, ok=False, error=str(err), rate_limited=isinstance(err, RateLimited),
                                latency_s=time.monotonic() - started)
            raise err

        usage = data.get("usage") or {}
        used_models = [m for m in (data.get("modelUsage") or {}) if "haiku" not in m] or list(
            data.get("modelUsage") or {}
        )
        get_ledger().record(self.name, ok=True, input_tokens=usage.get("input_tokens"),
                            output_tokens=usage.get("output_tokens"), cost_usd=data.get("total_cost_usd"),
                            latency_s=time.monotonic() - started)
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
    # Лимиты — с официальных страниц провайдеров на 04.09.2026 (поле docs). rpd/rpm — только явно опубликованные.
    {"model": "groq/qwen/qwen3.8-27b", "label": "Groq · Qwen 3.8 27B", "env": "GROQ_API_KEY",
     "free": "1000 запросов/день, 30 RPM, 8K токенов/мин, 200K токенов/день", "rpd": 1000, "rpm": 30,
     "reset": "по заголовкам x-ratelimit-reset-*", "docs": "https://console.groq.com/docs/rate-limits",
     "signup": "https://console.groq.com/keys"},
    {"model": "groq/openai/gpt-oss-120b", "label": "Groq · GPT-OSS 120B", "env": "GROQ_API_KEY",
     "free": "1000 запросов/день, 30 RPM, 8K токенов/мин, 200K токенов/день", "rpd": 1000, "rpm": 30,
     "reset": "по заголовкам x-ratelimit-reset-*", "docs": "https://console.groq.com/docs/rate-limits",
     "signup": "https://console.groq.com/keys"},
    {"model": "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast", "label": "Cloudflare · Llama 3.3 70B",
     "env": "CLOUDFLARE_API_KEY", "free": "10 000 нейронов/день ≈ 375K входных или 49K выходных токенов",
     "rpd": None, "rpm": None, "reset": "00:00 UTC",
     "docs": "https://developers.cloudflare.com/workers-ai/platform/pricing/", "signup": "https://dash.cloudflare.com"},
    {"model": "zai/glm-4.7-flash", "label": "Z.ai · GLM-4.7 Flash", "env": "ZAI_API_KEY",
     "free": "бессрочно бесплатна, 1 одновременный запрос; размышления выключены (иначе расход в 28 раз выше)",
     "rpd": None, "rpm": None, "reset": "не публикуется",
     "docs": "https://docs.z.ai/guides/llm/glm-4.7-flash", "signup": "https://z.ai/manage-apikey/apikey-list"},
    {"model": "gigachat/GigaChat-2", "label": "Сбер · GigaChat-2", "env": "GIGACHAT_AUTH_KEY",
     "free": "Freemium: 1 000 000 токенов, обновляется раз в 12 месяцев; 1 поток", "rpd": None, "rpm": None,
     "reset": "раз в 12 месяцев от даты регистрации",
     "docs": "https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs",
     "signup": "https://developers.sber.ru/studio"},
    {"model": "gigachat/GigaChat-2-Pro", "label": "Сбер · GigaChat-2 Pro", "env": "GIGACHAT_AUTH_KEY",
     "free": "из того же пула 1 000 000 токенов, расходуется быстрее", "rpd": None, "rpm": None,
     "reset": "раз в 12 месяцев от даты регистрации",
     "docs": "https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs",
     "signup": "https://developers.sber.ru/studio"},
    {"model": "openrouter/google/gemma-4-26b-a4b-it:free", "label": "OpenRouter · Gemma 4 26B (free)",
     "env": "OPENROUTER_API_KEY", "free": "50 запросов/день на все free-модели (1000 при пополнении ≥ $10), 20 RPM",
     "rpd": 50, "rpm": 20, "reset": "00:00 UTC", "docs": "https://openrouter.ai/docs/api-reference/limits",
     "signup": "https://openrouter.ai/keys"},
    {"model": "openrouter/google/gemma-4-31b-it:free", "label": "OpenRouter · Gemma 4 31B (free)",
     "env": "OPENROUTER_API_KEY", "free": "50 запросов/день на все free-модели (1000 при пополнении ≥ $10), 20 RPM",
     "rpd": 50, "rpm": 20, "reset": "00:00 UTC", "docs": "https://openrouter.ai/docs/api-reference/limits",
     "signup": "https://openrouter.ai/keys"},
    {"model": "mistral/mistral-small-latest", "label": "Mistral Small", "env": "MISTRAL_API_KEY",
     "free": "Free mode: $10 кредитов/мес, лимиты по моделям видны только в Admin Panel → API → Limits",
     "rpd": None, "rpm": None, "reset": "не публикуется",
     "docs": "https://docs.mistral.ai/admin/billing-usage/usage-limits", "signup": "https://console.mistral.ai/api-keys"},
    {"model": "cerebras/gpt-oss-120b", "label": "Cerebras · GPT-OSS 120B", "env": "CEREBRAS_API_KEY",
     "free": "триал $5 на 30 дней только после привязки карты; 5 RPM, 1M токенов/день", "rpd": None, "rpm": 5,
     "reset": "непрерывное пополнение (token bucket)", "docs": "https://inference-docs.cerebras.ai/support/rate-limits",
     "signup": "https://cloud.cerebras.ai"},
    {"model": "gemini/gemini-3.1-flash-lite", "label": "Google Gemini 3.1 Flash Lite", "env": "GEMINI_API_KEY",
     "free": "лимиты видны только в AI Studio; старшие Flash часто 503 «high demand»", "rpd": None, "rpm": None,
     "reset": "00:00 по тихоокеанскому времени (07:00 UTC)", "docs": "https://ai.google.dev/gemini-api/docs/rate-limits",
     "signup": "https://aistudio.google.com/apikey"},
    {"model": "anthropic/claude-haiku-4-5-20251001", "label": "Anthropic · Claude Haiku 4.5 (платно)",
     "env": "ANTHROPIC_API_KEY", "free": "нет, платный API", "rpd": None, "rpm": None, "reset": None,
     "docs": "https://docs.anthropic.com/en/api/rate-limits", "signup": "https://console.anthropic.com"},
    {"model": "claude-cli", "label": "Claude Code CLI (локальные эксперименты)", "env": None,
     "free": "в рамках подписки", "rpd": None, "rpm": None, "reset": None, "docs": None,
     "signup": "https://docs.anthropic.com/claude-code"},
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
        "cloudflare": "CLOUDFLARE_API_KEY", "dashscope": "DASHSCOPE_API_KEY", "gigachat": "GIGACHAT_AUTH_KEY",
        "zai": "ZAI_API_KEY",
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
    if spec.startswith("gigachat/"):
        return GigaChat(model=spec.split("/", 1)[1])
    if spec.split("/", 1)[0] in OPENAI_COMPAT:
        return OpenAICompat(model=spec)
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
    providers = [build_llm(spec) for spec in models]
    ledger = get_ledger()
    blocked = {p.name: ledger.blocked_until(p.name) for p in providers}
    skip_blocked = any(not blocked[p.name] for p in providers)  # если заблокированы все — всё равно пробуем
    for provider in providers:
        until = blocked.get(provider.name)
        if until and skip_blocked:
            attempts.append({"model": provider.name, "ok": False, "skipped": True,
                             "error": f"пропущена: лимит исчерпан до {until.strftime('%H:%M')} UTC"})
            continue
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
