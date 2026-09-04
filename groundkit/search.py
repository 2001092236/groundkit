"""Провайдеры веб-поиска с единым интерфейсом и фильтром по доменам.

Все провайдеры возвращают список ``SearchResult``. Фильтр по доменам применяется
на нашей стороне — так поведение одинаково у всех источников.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

log = logging.getLogger("groundkit.search")

USER_AGENT = "groundkit/0.1 (+https://github.com/2001092236/groundkit)"

# Параметры-трекеры, которые не влияют на содержимое страницы.
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "yclid"}


def normalize_url(url: str) -> str:
    """Приводит URL к каноничному виду для сравнения ссылок.

    Убирает схему, ``www.``, якорь, трекинговые параметры и завершающий слэш.
    ``https://www.Pravo.gov.ru/doc/?utm_source=x#top`` → ``pravo.gov.ru/doc``.
    """
    url = url.strip().rstrip(".,;:)»\"'")
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query) if k not in _TRACKING_PARAMS])
    path = parsed.path.rstrip("/")
    return urlunparse(("", host, path, "", query, "")).lstrip("/")


@dataclass
class SearchResult:
    """Один найденный документ."""

    title: str
    url: str
    snippet: str
    published: str | None = None
    index: int = 0
    content: str | None = None  # полный текст страницы, если его загружали

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.lower().removeprefix("www.")

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "snippet": self.snippet,
            "published": self.published,
            "has_content": bool(self.content),
        }


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int = 8) -> list[SearchResult]: ...


def _filter_domains(results: list[SearchResult], allowed: list[str] | None) -> list[SearchResult]:
    """Оставляет только результаты с разрешённых доменов (включая поддомены)."""
    if not allowed:
        return results
    normalized = [d.lower().strip().removeprefix("www.") for d in allowed if d.strip()]
    if not normalized:
        return results
    kept = []
    for r in results:
        if any(r.domain == d or r.domain.endswith("." + d) for d in normalized):
            kept.append(r)
    return kept


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique = []
    for r in results:
        if not r.url:
            continue
        key = normalize_url(r.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


@dataclass
class SearXNG:
    """Self-hosted метапоиск. Бесплатно и без ключа.

    Требует инстанс с включённым JSON-выводом в settings.yml::

        search:
          formats: [html, json]
    """

    base_url: str = field(default_factory=lambda: os.getenv("SEARXNG_URL", "http://localhost:8080"))
    language: str = "ru"
    engines: str | None = None  # например "google,bing,duckduckgo"
    timeout: float = 20.0
    name: str = "searxng"

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        params: dict[str, str] = {"q": query, "format": "json", "language": self.language}
        if self.engines:
            params["engines"] = self.engines
        with _client(self.timeout) as client:
            resp = client.get(f"{self.base_url.rstrip('/')}/search", params=params)
        resp.raise_for_status()
        payload = resp.json().get("results", [])
        return [
            SearchResult(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
                published=item.get("publishedDate"),
            )
            for item in payload[:limit]
        ]


@dataclass
class Exa:
    """Нейропоиск по смыслу. 20 000 запросов/мес бесплатно, отдаёт полный текст."""

    api_key: str = field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))
    timeout: float = 30.0
    name: str = "exa"

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("EXA_API_KEY не задан")
        body = {"query": query, "numResults": limit, "contents": {"text": {"maxCharacters": 3000}}}
        with _client(self.timeout) as client:
            resp = client.post("https://api.exa.ai/search", json=body, headers={"x-api-key": self.api_key})
        resp.raise_for_status()
        out = []
        for item in resp.json().get("results", []):
            text = item.get("text") or ""
            out.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=item.get("url") or "",
                    snippet=text[:500],
                    published=item.get("publishedDate"),
                    content=text or None,
                )
            )
        return out


@dataclass
class Brave:
    """Независимый индекс. 2000 запросов/мес бесплатно."""

    api_key: str = field(default_factory=lambda: os.getenv("BRAVE_API_KEY", ""))
    timeout: float = 20.0
    name: str = "brave"

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("BRAVE_API_KEY не задан")
        with _client(self.timeout) as client:
            resp = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": min(limit, 20), "search_lang": "ru"},
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            )
        resp.raise_for_status()
        items = resp.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("description") or "",
                published=item.get("page_age") or item.get("age"),
            )
            for item in items
        ]


@dataclass
class Jina:
    """Поиск Jina (s.jina.ai): нужен ключ (бесплатный стартовый баланс токенов), отдаёт текст страниц."""

    api_key: str = field(default_factory=lambda: os.getenv("JINA_API_KEY", ""))
    timeout: float = 40.0
    name: str = "jina"

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY не задан (с сентября 2026 s.jina.ai без ключа отвечает 401)")
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.api_key}"}
        with _client(self.timeout) as client:
            resp = client.get("https://s.jina.ai/", params={"q": query}, headers=headers)
        resp.raise_for_status()
        out = []
        for item in resp.json().get("data", [])[:limit]:
            text = item.get("content") or ""
            out.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=item.get("url") or "",
                    snippet=(item.get("description") or text[:500]),
                    published=item.get("publishedTime"),
                    content=text[:3000] or None,
                )
            )
        return out


@dataclass
class DuckDuckGo:
    """Запасной вариант без ключа. Требует пакет ``ddgs``."""

    region: str = "ru-ru"
    timeout: float = 20.0
    name: str = "ddg"

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Установите `pip install ddgs`") from exc

        hits = list(DDGS(timeout=self.timeout).text(query, region=self.region, max_results=limit))
        return [
            SearchResult(
                title=h.get("title") or "",
                url=h.get("href") or "",
                snippet=h.get("body") or "",
            )
            for h in hits
        ]


PROVIDERS: dict[str, type] = {
    "searxng": SearXNG,
    "exa": Exa,
    "brave": Brave,
    "jina": Jina,
    "ddg": DuckDuckGo,
}

# Что нужно, чтобы провайдер заработал (для CLI `groundkit providers` и веб-демо).
PROVIDER_INFO: dict[str, dict] = {
    "searxng": {"label": "SearXNG (self-hosted)", "env": "SEARXNG_URL", "free": "безлимитно", "needs_key": False},
    "ddg": {"label": "DuckDuckGo", "env": None, "free": "без ключа, мягкие лимиты", "needs_key": False},
    "jina": {"label": "Jina Search", "env": "JINA_API_KEY", "free": "10M токенов на новый ключ, 100 RPM",
             "needs_key": True},
    "exa": {"label": "Exa", "env": "EXA_API_KEY", "free": "$10 кредитов/мес ≈ 1400 поисков", "needs_key": True},
    "brave": {"label": "Brave Search", "env": "BRAVE_API_KEY", "free": "2000 запр/мес", "needs_key": True},
}


def provider_configured(name: str) -> bool:
    info = PROVIDER_INFO.get(name)
    if info is None:
        return False
    if name == "ddg":
        try:
            import ddgs  # noqa: F401
        except ImportError:
            return False
        return True
    if not info["needs_key"]:
        return True
    return bool(os.getenv(info["env"] or ""))


def build_provider(spec: str | SearchProvider) -> SearchProvider:
    if not isinstance(spec, str):
        return spec
    if spec not in PROVIDERS:
        raise ValueError(f"Неизвестный провайдер поиска: {spec}. Доступны: {list(PROVIDERS)}")
    return PROVIDERS[spec]()


@dataclass
class SearchRun:
    """Результат прогона цепочки провайдеров."""

    results: list[SearchResult]
    provider: str
    errors: dict[str, str] = field(default_factory=dict)
    query: str = ""


def run_search(
    query: str,
    providers: list[str | SearchProvider],
    allowed_domains: list[str] | None = None,
    limit: int = 8,
) -> SearchRun:
    """Пробует провайдеров по очереди, пока один не вернёт результаты.

    Возвращает результаты, имя сработавшего провайдера и ошибки остальных.
    """
    errors: dict[str, str] = {}
    for spec in providers:
        try:
            provider = build_provider(spec)
        except ValueError as exc:
            errors[str(spec)] = str(exc)
            continue
        name = getattr(provider, "name", str(spec))
        try:
            raw = provider.search(query, limit=limit * 3 if allowed_domains else limit)
        except Exception as exc:  # noqa: BLE001 — падение одного не должно ронять цепочку
            errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("Поиск %s не удался: %s", name, errors[name])
            continue
        filtered = _filter_domains(_dedupe(raw), allowed_domains)[:limit]
        if filtered:
            for i, r in enumerate(filtered, start=1):
                r.index = i
            return SearchRun(results=filtered, provider=name, errors=errors, query=query)
        errors.setdefault(name, "нет результатов" + (" по разрешённым доменам" if allowed_domains else ""))
    return SearchRun(results=[], provider="", errors=errors, query=query)


def search_with_fallback(
    query: str,
    providers: list[str | SearchProvider],
    allowed_domains: list[str] | None = None,
    limit: int = 8,
) -> tuple[list[SearchResult], str]:
    """Совместимая обёртка над ``run_search``: (результаты, имя провайдера).

    Если ни один провайдер не отработал без ошибок — поднимает RuntimeError.
    """
    run = run_search(query, providers, allowed_domains, limit)
    if not run.results and run.errors and all("нет результатов" not in e for e in run.errors.values()):
        raise RuntimeError(f"Все провайдеры поиска недоступны: {run.errors}")
    return run.results, run.provider
