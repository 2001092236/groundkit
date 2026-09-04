"""Загрузка текста найденных страниц, чтобы модель видела не 200 символов сниппета,
а сам документ. Простой экстрактор без зависимостей; при наличии ``trafilatura``
используется он.
"""

from __future__ import annotations

import html
import ipaddress
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx

from .search import USER_AGENT, SearchResult

log = logging.getLogger("groundkit.fetch")

_DROP_BLOCKS = re.compile(r"(?is)<(script|style|noscript|svg|nav|footer|header|form)\b.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n\s*\n+")


def html_to_text(raw: str) -> str:
    """Грубое HTML → текст: без скриптов/стилей/навигации, схлопнутые пробелы."""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(raw, include_comments=False, include_tables=True)
        if extracted:
            return extracted
    except ImportError:
        pass
    text = _DROP_BLOCKS.sub(" ", raw)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", text)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _NL.sub("\n\n", text).strip()


def _is_private_host(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host in {"localhost", ""} or host.endswith(".local"):
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def fetch_text(url: str, *, timeout: float = 10.0, max_chars: int = 4000) -> str | None:
    """Скачивает страницу и возвращает текст, либо None если не HTML/не удалось."""
    if _is_private_host(url):
        return None
    try:
        with httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en"}
        ) as client:
            with client.stream("GET", url) as resp:
                ctype = resp.headers.get("content-type", "")
                if resp.status_code >= 400 or not ("html" in ctype or "text/plain" in ctype):
                    return None
                chunks, size = [], 0
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > 1_500_000:
                        break
                raw = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — страница может быть недоступна, это нормально
        log.debug("Не удалось загрузить %s: %s", url, exc)
        return None
    text = raw if "text/plain" in ctype else html_to_text(raw)
    return text[:max_chars] or None


def enrich(
    results: list[SearchResult], *, max_pages: int = 5, max_chars: int = 4000, timeout: float = 10.0, workers: int = 5
) -> int:
    """Догружает текст первых ``max_pages`` результатов без контента. Возвращает число удач."""
    targets = [r for r in results if not r.content][:max_pages]
    if not targets:
        return 0
    with ThreadPoolExecutor(max_workers=min(workers, len(targets))) as pool:
        texts = list(pool.map(lambda r: fetch_text(r.url, timeout=timeout, max_chars=max_chars), targets))
    done = 0
    for r, text in zip(targets, texts, strict=True):
        if text and len(text) > len(r.snippet):
            r.content = text
            done += 1
    return done
