"""Связка «поиск → контекст → LLM → программная проверка ссылок»."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .fetch import enrich
from .llm import LLMError, LLMProvider, LLMResponse, complete_with_fallback, default_chain
from .search import SearchProvider, SearchResult, normalize_url, run_search

log = logging.getLogger("groundkit.answer")

SYSTEM_PROMPT = """Ты отвечаешь строго на основе предоставленных источников.

Правила:
1. Используй ТОЛЬКО информацию из источников ниже. Если ответа в них нет — так и скажи.
2. Ссылайся на источники номерами в квадратных скобках: [1], [2]. Каждое утверждение — с номером.
3. Указывай дату публикации источника, если она известна. Это критично: устаревшая
   редакция документа выглядит так же убедительно, как действующая.
4. Не придумывай URL. Любая ссылка должна быть из списка источников.
5. Если источники противоречат друг другу — скажи об этом прямо.
6. Отвечай на языке вопроса. Пиши по делу, без вступлений.
"""

REWRITE_PROMPT = """Преобразуй вопрос пользователя в короткий поисковый запрос (3–8 слов) на языке вопроса:
ключевые термины и юридические/предметные формулировки, без вопросительных слов, эмоций и лишних деталей.
Ответь только текстом запроса, без кавычек и пояснений."""

URL_RE = re.compile(r"https?://[^\s<>\)\]\"'»]+")
CITATION_RE = re.compile(r"\[(\d{1,2})\]")

# Обрезаем инструкции, которые может содержать сама веб-страница (prompt injection).
INJECTION_RE = re.compile(
    r"(?i)(ignore (all )?(previous|prior|above) instructions"
    r"|disregard (the )?(above|previous|prior)( instructions)?"
    r"|you are now [a-z ]{0,40}(assistant|ai|model)"
    r"|\bsystem prompt\b"
    r"|ты (теперь|должен теперь)"
    r"|(игнорируй|забудь) (все )?(предыдущие|прошлые|прежние) (инструкции|указания))"
)

MAX_CONTEXT_CHARS_PER_SOURCE = 2000
MAX_CONTEXT_CHARS_TOTAL = 12000  # ~5–6K токенов кириллицы: укладывается в 8K токенов/мин бесплатного Groq


@dataclass
class Answer:
    answer: str
    sources: list[SearchResult]
    search_provider: str
    model: str
    cited: list[int] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)
    dropped_indices: list[int] = field(default_factory=list)
    injection_flagged: bool = False
    search_query: str = ""
    search_errors: dict[str, str] = field(default_factory=dict)
    llm: LLMResponse | None = None
    pages_fetched: int = 0
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.sources) and bool(self.model)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "ok": self.ok,
            "sources": [s.to_dict() for s in self.sources],
            "cited": self.cited,
            "search_provider": self.search_provider,
            "search_query": self.search_query,
            "search_errors": self.search_errors,
            "model": self.model,
            "llm": self.llm.to_dict() if self.llm else None,
            "dropped_citations": self.dropped_citations,
            "dropped_indices": self.dropped_indices,
            "injection_flagged": self.injection_flagged,
            "pages_fetched": self.pages_fetched,
            "timing": self.timing,
        }


def sanitize(text: str) -> tuple[str, bool]:
    """Убирает из текста источника похожее на инструкции модели."""
    flagged = bool(INJECTION_RE.search(text))
    return INJECTION_RE.sub("[удалено]", text), flagged


def build_context(
    results: list[SearchResult],
    max_chars: int = MAX_CONTEXT_CHARS_PER_SOURCE,
    total_chars: int = MAX_CONTEXT_CHARS_TOTAL,
) -> tuple[str, bool]:
    """Нумерованный контекст для модели. Возвращает текст и флаг подозрения на инъекцию.

    Каждому источнику даётся не больше ``max_chars``, всем вместе — не больше ``total_chars``:
    бесплатные модели ограничены не только запросами в день, но и токенами в минуту.
    """
    blocks = []
    any_flagged = False
    per_source = max(300, min(max_chars, total_chars // max(len(results), 1)))
    for r in results:
        body = (r.content or r.snippet or "")[:per_source]
        clean, flagged = sanitize(body)
        any_flagged = any_flagged or flagged
        date = f" (дата: {r.published})" if r.published else ""
        blocks.append(f"[{r.index}] {r.title}{date}\nURL: {r.url}\n{clean}")
    return "\n\n".join(blocks), any_flagged


def verify_citations(text: str, results: list[SearchResult]) -> tuple[str, list[str], list[int], list[int]]:
    """Программная сверка ответа с источниками.

    * URL, которого не было в выдаче, заменяется пометкой — это и есть «выдуманная ссылка».
    * Номер [n] вне диапазона источников тоже вырезается.
    Возвращает (текст, отброшенные URL, отброшенные номера, использованные номера).
    """
    known = {normalize_url(r.url): r.url for r in results}
    valid_indices = {r.index for r in results}
    dropped_urls: list[str] = []
    dropped_idx: list[int] = []

    def replace_url(match: re.Match[str]) -> str:
        raw = match.group(0)
        if normalize_url(raw) in known:
            return raw
        dropped_urls.append(raw.rstrip(".,;:)"))
        return "[ссылка не подтверждена источниками]"

    def replace_idx(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if n in valid_indices:
            return match.group(0)
        dropped_idx.append(n)
        return ""

    text = URL_RE.sub(replace_url, text)
    text = CITATION_RE.sub(replace_idx, text)
    cited = sorted({int(m) for m in CITATION_RE.findall(text)})
    return text, dropped_urls, sorted(set(dropped_idx)), cited


class Answerer:
    """Задаёт вопрос модели, опираясь на результаты веб-поиска.

    Пример::

        qa = Answerer(allowed_domains=["pravo.gov.ru"])
        result = qa.ask("Требования к форме договора аренды?")
    """

    def __init__(
        self,
        allowed_domains: list[str] | None = None,
        search: str | SearchProvider | list[str | SearchProvider] = "searxng",
        model: str | LLMProvider | list[str | LLMProvider] | None = None,
        limit: int = 8,
        fetch_pages: bool = True,
        max_pages: int = 5,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.2,
        rewrite_query: bool = False,
        max_context_chars: int = MAX_CONTEXT_CHARS_TOTAL,
    ) -> None:
        self.allowed_domains = [d for d in (allowed_domains or []) if d.strip()] or None
        self.search_providers = search if isinstance(search, list) else [search]
        if model is None:
            self.models: list[str | LLMProvider] = list(default_chain())
        else:
            self.models = model if isinstance(model, list) else [model]
        self.limit = limit
        self.fetch_pages = fetch_pages
        self.max_pages = max_pages
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.rewrite_query = rewrite_query
        self.max_context_chars = max_context_chars

    def search(self, question: str):
        return run_search(question, self.search_providers, self.allowed_domains, self.limit)

    def rewrite(self, question: str) -> str:
        """Переформулирует разговорный вопрос в поисковый запрос. При любой ошибке — исходный вопрос."""
        messages = [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": question}]
        try:
            response = complete_with_fallback(messages, self.models, temperature=0.0, max_tokens=60)
        except LLMError:
            return question
        query = response.text.strip().strip('"«»\'').splitlines()[0].strip() if response.text.strip() else ""
        return query if 3 <= len(query) <= 200 else question

    def ask(self, question: str) -> Answer:
        timing: dict[str, float] = {}
        t0 = time.monotonic()
        query = question
        if self.rewrite_query:
            query = self.rewrite(question)
            timing["rewrite_s"] = round(time.monotonic() - t0, 2)
        t_search = time.monotonic()
        run = self.search(query)
        if not run.results and query != question:
            run = self.search(question)  # переформулировка не помогла — пробуем как есть
            query = question
        timing["search_s"] = round(time.monotonic() - t_search, 2)

        if not run.results:
            return Answer(
                answer="По разрешённым источникам ничего не найдено."
                if self.allowed_domains
                else "Поиск не вернул результатов.",
                sources=[],
                search_provider=run.provider,
                model="",
                search_query=query,
                search_errors=run.errors,
                timing=timing,
            )

        pages = 0
        if self.fetch_pages:
            t1 = time.monotonic()
            pages = enrich(run.results, max_pages=self.max_pages)
            timing["fetch_s"] = round(time.monotonic() - t1, 2)

        context, flagged = build_context(run.results, total_chars=self.max_context_chars)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Источники:\n\n{context}\n\nВопрос: {question}"},
        ]

        t2 = time.monotonic()
        response = complete_with_fallback(messages, self.models, temperature=self.temperature)
        timing["llm_s"] = round(time.monotonic() - t2, 2)

        verified, dropped_urls, dropped_idx, cited = verify_citations(response.text, run.results)
        timing["total_s"] = round(time.monotonic() - t0, 2)
        if dropped_urls or dropped_idx:
            log.info("Отброшено выдуманных ссылок: %s, номеров: %s", dropped_urls, dropped_idx)

        return Answer(
            answer=verified,
            sources=run.results,
            search_provider=run.provider,
            model=response.model,
            cited=cited,
            dropped_citations=dropped_urls,
            dropped_indices=dropped_idx,
            injection_flagged=flagged,
            search_query=query,
            search_errors=run.errors,
            llm=response,
            pages_fetched=pages,
            timing=timing,
        )
