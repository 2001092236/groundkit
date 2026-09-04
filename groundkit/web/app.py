"""Веб-демо и HTTP API: ``groundkit serve`` или ``uvicorn groundkit.web.app:app``.

Эндпоинты:
  GET  /               — страница демо
  GET  /api/config     — что настроено (поиск, модели, пресеты доменов, лимиты)
  POST /api/search     — только поиск
  POST /api/ask        — поиск + модель + проверка ссылок
  GET  /api/health
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..answer import Answerer
from ..llm import CLAUDE_CLI_ENV, LLMError, claude_cli_enabled, default_chain, list_models
from ..search import PROVIDER_INFO, provider_configured, run_search

log = logging.getLogger("groundkit.web")

STATIC = Path(__file__).parent / "static"

# Лимиты на IP, чтобы бесплатные квоты не выжрали боты. Настраиваются через env.
RPM = int(os.getenv("GROUNDKIT_RPM", "6"))
RPD = int(os.getenv("GROUNDKIT_RPD", "80"))
TRUST_PROXY = os.getenv("GROUNDKIT_TRUST_PROXY", "").lower() in {"1", "true", "yes"}
ACCESS_TOKEN = os.getenv("GROUNDKIT_ACCESS_TOKEN", "")
DEFAULT_SEARCH = [s.strip() for s in os.getenv("GROUNDKIT_SEARCH", "searxng,ddg").split(",") if s.strip()]
PUBLIC_URL = os.getenv("GROUNDKIT_PUBLIC_URL", "")

DOMAIN_PRESETS = {
    "Право РФ": ["pravo.gov.ru", "publication.pravo.gov.ru", "consultant.ru", "garant.ru", "sozd.duma.gov.ru",
                 "regulation.gov.ru", "kremlin.ru", "government.ru"],
    "Медицина": ["who.int", "minzdrav.gov.ru", "cr.minzdrav.gov.ru", "pubmed.ncbi.nlm.nih.gov",
                 "cochrane.org"],
    "Финансы РФ": ["cbr.ru", "minfin.gov.ru", "nalog.gov.ru", "moex.com"],
    "Python-документация": ["docs.python.org", "peps.python.org", "pypi.org", "docs.pydantic.dev",
                            "fastapi.tiangolo.com"],
    "Без ограничений": [],
}


class _RateLimiter:
    def __init__(self, rpm: int, rpd: int) -> None:
        self.rpm, self.rpd = rpm, rpd
        self._minute: dict[str, deque[float]] = defaultdict(deque)
        self._day: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, str]:
        now = time.monotonic()
        with self._lock:
            for bucket, window, cap in ((self._minute[key], 60, self.rpm), (self._day[key], 86400, self.rpd)):
                while bucket and now - bucket[0] > window:
                    bucket.popleft()
                if len(bucket) >= cap:
                    return False, (f"Лимит демо: не больше {self.rpm} запросов в минуту "
                                   f"и {self.rpd} в сутки с одного адреса.")
            self._minute[key].append(now)
            self._day[key].append(now)
        return True, ""


limiter = _RateLimiter(RPM, RPD)

app = FastAPI(title="groundkit demo", version=__version__, docs_url="/api/docs", redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _guard(request: Request) -> None:
    if ACCESS_TOKEN and request.headers.get("x-access-token") != ACCESS_TOKEN:
        raise HTTPException(401, "Нужен заголовок X-Access-Token")
    ok, reason = limiter.check(_client_ip(request))
    if not ok:
        raise HTTPException(429, reason)


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    domains: list[str] = Field(default_factory=list, max_length=30)
    search: list[str] = Field(default_factory=list, max_length=5)
    limit: int = Field(default=8, ge=1, le=15)


class AskIn(SearchIn):
    model: list[str] = Field(default_factory=list, max_length=5)
    fetch_pages: bool = True
    rewrite_query: bool = True


def _models_for_request(requested: list[str]) -> list[str]:
    allowed = {m["model"] for m in list_models()}
    chain = [m for m in requested if m in allowed] or default_chain()
    if not claude_cli_enabled():
        chain = [m for m in chain if not m.startswith("claude-cli")]
    return chain


def _search_for_request(requested: list[str]) -> list[str]:
    return [s for s in requested if s in PROVIDER_INFO] or DEFAULT_SEARCH


@app.api_route("/", methods=["GET", "HEAD"])
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


@app.get("/api/config")
def config() -> dict:
    models = list_models()
    if not claude_cli_enabled():
        models = [m for m in models if not m["model"].startswith("claude-cli")]
    return {
        "version": __version__,
        "search": [{"name": n, **info, "configured": provider_configured(n), "default": n in DEFAULT_SEARCH}
                   for n, info in PROVIDER_INFO.items()],
        "models": models,
        "default_chain": [m for m in default_chain() if claude_cli_enabled() or not m.startswith("claude-cli")],
        "presets": DOMAIN_PRESETS,
        "limits": {"rpm": RPM, "rpd": RPD},
        "claude_cli": claude_cli_enabled(),
        "claude_cli_env": CLAUDE_CLI_ENV,
        "access_token_required": bool(ACCESS_TOKEN),
        "public_url": PUBLIC_URL,
    }


@app.post("/api/search")
async def api_search(body: SearchIn, request: Request) -> dict:
    _guard(request)
    run = await run_in_threadpool(
        run_search, body.query, _search_for_request(body.search), body.domains or None, body.limit
    )
    return {"query": run.query, "provider": run.provider, "errors": run.errors,
            "results": [r.to_dict() for r in run.results]}


@app.post("/api/ask")
async def api_ask(body: AskIn, request: Request) -> JSONResponse:
    _guard(request)
    models = _models_for_request(body.model)
    qa = Answerer(
        allowed_domains=body.domains or None,
        search=_search_for_request(body.search),
        model=models or ["__none__"],
        limit=body.limit,
        fetch_pages=body.fetch_pages,
        rewrite_query=body.rewrite_query,
    )
    if not models:
        # Без модели всё равно показываем, что нашлось: поиск работает без ключей.
        run = await run_in_threadpool(qa.search, body.query)
        return JSONResponse(status_code=503, content={
            "error": "no_model",
            "detail": "Ни одна модель не настроена. Нужен хотя бы один ключ: GEMINI_API_KEY или GROQ_API_KEY.",
            "sources": [r.to_dict() for r in run.results],
            "search_provider": run.provider,
            "search_errors": run.errors,
        })
    try:
        result = await run_in_threadpool(qa.ask, body.query)
    except LLMError as exc:
        return JSONResponse(status_code=502, content={"error": "llm_failed", "detail": str(exc)[:600]})
    return JSONResponse(content=result.to_dict())
