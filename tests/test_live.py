"""Живые проверки: реальный поиск и реальная модель. Включаются GROUNDKIT_LIVE=1.

  GROUNDKIT_LIVE=1 pytest -m live -s
"""

import os
import shutil

import pytest

from groundkit import Answerer, ClaudeCLI
from groundkit.search import DuckDuckGo, run_search

pytestmark = pytest.mark.live
live = pytest.mark.skipif(os.getenv("GROUNDKIT_LIVE") != "1", reason="нужен GROUNDKIT_LIVE=1")


@live
def test_ddg_live_search_returns_results():
    run = run_search("Гражданский кодекс РФ форма договора аренды здания", ["ddg"], None, 5)
    assert run.provider == "ddg" and len(run.results) >= 1
    assert all(r.url.startswith("http") for r in run.results)


@live
def test_domain_whitelist_live():
    run = run_search("статья 651 ГК РФ", ["ddg"], ["consultant.ru", "garant.ru"], 5)
    assert run.results and all(r.domain.endswith(("consultant.ru", "garant.ru")) for r in run.results)


@live
@pytest.mark.skipif(shutil.which("claude") is None, reason="нет claude CLI")
def test_claude_cli_as_plain_llm():
    out = ClaudeCLI().complete([{"role": "system", "content": "Отвечай одним словом."},
                                {"role": "user", "content": "Столица Франции?"}])
    assert "париж" in out.text.lower() or "paris" in out.text.lower()
    assert out.provider == "claude-cli" and out.latency_s > 0


@live
@pytest.mark.skipif(shutil.which("claude") is None, reason="нет claude CLI")
def test_full_pipeline_ddg_plus_claude_cli():
    qa = Answerer(allowed_domains=["consultant.ru", "garant.ru", "pravo.gov.ru"], search=DuckDuckGo(),
                  model=ClaudeCLI(), limit=5)
    out = qa.ask("В какой форме заключается договор аренды здания по ГК РФ?")
    print("\nОТВЕТ:", out.answer, "\nИСТОЧНИКИ:", [s.url for s in out.sources], "\nTIMING:", out.timing)
    assert out.ok and out.cited, "модель должна сослаться хотя бы на один источник"
    assert not out.dropped_citations


@live
@pytest.mark.skipif(not (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")), reason="нет ключей Gemini/Groq")
def test_full_pipeline_free_models():
    qa = Answerer(search=DuckDuckGo(), limit=5)
    out = qa.ask("Когда был принят Гражданский кодекс РФ (часть первая)?")
    print("\nМОДЕЛЬ:", out.model, "\nОТВЕТ:", out.answer)
    assert out.ok and "1994" in out.answer
