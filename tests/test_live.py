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


@live
@pytest.mark.skipif(not os.getenv("GIGACHAT_AUTH_KEY"), reason="нет GIGACHAT_AUTH_KEY")
def test_gigachat_live():
    from groundkit import GigaChat

    out = GigaChat().complete([{"role": "system", "content": "Отвечай одним словом."},
                               {"role": "user", "content": "Столица Франции?"}])
    print("\nGigaChat:", out.text, out.model, f"{out.latency_s:.1f}s")
    assert "париж" in out.text.lower()
    assert out.provider == "gigachat" and out.input_tokens


@live
def test_pollinations_live_image():
    """Pollinations работает без ключа — картинка должна прийти настоящим JPEG/PNG."""
    from groundkit import generate_image

    img = generate_image("a red apple on a white table", ["pollinations"], size=(384, 384), seed=1)
    print("\nPollinations:", img.content_type, len(img.data), "байт", f"{img.latency_s:.1f}s")
    assert img.data[:2] in (b"\xff\xd8", b"\x89P") and len(img.data) > 3000
    assert img.content_type.startswith("image/")
