import httpx
import pytest
import respx

from groundkit.search import (
    Brave,
    DuckDuckGo,
    Exa,
    Jina,
    SearchResult,
    SearXNG,
    _dedupe,
    _filter_domains,
    normalize_url,
    run_search,
    search_with_fallback,
)


def test_normalize_url_strips_noise():
    assert normalize_url("https://www.Pravo.gov.ru/doc/?utm_source=x#top") == "pravo.gov.ru/doc"
    assert normalize_url("http://pravo.gov.ru/doc") == normalize_url("https://www.pravo.gov.ru/doc/")
    assert normalize_url("https://a.ru/x?id=5&utm_medium=m") == "a.ru/x?id=5"


def test_filter_domains_keeps_subdomains_and_ignores_www():
    rs = [
        SearchResult("a", "https://www.pravo.gov.ru/x", ""),
        SearchResult("b", "https://publication.pravo.gov.ru/y", ""),
        SearchResult("c", "https://notpravo.gov.ru/z", ""),
        SearchResult("d", "https://evil.com/pravo.gov.ru", ""),
    ]
    kept = _filter_domains(rs, ["Pravo.gov.ru"])
    assert [r.title for r in kept] == ["a", "b"]
    assert _filter_domains(rs, None) == rs
    assert _filter_domains(rs, ["", "  "]) == rs


def test_dedupe_by_normalized_url():
    rs = [SearchResult("a", "https://x.ru/p/", ""), SearchResult("b", "http://www.x.ru/p", ""),
          SearchResult("c", "", ""), SearchResult("d", "https://x.ru/q", "")]
    assert [r.title for r in _dedupe(rs)] == ["a", "d"]


class Fake:
    def __init__(self, name, results=None, error=None):
        self.name, self._results, self._error = name, results or [], error
        self.calls = []

    def search(self, query, limit=8):
        self.calls.append((query, limit))
        if self._error:
            raise self._error
        return list(self._results)


def test_run_search_falls_back_and_numbers_results():
    broken = Fake("broken", error=httpx.ConnectError("down"))
    empty = Fake("empty")
    good = Fake("good", [SearchResult("t1", "https://a.ru/1", "s"), SearchResult("t2", "https://b.ru/2", "s")])
    run = run_search("q", [broken, empty, good], allowed_domains=None, limit=5)
    assert run.provider == "good"
    assert [r.index for r in run.results] == [1, 2]
    assert "broken" in run.errors and "empty" in run.errors
    assert good.calls == [("q", 5)]


def test_run_search_requests_more_when_filtering_domains():
    good = Fake("good", [SearchResult("t", "https://a.ru/1", ""), SearchResult("t", "https://b.ru/2", "")])
    run = run_search("q", [good], allowed_domains=["b.ru"], limit=4)
    assert good.calls == [("q", 12)]
    assert [r.url for r in run.results] == ["https://b.ru/2"]
    assert run.results[0].index == 1


def test_run_search_unknown_provider_name_is_reported():
    run = run_search("q", ["nope"], None, 3)
    assert run.results == [] and "nope" in run.errors


def test_search_with_fallback_raises_when_everything_broken():
    with pytest.raises(RuntimeError):
        search_with_fallback("q", [Fake("x", error=RuntimeError("boom"))])


def test_search_with_fallback_returns_empty_when_just_no_results():
    assert search_with_fallback("q", [Fake("x")]) == ([], "")


@respx.mock
def test_searxng_parses_json():
    respx.get("http://sx:8080/search").mock(return_value=httpx.Response(200, json={"results": [
        {"title": "T", "url": "https://a.ru", "content": "C", "publishedDate": "2026-01-01"},
        {"title": None, "url": "https://b.ru", "content": None},
    ]}))
    out = SearXNG(base_url="http://sx:8080/").search("q", limit=5)
    assert out[0].title == "T" and out[0].published == "2026-01-01"
    assert out[1].title == "" and out[1].snippet == ""
    assert respx.calls.last.request.url.params["format"] == "json"


@respx.mock
def test_exa_needs_key_and_returns_full_text(monkeypatch):
    with pytest.raises(RuntimeError):
        Exa(api_key="").search("q")
    respx.post("https://api.exa.ai/search").mock(return_value=httpx.Response(200, json={"results": [
        {"title": "T", "url": "https://a.ru", "text": "x" * 1000, "publishedDate": "2025-05-05"}]}))
    out = Exa(api_key="k").search("q", limit=3)
    assert out[0].content and len(out[0].snippet) == 500
    assert respx.calls.last.request.headers["x-api-key"] == "k"


@respx.mock
def test_brave_parses_web_results():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(return_value=httpx.Response(200, json={
        "web": {"results": [{"title": "T", "url": "https://a.ru", "description": "D", "page_age": "2026-02-02"}]}}))
    out = Brave(api_key="k").search("q")
    assert out[0].published == "2026-02-02" and out[0].snippet == "D"


@respx.mock
def test_jina_requires_key_and_parses_data():
    with pytest.raises(RuntimeError):
        Jina(api_key="").search("q")
    respx.get("https://s.jina.ai/").mock(return_value=httpx.Response(200, json={"data": [
        {"title": "T", "url": "https://a.ru", "description": "D", "content": "full text", "publishedTime": "2026"}]}))
    out = Jina(api_key="k").search("q")
    assert out[0].content == "full text" and out[0].snippet == "D"
    assert respx.calls.last.request.headers["Authorization"] == "Bearer k"


def test_ddg_uses_ddgs_package(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, region=None, max_results=None):
            assert region == "ru-ru" and max_results == 2
            return [{"title": "T", "href": "https://a.ru", "body": "B"}]

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    out = DuckDuckGo().search("q", limit=2)
    assert out[0].url == "https://a.ru" and out[0].snippet == "B"
