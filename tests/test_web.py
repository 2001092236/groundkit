import pytest
from fastapi.testclient import TestClient

from groundkit.answer import Answer
from groundkit.llm import LLMResponse
from groundkit.search import SearchResult, SearchRun
from groundkit.web import app as webapp


@pytest.fixture
def client():
    webapp.limiter._minute.clear()
    webapp.limiter._day.clear()
    return TestClient(webapp.app)


def test_index_and_health(client):
    assert client.get("/api/health").json()["ok"] is True
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_config_lists_providers_and_presets(client, monkeypatch):
    cfg = client.get("/api/config").json()
    names = {s["name"] for s in cfg["search"]}
    assert {"searxng", "ddg", "jina", "exa", "brave"} <= names
    assert "Право РФ" in cfg["presets"] and cfg["limits"]["rpm"] > 0
    assert all(not m["model"].startswith("claude-cli") for m in cfg["models"])  # выключен по умолчанию
    monkeypatch.setenv("GROQ_API_KEY", "k")
    cfg = client.get("/api/config").json()
    assert cfg["default_chain"] == ["groq/llama-3.3-70b-versatile"]


def test_ask_without_models_returns_sources_and_503(client, monkeypatch):
    run = SearchRun([SearchResult("t", "https://a.ru", "s", index=1)], "ddg")
    monkeypatch.setattr(webapp, "run_search", lambda *a, **k: run)
    monkeypatch.setattr(webapp.Answerer, "search", lambda self, q: run)
    r = client.post("/api/ask", json={"query": "вопрос"})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "no_model" and body["sources"][0]["url"] == "https://a.ru"


def test_ask_happy_path(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    answer = Answer(answer="Ответ [1]", sources=[SearchResult("t", "https://a.ru", "s", index=1)], search_provider="ddg",
                    model="groq/llama-3.3-70b-versatile", cited=[1],
                    llm=LLMResponse("Ответ [1]", "groq/llama-3.3-70b-versatile", "litellm"))
    monkeypatch.setattr(webapp.Answerer, "ask", lambda self, q: answer)
    r = client.post("/api/ask", json={"query": "вопрос", "domains": ["a.ru"], "model": ["groq/llama-3.3-70b-versatile"]})
    assert r.status_code == 200 and r.json()["answer"] == "Ответ [1]" and r.json()["cited"] == [1]


def test_search_endpoint_and_validation(client, monkeypatch):
    monkeypatch.setattr(webapp, "run_search", lambda q, p, d, lim: SearchRun([SearchResult("t", "https://a.ru", "s", index=1)], p[0]))
    r = client.post("/api/search", json={"query": "вопрос", "search": ["ddg"]})
    assert r.status_code == 200 and r.json()["provider"] == "ddg"
    assert client.post("/api/search", json={"query": "x"}).status_code == 422
    assert client.post("/api/ask", json={"query": "q" * 501}).status_code == 422


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(webapp.limiter, "rpm", 2)
    monkeypatch.setattr(webapp, "run_search", lambda *a, **k: SearchRun([], ""))
    for _ in range(2):
        assert client.post("/api/search", json={"query": "вопрос"}).status_code == 200
    assert client.post("/api/search", json={"query": "вопрос"}).status_code == 429


def test_access_token(client, monkeypatch):
    monkeypatch.setattr(webapp, "ACCESS_TOKEN", "secret")
    monkeypatch.setattr(webapp, "run_search", lambda *a, **k: SearchRun([], ""))
    assert client.post("/api/search", json={"query": "вопрос"}).status_code == 401
    assert client.post("/api/search", json={"query": "вопрос"}, headers={"X-Access-Token": "secret"}).status_code == 200
