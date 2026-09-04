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
    assert cfg["default_chain"] == ["groq/qwen/qwen3.8-27b", "groq/openai/gpt-oss-120b"]


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
                    model="groq/qwen/qwen3.8-27b", cited=[1],
                    llm=LLMResponse("Ответ [1]", "groq/qwen/qwen3.8-27b", "litellm"))
    monkeypatch.setattr(webapp.Answerer, "ask", lambda self, q: answer)
    r = client.post("/api/ask", json={"query": "вопрос", "domains": ["a.ru"], "model": ["groq/qwen/qwen3.8-27b"]})
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


def test_usage_endpoint(client, monkeypatch):
    monkeypatch.setattr(webapp, "_openrouter_live", lambda: None)
    from groundkit.usage import get_ledger

    get_ledger().record("groq/qwen/qwen3.8-27b", ok=True, input_tokens=5, output_tokens=1)
    body = client.get("/api/usage").json()
    row = next(r for r in body["models"] if r["model"] == "groq/qwen/qwen3.8-27b")
    assert row["used_today"] == 1 and row["rpd"] == 1000 and row["remaining"] == 999
    assert "days" in body and body["openrouter"] is None


def test_image_endpoint_returns_data_uri(client, monkeypatch):
    from groundkit.images import ImageResult

    img = ImageResult(data=b"\xff\xd8bytes", content_type="image/jpeg", provider="pollinations",
                      model="sana", prompt="кот", width=768, height=512, seed=3, latency_s=1.2)
    seen = {}

    def fake(prompt, providers, *, size, seed):
        seen.update(prompt=prompt, providers=providers, size=size, seed=seed)
        return img

    monkeypatch.setattr(webapp, "generate_image", fake)
    r = client.post("/api/image", json={"prompt": "кот", "width": 768, "height": 512, "seed": 3,
                                        "provider": ["pollinations", "выдуманный"]})
    assert r.status_code == 200
    body = r.json()
    assert body["data_uri"].startswith("data:image/jpeg;base64,") and body["width"] == 768
    assert seen["providers"] == ["pollinations"] and seen["size"] == (768, 512) and seen["seed"] == 3


def test_image_endpoint_reports_failures(client, monkeypatch):
    from groundkit.images import ImageError, ImageRateLimited

    monkeypatch.setattr(webapp, "generate_image",
                        lambda *a, **k: (_ for _ in ()).throw(ImageRateLimited("подожди")))
    assert client.post("/api/image", json={"prompt": "кот"}).status_code == 429
    monkeypatch.setattr(webapp, "generate_image",
                        lambda *a, **k: (_ for _ in ()).throw(ImageError("всё сломалось")))
    r = client.post("/api/image", json={"prompt": "кот"})
    assert r.status_code == 502 and r.json()["error"] == "image_failed"


def test_image_endpoint_validates_and_can_be_disabled(client, monkeypatch):
    assert client.post("/api/image", json={"prompt": "к"}).status_code == 422
    assert client.post("/api/image", json={"prompt": "кот", "width": 4000}).status_code == 422
    monkeypatch.setattr(webapp, "IMAGES_ENABLED", False)
    r = client.post("/api/image", json={"prompt": "кот"})
    assert r.status_code == 503 and r.json()["error"] == "images_disabled"


def test_config_lists_image_providers(client):
    cfg = client.get("/api/config").json()
    assert cfg["images_enabled"] is True
    names = {p["name"]: p for p in cfg["image_providers"]}
    assert names["pollinations"]["configured"] is True      # работает без ключа
    assert names["cloudflare"]["configured"] is False       # ключей в тестах нет
