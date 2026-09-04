import base64

import httpx
import pytest
import respx

from groundkit.images import (
    CloudflareImages,
    ImageError,
    ImageNotConfigured,
    ImageRateLimited,
    ImageResult,
    Pollinations,
    build_image_provider,
    generate_image,
    image_provider_configured,
)
from groundkit.usage import get_ledger

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 200
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/красное яблоко"


@respx.mock
def test_pollinations_builds_request_and_returns_image():
    route = respx.get(url__startswith="https://image.pollinations.ai/prompt/").mock(
        return_value=httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"}))
    out = Pollinations(token="tok").generate("красное яблоко", size=(768, 512), seed=42)
    assert out.data == JPEG and out.content_type == "image/jpeg" and out.provider == "pollinations"
    assert out.width == 768 and out.height == 512 and out.seed == 42 and out.model == "sana"
    params = route.calls[0].request.url.params
    assert params["width"] == "768" and params["height"] == "512" and params["seed"] == "42"
    assert params["nologo"] == "true" and params["private"] == "true" and params["model"] == "sana"
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"
    assert "%D0%BA%D1%80%D0%B0%D1%81%D0%BD%D0%BE%D0%B5" in str(route.calls[0].request.url)


@respx.mock
def test_pollinations_without_token_sends_no_auth_header():
    route = respx.get(url__startswith="https://image.pollinations.ai/prompt/").mock(
        return_value=httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"}))
    Pollinations(token="").generate("apple")
    assert "authorization" not in route.calls[0].request.headers


@respx.mock
def test_pollinations_errors_are_classified():
    route = respx.get(url__startswith="https://image.pollinations.ai/prompt/")
    route.mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(ImageRateLimited):
        Pollinations().generate("x")
    route.mock(return_value=httpx.Response(402, text="Payment Required"))
    with pytest.raises(ImageNotConfigured):
        Pollinations().generate("x")
    # 200, но не картинка — тоже ошибка, а не «успех» с HTML внутри
    route.mock(return_value=httpx.Response(200, text="<html>oops</html>",
                                           headers={"content-type": "text/html"}))
    with pytest.raises(ImageError):
        Pollinations().generate("x")


@respx.mock
def test_cloudflare_decodes_base64(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "k")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    route = respx.post(url__startswith="https://api.cloudflare.com/client/v4/accounts/acc/ai/run/").mock(
        return_value=httpx.Response(200, json={"result": {"image": base64.b64encode(JPEG).decode()}}))
    out = CloudflareImages().generate("apple", size=(512, 512), seed=7)
    assert out.data == JPEG and out.provider == "cloudflare"
    assert route.calls[0].request.headers["authorization"] == "Bearer k"


def test_cloudflare_needs_keys():
    with pytest.raises(ImageNotConfigured):
        CloudflareImages(api_key="", account_id="").generate("x")


@respx.mock
def test_cloudflare_result_without_image_is_an_error(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "k")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    respx.post(url__startswith="https://api.cloudflare.com/").mock(
        return_value=httpx.Response(200, json={"result": {}}))
    with pytest.raises(ImageError):
        CloudflareImages().generate("x")


class FakeProvider:
    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.calls = 0

    def generate(self, prompt, *, size=(1024, 1024), seed=None):
        self.calls += 1
        if self.error:
            raise self.error
        return ImageResult(data=JPEG, content_type="image/jpeg", provider=self.name, model="m",
                           prompt=prompt, width=size[0], height=size[1], seed=seed)


def test_generate_image_falls_back():
    bad = FakeProvider("bad", error=ImageRateLimited("429"))
    good = FakeProvider("good")
    out = generate_image("apple", [bad, good])
    assert out.provider == "good" and bad.calls == 1 and good.calls == 1


def test_generate_image_reports_all_failures():
    with pytest.raises(ImageError, match="a — .*b — "):
        generate_image("x", [FakeProvider("a", error=ImageError("boom")),
                             FakeProvider("b", error=ImageNotConfigured("no key"))])


def test_generate_image_default_chain_without_keys(monkeypatch):
    """Без ключей остаётся один Pollinations — он не требует регистрации."""
    calls: list[str] = []
    monkeypatch.setattr("groundkit.images.build_image_provider",
                        lambda spec: calls.append(spec) or FakeProvider(str(spec)))
    generate_image("x")
    assert calls == ["pollinations"]


def test_build_image_provider_and_configured(monkeypatch):
    assert isinstance(build_image_provider("pollinations"), Pollinations)
    assert build_image_provider("cloudflare/@cf/other/model").model == "@cf/other/model"
    with pytest.raises(ValueError):
        build_image_provider("midjourney")
    assert image_provider_configured("pollinations")
    assert not image_provider_configured("cloudflare") and not image_provider_configured("nope")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "k")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    assert image_provider_configured("cloudflare")


def test_image_result_save_and_data_uri(tmp_path):
    img = ImageResult(data=JPEG, content_type="image/jpeg", provider="p", model="m", prompt="q",
                      width=8, height=8)
    path = img.save(tmp_path / "sub" / "pic")          # расширение подставляется само
    assert path.name == "pic.jpg" and path.read_bytes() == JPEG
    assert img.save(tmp_path / "named.png").name == "named.png"
    assert img.to_data_uri().startswith("data:image/jpeg;base64,")
    assert img.to_dict()["bytes"] == len(JPEG)


@respx.mock
def test_image_calls_land_in_usage_ledger():
    respx.get(url__startswith="https://image.pollinations.ai/prompt/").mock(
        return_value=httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"}))
    Pollinations().generate("apple")
    row = get_ledger().summary([{"model": "pollinations/sana", "label": "Pollinations"}])[0]
    assert row["used_today"] == 1 and row["ok_today"] == 1
