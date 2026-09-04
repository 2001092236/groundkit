"""Генерация картинок бесплатными провайдерами за одним интерфейсом.

Тот же принцип, что и у моделей текста: провайдеры пробуются по очереди, каждый вызов
попадает в журнал лимитов, ключи не обязательны.

    from groundkit import generate_image

    img = generate_image("уютная библиотека с юридическими книгами", size=(768, 512))
    img.save("library.jpg")

Провайдеры:
  * ``pollinations`` — без ключа. Токен (``POLLINATIONS_TOKEN``) поднимает тариф:
    анонимно один запрос в 15 секунд, с бесплатной регистрацией — в 5 секунд.
  * ``cloudflare`` — Workers AI, модель FLUX.1 schnell. Нужны ``CLOUDFLARE_API_KEY``
    и ``CLOUDFLARE_ACCOUNT_ID``; расходует те же 10 000 нейронов в день.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import httpx

from .usage import get_ledger

log = logging.getLogger("groundkit.images")

DEFAULT_SIZE = (1024, 1024)
_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


class ImageError(RuntimeError):
    """Картинку сгенерировать не удалось."""


class ImageRateLimited(ImageError):
    """Провайдер попросил подождать — стоит взять следующего."""


class ImageNotConfigured(ImageError):
    """Нет ключа или доступа."""


@dataclass
class ImageResult:
    """Готовая картинка в памяти."""

    data: bytes
    content_type: str
    provider: str
    model: str
    prompt: str
    width: int
    height: int
    seed: int | None = None
    latency_s: float = 0.0

    @property
    def extension(self) -> str:
        return _EXT.get(self.content_type.split(";")[0].strip(), ".bin")

    def save(self, path: str | Path) -> Path:
        """Сохраняет файл. Если у пути нет расширения — подставляет по типу картинки."""
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(self.extension)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.data)
        return target

    def to_data_uri(self) -> str:
        return f"data:{self.content_type};base64,{base64.b64encode(self.data).decode()}"

    def to_dict(self) -> dict:
        return {
            "provider": self.provider, "model": self.model, "prompt": self.prompt,
            "width": self.width, "height": self.height, "seed": self.seed,
            "content_type": self.content_type, "bytes": len(self.data),
            "latency_s": round(self.latency_s, 2),
        }


class ImageProvider(Protocol):
    name: str

    def generate(
        self, prompt: str, *, size: tuple[int, int] = DEFAULT_SIZE, seed: int | None = None
    ) -> ImageResult: ...


@dataclass
class Pollinations:
    """image.pollinations.ai — бесплатно и без ключа.

    Токен не обязателен: без него один запрос раз в 15 секунд, с бесплатным токеном
    (auth.pollinations.ai) — раз в 5 секунд. ``nologo`` и ``private`` работают только
    для зарегистрированных.
    """

    model: str = "sana"
    token: str = field(default_factory=lambda: os.getenv("POLLINATIONS_TOKEN", ""))
    referrer: str = field(default_factory=lambda: os.getenv("POLLINATIONS_REFERRER", "groundkit"))
    nologo: bool = True
    private: bool = True
    enhance: bool = False
    timeout: float = 180.0
    base_url: str = "https://image.pollinations.ai"

    @property
    def name(self) -> str:
        return f"pollinations/{self.model}"

    def models(self) -> list[str]:
        """Какие модели доступны этому токену."""
        resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [m if isinstance(m, str) else m.get("name", "") for m in data]

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def generate(
        self, prompt: str, *, size: tuple[int, int] = DEFAULT_SIZE, seed: int | None = None
    ) -> ImageResult:
        width, height = size
        params: dict[str, str] = {
            "width": str(width), "height": str(height), "model": self.model,
            "nologo": str(self.nologo).lower(), "private": str(self.private).lower(),
            "referrer": self.referrer,
        }
        if self.enhance:
            params["enhance"] = "true"
        if seed is not None:
            params["seed"] = str(seed)
        url = f"{self.base_url}/prompt/{quote(prompt, safe='')}"
        ledger = get_ledger()
        started = time.monotonic()
        try:
            resp = httpx.get(url, params=params, headers=self._headers(), timeout=self.timeout,
                             follow_redirects=True)
        except httpx.HTTPError as exc:
            err = ImageError(f"Pollinations: {type(exc).__name__}: {exc}")
            ledger.record(self.name, ok=False, latency_s=time.monotonic() - started, error=str(err))
            raise err from exc
        latency = time.monotonic() - started
        if resp.status_code >= 400 or not resp.headers.get("content-type", "").startswith("image/"):
            detail = resp.text[:200]
            if resp.status_code == 429:
                err = ImageRateLimited(f"Pollinations 429: {detail}")
            elif resp.status_code in (401, 402, 403):
                err = ImageNotConfigured(f"Pollinations {resp.status_code}: {detail}")
            else:
                err = ImageError(f"Pollinations {resp.status_code}: {detail}")
            ledger.record(self.name, ok=False, latency_s=latency, error=str(err),
                          rate_limited=resp.status_code == 429, headers=dict(resp.headers))
            raise err
        ledger.record(self.name, ok=True, latency_s=latency, headers=dict(resp.headers))
        return ImageResult(
            data=resp.content, content_type=resp.headers["content-type"], provider="pollinations",
            model=self.model, prompt=prompt, width=width, height=height, seed=seed, latency_s=latency,
        )


@dataclass
class CloudflareImages:
    """Workers AI: FLUX.1 schnell. Расходует общий дневной лимит нейронов."""

    model: str = "@cf/black-forest-labs/flux-1-schnell"
    api_key: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_API_KEY", ""))
    account_id: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    steps: int = 4
    timeout: float = 120.0

    @property
    def name(self) -> str:
        return f"cloudflare/{self.model}"

    def generate(
        self, prompt: str, *, size: tuple[int, int] = DEFAULT_SIZE, seed: int | None = None
    ) -> ImageResult:
        if not (self.api_key and self.account_id):
            raise ImageNotConfigured("Нужны CLOUDFLARE_API_KEY и CLOUDFLARE_ACCOUNT_ID")
        body: dict = {"prompt": prompt, "steps": self.steps}
        if seed is not None:
            body["seed"] = seed
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        ledger = get_ledger()
        started = time.monotonic()
        try:
            resp = httpx.post(url, json=body, headers={"Authorization": f"Bearer {self.api_key}"},
                              timeout=self.timeout)
        except httpx.HTTPError as exc:
            err = ImageError(f"Cloudflare: {type(exc).__name__}: {exc}")
            ledger.record(self.name, ok=False, latency_s=time.monotonic() - started, error=str(err))
            raise err from exc
        latency = time.monotonic() - started
        if resp.status_code >= 400:
            detail = resp.text[:200]
            if resp.status_code == 429:
                err = ImageRateLimited(f"Cloudflare 429: {detail}")
            elif resp.status_code in (401, 402, 403):
                err = ImageNotConfigured(f"Cloudflare {resp.status_code}: {detail}")
            else:
                err = ImageError(f"Cloudflare {resp.status_code}: {detail}")
            ledger.record(self.name, ok=False, latency_s=latency, error=str(err),
                          rate_limited=resp.status_code == 429, headers=dict(resp.headers))
            raise err
        payload = (resp.json() or {}).get("result") or {}
        encoded = payload.get("image")
        if not encoded:
            err = ImageError(f"Cloudflare вернул ответ без картинки: {resp.text[:200]}")
            ledger.record(self.name, ok=False, latency_s=latency, error=str(err))
            raise err
        ledger.record(self.name, ok=True, latency_s=latency, headers=dict(resp.headers))
        return ImageResult(
            data=base64.b64decode(encoded), content_type="image/jpeg", provider="cloudflare",
            model=self.model, prompt=prompt, width=size[0], height=size[1], seed=seed, latency_s=latency,
        )


IMAGE_PROVIDERS: dict[str, type] = {"pollinations": Pollinations, "cloudflare": CloudflareImages}

IMAGE_PROVIDER_INFO: dict[str, dict] = {
    "pollinations": {"label": "Pollinations (sana)", "env": None, "needs_key": False,
                     "free": "без ключа: 1 запрос в 15 с; с бесплатным токеном — в 5 с",
                     "docs": "https://auth.pollinations.ai"},
    "cloudflare": {"label": "Cloudflare · FLUX.1 schnell", "env": "CLOUDFLARE_API_KEY", "needs_key": True,
                   "free": "из общих 10 000 нейронов в день",
                   "docs": "https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/"},
}


def image_provider_configured(name: str) -> bool:
    info = IMAGE_PROVIDER_INFO.get(name)
    if info is None:
        return False
    if not info["needs_key"]:
        return True
    return bool(os.getenv("CLOUDFLARE_API_KEY") and os.getenv("CLOUDFLARE_ACCOUNT_ID"))


def build_image_provider(spec: str | ImageProvider) -> ImageProvider:
    if not isinstance(spec, str):
        return spec
    key, _, model = spec.partition("/")
    if key not in IMAGE_PROVIDERS:
        raise ValueError(f"Неизвестный провайдер картинок: {key}. Доступны: {list(IMAGE_PROVIDERS)}")
    return IMAGE_PROVIDERS[key](model=model) if model else IMAGE_PROVIDERS[key]()


def generate_image(
    prompt: str,
    providers: list[str | ImageProvider] | None = None,
    *,
    size: tuple[int, int] = DEFAULT_SIZE,
    seed: int | None = None,
) -> ImageResult:
    """Генерирует картинку, перебирая провайдеров по очереди.

    По умолчанию: Pollinations (без ключа), затем Cloudflare, если ключ есть.
    """
    specs = providers or ["pollinations", *(["cloudflare"] if image_provider_configured("cloudflare") else [])]
    errors: list[str] = []
    for spec in specs:
        try:
            provider = build_image_provider(spec)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        try:
            return provider.generate(prompt, size=size, seed=seed)
        except ImageError as exc:
            errors.append(f"{provider.name} — {exc}")
            log.warning("Провайдер картинок %s не сработал: %s", provider.name, exc)
    raise ImageError("Не удалось сгенерировать картинку: " + "; ".join(errors) if errors
                     else "Не задано ни одного провайдера картинок")
