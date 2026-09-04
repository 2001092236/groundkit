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
import inspect
import logging
import os
import struct
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


def image_size(data: bytes) -> tuple[int, int] | None:
    """Реальные размеры картинки из её байтов: JPEG, PNG, GIF, WEBP (VP8X/VP8L/VP8).

    Нужно потому, что провайдер не обязан выдать запрошенный размер: Cloudflare FLUX,
    например, всегда отдаёт 1024×1024, что бы у него ни просили.
    """
    if data[:2] == b"\xff\xd8":  # JPEG: ищем маркер SOF
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                i += 2
                continue
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", data[i + 5:i + 9])
                return width, height
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data[:3] == b"GIF" and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return width, height
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8X":
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if chunk == b"VP8 ":
            width, height = struct.unpack("<HH", data[26:30])
            return width & 0x3FFF, height & 0x3FFF
    return None


def _send(send, *, retries: int, delay: float, label: str) -> httpx.Response:
    """Выполняет запрос с повторами на обрывах связи.

    Провайдеры картинок держат соединение десятки секунд, и обрыв TLS на середине —
    обычное дело. Повторяем только сетевые сбои: ответ с кодом ошибки повторять незачем.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return send()
        except httpx.TransportError as exc:
            last = exc
            if attempt < retries:
                log.warning("%s: попытка %d не удалась (%s), повторяю", label, attempt + 1, exc)
                time.sleep(delay * (attempt + 1))
        except httpx.HTTPError as exc:
            last = exc
            break
    raise ImageError(f"{label}: {type(last).__name__}: {last}") from last


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
    retries: int = 2
    retry_delay: float = 2.0
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
            resp = _send(
                lambda: httpx.get(url, params=params, headers=self._headers(), timeout=self.timeout,
                                  follow_redirects=True),
                retries=self.retries, delay=self.retry_delay, label="Pollinations",
            )
        except ImageError as err:
            ledger.record(self.name, ok=False, latency_s=time.monotonic() - started, error=str(err))
            raise
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
        actual = image_size(resp.content) or (width, height)
        return ImageResult(
            data=resp.content, content_type=resp.headers["content-type"], provider="pollinations",
            model=self.model, prompt=prompt, width=actual[0], height=actual[1], seed=seed, latency_s=latency,
        )


@dataclass
class CloudflareImages:
    """Workers AI: FLUX.1 schnell. Расходует общий дневной лимит нейронов."""

    model: str = "@cf/black-forest-labs/flux-1-schnell"
    api_key: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_API_KEY", ""))
    account_id: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    steps: int = 4
    # FLUX.1 schnell отвергает запрос целиком, если прислать ему seed или размеры.
    supports_seed: bool = False
    timeout: float = 120.0
    retries: int = 2
    retry_delay: float = 2.0

    @property
    def name(self) -> str:
        return f"cloudflare/{self.model}"

    def generate(
        self, prompt: str, *, size: tuple[int, int] = DEFAULT_SIZE, seed: int | None = None
    ) -> ImageResult:
        if not (self.api_key and self.account_id):
            raise ImageNotConfigured("Нужны CLOUDFLARE_API_KEY и CLOUDFLARE_ACCOUNT_ID")
        body: dict = {"prompt": prompt, "steps": self.steps}
        if seed is not None and self.supports_seed:
            body["seed"] = seed
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        ledger = get_ledger()
        started = time.monotonic()
        try:
            resp = _send(
                lambda: httpx.post(url, json=body, headers={"Authorization": f"Bearer {self.api_key}"},
                                   timeout=self.timeout),
                retries=self.retries, delay=self.retry_delay, label="Cloudflare",
            )
        except ImageError as err:
            ledger.record(self.name, ok=False, latency_s=time.monotonic() - started, error=str(err))
            raise
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
        data = base64.b64decode(encoded)
        actual = image_size(data) or size   # FLUX сам решает размер и запрошенный игнорирует
        return ImageResult(
            data=data, content_type="image/jpeg", provider="cloudflare",
            model=self.model, prompt=prompt, width=actual[0], height=actual[1], seed=seed, latency_s=latency,
        )


@dataclass
class HuggingFaceSpace:
    """Бесплатный GPU в публичных пространствах Hugging Face (ZeroGPU).

    Считается не деньгами, а GPU-секундами: без токена квота на практике нулевая,
    бесплатный аккаунт даёт ~5 минут в сутки, чего хватает на несколько десятков
    картинок FLUX.1-schnell. Токен кладётся в ``HF_TOKEN``.

    Требует пакет ``gradio_client``: ``pip install "groundkit[spaces]"``.
    Схема вызова у каждого пространства своя и меняется вместе с его кодом,
    поэтому список запасных пространств задан явно.
    """

    space: str = "black-forest-labs/FLUX.1-schnell"
    token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    api_name: str = "/infer"
    steps: int = 4
    timeout: float = 180.0

    @property
    def name(self) -> str:
        return f"hf-space/{self.space}"

    def generate(
        self, prompt: str, *, size: tuple[int, int] = DEFAULT_SIZE, seed: int | None = None
    ) -> ImageResult:
        try:
            from gradio_client import Client
        except ImportError as exc:  # pragma: no cover
            raise ImageNotConfigured('Установите `pip install "groundkit[spaces]"`') from exc

        ledger = get_ledger()
        started = time.monotonic()
        try:
            # В gradio_client 1.x параметр назывался hf_token, в 2.x — token.
            kwargs = {"token" if "token" in inspect.signature(Client.__init__).parameters else "hf_token":
                      self.token or None}
            client = Client(self.space, verbose=False, **kwargs)
            out = client.predict(
                prompt=prompt, seed=seed or 0, randomize_seed=seed is None,
                width=size[0], height=size[1], num_inference_steps=self.steps, api_name=self.api_name,
            )
        except Exception as exc:  # noqa: BLE001 — gradio бросает свои типы ошибок
            text = str(exc)
            if "quota" in text.lower():
                err: ImageError = ImageRateLimited(f"ZeroGPU: квота исчерпана. {text[:200]}")
            elif "401" in text or "authenticate" in text.lower():
                err = ImageNotConfigured(f"ZeroGPU: {text[:200]}")
            else:
                err = ImageError(f"ZeroGPU {self.space}: {type(exc).__name__}: {text[:200]}")
            ledger.record(self.name, ok=False, latency_s=time.monotonic() - started, error=str(err),
                          rate_limited=isinstance(err, ImageRateLimited))
            raise err from exc

        latency = time.monotonic() - started
        path = out[0] if isinstance(out, (tuple, list)) else out
        try:
            data = Path(str(path)).read_bytes()
        except OSError as exc:
            err = ImageError(f"ZeroGPU {self.space}: не удалось прочитать файл {path}")
            ledger.record(self.name, ok=False, latency_s=latency, error=str(err))
            raise err from exc
        ledger.record(self.name, ok=True, latency_s=latency)
        actual = image_size(data) or size
        ctype = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return ImageResult(data=data, content_type=ctype, provider="hf-space", model=self.space,
                           prompt=prompt, width=actual[0], height=actual[1], seed=seed, latency_s=latency)


IMAGE_PROVIDERS: dict[str, type] = {
    "pollinations": Pollinations,
    "cloudflare": CloudflareImages,
    "hf-space": HuggingFaceSpace,
}

IMAGE_PROVIDER_INFO: dict[str, dict] = {
    "pollinations": {"label": "Pollinations (sana)", "env": None, "needs_key": False,
                     "free": "без ключа: 1 запрос в 15 с; с бесплатным токеном — в 5 с",
                     "docs": "https://auth.pollinations.ai"},
    "cloudflare": {"label": "Cloudflare · FLUX.1 schnell", "env": "CLOUDFLARE_API_KEY", "needs_key": True,
                   "free": "из общих 10 000 нейронов в день",
                   "docs": "https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/"},
    "hf-space": {"label": "Hugging Face ZeroGPU · FLUX.1 schnell", "env": "HF_TOKEN", "needs_key": True,
                 "free": "GPU-секунды: без токена на практике 0, с бесплатным аккаунтом ~5 мин/сутки",
                 "docs": "https://huggingface.co/docs/hub/spaces-zerogpu"},
}


def image_provider_configured(name: str) -> bool:
    info = IMAGE_PROVIDER_INFO.get(name)
    if info is None:
        return False
    if not info["needs_key"]:
        return True
    if name == "hf-space":
        return bool(os.getenv("HF_TOKEN"))
    return bool(os.getenv("CLOUDFLARE_API_KEY") and os.getenv("CLOUDFLARE_ACCOUNT_ID"))


def build_image_provider(spec: str | ImageProvider) -> ImageProvider:
    if not isinstance(spec, str):
        return spec
    key, _, model = spec.partition("/")
    if key not in IMAGE_PROVIDERS:
        raise ValueError(f"Неизвестный провайдер картинок: {key}. Доступны: {list(IMAGE_PROVIDERS)}")
    if key == "hf-space":                       # у пространств id из двух частей: org/space
        return HuggingFaceSpace(space=model) if model else HuggingFaceSpace()
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
    specs = providers or [
        "pollinations",
        *(["cloudflare"] if image_provider_configured("cloudflare") else []),
        *(["hf-space"] if image_provider_configured("hf-space") else []),
    ]
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
