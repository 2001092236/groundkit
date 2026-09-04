"""Учёт лимитов: сколько запросов ушло сегодня по каждой модели, что провайдер сообщил в
заголовках о остатке, когда сброс. Простой JSON-файл, без внешних сервисов.

Файл: ``GROUNDKIT_USAGE_FILE`` (по умолчанию ``~/.groundkit/usage.json``).
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

KEEP_DAYS = 14
DEFAULT_COOLDOWN_S = 600  # если провайдер ответил 429, но не сказал когда можно снова

_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+)ms)?$")


def default_path() -> Path:
    return Path(os.getenv("GROUNDKIT_USAGE_FILE") or Path.home() / ".groundkit" / "usage.json")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_reset(value: str, now: datetime) -> str | None:
    """Заголовок сброса бывает длительностью («2m59.56s», «7.66s», «30»), epoch в секундах или мс."""
    v = str(value).strip()
    if not v:
        return None
    if v.replace(".", "", 1).isdigit():
        num = float(v)
        if num > 1e12:  # epoch в миллисекундах
            return datetime.fromtimestamp(num / 1000, UTC).isoformat()
        if num > 1e9:  # epoch в секундах
            return datetime.fromtimestamp(num, UTC).isoformat()
        return (now + timedelta(seconds=num)).isoformat()
    m = _DURATION_RE.match(v)
    if m and any(m.groups()):
        h, mi, s, ms = (float(x) if x else 0.0 for x in m.groups())
        return (now + timedelta(hours=h, minutes=mi, seconds=s, milliseconds=ms)).isoformat()
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(UTC).isoformat()
    except ValueError:
        return None


def parse_ratelimit_headers(headers: dict | None, now: datetime | None = None) -> dict:
    """Вытаскивает из заголовков ответа остаток/лимит/сброс по запросам и токенам.

    Понимает Groq и Cerebras (x-ratelimit-*-requests[-day]), OpenRouter (x-ratelimit-limit/remaining/reset),
    Mistral (x-ratelimitbysize-*) и retry-after. Неизвестное складывается в ``raw``.
    """
    now = now or _utc_now()
    out: dict = {"raw": {}}
    if not headers:
        return out
    for key, value in headers.items():
        k = key.lower().removeprefix("llm_provider-")
        if "ratelimit" not in k and k != "retry-after":
            continue
        out["raw"][k] = str(value)
        if k == "retry-after":
            out["retry_at"] = _parse_reset(str(value), now)
            continue
        kind = "tokens" if "token" in k else "requests"
        window = "day" if "day" in k else ("minute" if ("minute" in k or "bysize" in k) else "")
        suffix = f"_{window}" if window else ""
        rest = k.replace("ratelimitbysize", "").replace("ratelimit", "")  # иначе «limit» находится в «ratelimit»
        if "remaining" in rest:
            out[f"remaining_{kind}{suffix}"] = _to_int(value)
        elif "reset" in rest:
            out[f"reset_{kind}{suffix}_at"] = _parse_reset(str(value), now)
        elif "limit" in rest:
            out[f"limit_{kind}{suffix}"] = _to_int(value)
    return out


def _to_int(value) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


class UsageLedger:
    """Потокобезопасный журнал вызовов моделей в JSON-файле."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "days" in data:
                data.setdefault("providers", {})
                return data
        except (OSError, ValueError):
            pass
        return {"days": {}, "providers": {}}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # учёт не должен ронять основной вызов

    def record(
        self,
        model: str,
        *,
        ok: bool,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_s: float = 0.0,
        rate_limited: bool = False,
        headers: dict | None = None,
        error: str | None = None,
    ) -> None:
        now = _utc_now()
        day = now.date().isoformat()
        with self._lock:
            rec = self._data["days"].setdefault(day, {}).setdefault(model, {
                "requests": 0, "ok": 0, "errors": 0, "rate_limited": 0,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "seconds": 0.0,
            })
            rec["requests"] += 1
            rec["ok" if ok else "errors"] += 1
            rec["rate_limited"] += int(rate_limited)
            rec["input_tokens"] += int(input_tokens or 0)
            rec["output_tokens"] += int(output_tokens or 0)
            rec["cost_usd"] = round(rec["cost_usd"] + float(cost_usd or 0.0), 6)
            rec["seconds"] = round(rec["seconds"] + float(latency_s or 0.0), 2)
            if error:
                rec["last_error"] = error[:200]
                rec["last_error_at"] = now.isoformat()

            prov = self._data["providers"].setdefault(model, {})
            parsed = parse_ratelimit_headers(headers, now)
            if parsed.get("raw"):
                prov.update({k: v for k, v in parsed.items() if k != "raw"})
                prov["raw"] = parsed["raw"]
                prov["observed_at"] = now.isoformat()
            if rate_limited:
                # retry-after — авторитетно; иначе ближайший из сбросов (429 обычно про самый короткий счётчик)
                resets = [v for k, v in parsed.items()
                          if k.startswith("reset_") and v and datetime.fromisoformat(v) > now]
                until = parsed.get("retry_at") or (min(resets) if resets else None)
                if not until or datetime.fromisoformat(until) <= now:
                    until = (now + timedelta(seconds=DEFAULT_COOLDOWN_S)).isoformat()
                prov["blocked_until"] = until
            elif ok:
                prov.pop("blocked_until", None)

            cutoff = (now - timedelta(days=KEEP_DAYS)).date().isoformat()
            for old in [d for d in self._data["days"] if d < cutoff]:
                del self._data["days"][old]
            self._save()

    def blocked_until(self, model: str) -> datetime | None:
        until = self._data["providers"].get(model, {}).get("blocked_until")
        if not until:
            return None
        dt = datetime.fromisoformat(until)
        return dt if dt > _utc_now() else None

    def today(self, model: str) -> dict:
        return dict(self._data["days"].get(_utc_now().date().isoformat(), {}).get(model, {}))

    def summary(self, known: list[dict]) -> list[dict]:
        """Сводка по известным моделям + всем, что встречались в журнале."""
        now = _utc_now()
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seen = {m["model"]: m for m in known}
        for day in self._data["days"].values():
            for model in day:
                seen.setdefault(model, {"model": model, "label": model})
        rows = []
        for model, meta in seen.items():
            today = self.today(model)
            prov = self._data["providers"].get(model, {})
            used = today.get("requests", 0)
            rpd = meta.get("rpd")
            remaining_by_provider = prov.get("remaining_requests_day", prov.get("remaining_requests"))
            if remaining_by_provider is not None:
                remaining = remaining_by_provider
            else:
                remaining = max(rpd - used, 0) if rpd else None
            blocked = self.blocked_until(model)
            rows.append({
                "model": model,
                "label": meta.get("label", model),
                "configured": meta.get("configured"),
                "used_today": used,
                "ok_today": today.get("ok", 0),
                "errors_today": today.get("errors", 0),
                "rate_limited_today": today.get("rate_limited", 0),
                "tokens_today": today.get("input_tokens", 0) + today.get("output_tokens", 0),
                "cost_today_usd": today.get("cost_usd", 0.0),
                "rpd": rpd,
                "rpm": meta.get("rpm"),
                "free_note": meta.get("free"),
                "reset_note": meta.get("reset"),
                "docs": meta.get("docs"),
                "remaining": remaining,
                "remaining_source": "provider" if remaining_by_provider is not None else ("estimate" if rpd else None),
                "provider": {k: v for k, v in prov.items() if k != "raw"},
                "resets_at": prov.get("reset_requests_day_at") or prov.get("reset_requests_at") or midnight.isoformat(),
                "blocked_until": blocked.isoformat() if blocked else None,
                "last_error": today.get("last_error"),
            })
        return rows

    def days(self) -> dict:
        return json.loads(json.dumps(self._data["days"]))


_ledger: UsageLedger | None = None
_ledger_lock = threading.Lock()


def get_ledger() -> UsageLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None or _ledger.path != default_path():
            _ledger = UsageLedger()
        return _ledger
