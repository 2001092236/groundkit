"""Скриншоты демо для README: python scripts/screenshots.py [http://127.0.0.1:8765]

Нужен Playwright: pip install playwright && playwright install chromium
"""

import sys
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
QUESTION = "В какой форме заключается договор аренды здания по ГК РФ?"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/?{urlencode({'q': QUESTION, 'preset': 'Право РФ', 'run': '1'})}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_selector("#report:not([hidden]) .kpi, #notice .notice", timeout=240_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "demo.png"), full_page=True)
        result = page.evaluate(
            "document.querySelector('#report .kpi .v')?.textContent || document.querySelector('#notice')?.textContent"
        )
        print("desktop:", OUT / "demo.png", "|", (result or "").strip()[:80])

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.goto(f"{BASE}/", wait_until="load")
        mobile.wait_for_timeout(1500)
        mobile.screenshot(path=str(OUT / "demo-mobile.png"), full_page=False)
        print("mobile:", OUT / "demo-mobile.png")
        browser.close()


if __name__ == "__main__":
    main()
