"""
Search provider abstraction + Headless Google Lens + Free Reverse Search implementations.

Provides:
    SearchProvider                 — abstract base
    HeadlessLensProvider           — self-contained headless Google Lens search (with automatic fallback)
    DirectYandexProvider           — direct free reverse image search via Yandex Images
    FreeMultiEngineSearchProvider  — orchestrates Headless Lens + Direct Yandex + DDGS OSINT
    SerpAPIProvider                — legacy reverse-image search via SerpAPI
    MockProvider                   — for unit tests ONLY
"""

from __future__ import annotations

import abc
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin
from contextlib import contextmanager
import threading
import time

import numpy as np
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single search-result candidate."""
    image_url: str
    source_url: str
    title: str = ""
    domain: str = ""


@dataclass
class SearchResult:
    """Aggregated search results from a provider."""
    candidates: list[Candidate] = field(default_factory=list)
    provider: str = ""
    searched_at: str = ""
    raw_response: Optional[dict] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SearchProvider(abc.ABC):
    """Interface every search provider must implement."""

    @abc.abstractmethod
    def search(self, image_path: str | Path) -> SearchResult:
        """
        Perform a reverse-image / face search using *image_path*.

        Must return dynamically retrieved results — never hardcoded data.
        """
        ...


BLOCKED_KEYWORDS = {
    "xhamster", "loveplanet", "zamantika", "mybro.tv", "porn", "adult", "escort",
    "webcam", "stripchat", "bongacams", "chaturbate", "livejasmin", "onlyfans",
    "dating", "hookup", "sexy", "nsfw", "xxx", "erotic"
}

SOCIAL_DOMAINS = [
    "x.com", "twitter.com", "instagram.com", "linkedin.com", "reddit.com",
    "facebook.com", "threads.net", "youtube.com", "github.com", "pinterest.com",
    "tiktok.com", "medium.com", "quora.com"
]


def filter_and_prioritize_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    1. Removes any candidate from adult, shady, or spam domains.
    2. Prioritizes legitimate social media platforms (Twitter/X, LinkedIn, Instagram, Reddit, etc.)
       at the top of the candidate list for facial analysis.
    """
    clean: list[Candidate] = []
    seen: set[str] = set()

    for c in candidates:
        if not c.source_url or c.source_url in seen:
            continue
        url_lower = c.source_url.lower()
        domain_lower = (c.domain or "").lower()

        # Discard blocked domains
        if any(bad in url_lower or bad in domain_lower for bad in BLOCKED_KEYWORDS):
            continue

        seen.add(c.source_url)
        clean.append(c)

    # Sort so that social media domains come first
    def _social_priority(c: Candidate) -> int:
        d = (c.domain or "").lower()
        u = (c.source_url or "").lower()
        for idx, s in enumerate(SOCIAL_DOMAINS):
            if s in d or s in u:
                return idx
        return 999

    clean.sort(key=_social_priority)
    return clean


# ---------------------------------------------------------------------------
# SerpAPI Google Lens provider
# ---------------------------------------------------------------------------

class SerpAPIProvider(SearchProvider):
    """
    Genuine reverse-image search via SerpAPI's Google Lens endpoint.

    Uses the ``serpapi`` Python client to:
    1. Upload the local image file to SerpAPI's image API (returns a temporary image_id).
    2. Perform a Google Lens search using that image_id.
    3. Parse ``visual_matches`` from the response.

    Requires the ``SERPAPI_KEY`` environment variable.
    """

    PROVIDER_NAME = "SerpAPI Google Lens"

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            self._api_key = os.getenv("SERPAPI_KEY", "")
        else:
            self._api_key = api_key
        if not self._api_key or self._api_key.startswith("your_"):
            raise RuntimeError(
                "SERPAPI_KEY is not configured. "
                "Get a key at https://serpapi.com and set it in your .env file."
            )

    def search(self, image_path: str | Path) -> SearchResult:
        """Upload local image to Google Lens via SerpAPI and return candidates."""
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            import serpapi as serpapi_mod
        except ImportError:
            raise RuntimeError(
                "serpapi package not installed. Run: pip install serpapi"
            )

        client = serpapi_mod.Client(api_key=self._api_key)

        # Step 1: Sanitize and upload local image to get a temporary image_id
        # SerpApi's Image API rejects files >500KB, so re-encode any file that
        # is not already a small-enough baseline JPEG (not just non-JPEG input).
        upload_path = str(image_path)
        temp_upload_file = None
        try:
            import os as _os
            from PIL import Image
            needs_reencode = True
            try:
                if _os.path.getsize(upload_path) <= 500_000:
                    with Image.open(upload_path) as _im_chk:
                        if _im_chk.format == "JPEG" and _im_chk.mode == "RGB":
                            needs_reencode = False
            except Exception:
                needs_reencode = True
            if needs_reencode:
                import tempfile
                with Image.open(str(image_path)) as im:
                    rgb = im.convert("RGB")
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    temp_upload_file = tmp.name
                    tmp.close()
                    quality = 92
                    while quality >= 30:
                        rgb.save(temp_upload_file, "JPEG", quality=quality)
                        if _os.path.getsize(temp_upload_file) <= 500_000:
                            break
                        quality -= 10
                    if _os.path.getsize(temp_upload_file) > 500_000:
                        # Quality floor reached; halve resolution and retry once
                        w, h = rgb.size
                        rgb = rgb.resize((max(1, w // 2), max(1, h // 2)))
                        rgb.save(temp_upload_file, "JPEG", quality=85)
                upload_path = temp_upload_file
        except Exception:
            pass

        import time
        upload_result = None
        image_id = None
        try:
            for attempt in range(3):
                try:
                    # POST the image directly to SerpApi's Image API
                    # (wire format identical to serpapi>=1.1.0 client.upload_image,
                    #  which is unavailable in serpapi 1.0.x)
                    with open(upload_path, "rb") as img_f:
                        resp_up = requests.post(
                            "https://serpapi.com/image",
                            data={"api_key": self._api_key},
                            files={"image": ("image.jpg", img_f, "image/jpeg")},
                            timeout=30,
                        )
                    if resp_up.status_code == 429 and attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    resp_up.raise_for_status()
                    upload_result = resp_up.json()
                    image_id = upload_result.get("image_id")
                    if image_id:
                        break
                    raise RuntimeError(f"Upload succeeded but no image_id returned: {upload_result}")
                except RuntimeError:
                    raise
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Image upload to SerpAPI failed: {e}")

            if not image_id:
                raise RuntimeError(f"Upload succeeded but no image_id returned: {upload_result}")
        finally:
            if temp_upload_file and Path(temp_upload_file).exists():
                try:
                    Path(temp_upload_file).unlink()
                except OSError:
                    pass

        # Step 2: Google Lens search using image_id
        try:
            raw = client.search({
                "engine": "google_lens",
                "image_id": image_id,
            })
        except Exception as e:
            raise RuntimeError(f"SerpAPI Google Lens search failed: {e}")

        # Convert SerpResults to dict if needed
        if hasattr(raw, "as_dict"):
            raw = raw.as_dict()
        elif not isinstance(raw, dict):
            raw = dict(raw)

        # Check for API errors
        if "error" in raw:
            raise RuntimeError(f"SerpAPI error: {raw['error']}")

        candidates: list[Candidate] = []

        # Extract from visual_matches (Google Lens primary results)
        for item in raw.get("visual_matches", []):
            img_url = item.get("thumbnail", "") or item.get("image", "")
            src_url = item.get("link", "")
            title = item.get("title", "")
            domain = item.get("source", "") or item.get("displayed_link", "")
            if img_url and src_url:
                candidates.append(Candidate(
                    image_url=img_url,
                    source_url=src_url,
                    title=title,
                    domain=domain,
                ))

        # Also check image_sources if present
        for item in raw.get("image_sources", []):
            img_url = item.get("thumbnail", "") or item.get("image", "")
            src_url = item.get("link", "")
            if img_url and src_url:
                candidates.append(Candidate(
                    image_url=img_url,
                    source_url=src_url,
                    title=item.get("title", ""),
                    domain=item.get("source", ""),
                ))

        # Filter blocked domains and prioritize social media
        filtered = filter_and_prioritize_candidates(candidates)

        return SearchResult(
            candidates=filtered,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Direct Yandex Reverse Search Provider (100% Free - No SerpAPI required)
# ---------------------------------------------------------------------------

class DirectYandexProvider(SearchProvider):
    """
    Direct, free reverse image search via Yandex Images without SerpAPI.
    Uploads local image to a temporary direct host (freeimage.host) to obtain
    a public image URL, then queries Yandex's reverse search directly and extracts
    candidate faces across indexed web appearances and social platforms.
    """
    PROVIDER_NAME = "Direct Yandex Images (Free)"

    def __init__(self, timeout: float = 25.0):
        self._timeout = timeout

    def _upload_to_public_url(self, image_path: Path) -> str:
        """Uploads local image to a temporary direct host (Catbox with freeimage.host fallback)."""
        # 1. Try Catbox first (fast, reliable direct links, no API key required)
        try:
            with open(str(image_path), "rb") as f:
                cat_resp = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": ("image.jpg", f, "image/jpeg")},
                    timeout=self._timeout,
                )
            if cat_resp.status_code == 200 and cat_resp.text.strip().startswith("http"):
                return cat_resp.text.strip()
        except Exception:
            pass

        # 2. Fallback to freeimage.host
        import base64
        import tempfile
        from PIL import Image

        temp_jpeg = None
        try:
            with Image.open(str(image_path)) as im:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_jpeg = tmp.name
                tmp.close()
                im.convert("RGB").save(temp_jpeg, "JPEG", quality=92)
                with open(temp_jpeg, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            if temp_jpeg and Path(temp_jpeg).exists():
                try:
                    Path(temp_jpeg).unlink()
                except OSError:
                    pass

        resp = requests.post(
            "https://freeimage.host/api/1/upload",
            data={
                "key": "6d207e02198a847aa98d0a2a901485a5",
                "action": "upload",
                "source": b64,
                "format": "json",
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        url = resp.json().get("image", {}).get("url")
        if not url:
            raise RuntimeError(f"FreeImage upload failed: {resp.text}")
        return url

    def search(self, image_path: str | Path) -> SearchResult:
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            public_url = self._upload_to_public_url(image_path)
        except Exception as e:
            return SearchResult(
                candidates=[],
                provider=self.PROVIDER_NAME,
                searched_at=timestamp,
                raw_response={"error": f"Image host upload failed: {e}"},
            )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        yandex_url = f"https://yandex.com/images/search?rpt=imageview&url={public_url}"

        candidates: list[Candidate] = []
        raw_info: dict = {"public_url": public_url, "yandex_url": yandex_url}
        try:
            resp = requests.get(yandex_url, headers=headers, timeout=self._timeout)
            if resp.status_code == 200 and resp.text:
                soup = BeautifulSoup(resp.text, "html.parser")
                seen_img: set[str] = set()
                from urllib.parse import unquote, urlparse

                # 1. Parse similar images and site matches with img_url parameter
                for a in soup.find_all("a"):
                    href = a.get("href", "")
                    if "img_url=" in href:
                        m_img = re.search(r"img_url=([^&]+)", href)
                        if m_img:
                            img_u = unquote(m_img.group(1))
                            if not img_u or img_u in seen_img or not img_u.startswith("http"):
                                continue
                            seen_img.add(img_u)

                            m_rurl = re.search(r"rurl=([^&]+)", href)
                            source_u = unquote(m_rurl.group(1)) if m_rurl else href
                            if not source_u.startswith("http"):
                                source_u = urljoin("https://yandex.com", source_u)

                            title = a.get_text().strip()
                            domain = urlparse(source_u).netloc or urlparse(img_u).netloc
                            candidates.append(Candidate(
                                image_url=img_u,
                                source_url=source_u,
                                title=title,
                                domain=domain,
                            ))

                # 2. Parse direct site links in "Sites with information about the image"
                for region in soup.find_all(["div", "section"]):
                    heading = region.find(["h2", "h3"])
                    if heading and "site" in heading.get_text().lower():
                        for a in region.find_all("a"):
                            href = a.get("href", "")
                            if href.startswith("http") and "yandex" not in href:
                                title = a.get_text().strip()
                                domain = urlparse(href).netloc
                                img_el = a.find("img")
                                img_src = img_el.get("src") if img_el else ""
                                if img_src and img_src.startswith("http") and img_src not in seen_img:
                                    seen_img.add(img_src)
                                    candidates.append(Candidate(
                                        image_url=img_src,
                                        source_url=href,
                                        title=title,
                                        domain=domain,
                                    ))
        except Exception as e:
            raw_info["error"] = str(e)

        filtered = filter_and_prioritize_candidates(candidates)
        raw_info["count"] = len(filtered)

        return SearchResult(
            candidates=filtered,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response=raw_info,
        )


# ---------------------------------------------------------------------------
# Headless Google Lens Provider (Self-hosted automated visual discovery)
# ---------------------------------------------------------------------------

class HeadlessLensProvider(SearchProvider):
    """
    Self-contained headless Google Lens reverse visual search engine.
    Uploads local image directly to Google Lens v3 upload endpoint to bypass in-page
    bot verification, then orchestrates offscreen Chrome via Playwright to extract
    visual matches with zero CAPTCHA friction.
    Automatically delegates to DirectYandexProvider if Google Lens fails.
    """
    PROVIDER_NAME = "Headless Google Lens"

    def __init__(
        self,
        headless: bool = True,
        timeout: float = 25.0,
        user_data_dir: str | Path | None = None,
        fallback_on_captcha: bool = True,
    ):
        self.headless = headless
        self.timeout = timeout
        if user_data_dir is None:
            import tempfile
            self.user_data_dir = Path(tempfile.gettempdir()) / "social_detective_lens_profile"
        else:
            self.user_data_dir = Path(user_data_dir).resolve()
        self.fallback_on_captcha = fallback_on_captcha
        # Resolve a real Chrome/Chromium binary for Playwright.
        # channel="chrome" only finds Google Chrome; on distros with only
        # Chromium (e.g. Arch), point Playwright at the system binary instead.
        import shutil
        self._chromium_executable = os.getenv("CHROMIUM_PATH") or shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("google-chrome-stable") or ""

    def search(self, image_path: str | Path) -> SearchResult:
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        timestamp = datetime.now(timezone.utc).isoformat()
        candidates: list[Candidate] = []
        raw_info: dict = {"image": str(image_path)}

        # 1. Upload directly to Google Lens v3 upload endpoint to get search Location URL
        # This completely bypasses Google's in-page bot detection and captcha triggers!
        location = None
        session = requests.Session()
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
            with open(image_path, "rb") as f:
                files = {"encoded_image": ("image.jpg", f, "image/jpeg")}
                resp = session.post(
                    "https://lens.google.com/v3/upload",
                    files=files,
                    headers=headers,
                    allow_redirects=False,
                    timeout=self.timeout,
                )
            location = resp.headers.get("Location")
        except Exception as e:
            raw_info["upload_error"] = str(e)

        # 2. Render search results page via offscreen Chrome
        html = ""
        if location:
            try:
                from playwright.sync_api import sync_playwright
                self.user_data_dir.mkdir(parents=True, exist_ok=True)

                args = [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-infobars",
                ]
                if self.headless:
                    # Offscreen coordinates ensure the window is invisible while running
                    # in full headful mode, avoiding Google's headless bot detection
                    args.extend(["--window-position=-2400,-2400", "--window-size=1920,1080"])

                with sync_playwright() as p:
                    launch_kwargs = {}
                    if self._chromium_executable:
                        launch_kwargs["executable_path"] = self._chromium_executable
                    else:
                        launch_kwargs["channel"] = "chrome"
                    context = p.chromium.launch_persistent_context(
                        user_data_dir=str(self.user_data_dir),
                        **launch_kwargs,
                        headless=False,
                        ignore_default_args=["--enable-automation"],
                        args=args,
                        viewport={"width": 1920, "height": 1080},
                    )
                    page = context.pages[0] if context.pages else context.new_page()

                    cookies_to_add = []
                    for c in session.cookies:
                        cookies_to_add.append({
                            "name": c.name,
                            "value": c.value,
                            "domain": c.domain,
                            "path": c.path,
                        })
                    if cookies_to_add:
                        context.add_cookies(cookies_to_add)

                    page.goto(location, wait_until="load", timeout=int(self.timeout * 1000))
                    try:
                        page.wait_for_selector("div[data-item-id], [jsname], script", timeout=3000)
                    except Exception:
                        pass
                    page.evaluate("window.scrollTo(0, 1500)")
                    time.sleep(0.5)

                    html = page.content()
                    raw_info["final_url"] = page.url
                    context.close()

                # 3. Parse Google Lens Visual Matches from embedded JSON
                def clean_str(s: str) -> str:
                    s = s.replace(r"\u003d", "=").replace(r"\u0026", "&").replace(r"\/", "/")
                    s = s.replace(r"\u003c", "<").replace(r"\u003e", ">")
                    return s

                pattern = re.compile(
                    r'\["(https:[^"]+)",\s*(\d+),\s*(\d+)\],\s*null,\s*\d+,\s*\{[^}]*"2003":\s*\[[^,]*,\s*"[^"]*",\s*"([^"]+)",\s*"([^"]*)"',
                    re.DOTALL
                )

                seen: set[str] = set()
                from urllib.parse import urlparse
                for m in pattern.finditer(html):
                    orig_img = clean_str(m.group(1))
                    source_url = clean_str(m.group(4))
                    title = clean_str(m.group(5))
                    domain = urlparse(source_url).netloc
                    if source_url not in seen:
                        seen.add(source_url)
                        candidates.append(Candidate(
                            image_url=orig_img,
                            source_url=source_url,
                            title=title,
                            domain=domain,
                        ))

            except Exception as e:
                raw_info["browser_error"] = str(e)

        # 4. Fallback to DirectYandexProvider if Google Lens returned no candidates
        if not candidates and self.fallback_on_captcha:
            print("        [i] Seamlessly activating Direct Free Visual Search fallback...")
            yandex = DirectYandexProvider(timeout=self.timeout)
            y_res = yandex.search(image_path)
            if y_res.candidates:
                return SearchResult(
                    candidates=y_res.candidates,
                    provider=f"{self.PROVIDER_NAME} (with Direct Yandex Fallback)",
                    searched_at=timestamp,
                    raw_response={"lens": raw_info, "yandex": y_res.raw_response},
                )

        filtered = filter_and_prioritize_candidates(candidates)
        raw_info["candidate_count"] = len(filtered)
        return SearchResult(
            candidates=filtered,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response=raw_info,
        )


# ---------------------------------------------------------------------------
# Free Multi-Engine Visual Search Provider
# ---------------------------------------------------------------------------

class FreeMultiEngineSearchProvider(SearchProvider):
    """
    Combined visual reverse search provider:
    Orchestrates Headless Google Lens with Direct Yandex Visual Search
    and DuckDuckGo social OSINT correlation without requiring any paid API keys.
    """
    PROVIDER_NAME = "Free Multi-Engine Visual Search"

    def __init__(self, headless: bool = True, timeout: float = 25.0):
        self.lens = HeadlessLensProvider(headless=headless, timeout=timeout, fallback_on_captcha=True)
        self.yandex = DirectYandexProvider(timeout=timeout)

    def search(self, image_path: str | Path) -> SearchResult:
        res = self.lens.search(image_path)
        if len(res.candidates) < 5:
            try:
                y_res = self.yandex.search(image_path)
                existing_urls = {c.image_url for c in res.candidates}
                for c in y_res.candidates:
                    if c.image_url not in existing_urls:
                        res.candidates.append(c)
                        existing_urls.add(c.image_url)
            except Exception:
                pass
        res.candidates = filter_and_prioritize_candidates(res.candidates)
        return res


# ---------------------------------------------------------------------------
# Yandex Images Provider — Deep biometric social media reverse search
# ---------------------------------------------------------------------------

class YandexProvider(SearchProvider):
    """
    Reverse image search using SerpAPI's Yandex Images engine.
    Yandex performs biometric cross-social-media face matching where Google Lens
    is restricted.
    """
    PROVIDER_NAME = "SerpAPI Yandex Images"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self._api_key = api_key
        self._timeout = timeout

        if not self._api_key or self._api_key.strip() == "" or "your_" in self._api_key:
            raise RuntimeError(
                "SERPAPI_KEY is not set or contains a placeholder. "
                "Set a valid SERPAPI_KEY in your .env file."
            )

    def _upload_to_public_url(self, image_path: Path) -> str:
        """
        Uploads local image to a temporary direct host (freeimage.host) to obtain
        a public image URL required by SerpAPI's yandex_images engine.
        """
        import base64
        import tempfile
        import requests
        from PIL import Image

        temp_jpeg = None
        try:
            with Image.open(str(image_path)) as im:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_jpeg = tmp.name
                tmp.close()
                im.convert("RGB").save(temp_jpeg, "JPEG", quality=92)
                with open(temp_jpeg, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            if temp_jpeg and Path(temp_jpeg).exists():
                try:
                    Path(temp_jpeg).unlink()
                except OSError:
                    pass

        resp = requests.post("https://freeimage.host/api/1/upload", data={
            "key": "6d207e02198a847aa98d0a2a901485a5",
            "action": "upload",
            "source": b64,
            "format": "json"
        }, timeout=self._timeout)
        resp.raise_for_status()
        url = resp.json().get("image", {}).get("url")
        if not url:
            raise RuntimeError(f"FreeImage upload failed: {resp.text}")
        return url

    def search(self, image_path: str | Path) -> SearchResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            import serpapi as serpapi_mod
        except ImportError:
            raise RuntimeError("serpapi package not installed. Run: pip install serpapi")

        client = serpapi_mod.Client(api_key=self._api_key)
        public_url = self._upload_to_public_url(image_path)

        try:
            raw = client.search({
                "engine": "yandex_images",
                "url": public_url,
            })
        except Exception as e:
            raise RuntimeError(f"SerpAPI Yandex Images search failed: {e}")

        if hasattr(raw, "as_dict"):
            raw = raw.as_dict()
        elif not isinstance(raw, dict):
            raw = dict(raw)

        if "error" in raw:
            raise RuntimeError(f"SerpAPI Yandex error: {raw['error']}")

        def _extract_link(val) -> str:
            if isinstance(val, dict):
                return val.get("link", "") or val.get("url", "") or ""
            return str(val) if isinstance(val, str) else ""

        candidates: list[Candidate] = []
        for item in raw.get("image_results", []):
            img_url = (
                _extract_link(item.get("original_image"))
                or _extract_link(item.get("thumbnail"))
                or _extract_link(item.get("image"))
            )
            src_url = item.get("link", "")
            title = item.get("title", "")
            domain = item.get("source", "")
            if img_url:
                candidates.append(Candidate(
                    image_url=img_url,
                    source_url=src_url,
                    title=title,
                    domain=domain,
                ))

        for item in raw.get("similar_images", []):
            img_url = (
                _extract_link(item.get("image"))
                or _extract_link(item.get("thumbnail"))
            )
            src_url = item.get("link", "")
            title = item.get("title", "")
            domain = item.get("source", "")
            if img_url and not any(c.image_url == img_url for c in candidates):
                candidates.append(Candidate(
                    image_url=img_url,
                    source_url=src_url,
                    title=title,
                    domain=domain,
                ))

        # Filter blocked domains and prioritize social media
        filtered = filter_and_prioritize_candidates(candidates)

        return SearchResult(
            candidates=filtered,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Target URL Provider — Direct social post / webpage inspection
# ---------------------------------------------------------------------------

class TargetURLProvider(SearchProvider):
    """
    Directly extracts media images from a target webpage or social post URL.
    Used for targeted facial verification against suspected online appearances.
    """
    PROVIDER_NAME = "Target URL Inspector"

    def __init__(self, target_url: str, timeout: float = 15.0):
        self.target_url = target_url
        self._timeout = timeout

    def search(self, image_path: str | Path | None = None) -> SearchResult:
        import re
        from urllib.parse import urljoin, urlparse
        import requests
        from bs4 import BeautifulSoup

        timestamp = datetime.now(timezone.utc).isoformat()
        domain = urlparse(self.target_url).netloc.lower()
        candidates: list[Candidate] = []
        page_title = ""

        # Special handling: Instagram post via Instaloader
        if "instagram.com" in domain and ("/p/" in self.target_url or "/reel/" in self.target_url):
            try:
                import instaloader
                shortcode_match = re.search(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)", self.target_url)
                if shortcode_match:
                    shortcode = shortcode_match.group(1)
                    L = instaloader.Instaloader()
                    post = instaloader.Post.from_shortcode(L.context, shortcode)
                    owner = post.owner_username
                    caption = (post.caption or "").strip()
                    title = f"{owner} on Instagram: \"{caption[:80]}...\"" if caption else f"Instagram post by {owner}"

                    if post.typename == "GraphSidecar":
                        for i, node in enumerate(post.get_sidecar_nodes()):
                            candidates.append(Candidate(
                                image_url=node.display_url,
                                source_url=self.target_url,
                                title=f"{title} (Slide {i+1})",
                                domain="www.instagram.com",
                            ))
                    else:
                        candidates.append(Candidate(
                            image_url=post.url,
                            source_url=self.target_url,
                            title=title,
                            domain="www.instagram.com",
                        ))

                    if candidates:
                        return SearchResult(
                            candidates=candidates,
                            provider="Instaloader (Instagram)",
                            searched_at=timestamp,
                            raw_response={"target_url": self.target_url, "image_count": len(candidates), "owner": owner},
                        )
            except Exception:
                pass  # Fall back to HTML scraper if instaloader fails

        # Check if the target_url itself is a direct image file
        image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
        parsed_path = urlparse(self.target_url).path.lower()
        if any(parsed_path.endswith(ext) for ext in image_extensions):
            candidates.append(Candidate(
                image_url=self.target_url,
                source_url=self.target_url,
                title=f"Direct Image ({Path(parsed_path).name})",
                domain=domain,
            ))
            return SearchResult(
                candidates=candidates,
                provider=self.PROVIDER_NAME,
                searched_at=timestamp,
                raw_response={"target_url": self.target_url, "image_count": 1},
            )

        user_agent = (
            "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"
            if "instagram.com" in domain
            else (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        headers = {
            "User-Agent": user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        }

        resp = requests.get(self.target_url, headers=headers, timeout=self._timeout)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # Extract page title
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            page_title = title_tag.string.strip()
        elif soup.find("meta", property="og:title"):
            page_title = soup.find("meta", property="og:title").get("content", "").strip()

        discovered_image_urls: set[str] = set()

        # 1. Look for Twitter/X media links (pbs.twimg.com/media/...)
        twimg_matches = re.findall(
            r"https://pbs\.twimg\.com/media/([A-Za-z0-9_-]+)(?:\.[a-zA-Z0-9]+|\?format=[a-zA-Z0-9]+)?",
            html
        )
        for media_id in twimg_matches:
            # Normalize to direct high-res JPG URL
            discovered_image_urls.add(f"https://pbs.twimg.com/media/{media_id}.jpg")

        # 2. Look for Twitter profile images if available
        profile_matches = re.findall(
            r"https://pbs\.twimg\.com/profile_images/([A-Za-z0-9_/-]+?)(?:_normal|_400x400)?\.(?:jpg|png|jpeg)",
            html
        )
        for p_id in profile_matches:
            discovered_image_urls.add(f"https://pbs.twimg.com/profile_images/{p_id}_400x400.jpg")

        # 3. OpenGraph and Twitter Card meta tags
        for meta_prop in ["og:image", "og:image:secure_url", "twitter:image", "twitter:image:src"]:
            meta_tag = soup.find("meta", attrs={"property": meta_prop}) or soup.find("meta", attrs={"name": meta_prop})
            if meta_tag and meta_tag.get("content"):
                u = meta_tag["content"].strip()
                if u.startswith("http"):
                    discovered_image_urls.add(u)

        # 4. Standard HTML <img> tags
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                full_url = urljoin(self.target_url, src)
                # Filter out small tracking pixels / data URIs / svgs
                if (
                    full_url.startswith("http")
                    and not any(x in full_url.lower() for x in ["favicon", "icon", "analytics", "tracking", ".svg"])
                ):
                    discovered_image_urls.add(full_url)

        for img_url in discovered_image_urls:
            candidates.append(Candidate(
                image_url=img_url,
                source_url=self.target_url,
                title=page_title,
                domain=domain,
            ))

        return SearchResult(
            candidates=candidates,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response={"target_url": self.target_url, "image_count": len(candidates)},
        )


# ---------------------------------------------------------------------------
# Twitter/X Public Profile Timeline Provider (OSINT Pivot)
# ---------------------------------------------------------------------------

class TwitterProfileProvider(SearchProvider):
    """
    Sweeps a Twitter/X public user profile timeline via SSR HTML.
    Extracts all tweet status URLs and their high-res attached media
    without requiring an official Twitter API key or user authentication.
    """

    PROVIDER_NAME = "Twitter / X Profile Discovery"

    def __init__(self, handle: str, timeout: float = 6.0):
        self.handle = handle.lstrip("@").strip()
        self.timeout = timeout

    def search(self, image_path: str | Path | None = None) -> SearchResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        if not self.handle:
            return SearchResult(candidates=[], provider=self.PROVIDER_NAME, searched_at=timestamp)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        url = f"https://x.com/{self.handle}"
        candidates: list[Candidate] = []
        seen_status: set[str] = set()

        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200 and resp.text:
                soup = BeautifulSoup(resp.text, "html.parser")

                # Twitter embeds links to tweets: /<handle>/status/<id>/photo/1 with an <img> tag inside
                for a in soup.find_all("a"):
                    href = a.get("href", "")
                    if "/status/" in href:
                        img = a.find("img")
                        if img and img.get("src"):
                            img_src = img["src"]
                            status_id = href.split("/status/")[1].split("/")[0].split("?")[0]
                            full_tweet_url = f"https://x.com/{self.handle}/status/{status_id}"

                            if full_tweet_url not in seen_status:
                                seen_status.add(full_tweet_url)
                                candidates.append(Candidate(
                                    image_url=img_src,
                                    source_url=full_tweet_url,
                                    title=f"Tweet by @{self.handle}",
                                    domain="x.com"
                                ))
        except Exception:
            pass

        # Fallback: query FxTwitter API for high-res avatar if direct scrape returned 0 items
        if not candidates:
            try:
                r_fx = requests.get(f"https://api.fxtwitter.com/{self.handle}", timeout=min(4.0, self.timeout))
                if r_fx.status_code == 200:
                    data = r_fx.json()
                    user = data.get("user", {})
                    avatar = user.get("avatar_url")
                    if avatar:
                        profile_url = f"https://x.com/{self.handle}"
                        if profile_url not in seen_status:
                            candidates.append(Candidate(
                                image_url=avatar.replace("_normal", "_400x400"),
                                source_url=profile_url,
                                title=f"Profile of @{self.handle} ({user.get('name', '')})",
                                domain="x.com"
                            ))
            except Exception:
                pass

        return SearchResult(
            candidates=candidates,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response={"handle": self.handle, "count": len(candidates)},
        )


def extract_social_handles(candidates: list[Candidate]) -> list[str]:
    """
    Extracts potential social usernames/handles from candidate URLs and titles.
    Supports LinkedIn, Instagram, GitHub, Twitter/X, and direct handle formats.
    """
    handles: set[str] = set()
    RESERVED = {
        "posts", "reel", "reels", "p", "share", "explore", "home", "status",
        "about", "login", "signup", "search", "hashtag", "direct", "stories",
        "in", "pub", "feed", "jobs", "learning", "events", "company", "groups",
        "intent", "i", "privacy", "tos", "help", "settings"
    }

    for c in candidates:
        urls = [c.source_url or "", c.image_url or ""]
        for u in urls:
            if not u:
                continue
            # LinkedIn /in/<handle> or /posts/<handle>_...
            m_li_in = re.search(r"linkedin\.com/in/([A-Za-z0-9_-]+)", u, re.IGNORECASE)
            if m_li_in:
                h = m_li_in.group(1).lower().strip()
                if h and h not in RESERVED and len(h) >= 3:
                    handles.add(h)
            m_li_post = re.search(r"linkedin\.com/posts/([A-Za-z0-9_-]+)_", u, re.IGNORECASE)
            if m_li_post:
                h = m_li_post.group(1).lower().strip()
                if h and h not in RESERVED and len(h) >= 3:
                    handles.add(h)

            # Instagram /<handle> or /p/...
            m_ig = re.search(r"instagram\.com/([A-Za-z0-9_.-]+)", u, re.IGNORECASE)
            if m_ig:
                h = m_ig.group(1).lower().strip().rstrip("/")
                if h and h not in RESERVED and len(h) >= 3:
                    handles.add(h)

            # GitHub /<handle>
            m_gh = re.search(r"github\.com/([A-Za-z0-9_-]+)", u, re.IGNORECASE)
            if m_gh:
                h = m_gh.group(1).lower().strip().rstrip("/")
                if h and h not in RESERVED and len(h) >= 3:
                    handles.add(h)

            # X / Twitter /<handle>
            m_x = re.search(r"(?:x|twitter)\.com/([A-Za-z0-9_]+)", u, re.IGNORECASE)
            if m_x:
                h = m_x.group(1).lower().strip()
                if h and h not in RESERVED and len(h) >= 3:
                    handles.add(h)

        # Also look in candidate title for @handle or "<handle> on Instagram"
        title = c.title or ""
        for word in title.split():
            if word.startswith("@") and len(word) > 2:
                clean_h = re.sub(r"[^A-Za-z0-9_]", "", word).lower()
                if clean_h and clean_h not in RESERVED and len(clean_h) >= 3:
                    handles.add(clean_h)

        for m_ig_name in re.findall(r"([A-Za-z0-9_.]{3,30})\s+on Instagram", title, re.IGNORECASE):
            h_clean = m_ig_name.lower().strip()
            if h_clean not in RESERVED and len(h_clean) >= 3:
                handles.add(h_clean)

        for m_ig_by in re.findall(r"Instagram post by\s+([A-Za-z0-9_.]{3,30})", title, re.IGNORECASE):
            h_clean = m_ig_by.lower().strip()
            if h_clean not in RESERVED and len(h_clean) >= 3:
                handles.add(h_clean)

    return sorted(handles)


_MEMORY_EMB_CACHE: dict[str, np.ndarray] = {}


def find_subject_memory_leads(
    query_embedding: np.ndarray,
    results_dir: str | Path = "data/results",
    similarity_threshold: float = 0.65,
    fp: Any | None = None,
) -> tuple[list[str], list[Candidate]]:
    """
    Forensics Identity Correlation:
    If an unindexed face matches a previously verified subject (>= similarity_threshold),
    recalls that subject's known social handles AND past verified appearances.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return [], []

    discovered_handles: set[str] = set()
    recalled_candidates: list[Candidate] = []
    seen_urls: set[str] = set()

    try:
        from app.matcher import cosine_similarity
        if fp is None:
            from app.face import FaceProcessor
            fp = FaceProcessor()
    except Exception:
        return [], []

    cache_file = results_path / ".subject_embeddings.pkl"
    if cache_file.exists() and not _MEMORY_EMB_CACHE:
        try:
            import pickle
            with open(cache_file, "rb") as f:
                _MEMORY_EMB_CACHE.update(pickle.load(f))
        except Exception:
            pass

    cache_updated = False
    RESERVED = {
        "posts", "reel", "reels", "p", "share", "explore", "home", "status",
        "about", "login", "signup", "search", "hashtag", "direct", "stories",
        "in", "pub", "feed", "jobs", "company", "intent", "i"
    }

    for record_file in sorted(results_path.glob("*.json"), reverse=True):
        try:
            with open(record_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            stored_image_path = data.get("query", {}).get("image")
            if not stored_image_path or not Path(stored_image_path).exists():
                continue

            if stored_image_path in _MEMORY_EMB_CACHE:
                stored_emb = _MEMORY_EMB_CACHE[stored_image_path]
            else:
                stored_emb = fp.get_embedding(stored_image_path)
                if stored_emb is not None:
                    _MEMORY_EMB_CACHE[stored_image_path] = stored_emb
                    cache_updated = True

            if stored_emb is None:
                continue

            sim = cosine_similarity(query_embedding, stored_emb)

            if sim >= similarity_threshold:
                # 1. Recall known verified appearances directly!
                urls_to_inspect = []
                for u in [
                    data.get("match", {}).get("source_url"),
                    data.get("content", {}).get("source_url"),
                    data.get("search", {}).get("target_url"),
                ]:
                    if u and u not in seen_urls:
                        urls_to_inspect.append(u)
                        seen_urls.add(u)

                # Direct verified image URLs
                for img_u in [
                    data.get("match", {}).get("image_url"),
                    data.get("content", {}).get("image_url"),
                ]:
                    if img_u and img_u not in seen_urls:
                        seen_urls.add(img_u)
                        recalled_candidates.append(Candidate(
                            image_url=img_u,
                            source_url=data.get("match", {}).get("source_url") or img_u,
                            title=data.get("match", {}).get("title") or "Verified Subject Appearance",
                            domain=data.get("match", {}).get("domain") or "Verified Identity Memory",
                        ))

                for u in urls_to_inspect:
                    try:
                        tp = TargetURLProvider(u, timeout=8.0)
                        t_res = tp.search()
                        if t_res and t_res.candidates:
                            for c in t_res.candidates:
                                if c.image_url not in seen_urls:
                                    seen_urls.add(c.image_url)
                                    recalled_candidates.append(c)
                    except Exception:
                        pass

                # 2. Extract social handles
                for auth_field in [
                    data.get("match", {}).get("author"),
                    data.get("content", {}).get("author"),
                    data.get("search", {}).get("handle"),
                ]:
                    if auth_field and isinstance(auth_field, str):
                        clean_a = auth_field.lstrip("@").strip().lower()
                        if clean_a not in RESERVED and len(clean_a) >= 3:
                            discovered_handles.add(clean_a)

                text_to_scan = " ".join([
                    data.get("match", {}).get("source_url", ""),
                    data.get("content", {}).get("source_url", ""),
                    data.get("content", {}).get("text", ""),
                    data.get("content", {}).get("title", ""),
                    data.get("match", {}).get("title", ""),
                ])

                at_handles = re.findall(r"@([A-Za-z0-9_]+)", text_to_scan)
                for h in at_handles:
                    if len(h) >= 3:
                        discovered_handles.add(h.lower())

                for pattern in [
                    r"linkedin\.com/in/([A-Za-z0-9_-]+)",
                    r"linkedin\.com/posts/([A-Za-z0-9_-]+)_",
                    r"instagram\.com/([A-Za-z0-9_.-]+)",
                    r"github\.com/([A-Za-z0-9_-]+)",
                    r"(?:x|twitter)\.com/([A-Za-z0-9_]+)",
                    r"([A-Za-z0-9_.]{3,30})\s+on Instagram",
                    r"Instagram post by\s+([A-Za-z0-9_.]{3,30})",
                    r"\(@([A-Za-z0-9_.]{3,30})\)\s+on Instagram",
                ]:
                    for m in re.findall(pattern, text_to_scan, re.IGNORECASE):
                        discovered_handles.add(m.lower())
        except Exception:
            continue

    if cache_updated:
        try:
            import pickle
            with open(cache_file, "wb") as f:
                pickle.dump(_MEMORY_EMB_CACHE, f)
        except Exception:
            pass

    clean_handles = [h for h in sorted(discovered_handles) if h not in RESERVED]
    return clean_handles, recalled_candidates


def find_social_handles_from_subject_memory(
    query_embedding: np.ndarray,
    results_dir: str | Path = "data/results",
    similarity_threshold: float = 0.58,
    fp: Any | None = None,
) -> list[str]:
    """Compatibility wrapper for returning only handles."""
    handles, _ = find_subject_memory_leads(
        query_embedding, results_dir=results_dir, similarity_threshold=similarity_threshold, fp=fp
    )
    return handles


def search_web_leads(query: str, max_results: int = 5) -> list[Candidate]:
    """
    OSINT Web Pivot:
    Searches DuckDuckGo for an identity handle or person name,
    inspects the top candidate web pages (portfolios, projects, profiles),
    and extracts candidate media images using TargetURLProvider.
    """
    from urllib.parse import urlparse
    from ddgs import DDGS
    candidates: list[Candidate] = []
    seen_urls: set[str] = set()

    clean_q = query.strip().lstrip("@")
    if len(clean_q) < 3:
        return []

    q_str = f'"{clean_q}"' if (" " in clean_q or "-" in clean_q) else clean_q
    try:
        ddgs = DDGS()
        results = list(ddgs.text(q_str, max_results=max_results))
    except Exception:
        results = []

    SKIP_DOMAINS = {
        "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
        "facebook.com", "instagram.com", "twitter.com", "x.com",
        "youtube.com", "pinterest.com"
    }

    for r in results:
        href = r.get("href", "")
        if not href or href in seen_urls:
            continue
        seen_urls.add(href)
        domain = urlparse(href).netloc.lower()
        if any(d in domain for d in SKIP_DOMAINS):
            continue

        try:
            tp = TargetURLProvider(href, timeout=8.0)
            t_res = tp.search()
            if t_res and t_res.candidates:
                candidates.extend(t_res.candidates)
        except Exception:
            continue

    return candidates


# ---------------------------------------------------------------------------
# LinkedIn Post Provider — Deep public post discovery via targeted search
# ---------------------------------------------------------------------------

class LinkedInPostProvider(SearchProvider):
    """
    Discovers candidate LinkedIn posts for investigation targets or associate leads.
    Searches public posts via SerpAPI (DuckDuckGo engine, with Google fallback),
    extracts open-graph images and metadata without hitting LinkedIn's authwall.
    """

    PROVIDER_NAME = "LinkedIn Post Discovery"

    def __init__(self, api_key: str | None = None, timeout: float = 6.0, allow_free: bool = False):
        self._api_key = api_key
        self._timeout = timeout
        self._allow_free = allow_free

        if not self._allow_free and (not self._api_key or self._api_key.strip() == "" or "your_" in self._api_key):
            raise RuntimeError(
                "SERPAPI_KEY is not set or contains a placeholder. "
                "Set a valid SERPAPI_KEY in your .env file."
            )

    def search_leads(self, names: list[str], contexts: list[str] | None = None) -> SearchResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        if not names:
            return SearchResult(candidates=[], provider=self.PROVIDER_NAME, searched_at=timestamp)

        client = None
        if self._api_key and not self._api_key.startswith("your_"):
            try:
                import serpapi as serpapi_mod
                client = serpapi_mod.Client(api_key=self._api_key)
            except ImportError:
                if not self._allow_free:
                    raise RuntimeError("serpapi package not installed. Run: pip install serpapi")
                client = None
        elif not self._allow_free:
            raise RuntimeError("SERPAPI_KEY is not set or contains a placeholder.")

        discovered_urls: set[str] = set()

        # Build search queries combining associate names and event contexts
        raw_queries: list[str] = []
        valid_contexts = [c for c in (contexts or []) if any(k in c.lower() for k in ["hack", "hazard", "namespace", "build"])]
        if not valid_contexts and contexts:
            valid_contexts = contexts[:2]

        for name in names:
            for ctx in valid_contexts:
                raw_queries.append(f"site:linkedin.com/posts/ {name} {ctx}")
            raw_queries.append(f"site:linkedin.com/posts/ {name}")

        queries = list(dict.fromkeys(raw_queries))

        from concurrent.futures import ThreadPoolExecutor

        def _run_query(q: str) -> list[str]:
            urls = []
            if client is not None:
                try:
                    res = client.search({"engine": "duckduckgo", "q": q})
                    items = res.get("organic_results", [])
                    if not items:
                        res = client.search({"engine": "google", "q": q})
                        items = res.get("organic_results", [])
                    for it in items:
                        link = it.get("link", "")
                        if "linkedin.com/posts/" in link:
                            clean_url = link.split("?")[0].rstrip("/")
                            urls.append(clean_url)
                except Exception:
                    pass
            if not urls:
                try:
                    for it in _safe_ddgs_text(q, max_results=15):
                        href = it.get("href", "")
                        if "linkedin.com/posts/" in href:
                            clean_url = href.split("?")[0].rstrip("/")
                            urls.append(clean_url)
                except Exception:
                    pass
            return urls

        with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as pool:
            for url_list in pool.map(_run_query, queries):
                for u in url_list:
                    discovered_urls.add(u)

        if not discovered_urls:
            return SearchResult(candidates=[], provider=self.PROVIDER_NAME, searched_at=timestamp)

        # Concurrently fetch og:image from discovered LinkedIn post URLs
        candidates: list[Candidate] = []

        def _fetch_og(post_url: str) -> Candidate | None:
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                }
                resp = requests.get(post_url, headers=headers, timeout=self._timeout)
                if resp.status_code == 200 and resp.text:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    og_img = soup.find("meta", property="og:image")
                    og_title = soup.find("meta", property="og:title")
                    if og_img and og_img.get("content"):
                        title = og_title["content"] if og_title and og_title.get("content") else (soup.title.string if soup.title else "")
                        return Candidate(
                            image_url=og_img["content"],
                            source_url=post_url,
                            title=title.strip(),
                            domain="linkedin.com",
                        )
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=min(len(discovered_urls), 8)) as pool:
            for cand in pool.map(_fetch_og, sorted(discovered_urls)):
                if cand:
                    candidates.append(cand)

        return SearchResult(
            candidates=candidates,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response={"queries": queries, "count": len(candidates)},
        )

    def search(self, image_path: str | Path | None = None) -> SearchResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        return SearchResult(candidates=[], provider=self.PROVIDER_NAME, searched_at=timestamp)


# ---------------------------------------------------------------------------
# Instagram Profile & Post Provider (OSINT Social Pivot)
# ---------------------------------------------------------------------------

_C_STDERR_LOCK = threading.Lock()
_DDGS_LOCK = threading.Lock()

@contextmanager
def _suppress_c_stderr():
    """
    Suppresses OS-level C/Rust file descriptor 2 (stderr) output.
    Used to silence low-level rustls/h2 EOF messages printed directly to stderr
    by the primp HTTP client when servers close connections without close_notify.
    """
    with _C_STDERR_LOCK:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(devnull, 2)
            os.close(devnull)
            try:
                yield
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
        except Exception:
            yield


def _safe_ddgs_text(q: str, max_results: int = 15) -> list[dict]:
    """
    Thread-safe and stderr-sanitized wrapper around DDGS.text().
    Prevents concurrent connection flooding and silences rustls/h2 EOF warnings.
    """
    with _DDGS_LOCK:
        with _suppress_c_stderr():
            try:
                from ddgs import DDGS
                d = DDGS(timeout=5)
                return list(d.text(q, max_results=max_results))
            except Exception:
                return []


class InstagramProfileProvider(SearchProvider):
    """
    Discovers public Instagram posts and reels for suspected handles and their associates.
    Uses targeted search engine dorks (DuckDuckGo engine via SerpAPI, with Google fallback)
    and unpacks multi-slide carousels (GraphSidecar) via Instaloader to evaluate all faces.
    """

    PROVIDER_NAME = "Instagram Profile & Post Discovery"

    def __init__(
        self,
        handle: str | None = None,
        api_key: str | None = None,
        timeout: float = 6.0,
        allow_free: bool = False,
    ):
        self.handle = handle.lstrip("@").strip() if handle else None
        self._api_key = api_key
        self._timeout = timeout
        self._allow_free = allow_free

        if not self._allow_free and (not self._api_key or self._api_key.strip() == "" or "your_" in self._api_key):
            raise RuntimeError(
                "SERPAPI_KEY is not set or contains a placeholder. "
                "Set a valid SERPAPI_KEY in your .env file."
            )

    def search_handles(
        self,
        handles: list[str],
        contexts: list[str] | None = None,
        max_handles: int = 6,
    ) -> SearchResult:
        """
        Sweeps multiple candidate handles and pivots on tagged associates.
        """
        from concurrent.futures import ThreadPoolExecutor
        timestamp = datetime.now(timezone.utc).isoformat()
        if not handles:
            return SearchResult(candidates=[], provider=self.PROVIDER_NAME, searched_at=timestamp)

        client = None
        if self._api_key and not self._api_key.startswith("your_"):
            try:
                import serpapi as serpapi_mod
                client = serpapi_mod.Client(api_key=self._api_key)
            except ImportError:
                if not self._allow_free:
                    raise RuntimeError("serpapi package not installed. Run: pip install serpapi")
                client = None
        elif not self._allow_free:
            raise RuntimeError("SERPAPI_KEY is not set or contains a placeholder.")

        clean_handles = [h.lstrip("@").strip() for h in handles if h and h.strip()]
        # Prioritize personal handles over generic terms
        clean_handles = [h for h in clean_handles if h.lower() not in {"popular", "instagram", "posts", "post"}]
        clean_handles = clean_handles[:max_handles]

        # Build initial queries
        queries: list[str] = []
        for h in clean_handles:
            queries.append(f"site:instagram.com {h}")
            queries.append(f"site:instagram.com/p/ {h}")
            queries.append(f"site:instagram.com/reel/ {h}")
            if contexts:
                for ctx in contexts[:2]:
                    queries.append(f"site:instagram.com {h} {ctx}")

        discovered_shortcodes: set[str] = set()
        discovered_tags: set[str] = set()

        def _extract_sc_from_url(u: str) -> str | None:
            if not u:
                return None
            m = re.search(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)", u)
            if m:
                return m.group(1)
            if "google.com" in u:
                try:
                    from urllib.parse import urlparse, parse_qs, unquote
                    parsed = urlparse(u)
                    qs = parse_qs(parsed.query)
                    for param in ["url", "q"]:
                        if param in qs:
                            target_u = unquote(qs[param][0])
                            m_target = re.search(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)", target_u)
                            if m_target:
                                return m_target.group(1)
                except Exception:
                    pass
                try:
                    r_head = requests.head(u, allow_redirects=False, timeout=3.0)
                    loc = r_head.headers.get("Location")
                    if loc:
                        m_loc = re.search(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)", loc)
                        if m_loc:
                            return m_loc.group(1)
                except Exception:
                    pass
            return None

        def _execute_query(q: str):
            res_codes = set()
            res_tags = set()
            # 1. Try SerpAPI if available and not exhausted
            if client is not None:
                try:
                    for engine in ["duckduckgo", "google"]:
                        for attempt in range(2):
                            try:
                                res = client.search({"engine": engine, "q": q})
                                for item in res.get("organic_results", []):
                                    link = item.get("link", "")
                                    title = item.get("title", "")
                                    snippet = item.get("snippet", "")
                                    disp_link = item.get("displayed_link", "")

                                    sc = _extract_sc_from_url(link)
                                    if not sc:
                                        sc_text = re.search(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)", f"{disp_link} {title} {snippet}")
                                        if sc_text:
                                            sc = sc_text.group(1)
                                    if sc:
                                        res_codes.add(sc)

                                    tags = re.findall(r"@([A-Za-z0-9_.]{3,30})", f"{title} {snippet}")
                                    for m_t in re.findall(r"([A-Za-z0-9_.]{3,30})\s+on Instagram", f"{title} {snippet}", re.IGNORECASE):
                                        tags.append(m_t)
                                    for t in tags:
                                        t_clean = t.lower()
                                        if t_clean not in [h.lower() for h in clean_handles] and t_clean not in {"instagram", "p", "reel", "reels"}:
                                            res_tags.add(t)
                                if res_codes:
                                    break
                            except Exception as e:
                                if "429" in str(e):
                                    time.sleep(1.0 * (attempt + 1))
                                    continue
                                break
                        if res_codes:
                            break
                except Exception:
                    pass

            # 2. Free DDGS fallback if no codes yet or SerpAPI is out of quota
            if not res_codes:
                try:
                    for it in _safe_ddgs_text(q, max_results=15):
                        href = it.get("href", "")
                        title = it.get("title", "")
                        body = it.get("body", "")
                        sc = _extract_sc_from_url(href)
                        if not sc:
                            sc_text = re.search(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)", f"{title} {body}")
                            if sc_text:
                                sc = sc_text.group(1)
                        if sc:
                            res_codes.add(sc)
                        tags = re.findall(r"@([A-Za-z0-9_.]{3,30})", f"{title} {body}")
                        for m_t in re.findall(r"([A-Za-z0-9_.]{3,30})\s+on Instagram", f"{title} {body}", re.IGNORECASE):
                            tags.append(m_t)
                        for t in tags:
                            t_clean = t.lower()
                            if t_clean not in [h.lower() for h in clean_handles] and t_clean not in {"instagram", "p", "reel", "reels"}:
                                res_tags.add(t)
                except Exception:
                    pass

            return res_codes, res_tags

        # Execute hop 1 queries concurrently
        with ThreadPoolExecutor(max_workers=min(len(queries), 6) or 1) as pool:
            for codes, tags in pool.map(_execute_query, queries):
                discovered_shortcodes.update(codes)
                discovered_tags.update(tags)

        # 2nd-hop pivot on discovered collaborator tags (co-occurring with handles)
        valid_tags = [
            t for t in discovered_tags
            if not any(k in t.lower() for k in ["cess", "group", "titans", "club", "event", "community"])
        ][:6]

        if valid_tags:
            hop2_queries = []
            for t in valid_tags:
                hop2_queries.append(f"site:instagram.com {t}")
                for h in clean_handles[:2]:
                    hop2_queries.append(f"site:instagram.com {h} {t}")

            with ThreadPoolExecutor(max_workers=min(len(hop2_queries), 6) or 1) as pool:
                for codes, _ in pool.map(_execute_query, hop2_queries):
                    discovered_shortcodes.update(codes)

        # Now unpack discovered shortcodes into Candidates (including carousels)
        candidates: list[Candidate] = []

        def _unpack_shortcode(sc: str) -> list[Candidate]:
            items: list[Candidate] = []
            try:
                import instaloader
                L = instaloader.Instaloader()
                post = instaloader.Post.from_shortcode(L.context, sc)
                caption = (post.caption or "").strip()
                caption_snip = caption[:80].replace("\n", " ")

                if post.typename == "GraphSidecar":
                    for i, node in enumerate(post.get_sidecar_nodes()):
                        items.append(Candidate(
                            image_url=node.display_url,
                            source_url=f"https://www.instagram.com/p/{sc}/?img_index={i+1}",
                            title=f"{post.owner_username} on Instagram: \"{caption_snip}...\"",
                            domain="instagram.com",
                        ))
                elif post.typename == "GraphImage" or post.typename == "GraphVideo":
                    items.append(Candidate(
                        image_url=post.url,
                        source_url=f"https://www.instagram.com/p/{sc}/" if post.typename == "GraphImage" else f"https://www.instagram.com/reel/{sc}/",
                        title=f"{post.owner_username} on Instagram: \"{caption_snip}...\"",
                        domain="instagram.com",
                    ))
            except Exception:
                # Fallback to public embed endpoint
                try:
                    embed_url = f"https://www.instagram.com/p/{sc}/embed/"
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                        )
                    }
                    r = requests.get(embed_url, headers=headers, timeout=self._timeout)
                    if r.status_code == 200:
                        soup = BeautifulSoup(r.text, "html.parser")
                        for img in soup.find_all("img"):
                            src = img.get("src")
                            if src and ("fbcdn" in src or "cdninstagram" in src):
                                items.append(Candidate(
                                    image_url=src,
                                    source_url=f"https://www.instagram.com/p/{sc}/",
                                    title=f"Instagram Post {sc}",
                                    domain="instagram.com",
                                ))
                                break
                except Exception:
                    pass
            return items

        # Unpack top discovered shortcodes into Candidates (capped at 8 for performance)
        target_shortcodes = sorted(discovered_shortcodes)[:8]
        with ThreadPoolExecutor(max_workers=min(len(target_shortcodes), 6) or 1) as pool:
            for unpacked_list in pool.map(_unpack_shortcode, target_shortcodes):
                candidates.extend(unpacked_list)

        return SearchResult(
            candidates=candidates,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response={
                "shortcodes": list(discovered_shortcodes),
                "count": len(candidates),
                "tags": list(discovered_tags),
            },
        )

    def search(self, image_path: str | Path | None = None) -> SearchResult:
        if self.handle:
            return self.search_handles([self.handle])
        timestamp = datetime.now(timezone.utc).isoformat()
        return SearchResult(candidates=[], provider=self.PROVIDER_NAME, searched_at=timestamp)


def extract_associate_network_leads(
    results_dir: str | Path = "data/results",
) -> tuple[list[str], list[str]]:
    """
    Extracts associated names and event/project contexts from verified investigation records.
    Used for OSINT Associate Network Pivoting when direct visual reverse search yields 0 hits.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return [], []

    collaborators: set[str] = set()
    other_names: set[str] = set()
    contexts: set[str] = set()

    STOP_WORDS = {
        "The World", "New Startup", "Summer Break", "Great Opportunity", "Top Content",
        "Sign In", "Join Now", "View Profile", "Report Post", "Report Comment",
        "Open Source", "Social Detective", "Face Search", "Blockchain Verification",
        "Content Retriever", "Ethereum Sepolia", "Content Verified", "Photo Gallery",
        "User Profile", "Check Out", "Also Thanks", "Global Hackathon",
    }

    for p in sorted(results_path.glob("*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        content_text = " ".join([
            data.get("content", {}).get("text", ""),
            data.get("content", {}).get("title", ""),
            data.get("match", {}).get("title", ""),
        ])

        # 1. Direct collaborator blocks (highest priority)
        collab_blocks = re.findall(
            r"(?:with|teaming up with|team up with|along with)\s+([A-Z][a-z]+ [A-Z][a-z]+(?:(?:,\s*(?:and\s+)?|\s+and\s+)[A-Z][a-z]+ [A-Z][a-z]+)*)",
            content_text,
        )
        for block in collab_blocks:
            for n in re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", block):
                n_clean = n.strip()
                if n_clean not in STOP_WORDS and len(n_clean) >= 4:
                    collaborators.add(n_clean)

        pipe_names = re.findall(r"\|\s*([A-Z][a-z]+ [A-Z][a-z]+)", content_text)
        for n in pipe_names:
            n_clean = n.strip()
            if n_clean not in STOP_WORDS and len(n_clean) >= 4:
                other_names.add(n_clean)

        # 2. Extract key event / campaign contexts
        for ctx in ["Hackhazards", "Namespace", "Hackhazards 26", "Sarvam AI", "Blinky", "Hacker House"]:
            if ctx.lower() in content_text.lower():
                contexts.add(ctx)

    # Collaborators take top priority
    ordered_names = sorted(collaborators) + [n for n in sorted(other_names) if n not in collaborators]
    return ordered_names, sorted(contexts)


# ---------------------------------------------------------------------------
# Mock provider — unit tests ONLY
# ---------------------------------------------------------------------------

class MockSearchProvider(SearchProvider):
    """
    Returns pre-configured candidates.

    **FOR UNIT TESTS ONLY** — never used in the real pipeline.
    """

    PROVIDER_NAME = "Mock (test only)"

    def __init__(self, candidates: list[Candidate] | None = None):
        self._candidates = candidates or []

    def search(self, image_path: str | Path) -> SearchResult:
        return SearchResult(
            candidates=list(self._candidates),
            provider=self.PROVIDER_NAME,
            searched_at=datetime.now(timezone.utc).isoformat(),
        )
