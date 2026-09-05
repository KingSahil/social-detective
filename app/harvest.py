"""
Public content harvester for social accounts.

Extracts public profile avatars and og:images from discovered account hits:
    - Tier 1 structured extractors (GitHub, Dev.to, Gravatar, YouTube, Devpost, Pinterest, Duolingo)
    - Generic HTML fallback (meta og:image, profile <img> tags)
    - Browser escalation hook (for bot-walled profiles when enabled)
    - Biometric media validation (raster checks, dimension limits, WebP/GIF handling, generic logo filtering)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image

from app.config import (
    DATA_DIR,
    PIVOT_BROWSER_FALLBACK,
    PIVOT_MAX_CANDIDATES,
    PIVOT_MAX_WORKERS,
    PIVOT_TIMEOUT,
)
from app.identity import AccountHit, DEFAULT_USER_AGENT
from app.search import Candidate

logger = logging.getLogger(__name__)

# Minimum pixel dimensions for biometric face analysis (EC13)
MIN_IMAGE_DIM = 60

# Generic default placeholder hashes or sub-strings to ignore (EC15)
GENERIC_AVATAR_PATTERNS = [
    "identicon",
    "default_avatar",
    "default-avatar",
    "avatar_default",
    "placeholder",
    "default-user",
    "user-placeholder",
    "default.jpg",
    "default.png",
    "gravatar.com/avatar/00000000000000000000000000000000",
    "d=mp",
    "d=identicon",
    "d=blank",
    "blank.gif",
    "1x1.trans.gif",
]


# ---------------------------------------------------------------------------
# Image Validation & Decoding (EC12, EC13, EC14, EC15)
# ---------------------------------------------------------------------------

def is_generic_placeholder(url: str) -> bool:
    """Check if the image URL points to a known generic avatar placeholder."""
    url_lower = url.lower()
    for pat in GENERIC_AVATAR_PATTERNS:
        if pat in url_lower:
            return True
    return False


def validate_and_decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Validates image bytes for biometric face matching:
      - Rejects non-raster / SVG data (EC12)
      - Decodes raster images with cv2, falling back to PIL for WebP/GIF (EC14)
      - Enforces minimum width and height >= 60px (EC13)
    Returns an RGB/BGR numpy array or None if invalid.
    """
    # Trivial garbage filter only (very small floor: valid tiny WebP/GIF
    # thumbnails can be well under 100 bytes). Real validation happens via
    # decode + minimum-dimension checks below.
    if not image_bytes or len(image_bytes) < 50:
        return None

    # EC12: Check if payload is SVG text
    head = image_bytes[:256].strip().lower()
    if b"<svg" in head or b"<?xml" in head:
        return None

    # Try cv2.imdecode first (fastest)
    img_array = None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        img_array = None

    # EC14: Fallback to PIL (handles WebP, animated GIFs, uncommon colorspaces)
    if img_array is None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                rgb_img = pil_img.convert("RGB")
                img_array = np.array(rgb_img)[:, :, ::-1]  # RGB to BGR for cv2 consistency
        except Exception:
            return None

    if img_array is None or img_array.size == 0:
        return None

    h, w = img_array.shape[:2]
    # EC13: Check minimum dimensions
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        return None

    return img_array


# ---------------------------------------------------------------------------
# Tier 1 Structured Extractors
# ---------------------------------------------------------------------------

def extract_github_avatar(hit: AccountHit, session: requests.Session, timeout: float) -> Optional[str]:
    """GitHub avatar: https://github.com/{user}.png (302 redirects to CDN avatar)."""
    user = hit.profile_url.rstrip("/").split("/")[-1]
    user = user.lstrip("@")
    if not user:
        return None
    avatar_url = f"https://github.com/{user}.png"
    return avatar_url


def extract_gravatar_avatar(hit: AccountHit, session: requests.Session, timeout: float) -> Optional[str]:
    """Gravatar: /avatar/{md5} or json endpoint."""
    profile_url = hit.profile_url
    if "gravatar.com/" in profile_url:
        match = re.search(r"gravatar\.com/([a-f0-9]{32,64})", profile_url)
        if match:
            hash_val = match.group(1)
            return f"https://gravatar.com/avatar/{hash_val}?s=400&d=404"
        # Or check json
        user = profile_url.rstrip("/").split("/")[-1]
        try:
            resp = session.get(f"https://gravatar.com/{user}.json", timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT})
            if resp.status_code == 200:
                data = resp.json()
                entry = data.get("entry", [])
                if entry and isinstance(entry, list) and "thumbnailUrl" in entry[0]:
                    return entry[0]["thumbnailUrl"]
        except Exception:
            pass
    return None


def extract_duolingo_avatar(hit: AccountHit, session: requests.Session, timeout: float) -> Optional[str]:
    """Duolingo metadata API."""
    user = hit.profile_url.rstrip("/").split("/")[-1]
    if not user:
        return None
    try:
        resp = session.get(
            f"https://www.duolingo.com/2017-06-30/users?username={user}",
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        if resp.status_code == 200:
            data = resp.json()
            users = data.get("users", [])
            if users and users[0].get("picture"):
                pic = users[0]["picture"]
                if not pic.startswith("http"):
                    pic = "https:" + pic
                return pic
    except Exception:
        pass
    return None


def extract_devto_avatar(hit: AccountHit, session: requests.Session, timeout: float) -> Optional[str]:
    """Dev.to og:image / profile avatar."""
    return None  # Let generic HTML extractor fetch the og:image from dev.to profile


TIER1_SPECIAL_EXTRACTORS: dict[str, Callable[[AccountHit, requests.Session, float], Optional[str]]] = {
    "GitHub": extract_github_avatar,
    "Gravatar": extract_gravatar_avatar,
    "Duolingo": extract_duolingo_avatar,
}


# ---------------------------------------------------------------------------
# Generic HTML & Browser Extractors
# ---------------------------------------------------------------------------

def extract_og_image_from_html(html: str, base_url: str) -> list[str]:
    """Extract candidate og:image and high-signal profile images from HTML."""
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    image_urls: list[str] = []

    # 1. Look for meta og:image or twitter:image
    for meta in soup.find_all("meta"):
        raw_prop = meta.get("property") or ""
        prop = " ".join(raw_prop).lower() if isinstance(raw_prop, list) else str(raw_prop).lower()
        raw_name = meta.get("name") or ""
        name = " ".join(raw_name).lower() if isinstance(raw_name, list) else str(raw_name).lower()
        raw_content = meta.get("content") or ""
        content = " ".join(raw_content).strip() if isinstance(raw_content, list) else str(raw_content).strip()
        if not content:
            continue
        if prop in ("og:image", "og:image:url", "og:image:secure_url") or name in ("twitter:image", "twitter:image:src"):
            full_url = urllib.parse.urljoin(base_url, content)
            if not is_generic_placeholder(full_url) and not full_url.lower().endswith(".svg"):
                image_urls.append(full_url)

    # 2. Look for profile / avatar <img> tags
    for img in soup.find_all("img"):
        raw_src = img.get("src") or img.get("data-src") or ""
        src = " ".join(raw_src).strip() if isinstance(raw_src, list) else str(raw_src).strip()
        if not src:
            continue
        raw_class = img.get("class") or ""
        img_class = " ".join(raw_class).lower() if isinstance(raw_class, list) else str(raw_class).lower()
        img_id = str(img.get("id") or "").lower()
        img_alt = str(img.get("alt") or "").lower()

        is_avatar = any(
            kw in img_class or kw in img_id or kw in img_alt
            for kw in ("avatar", "profile", "user-image", "author-photo", "photo")
        )
        if is_avatar:
            full_url = urllib.parse.urljoin(base_url, src)
            if not is_generic_placeholder(full_url) and not full_url.lower().endswith(".svg"):
                image_urls.append(full_url)

    return image_urls


def default_browser_fetch(url: str, timeout: float = 15.0) -> str:
    """
    Playwright-based browser fallback renderer for protected/bot-walled profile pages.
    """
    executable = (
        os.getenv("CHROMIUM_PATH")
        or shutil.which("chromium")
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or ""
    )

    try:
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as temp_profile:
            with sync_playwright() as p:
                launch_kwargs = {
                    "user_data_dir": temp_profile,
                    "headless": True,
                    "args": [
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--window-position=-2400,-2400",
                        "--window-size=1280,720",
                    ],
                }
                if executable:
                    launch_kwargs["executable_path"] = executable
                else:
                    launch_kwargs["channel"] = "chrome"

                context = p.chromium.launch_persistent_context(**launch_kwargs)
                page = context.new_page()
                page.set_default_timeout(int(timeout * 1000))
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                content = page.content()
                context.close()
                return content
    except Exception as e:
        logger.debug("Browser fallback failed for %s: %s", url, e)
        return ""


# ---------------------------------------------------------------------------
# PublicContentHarvester
# ---------------------------------------------------------------------------

class PublicContentHarvester:
    """
    Harvester that resolves AccountHit instances into verified Candidate image records.
    """

    def __init__(
        self,
        timeout: float = PIVOT_TIMEOUT,
        max_workers: int = PIVOT_MAX_WORKERS,
        max_candidates: int = PIVOT_MAX_CANDIDATES,
        browser_fallback: bool = PIVOT_BROWSER_FALLBACK,
        browser_fetcher: Optional[Callable[[str, float], str]] = None,
    ):
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_candidates = max_candidates
        self.browser_fallback = browser_fallback
        self.browser_fetcher = browser_fetcher or default_browser_fetch

    def _fetch_and_validate_image(
        self,
        image_url: str,
        session: requests.Session,
    ) -> bool:
        """Fetch image bytes and verify raster validity + dimensions."""
        if not image_url or image_url.lower().endswith(".svg"):
            return False
        if is_generic_placeholder(image_url):
            return False

        try:
            resp = session.get(
                image_url,
                timeout=self.timeout,
                headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"},
            )
            if resp.status_code != 200:
                return False
            content_type = resp.headers.get("Content-Type", "").lower()
            if "svg" in content_type:
                return False

            decoded = validate_and_decode_image(resp.content)
            return decoded is not None
        except Exception as e:
            logger.debug("Failed to validate image %s: %s", image_url, e)
            return False

    def harvest_hit(
        self,
        hit: AccountHit,
        session: requests.Session,
    ) -> list[Candidate]:
        """Harvest images for a single AccountHit."""
        candidates: list[Candidate] = []
        domain = urllib.parse.urlparse(hit.profile_url).netloc
        title = f"{hit.site_name} Profile"

        # 1. Direct structured extractor
        if hit.site_name in TIER1_SPECIAL_EXTRACTORS:
            extractor = TIER1_SPECIAL_EXTRACTORS[hit.site_name]
            try:
                img_url = extractor(hit, session, self.timeout)
                if img_url and self._fetch_and_validate_image(img_url, session):
                    candidates.append(
                        Candidate(
                            image_url=img_url,
                            source_url=hit.profile_url,
                            title=title,
                            domain=domain,
                        )
                    )
                    return candidates
            except Exception as e:
                logger.debug("Structured extractor failed for %s: %s", hit.site_name, e)

        # 2. Generic HTML fetch & og:image extraction
        html = ""
        try:
            resp = session.get(
                hit.profile_url,
                timeout=self.timeout,
                headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
            )
            if resp.status_code == 200:
                html = resp.text
            elif resp.status_code == 403 and self.browser_fallback:
                # Escalate to browser if enabled
                html = self.browser_fetcher(hit.profile_url, self.timeout)
        except Exception:
            if self.browser_fallback:
                try:
                    html = self.browser_fetcher(hit.profile_url, self.timeout)
                except Exception:
                    html = ""

        if html:
            extracted_imgs = extract_og_image_from_html(html, hit.profile_url)
            for img_url in extracted_imgs[:2]:  # cap at top 2 images per hit
                if self._fetch_and_validate_image(img_url, session):
                    candidates.append(
                        Candidate(
                            image_url=img_url,
                            source_url=hit.profile_url,
                            title=title,
                            domain=domain,
                        )
                    )
                    if len(candidates) >= 2:
                        break

        return candidates

    def harvest(
        self,
        hits: list[AccountHit],
        session: Optional[requests.Session] = None,
    ) -> list[Candidate]:
        """
        Harvests candidates across all supplied AccountHits concurrently.
        Deduplicates by image_url (EC16).
        Caps output to max_candidates.
        """
        if not hits:
            return []

        candidates: list[Candidate] = []
        seen_images: set[str] = set()

        owns_session = False
        if session is None:
            session = requests.Session()
            owns_session = True

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_hit = {
                    executor.submit(self.harvest_hit, hit, session): hit
                    for hit in hits
                }

                for future in as_completed(future_to_hit):
                    try:
                        hit_candidates = future.result()
                        for c in hit_candidates:
                            if c.image_url not in seen_images:
                                seen_images.add(c.image_url)
                                candidates.append(c)
                                if len(candidates) >= self.max_candidates:
                                    break
                        if len(candidates) >= self.max_candidates:
                            for pending in future_to_hit:
                                pending.cancel()
                            break
                    except Exception as e:
                        logger.debug("Harvest hit error: %s", e)
        finally:
            if owns_session:
                try:
                    session.close()
                except Exception:
                    pass

        return candidates
