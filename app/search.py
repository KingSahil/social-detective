"""
Search provider abstraction + SerpAPI Google Lens implementation.

Provides:
    SearchProvider   — abstract base
    SerpAPIProvider  — genuine reverse-image search via Google Lens
    MockProvider     — for unit tests ONLY
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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

        # Step 1: Upload local image to get a temporary image_id
        try:
            upload_result = client.upload_image(str(image_path))
            image_id = upload_result.get("image_id")
            if not image_id:
                raise RuntimeError(f"Upload succeeded but no image_id returned: {upload_result}")
        except Exception as e:
            raise RuntimeError(f"Image upload to SerpAPI failed: {e}")

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

        # Deduplicate by source_url
        seen: set[str] = set()
        unique: list[Candidate] = []
        for c in candidates:
            if c.source_url not in seen:
                seen.add(c.source_url)
                unique.append(c)

        return SearchResult(
            candidates=unique,
            provider=self.PROVIDER_NAME,
            searched_at=timestamp,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Target URL Provider — Direct social post / webpage inspection
# ---------------------------------------------------------------------------

class TargetURLProvider(SearchProvider):
    """
    Directly inspects a specific target URL (e.g., an X/Twitter post, Reddit
    thread, Instagram link, or webpage) and extracts candidate media images
    dynamically for face matching and blockchain notarization.

    Zero hardcoding: dynamically parses media CDN links (e.g., pbs.twimg.com),
    OpenGraph tags, Twitter Card images, JSON-LD schemas, and HTML img elements.
    """

    PROVIDER_NAME = "Target URL Inspector"

    def __init__(self, target_url: str, timeout: int = 15):
        self.target_url = target_url.strip()
        self._timeout = timeout

    def search(self, image_path: str | Path) -> SearchResult:
        """Extract all candidate media images from the target URL."""
        import re
        from urllib.parse import urljoin, urlparse
        import requests
        from bs4 import BeautifulSoup

        timestamp = datetime.now(timezone.utc).isoformat()
        domain = urlparse(self.target_url).netloc.lower()
        candidates: list[Candidate] = []
        page_title = ""

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

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
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
