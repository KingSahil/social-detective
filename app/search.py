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
        upload_path = str(image_path)
        temp_upload_file = None
        try:
            from PIL import Image
            with Image.open(str(image_path)) as im:
                if im.format != "JPEG" or im.mode != "RGB":
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    temp_upload_file = tmp.name
                    tmp.close()
                    im.convert("RGB").save(temp_upload_file, "JPEG", quality=95)
                    upload_path = temp_upload_file
        except Exception:
            pass

        try:
            upload_result = client.upload_image(upload_path)
            image_id = upload_result.get("image_id")
            if not image_id:
                raise RuntimeError(f"Upload succeeded but no image_id returned: {upload_result}")
        except Exception as e:
            raise RuntimeError(f"Image upload to SerpAPI failed: {e}")
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
