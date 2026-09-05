"""
LinkedIn public-post harvester.

Verified platform behavior (probed 2026-09-04, requests + headful chromium):
  - Public POST pages (linkedin.com/posts/{slug}_{activity_id}) are served to
    guests: HTML contains og:image, embedded media.licdn.com photo URLs
    (feedshare post images + profile-displayphoto avatars) and associate
    profile links (https://{cc}.linkedin.com/in/{slug}).
    Plain requests with a browser UA works; browser escalation is an optional
    fallback for intermittent guest-wall variance.
  - PROFILE pages (linkedin.com/in/{slug}) serve HTTP 999 to scripts and
    redirect browsers to /authwall. They are NOT harvested: the pipeline is
    public-data-only and never authenticates or bypasses the login wall.

Extraction contract (all URLs deduped, feedshare images ranked first):
    photos         -> feedshare post images, then profile-displayphoto avatars
    profile_slugs  -> /in/{slug} links found on the page (associates + author)
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

from app.identity import DEFAULT_USER_AGENT
from app.search import Candidate

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_MEDIA_URL_PATTERN = re.compile(
    r"https://media\.licdn\.com/dms/image/[^\s\"'\\<>]+"
)
_PROFILE_LINK_PATTERN = re.compile(
    r"https://[a-z]{2,4}\.linkedin\.com/in/([A-Za-z0-9_-]{3,60})"
)
_POST_SLUG_PATTERN = re.compile(r"/posts/([A-Za-z0-9_-]{3,60})_")

# Path-like words that appear in /in/ links but are not real member slugs
_RESERVED_SLUGS = {
    "in", "pub", "dir", "search", "login", "signup", "feed", "jobs",
    "company", "posts", "pulse", "groups", "school", "page",
}

# Display-photo variants ranked below real post images for matching purposes
_FEEDSHARE_MARK = "feedshare"
_DISPLAYPHOTO_MARK = "profile-displayphoto"


@dataclass
class LinkedInPostContent:
    """Extracted public content from one LinkedIn post page."""
    source_url: str = ""
    title: str = ""
    text: str = ""
    photos: list[str] = field(default_factory=list)      # ranked: feedshare first
    profile_slugs: list[str] = field(default_factory=list)  # author + associates
    raw_html_len: int = 0


def extract_linkedin_content(html: str, source_url: str) -> LinkedInPostContent:
    """
    Parse a public LinkedIn post page (guest HTML) into ranked media URLs,
    associate profile slugs, and post metadata. Pure function, no network.
    """
    content = LinkedInPostContent(source_url=source_url, raw_html_len=len(html or ""))
    if not html:
        return content

    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("meta", property="og:title")
    if title_tag and title_tag.get("content"):
        content.title = " ".join(str(title_tag["content"]).split())[:200]
    if not content.title and soup.title and soup.title.string:
        content.title = " ".join(soup.title.string.split())[:200]

    desc_tag = soup.find("meta", property="og:description")
    if desc_tag and desc_tag.get("content"):
        content.text = " ".join(str(desc_tag["content"]).split())[:500]

    # Collect and rank media URLs (unescape \u0026 and HTML entities)
    raw_urls = {html_lib.unescape(u) for u in _MEDIA_URL_PATTERN.findall(html)}
    feedshare: list[str] = []
    displayphoto: list[str] = []
    other: list[str] = []
    for u in raw_urls:
        base = u.split("?")[0]
        if base.endswith(".svg") or "framing" in base.lower():
            continue
        if _FEEDSHARE_MARK in u:
            feedshare.append(u)
        elif _DISPLAYPHOTO_MARK in u:
            # dedupe by photo id path (scale variants duplicate)
            key = base.rsplit("/", 0)[0]
            displayphoto.append(u)
        else:
            other.append(u)

    # Dedupe display photos by their unique id segment (before /0/ or /1/)
    seen_ids: set[str] = set()
    uniq_display: list[str] = []
    for u in displayphoto:
        m = re.search(r"/dms/image/(?:v2/)?([A-Za-z0-9_-]+)/profile-displayphoto", u)
        key = m.group(1) if m else u
        if key not in seen_ids:
            seen_ids.add(key)
            uniq_display.append(u)

    content.photos = feedshare + uniq_display + other[:2]

    # Associate profile slugs (author + tagged/commenting members)
    slugs: list[str] = []
    seen_slugs: set[str] = set()
    for slug in _PROFILE_LINK_PATTERN.findall(html):
        s = slug.strip().lower()
        if s in _RESERVED_SLUGS or s in seen_slugs:
            continue
        seen_slugs.add(s)
        slugs.append(s)
    content.profile_slugs = slugs

    return content


def _post_slug_from_url(post_url: str) -> Optional[str]:
    m = _POST_SLUG_PATTERN.search(post_url or "")
    return m.group(1) if m else None


def is_linkedin_post_url(url: str) -> bool:
    return bool(url) and "linkedin.com/posts/" in url


def harvest_linkedin_post(
    post_url: str,
    session: Optional[requests.Session] = None,
    timeout: float = 12.0,
    browser_fetcher: Optional[Callable[[str, float], str]] = None,
    max_photos: int = 4,
) -> list[Candidate]:
    """
    Fetch one public LinkedIn post URL and return validated image Candidates.

    requests is the primary tier (guest HTML works). If the response is
    blocked/empty and a browser_fetcher is supplied, the page is re-rendered
    through it. Profile pages are intentionally out of scope (login wall).
    """
    if not is_linkedin_post_url(post_url):
        return []

    owns_session = False
    if session is None:
        session = requests.Session()
        owns_session = True

    html = ""
    try:
        resp = session.get(
            post_url,
            timeout=timeout,
            headers={
                "User-Agent": _BROWSER_UA,
                "Accept-Language": "en-US,en;q=0.9",
            },
            allow_redirects=True,
        )
        if resp.status_code == 200 and "media.licdn.com" in resp.text:
            html = resp.text
        else:
            logger.debug(
                "LinkedIn post fetch degraded: status=%s url=%s",
                resp.status_code, resp.url if resp else "?",
            )
    except Exception as e:
        logger.debug("LinkedIn post fetch failed: %s", e)
    finally:
        if owns_session:
            try:
                session.close()
            except Exception:
                pass

    if not html and browser_fetcher is not None:
        try:
            html = browser_fetcher(post_url, timeout)
        except Exception as e:
            logger.debug("LinkedIn browser escalation failed: %s", e)

    if not html:
        return []

    content = extract_linkedin_content(html, post_url)
    author_slug = _post_slug_from_url(post_url) or ""

    candidates: list[Candidate] = []
    for img_url in content.photos[:max_photos]:
        title = content.title or "LinkedIn Post"
        if _DISPLAYPHOTO_MARK in img_url:
            label = f"LinkedIn Profile Photo ({author_slug or 'member'})"
        else:
            label = f"LinkedIn Post Image ({author_slug or 'member'})"
        candidates.append(
            Candidate(
                image_url=img_url,
                source_url=post_url,
                title=label,
                domain="linkedin.com",
            )
        )
    _ = content.text  # retained on LinkedInPostContent for future forensic use
    return candidates


def harvest_associate_slugs(
    post_url: str,
    session: Optional[requests.Session] = None,
    timeout: float = 12.0,
    exclude: Optional[set[str]] = None,
) -> list[str]:
    """
    Return member slugs (author + associates) discovered on a public post page.
    These feed cross-platform handle sweeps and associate forensics.
    """
    owns = False
    if session is None:
        session = requests.Session()
        owns = True
    try:
        resp = session.get(
            post_url,
            timeout=timeout,
            headers={"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        content = extract_linkedin_content(resp.text, post_url)
    except Exception:
        return []
    finally:
        if owns:
            try:
                session.close()
            except Exception:
                pass

    ex = {e.strip().lower() for e in (exclude or set())}
    return [s for s in content.profile_slugs if s not in ex]
