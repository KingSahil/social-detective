"""
Content retrieval and deterministic canonicalization.

After a matching candidate is selected, this module:
1. Fetches the public page at the source URL.
2. Extracts available metadata (title, description, OG tags).
3. Downloads the actual matched image bytes.
4. Builds a deterministic canonical representation for hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class DiscoveredContent:
    """Public content discovered at a candidate's source URL."""
    source_url: str = ""
    image_url: str = ""
    platform: str = ""
    title: str = ""
    description: str = ""
    text: str = ""
    author: str = ""
    retrieved_at: str = ""
    image_bytes: bytes = field(default=b"", repr=False)


class ContentRetriever:
    """Fetch and canonicalize discovered web content."""

    def __init__(self, timeout: int = 15):
        self._timeout = timeout

    def retrieve(self, source_url: str, image_url: str) -> DiscoveredContent:
        """
        Fetch the source page and image, returning a ``DiscoveredContent``.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        domain = urlparse(source_url).netloc if source_url else ""

        content = DiscoveredContent(
            source_url=source_url,
            image_url=image_url,
            platform=domain,
            retrieved_at=timestamp,
        )

        # Fetch the source page
        try:
            user_agent = (
                "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"
                if "instagram.com" in domain
                else (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            resp = requests.get(source_url, timeout=self._timeout, headers={
                "User-Agent": user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            })
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Title
            title_tag = soup.find("title")
            if title_tag:
                content.title = title_tag.get_text(strip=True)

            # Meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content.description = meta_desc["content"]

            # Open Graph / Twitter description
            og_desc = (
                soup.find("meta", attrs={"property": "og:description"})
                or soup.find("meta", attrs={"name": "twitter:description"})
            )
            if og_desc and og_desc.get("content"):
                content.text = og_desc["content"]
            elif content.description:
                content.text = content.description

            # Check JSON-LD for richer text if available
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string or "")
                    if isinstance(data, dict):
                        text_val = data.get("articleBody") or data.get("text") or data.get("description")
                        if text_val and len(str(text_val)) > len(content.text):
                            content.text = str(text_val)
                except Exception:
                    pass

        except Exception:
            # Page may be unavailable — we still have the image URL
            pass

        # Special handling for Instagram posts via Instaloader
        if "instagram.com" in domain and ("/p/" in source_url or "/reel/" in source_url):
            try:
                import instaloader
                import re
                shortcode_match = re.search(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)", source_url)
                if shortcode_match:
                    shortcode = shortcode_match.group(1)
                    L = instaloader.Instaloader()
                    post = instaloader.Post.from_shortcode(L.context, shortcode)
                    content.author = post.owner_username
                    content.text = post.caption or content.text
                    caption_preview = f': "{post.caption[:100]}..."' if post.caption else ""
                    content.title = f"{post.owner_username} on Instagram{caption_preview}"
            except Exception:
                pass

        # Special handling for Twitter / X posts via oEmbed and FxTwitter
        if ("x.com" in domain or "twitter.com" in domain) and "/status/" in source_url:
            try:
                oembed_url = f"https://publish.twitter.com/oembed?url={source_url}&omit_script=true"
                r_oe = requests.get(oembed_url, timeout=6)
                if r_oe.status_code == 200:
                    oe_data = r_oe.json()
                    content.author = oe_data.get("author_name")
                    raw_html = oe_data.get("html", "")
                    clean_text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
                    if clean_text:
                        content.text = clean_text
                        content.title = f"{content.author} on X: \"{clean_text[:100]}...\""
            except Exception:
                pass

            if not content.text or content.text == content.description:
                try:
                    import re
                    m_tw = re.search(r"(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/(\d+)", source_url)
                    if m_tw:
                        user_h, status_id = m_tw.group(1), m_tw.group(2)
                        fx_url = f"https://api.fxtwitter.com/{user_h}/status/{status_id}"
                        r_fx = requests.get(fx_url, timeout=6)
                        if r_fx.status_code == 200:
                            fx_data = r_fx.json().get("tweet", {})
                            content.author = fx_data.get("author", {}).get("name") or user_h
                            if fx_data.get("text"):
                                content.text = fx_data["text"]
                                content.title = f"{content.author} on X: \"{content.text[:100]}...\""
                except Exception:
                    pass

        # Special handling for LinkedIn posts
        if "linkedin.com" in domain and ("/posts/" in source_url or "/in/" in source_url):
            if not content.author:
                if content.title and "|" in content.title:
                    content.author = content.title.split("|")[-1].strip()
                elif "-activity-" in source_url:
                    import re
                    m_li = re.search(r"linkedin\.com/posts/([A-Za-z0-9_-]+?)_(?:.*)-activity-", source_url)
                    if m_li:
                        author_slug = m_li.group(1)
                        content.author = re.sub(r"-[0-9a-fA-F]+$", "", author_slug).replace("-", " ").title()

        # Download the actual image bytes (for content hashing)
        try:
            img_headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0.0.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            }
            if source_url:
                img_headers["Referer"] = source_url
            img_resp = requests.get(image_url, headers=img_headers, timeout=self._timeout)
            img_resp.raise_for_status()
            content.image_bytes = img_resp.content
        except Exception:
            pass

        return content

    @staticmethod
    def canonicalize(content: DiscoveredContent) -> str:
        """
        Build a deterministic canonical string from the discovered content.

        Includes a hash of the image bytes so that any change to the actual
        image is detected, not just metadata changes.
        """
        # Hash the image bytes so the canonical form captures image content
        image_hash = ""
        if content.image_bytes:
            image_hash = hashlib.sha256(content.image_bytes).hexdigest()

        canonical_dict = {
            "image_hash": image_hash,
            "image_url": content.image_url,
            "source_url": content.source_url,
            "text": content.text,
        }
        # Deterministic: sorted keys, no extra whitespace
        return json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def canonicalize_from_record(record: dict) -> str:
        """
        Reconstruct the canonical string from a saved record JSON.

        The record's ``content`` section must contain the fields used
        during original canonicalization, including ``image_hash``.
        """
        content_section = record.get("content", {})
        match_section = record.get("match", {})

        canonical_dict = {
            "image_hash": content_section.get("image_hash", ""),
            "image_url": match_section.get("image_url", "") or content_section.get("image_url", ""),
            "source_url": match_section.get("source_url", "") or content_section.get("source_url", ""),
            "text": content_section.get("text", ""),
        }
        return json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
