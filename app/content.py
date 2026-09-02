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
            resp = requests.get(source_url, timeout=self._timeout, headers={
                "User-Agent": "Mozilla/5.0 (compatible; FaceTrace/1.0)"
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

            # Open Graph text
            og_desc = soup.find("meta", attrs={"property": "og:description"})
            if og_desc and og_desc.get("content"):
                content.text = og_desc["content"]
            elif content.description:
                content.text = content.description

        except Exception:
            # Page may be unavailable — we still have the image URL
            pass

        # Download the actual image bytes (for content hashing)
        try:
            img_resp = requests.get(image_url, timeout=self._timeout)
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
