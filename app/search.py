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
