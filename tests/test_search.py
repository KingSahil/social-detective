"""Tests for search providers (mock only — no API key required)."""

import pytest
from datetime import datetime

from app.search import (
    Candidate,
    SearchResult,
    MockSearchProvider,
    SerpAPIProvider,
)


class TestCandidate:
    def test_creation(self):
        c = Candidate(image_url="http://img.com/1.jpg", source_url="http://example.com")
        assert c.image_url == "http://img.com/1.jpg"
        assert c.source_url == "http://example.com"
        assert c.title == ""
        assert c.domain == ""


class TestMockSearchProvider:
    def test_returns_configured_candidates(self):
        candidates = [
            Candidate("http://img.com/1.jpg", "http://a.com", "A", "a.com"),
            Candidate("http://img.com/2.jpg", "http://b.com", "B", "b.com"),
        ]
        provider = MockSearchProvider(candidates)
        result = provider.search("dummy_path.jpg")

        assert isinstance(result, SearchResult)
        assert len(result.candidates) == 2
        assert result.provider == "Mock (test only)"
        assert result.searched_at  # should be non-empty

    def test_empty_candidates(self):
        provider = MockSearchProvider()
        result = provider.search("dummy_path.jpg")
        assert len(result.candidates) == 0


class TestSerpAPIProviderInit:
    def test_missing_api_key(self):
        with pytest.raises(RuntimeError, match="SERPAPI_KEY"):
            SerpAPIProvider(api_key="")

    def test_placeholder_api_key(self):
        with pytest.raises(RuntimeError, match="SERPAPI_KEY"):
            SerpAPIProvider(api_key="your_serpapi_key_here")
