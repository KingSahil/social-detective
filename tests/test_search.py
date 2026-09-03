"""Tests for search providers (mock only — no API key required)."""

import pytest
from datetime import datetime

from app.search import (
    Candidate,
    SearchResult,
    MockSearchProvider,
    SerpAPIProvider,
    TargetURLProvider,
)
from unittest.mock import patch, MagicMock


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


class TestTargetURLProvider:
    def test_direct_image_url(self):
        provider = TargetURLProvider("https://example.com/avatar.jpg")
        result = provider.search("dummy.jpg")
        assert len(result.candidates) == 1
        assert result.candidates[0].image_url == "https://example.com/avatar.jpg"
        assert result.candidates[0].domain == "example.com"

    @patch("requests.get")
    def test_extract_social_media(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = """
        <html>
        <head>
            <title>Test Post on X</title>
            <meta property="og:image" content="https://example.com/og_preview.jpg" />
        </head>
        <body>
            <img src="https://pbs.twimg.com/media/HPQytT_bUAA24UV.jpg" />
            <img src="https://example.com/photo2.png" />
            <img src="https://example.com/icon.svg" />
        </body>
        </html>
        """
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = TargetURLProvider("https://x.com/user/status/12345")
        result = provider.search("dummy.jpg")

        assert result.provider == "Target URL Inspector"
        image_urls = [c.image_url for c in result.candidates]
        assert "https://pbs.twimg.com/media/HPQytT_bUAA24UV.jpg" in image_urls
        assert "https://example.com/og_preview.jpg" in image_urls
        assert "https://example.com/photo2.png" in image_urls
        # SVG and icon filtered
        assert not any("icon.svg" in u for u in image_urls)

    @patch("instaloader.Post.from_shortcode")
    def test_instagram_carousel_extraction(self, mock_from_shortcode):
        mock_post = MagicMock()
        mock_post.owner_username = "test_user"
        mock_post.caption = "Test caption"
        mock_post.typename = "GraphSidecar"

        node1 = MagicMock()
        node1.display_url = "https://instagram.com/slide1.jpg"
        node2 = MagicMock()
        node2.display_url = "https://instagram.com/slide2.jpg"
        mock_post.get_sidecar_nodes.return_value = [node1, node2]
        mock_from_shortcode.return_value = mock_post

        provider = TargetURLProvider("https://www.instagram.com/p/ABC123xyz/?img_index=1")
        result = provider.search("dummy.jpg")

        assert result.provider == "Instaloader (Instagram)"
        assert len(result.candidates) == 2
        assert result.candidates[0].image_url == "https://instagram.com/slide1.jpg"
        assert result.candidates[1].image_url == "https://instagram.com/slide2.jpg"
        assert "Slide 1" in result.candidates[0].title
        assert "Slide 2" in result.candidates[1].title


class TestYandexProviderInit:
    def test_missing_api_key(self):
        from app.search import YandexProvider
        with pytest.raises(RuntimeError, match="SERPAPI_KEY"):
            YandexProvider(api_key="")

    def test_placeholder_api_key(self):
        from app.search import YandexProvider
        with pytest.raises(RuntimeError, match="SERPAPI_KEY"):
            YandexProvider(api_key="your_serpapi_key_here")

