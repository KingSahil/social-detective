"""Tests for search providers (mock only — no API key required)."""

import json
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


class TestTwitterProfileProvider:
    @patch("requests.get")
    def test_twitter_timeline_extraction(self, mock_get):
        from app.search import TwitterProfileProvider

        html_mock = """
        <html>
          <body>
            <a href="/testuser/status/1234567890/photo/1">
              <img src="https://pbs.twimg.com/media/sample_media_1?format=jpg&name=large" />
            </a>
            <a href="/testuser/status/9876543210/photo/1">
              <img src="https://pbs.twimg.com/media/sample_media_2.jpg" />
            </a>
          </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html_mock
        mock_get.return_value = mock_resp

        provider = TwitterProfileProvider("@testuser")
        res = provider.search()

        assert res.provider == "Twitter / X Profile Discovery"
        assert len(res.candidates) == 2
        assert res.candidates[0].source_url == "https://x.com/testuser/status/1234567890"
        assert "sample_media_1" in res.candidates[0].image_url
        assert res.candidates[1].source_url == "https://x.com/testuser/status/9876543210"
        assert "sample_media_2" in res.candidates[1].image_url

    def test_extract_social_handles(self):
        from app.search import extract_social_handles, Candidate

        candidates = [
            Candidate(image_url="http://example.com/1.jpg", source_url="https://in.linkedin.com/in/khannasparsh", domain="linkedin.com"),
            Candidate(image_url="http://example.com/2.jpg", source_url="https://www.linkedin.com/posts/kingsahil_startup-activity-12345", domain="linkedin.com"),
            Candidate(image_url="http://example.com/3.jpg", source_url="https://www.instagram.com/supreme__sahil/", domain="instagram.com"),
            Candidate(image_url="http://example.com/4.jpg", source_url="https://x.com/Aryannn_6476476/status/1234", domain="x.com"),
            Candidate(image_url="http://example.com/5.jpg", source_url="https://github.com/torvalds", domain="github.com"),
            Candidate(image_url="http://example.com/6.jpg", source_url="https://example.com/blog", title="Check out @cooldev on twitter", domain="example.com"),
        ]

        handles = extract_social_handles(candidates)
        assert "khannasparsh" in handles
        assert "kingsahil" in handles
        assert "supreme__sahil" in handles
        assert "aryannn_6476476" in handles
        assert "torvalds" in handles
        assert "cooldev" in handles

    def test_find_social_handles_from_subject_memory(self, tmp_path):
        import json
        import numpy as np
        from app.search import find_social_handles_from_subject_memory

        dummy_img = tmp_path / "sample.jpg"
        dummy_img.write_text("fake image content")

        record = {
            "query": {"image": str(dummy_img)},
            "match": {"source_url": "https://www.linkedin.com/posts/kingsahil_post-123"},
            "content": {
                "source_url": "https://x.com/supreme__sahil/status/11223344",
                "text": "Check out @blinky_ai on IG!",
                "title": "Sahil on X",
            },
        }
        rec_file = tmp_path / "record1.json"
        rec_file.write_text(json.dumps(record), encoding="utf-8")

        mock_fp = MagicMock()
        emb = np.ones(512, dtype=np.float32)
        mock_fp.get_embedding.return_value = emb

        recalled = find_social_handles_from_subject_memory(
            query_embedding=emb,
            results_dir=tmp_path,
            similarity_threshold=0.58,
            fp=mock_fp,
        )

        assert "kingsahil" in recalled
        assert "supreme__sahil" in recalled
        assert "blinky_ai" in recalled


    def test_extract_associate_network_leads(self, tmp_path):
        from app.search import extract_associate_network_leads

        record = {
            "content": {
                "title": "#hackhazards26 by Namespace | Sahil Gupta",
                "text": "Excited to compete with Sparsh Khanna, Gourish Julka, and Meharwan Singh!",
            },
            "match": {"title": "Sahil Gupta on LinkedIn"},
        }
        rec_file = tmp_path / "rec.json"
        rec_file.write_text(json.dumps(record), encoding="utf-8")

        names, contexts = extract_associate_network_leads(results_dir=tmp_path)
        assert "Sparsh Khanna" in names
        assert "Gourish Julka" in names
        assert "Meharwan Singh" in names
        assert "Hackhazards 26" in contexts or "Namespace" in contexts


    def test_linkedin_post_provider_requires_api_key(self):
        from app.search import LinkedInPostProvider
        with pytest.raises(RuntimeError, match="SERPAPI_KEY is not set"):
            LinkedInPostProvider(api_key=None)


    def test_linkedin_post_provider_mocked_search(self, monkeypatch):
        from app.search import LinkedInPostProvider
        import types

        mock_serp = MagicMock()
        mock_client = MagicMock()
        mock_serp.Client.return_value = mock_client
        mock_client.search.return_value = {
            "organic_results": [
                {
                    "link": "https://www.linkedin.com/posts/khannasparsh_hackhazards-activity-7467434164741517312-abcd",
                    "title": "Sparsh post",
                }
            ]
        }
        monkeypatch.setattr("serpapi.Client", mock_serp.Client)

        # Mock requests.get for og:image
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <html><head>
        <meta property="og:image" content="https://media.licdn.com/dms/image/v2/test.jpg" />
        <meta property="og:title" content="Sparsh Khanna's Post" />
        </head></html>
        """
        monkeypatch.setattr("requests.get", lambda *a, **kw: mock_resp)

        prov = LinkedInPostProvider(api_key="valid_key")
        res = prov.search_leads(names=["Sparsh Khanna"], contexts=["Hackhazards"])
        assert len(res.candidates) == 1
        assert res.candidates[0].domain == "linkedin.com"
        assert res.candidates[0].image_url == "https://media.licdn.com/dms/image/v2/test.jpg"
        assert "7467434164741517312" in res.candidates[0].source_url


    def test_instagram_profile_provider_requires_api_key(self):
        from app.search import InstagramProfileProvider
        with pytest.raises(RuntimeError, match="SERPAPI_KEY is not set"):
            InstagramProfileProvider(api_key=None)


    def test_instagram_profile_provider_mocked_search(self, monkeypatch):
        from app.search import InstagramProfileProvider
        import instaloader

        mock_serp = MagicMock()
        mock_client = MagicMock()
        mock_serp.Client.return_value = mock_client
        mock_client.search.return_value = {
            "organic_results": [
                {
                    "link": "https://www.instagram.com/p/DNQM2qFvvTv/",
                    "title": "Sahil Gupta | @raghavsharma1504 , @shekhar_shashank07 | Instagram",
                    "snippet": "Friends meetup at GNDU",
                }
            ]
        }
        monkeypatch.setattr("serpapi.Client", mock_serp.Client)

        # Mock instaloader Post
        mock_node1 = MagicMock()
        mock_node1.display_url = "https://instagram.fluh/slide1.webp"
        mock_node2 = MagicMock()
        mock_node2.display_url = "https://instagram.fluh/slide2.webp"

        mock_post = MagicMock()
        mock_post.typename = "GraphSidecar"
        mock_post.owner_username = "supreme__sahil"
        mock_post.caption = "@raghavsharma1504 , @shekhar_shashank07"
        mock_post.get_sidecar_nodes.return_value = [mock_node1, mock_node2]

        monkeypatch.setattr("instaloader.Post.from_shortcode", lambda ctx, sc: mock_post)

        prov = InstagramProfileProvider(api_key="valid_key")
        res = prov.search_handles(["supreme__sahil"])

        assert len(res.candidates) == 2
        assert res.candidates[0].domain == "instagram.com"
        assert "DNQM2qFvvTv" in res.candidates[0].source_url
        assert "img_index=1" in res.candidates[0].source_url
        assert res.candidates[0].image_url == "https://instagram.fluh/slide1.webp"
        assert "img_index=2" in res.candidates[1].source_url

    def test_instagram_profile_provider_unwraps_google_redirect(self, monkeypatch):
        from app.search import InstagramProfileProvider

        mock_serp = MagicMock()
        mock_client = MagicMock()
        mock_serp.Client.return_value = mock_client
        mock_client.search.return_value = {
            "organic_results": [
                {
                    "link": "https://www.google.com/url?q=https%3A%2F%2Fwww.instagram.com%2Freel%2FDbvdVHXOLSG%2F&sa=U",
                    "title": "supreme__sahil on Instagram: ML demo",
                    "snippet": "Custom implementation",
                    "displayed_link": "instagram.com/reel",
                }
            ]
        }
        monkeypatch.setattr("serpapi.Client", mock_serp.Client)

        mock_post = MagicMock()
        mock_post.typename = "GraphVideo"
        mock_post.owner_username = "supreme__sahil"
        mock_post.caption = "Custom implementation"
        mock_post.url = "https://instagram.fluh/cover.jpg"

        monkeypatch.setattr("instaloader.Post.from_shortcode", lambda ctx, sc: mock_post)

        prov = InstagramProfileProvider(api_key="valid_key")
        res = prov.search_handles(["supreme__sahil"])

        assert len(res.candidates) == 1
        assert "DbvdVHXOLSG" in res.candidates[0].source_url
        assert res.candidates[0].image_url == "https://instagram.fluh/cover.jpg"

    def test_extract_social_handles_from_instagram_titles(self):
        from app.search import Candidate, extract_social_handles

        candidates = [
            Candidate(
                image_url="https://cdn.example.com/1.jpg",
                source_url="https://www.instagram.com/p/DbvdVHXOLSG/",
                title="supreme__sahil on Instagram: \"Cool project\"",
                domain="instagram.com",
            ),
            Candidate(
                image_url="https://cdn.example.com/2.jpg",
                source_url="https://www.instagram.com/reel/XYZ123/",
                title="Instagram post by dev_expert",
                domain="instagram.com",
            ),
        ]
        handles = extract_social_handles(candidates)
        assert "supreme__sahil" in handles
        assert "dev_expert" in handles



