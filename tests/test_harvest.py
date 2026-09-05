"""
Tests for public content harvester, image validation, extractors, and Candidate generation.
100% offline — pure mocks and synthetic image generation.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from app.identity import AccountHit
from app.harvest import (
    validate_and_decode_image,
    is_generic_placeholder,
    extract_og_image_from_html,
    extract_github_avatar,
    extract_gravatar_avatar,
    extract_duolingo_avatar,
    PublicContentHarvester,
)
from app.search import UsernameSweepProvider


def _create_synthetic_image(width=100, height=100, fmt="JPEG") -> bytes:
    """Helper creating valid in-memory image bytes."""
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestImageValidation:
    def test_valid_jpeg(self):
        img_bytes = _create_synthetic_image(120, 120, "JPEG")
        arr = validate_and_decode_image(img_bytes)
        assert arr is not None
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (120, 120, 3)

    def test_valid_png(self):
        img_bytes = _create_synthetic_image(80, 80, "PNG")
        arr = validate_and_decode_image(img_bytes)
        assert arr is not None
        assert arr.shape == (80, 80, 3)

    def test_valid_webp(self):
        img_bytes = _create_synthetic_image(90, 90, "WEBP")
        arr = validate_and_decode_image(img_bytes)
        assert arr is not None
        assert arr.shape == (90, 90, 3)

    def test_reject_svg(self):
        svg_bytes = b"<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'><circle cx='50' cy='50' r='40'/></svg>"
        assert validate_and_decode_image(svg_bytes) is None

    def test_reject_sub_minimum_dimensions(self):
        # 40x40 is below MIN_IMAGE_DIM=60
        tiny_bytes = _create_synthetic_image(40, 40, "JPEG")
        assert validate_and_decode_image(tiny_bytes) is None

    def test_generic_placeholders(self):
        assert is_generic_placeholder("https://gravatar.com/avatar/00000000000000000000000000000000") is True
        assert is_generic_placeholder("https://github.com/identicons/user.png") is True
        assert is_generic_placeholder("https://site.com/default-avatar.jpg") is True
        assert is_generic_placeholder("https://pbs.twimg.com/profile_images/123/my_photo.jpg") is False


class TestStructuredExtractors:
    def test_github_avatar_extractor(self):
        hit = AccountHit(
            site_name="GitHub",
            category="coding",
            profile_url="https://github.com/torvalds",
        )
        session = MagicMock()
        url = extract_github_avatar(hit, session, 5.0)
        assert url == "https://github.com/torvalds.png"

    def test_gravatar_extractor(self):
        hit = AccountHit(
            site_name="Gravatar",
            category="social",
            profile_url="https://gravatar.com/205e460b479e2e5b48aec07710c08d50",
        )
        session = MagicMock()
        url = extract_gravatar_avatar(hit, session, 5.0)
        assert url == "https://gravatar.com/avatar/205e460b479e2e5b48aec07710c08d50?s=400&d=404"

    def test_duolingo_extractor(self):
        hit = AccountHit(
            site_name="Duolingo",
            category="social",
            profile_url="https://www.duolingo.com/profile/duo_user",
        )
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"users": [{"picture": "//images.duolingo.com/avatar123"}]}
        session.get.return_value = resp

        url = extract_duolingo_avatar(hit, session, 5.0)
        assert url == "https://images.duolingo.com/avatar123"


class TestGenericHTMLExtraction:
    def test_og_image_parsed(self):
        html = """
        <html>
        <head>
            <meta property="og:image" content="https://example.com/social_preview.jpg" />
        </head>
        <body>
            <img class="user-avatar-large" src="/avatar_pic.png" />
            <img class="footer-logo" src="/logo.svg" />
        </body>
        </html>
        """
        imgs = extract_og_image_from_html(html, "https://example.com/user/profile")
        assert "https://example.com/social_preview.jpg" in imgs
        assert "https://example.com/avatar_pic.png" in imgs
        assert not any("logo.svg" in u for u in imgs)


class TestPublicContentHarvester:
    def test_harvester_resolves_and_validates_candidates(self):
        valid_img = _create_synthetic_image(100, 100, "JPEG")

        session = MagicMock()

        def fake_get(url, *args, **kwargs):
            resp = MagicMock()
            if "github.com" in url and url.endswith(".png"):
                resp.status_code = 200
                resp.content = valid_img
                resp.headers = {"Content-Type": "image/png"}
            elif "dev.to" in url:
                resp.status_code = 200
                resp.text = '<html><head><meta property="og:image" content="https://dev.to/avatar.jpg" /></head></html>'
                resp.headers = {"Content-Type": "text/html"}
            elif "dev.to/avatar.jpg" in url:
                resp.status_code = 200
                resp.content = valid_img
                resp.headers = {"Content-Type": "image/jpeg"}
            else:
                resp.status_code = 404
                resp.content = b""
                resp.headers = {}
            return resp

        session.get.side_effect = fake_get

        hits = [
            AccountHit(site_name="GitHub", category="coding", profile_url="https://github.com/johndoe"),
            AccountHit(site_name="dev.to", category="blog", profile_url="https://dev.to/johndoe"),
        ]

        harvester = PublicContentHarvester(max_workers=2)
        candidates = harvester.harvest(hits, session=session)

        assert len(candidates) >= 1
        urls = [c.image_url for c in candidates]
        assert "https://github.com/johndoe.png" in urls

    def test_browser_fallback_trigger_when_enabled(self):
        valid_img = _create_synthetic_image(100, 100, "JPEG")
        fake_browser_fetch = MagicMock()
        fake_browser_fetch.return_value = '<html><head><meta property="og:image" content="https://botwall.com/pic.jpg" /></head></html>'

        session = MagicMock()

        def fake_get(url, *args, **kwargs):
            resp = MagicMock()
            if "botwall.com/user" in url:
                resp.status_code = 403  # bot block
                resp.text = "Forbidden"
            elif "botwall.com/pic.jpg" in url:
                resp.status_code = 200
                resp.content = valid_img
                resp.headers = {"Content-Type": "image/jpeg"}
            return resp

        session.get.side_effect = fake_get

        hits = [
            AccountHit(site_name="BotWallSite", category="social", profile_url="https://botwall.com/user"),
        ]

        harvester = PublicContentHarvester(
            browser_fallback=True,
            browser_fetcher=fake_browser_fetch,
            max_workers=1,
        )
        candidates = harvester.harvest(hits, session=session)

        assert fake_browser_fetch.called
        assert len(candidates) == 1
        assert candidates[0].image_url == "https://botwall.com/pic.jpg"


class TestUsernameSweepProvider:
    @patch("app.identity.UsernameSweepEngine.sweep")
    @patch("app.harvest.PublicContentHarvester.harvest")
    def test_provider_search_flow(self, mock_harvest, mock_sweep):
        from app.search import Candidate

        mock_sweep.return_value = [
            AccountHit(site_name="GitHub", category="coding", profile_url="https://github.com/johndoe")
        ]
        mock_harvest.return_value = [
            Candidate(
                image_url="https://github.com/johndoe.png",
                source_url="https://github.com/johndoe",
                title="GitHub Profile",
                domain="github.com",
            )
        ]

        provider = UsernameSweepProvider(handles=["johndoe"])
        res = provider.search()

        assert res.provider == "Username Sweep (WhatsMyName)"
        assert len(res.candidates) == 1
        assert res.candidates[0].domain == "github.com"
        assert res.raw_response is not None
        assert res.raw_response["hits_count"] == 1

    def test_provider_empty_handles(self):
        provider = UsernameSweepProvider(handles=[])
        res = provider.search()
        assert res.candidates == []
        assert res.raw_response is not None
        assert res.raw_response["hits_count"] == 0
