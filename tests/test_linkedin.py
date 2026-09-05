"""Tests for the LinkedIn public-post harvester (100% offline mocks)."""

import pytest

from app.linkedin import (
    LinkedInPostContent,
    extract_linkedin_content,
    harvest_linkedin_post,
    is_linkedin_post_url,
)
from app.search import Candidate


POST_URL = (
    "https://www.linkedin.com/posts/gourish-julka-472a1632b_well-long-story-"
    "short-activity-7467494971043057664-KWKL"
)

GUEST_HTML = """
<html><head>
<title>Some Post | LinkedIn</title>
<meta property="og:title" content="Well, long story short | Gourish Julka" />
<meta property="og:description" content="A pal named Sahil asked me..." />
<meta property="og:image" content="https://media.licdn.com/dms/image/v2/D5622AQE-AAAA/feedshare-shrink_800/B56ZZ/0/123?e=9&amp;v=beta&amp;t=x" />
</head><body>
<img src="https://media.licdn.com/dms/image/v2/D4E03AQAAAA/profile-displayphoto-scale_400_400/B4EZZ/0/1756715600336?e=9&amp;v=beta" alt="member photo" />
<img src="https://media.licdn.com/dms/image/v2/D4E03AQAAAA/profile-displayphoto-scale_200_200/B4EZZ/0/1756715600336?e=9&amp;v=beta" alt="member photo small" />
<a href="https://in.linkedin.com/in/gourish-julka-472a1632b?trk=x">Gourish Julka</a>
<a href="https://in.linkedin.com/in/kingsahil?trk=y">Sahil Gupta</a>
<a href="https://in.linkedin.com/in/khannasparsh?trk=z">Sparsh Khanna</a>
<a href="https://www.linkedin.com/in/kingsahil?trk=dup">Sahil duplicate domain</a>
<a href="https://www.linkedin.com/feed/">feed link (reserved slug)</a>
</body></html>
"""


class TestURLClassification:
    def test_post_url_detected(self):
        assert is_linkedin_post_url(POST_URL) is True

    def test_profile_url_rejected(self):
        assert is_linkedin_post_url("https://www.linkedin.com/in/kingsahil") is False

    def test_empty_url_rejected(self):
        assert is_linkedin_post_url("") is False

    def test_other_domain_rejected(self):
        assert is_linkedin_post_url("https://example.com/posts/abc_123") is False


class TestExtraction:
    def test_feedshare_photo_ranked_first(self):
        c = extract_linkedin_content(GUEST_HTML, POST_URL)
        assert c.photos, "should extract photos"
        assert "feedshare" in c.photos[0]

    def test_displayphoto_deduped_by_id(self):
        c = extract_linkedin_content(GUEST_HTML, POST_URL)
        display = [p for p in c.photos if "profile-displayphoto" in p]
        # scale_400 and scale_200 share the same id segment -> 1 unique
        assert len(display) == 1

    def test_og_image_html_entities_unescaped(self):
        c = extract_linkedin_content(GUEST_HTML, POST_URL)
        assert all("&amp;" not in p for p in c.photos)

    def test_associate_slugs_deduped_and_reserved_filtered(self):
        c = extract_linkedin_content(GUEST_HTML, POST_URL)
        assert c.profile_slugs == ["gourish-julka-472a1632b", "kingsahil", "khannasparsh"]

    def test_title_and_text_from_og(self):
        c = extract_linkedin_content(GUEST_HTML, POST_URL)
        assert "Gourish Julka" in c.title
        assert "Sahil" in c.text

    def test_empty_html(self):
        c = extract_linkedin_content("", POST_URL)
        assert c.photos == [] and c.profile_slugs == []

    def test_svg_excluded(self):
        html = '<meta property="og:image" content="https://media.licdn.com/dms/image/logo.svg" />'
        c = extract_linkedin_content(html, POST_URL)
        assert c.photos == []


class TestHarvestMocked:
    def test_harvest_builds_candidates(self, monkeypatch):
        class FakeResp:
            status_code = 200
            text = GUEST_HTML

        class FakeSession:
            def get(self, url, **kw):
                return FakeResp()

        cands = harvest_linkedin_post(POST_URL, session=FakeSession())
        assert len(cands) >= 1
        assert all(isinstance(c, Candidate) for c in cands)
        assert all(c.domain == "linkedin.com" for c in cands)
        assert all(c.source_url == POST_URL for c in cands)
        # feedshare (post image) ranked above profile photos
        assert "Post Image" in cands[0].title

    def test_non_post_url_returns_empty(self):
        assert harvest_linkedin_post("https://www.linkedin.com/in/someone") == []

    def test_blocked_response_with_no_fallback_returns_empty(self, monkeypatch):
        import app.linkedin as li

        class BlockedResp:
            status_code = 999
            text = ""

        class FakeSession:
            def get(self, url, **kw):
                return BlockedResp()

        calls = []
        monkeypatch.setattr(
            li, "extract_linkedin_content", lambda *a, **k: calls.append(a)
        )
        assert harvest_linkedin_post(POST_URL, session=FakeSession()) == []
        assert calls == [], "parser must not run on blocked responses"

    def test_browser_fallback_used_when_requests_blocked(self, monkeypatch):
        import app.linkedin as li

        class BlockedResp:
            status_code = 999
            text = ""

        class FakeSession:
            def get(self, url, **kw):
                return BlockedResp()

        def fake_fetcher(url, timeout):
            return GUEST_HTML

        cands = harvest_linkedin_post(
            POST_URL, session=FakeSession(), browser_fetcher=fake_fetcher
        )
        assert len(cands) >= 1

    def test_max_photos_cap(self, monkeypatch):
        class FakeResp:
            status_code = 200
            text = GUEST_HTML

        class FakeSession:
            def get(self, url, **kw):
                return FakeResp()

        cands = harvest_linkedin_post(POST_URL, session=FakeSession(), max_photos=1)
        assert len(cands) == 1
