"""
Tests for identity models, WMN rule evaluator, handle sanitizer, and sweep engine.
100% offline — pure mocks and deterministic fixtures.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.identity import (
    AccountHit,
    sanitize_handle,
    load_wmn_sites,
    doctor_pivot,
    evaluate_wmn_rule,
    UsernameSweepEngine,
    HANDLE_STOP_WORDS,
)


class TestHandleSanitization:
    def test_basic_handle(self):
        assert sanitize_handle("johndoe") == "johndoe"
        assert sanitize_handle("@johndoe") == "johndoe"
        assert sanitize_handle("  @John_Doe-123  ") == "john_doe-123"

    def test_unicode_normalization(self):
        # Fullwidth latin characters normalized by NFKC
        # NOTE: 'john' alone is a stop-word by design (Pass 2), so use a
        # non-stop-word handle to prove NFKC normalization works.
        assert sanitize_handle("＠ｊｏｈｎｄｏｅ") == "johndoe"

    def test_trailing_punctuation(self):
        assert sanitize_handle("johndoe...") == "johndoe"
        assert sanitize_handle("@johndoe,") == "johndoe"
        assert sanitize_handle("('johndoe')") == "johndoe"

    def test_length_limits(self):
        assert sanitize_handle("ab") is None  # too short (< 3)
        assert sanitize_handle("a" * 31) is None  # too long (> 30)
        assert sanitize_handle("abc") == "abc"
        assert sanitize_handle("a" * 30) == "a" * 30

    def test_numeric_handles(self):
        assert sanitize_handle("123") is None  # numeric < 5 digits rejected
        assert sanitize_handle("8080") is None  # port artifact rejected
        assert sanitize_handle("12345") == "12345"  # 5 digits allowed
        assert sanitize_handle("user123") == "user123"  # alphanumeric allowed

    def test_stop_words(self):
        for word in ("admin", "support", "official", "login", "signup", "explore"):
            assert sanitize_handle(word) is None
            assert sanitize_handle(f"@{word}") is None


class TestWMNRuleEvaluation:
    def test_status_only_positive(self):
        site = {"e_code": 200, "e_string": "", "m_code": 404, "m_string": ""}
        matched, ev = evaluate_wmn_rule(200, "<html>Profile content</html>", "https://site.com/user", site)
        assert matched is True
        assert "e_code=200" in ev

    def test_status_only_negative(self):
        site = {"e_code": 200, "e_string": "", "m_code": 404, "m_string": ""}
        matched, ev = evaluate_wmn_rule(404, "<html>Not Found</html>", "https://site.com/user", site)
        assert matched is False

    def test_status_and_estring_positive(self):
        site = {"e_code": 200, "e_string": "user-avatar", "m_code": 404, "m_string": ""}
        matched, ev = evaluate_wmn_rule(200, "<html><div class='user-avatar'></div></html>", "https://site.com/user", site)
        assert matched is True
        assert "e_string_matched" in ev

    def test_status_and_estring_negative(self):
        site = {"e_code": 200, "e_string": "user-avatar", "m_code": 404, "m_string": ""}
        matched, ev = evaluate_wmn_rule(200, "<html><div>Generic page</div></html>", "https://site.com/user", site)
        assert matched is False
        assert "e_string_missing" in ev

    def test_mstring_detection(self):
        site = {"e_code": 200, "e_string": "", "m_code": 404, "m_string": "User does not exist"}
        matched, ev = evaluate_wmn_rule(200, "<html>User does not exist</html>", "https://site.com/user", site)
        assert matched is False
        assert ev == "m_string_present"

    def test_soft404_login_redirect(self):
        site = {"e_code": 200, "e_string": "profile_header", "m_code": 404, "m_string": ""}
        # Redirected to login page without e_string
        matched, ev = evaluate_wmn_rule(200, "<html>Please log in</html>", "https://site.com/login", site)
        assert matched is False
        assert ev == "auth_redirect_without_e_string"


class TestWMNDatasetAndDoctor:
    def test_doctor_pivot_offline(self):
        report = doctor_pivot()
        assert report["wmn_data_exists"] is True
        assert report["wmn_metadata_exists"] is True
        assert report["attribution_exists"] is True
        assert report["valid_site_count"] >= 500
        assert "social" in report["categories"]
        assert report["status"] == "ok"

    def test_load_wmn_sites(self):
        sites = load_wmn_sites()
        assert len(sites) >= 500
        assert all("name" in s and "uri_check" in s for s in sites)


class TestUsernameSweepEngine:
    def test_sweep_filters_protected_sites_by_default(self):
        fake_sites = [
            {"name": "SiteOpen", "cat": "social", "uri_check": "https://open.com/{account}", "e_code": 200, "protection": []},
            {"name": "SiteCloudflare", "cat": "social", "uri_check": "https://cf.com/{account}", "e_code": 200, "protection": ["cloudflare"]},
        ]
        engine = UsernameSweepEngine(sites=fake_sites, browser_fallback=False)
        targets = engine._filter_and_order_sites()
        assert len(targets) == 1
        assert targets[0]["name"] == "SiteOpen"

    def test_sweep_executes_and_returns_hits(self):
        fake_sites = [
            {"name": "TestSocial", "cat": "social", "uri_check": "https://test.com/{account}", "e_code": 200, "e_string": "welcome"},
            {"name": "MissingSocial", "cat": "social", "uri_check": "https://missing.com/{account}", "e_code": 200, "m_code": 404},
        ]
        engine = UsernameSweepEngine(sites=fake_sites, max_workers=2)

        session = MagicMock()

        def fake_get(url, *args, **kwargs):
            resp = MagicMock()
            if "test.com" in url:
                resp.status_code = 200
                resp.url = url
                resp.iter_content.return_value = [b"<html>welcome to user profile</html>"]
                resp.encoding = "utf-8"
            else:
                resp.status_code = 404
                resp.url = url
                resp.iter_content.return_value = [b"<html>not found</html>"]
                resp.encoding = "utf-8"
            return resp

        session.get.side_effect = fake_get

        hits = engine.sweep("johndoe", session=session)
        assert len(hits) == 1
        assert hits[0].site_name == "TestSocial"
        assert hits[0].profile_url == "https://test.com/johndoe"
        assert hits[0].category == "social"

    def test_sweep_handles_site_exceptions_gracefully(self):
        fake_sites = [
            {"name": "ErrorSite", "cat": "social", "uri_check": "https://error.com/{account}", "e_code": 200},
            {"name": "GoodSite", "cat": "social", "uri_check": "https://good.com/{account}", "e_code": 200, "e_string": "active"},
        ]
        engine = UsernameSweepEngine(sites=fake_sites, max_workers=2)

        session = MagicMock()

        def fake_get(url, *args, **kwargs):
            if "error.com" in url:
                raise requests.ConnectionError("DNS failure")
            resp = MagicMock()
            resp.status_code = 200
            resp.url = url
            resp.iter_content.return_value = [b"active user"]
            resp.encoding = "utf-8"
            return resp

        import requests
        session.get.side_effect = fake_get

        hits = engine.sweep("testuser", session=session)
        assert len(hits) == 1
        assert hits[0].site_name == "GoodSite"
