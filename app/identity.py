"""
Identity pivot & username sweep engine.

Provides:
    AccountHit            — discovered account metadata
    sanitize_handle       — handle sanitization & safety guard
    load_wmn_sites        — loader for vendored WhatsMyName dataset
    evaluate_wmn_rule     — rule checker for WMN sites
    UsernameSweepEngine   — concurrent sweep engine across WMN sites
    doctor_pivot          — offline integrity validation for vendored data
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from app.config import (
    DATA_DIR,
    PIVOT_BROWSER_FALLBACK,
    PIVOT_EXHAUSTIVE,
    PIVOT_MAX_ACCOUNTS,
    PIVOT_MAX_SITES,
    PIVOT_MAX_WORKERS,
    PIVOT_SWEEP_TIMEOUT,
    PIVOT_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Sets
# ---------------------------------------------------------------------------

MAX_HTML_BYTES = 512 * 1024  # 512 KB

HANDLE_STOP_WORDS = {
    "admin", "administrator", "contact", "support", "official", "user", "users",
    "john", "alex", "test", "home", "about", "login", "signup", "search",
    "explore", "profile", "account", "settings", "null", "undefined", "anonymous",
    "help", "privacy", "terms", "root", "guest", "info", "security"
}

# Priority categories for Tier 1 default sweep
TIER1_CATEGORIES = [
    "social", "coding", "tech", "images", "art", "blog", "music", "video", "gaming", "hobby"
]

# Protections that must be skipped in direct HTTP mode unless browser escalation is active
BLOCKED_PROTECTIONS = {"captcha", "cloudflare", "multiple", "ddos-guard", "anubis"}

# Default User-Agent to avoid immediate 403 blocks on basic web servers
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AccountHit:
    """Discovered account representation from a username sweep."""
    site_name: str
    category: str
    profile_url: str
    avatar_url: str = ""
    evidence_type: str = ""
    raw_evidence: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Handle Sanitization (EC1 - EC4)
# ---------------------------------------------------------------------------

def sanitize_handle(handle: str) -> Optional[str]:
    """
    Sanitize and validate a candidate social username:
      - NFKC Unicode normalization
      - Strip leading '@', whitespace, and trailing punctuation
      - Keep alphanumeric, dot, underscore, dash
      - Enforce bounds: 3 <= len <= 30
      - Reject pure numeric handles < 5 digits
      - Reject known generic stop words
    """
    if not handle or not isinstance(handle, str):
        return None

    # NFKC normalization
    h = unicodedata.normalize("NFKC", handle.strip())

    # Strip leading @ or common prefixes
    h = h.lstrip("@").strip()

    # Strip trailing punctuation (e.g. from title regex captures like "Jane Doe (@janedoe).")
    h = h.rstrip(".,:;!?'\"`()[]{}")

    # Regex filter: keep [A-Za-z0-9._-]
    h = re.sub(r"[^\w.-]", "", h)

    # Convert to lowercase for uniform processing and deduplication
    h_lower = h.lower()

    # Bounds check
    if not (3 <= len(h_lower) <= 30):
        return None

    # Reject pure numeric handles shorter than 5 chars (often ports/status codes/years)
    if h_lower.isdigit() and len(h_lower) < 5:
        return None

    # Reject stop words
    if h_lower in HANDLE_STOP_WORDS:
        return None

    return h_lower


# ---------------------------------------------------------------------------
# Dataset loader & validation
# ---------------------------------------------------------------------------

def load_wmn_sites(data_path: Optional[str | Path] = None) -> list[dict]:
    """
    Loads vendored WhatsMyName site rules from disk.
    Performs offline schema validation.
    """
    if data_path is None:
        path = DATA_DIR / "wmn" / "wmn-data.json"
    else:
        path = Path(data_path)

    if not path.exists():
        logger.warning("WhatsMyName dataset not found at %s", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load WhatsMyName dataset: %s", e)
        return []

    sites = data.get("sites", [])
    valid_sites: list[dict] = []

    for s in sites:
        if not isinstance(s, dict):
            continue
        # Check required fields
        if not s.get("name") or not s.get("uri_check"):
            continue
        if "e_code" not in s:
            continue
        valid_sites.append(s)

    return valid_sites


def doctor_pivot() -> dict:
    """
    Offline health check verifying dataset integrity and availability.
    """
    wmn_path = DATA_DIR / "wmn" / "wmn-data.json"
    meta_path = DATA_DIR / "wmn" / "wmn-metadata.json"
    attr_path = DATA_DIR / "wmn" / "ATTRIBUTION.md"

    report = {
        "wmn_data_exists": wmn_path.exists(),
        "wmn_metadata_exists": meta_path.exists(),
        "attribution_exists": attr_path.exists(),
        "valid_site_count": 0,
        "categories": [],
        "status": "fail",
    }

    if not wmn_path.exists():
        return report

    try:
        sites = load_wmn_sites(wmn_path)
        report["valid_site_count"] = len(sites)
        cats = sorted({s.get("cat", "misc") for s in sites if s.get("cat")})
        report["categories"] = cats
        if len(sites) >= 500:
            report["status"] = "ok"
    except Exception as e:
        report["error"] = str(e)

    return report


# ---------------------------------------------------------------------------
# WMN Rule Evaluation (EC5 - EC8, EC10, EC11)
# ---------------------------------------------------------------------------

def _fetch_url_safe(
    session: requests.Session,
    url: str,
    timeout: float = PIVOT_TIMEOUT,
    headers: Optional[dict] = None,
) -> tuple[int, str, str]:
    """
    Perform a bounded GET request streaming up to MAX_HTML_BYTES.
    Returns (status_code, text, final_url).
    """
    req_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    if headers:
        req_headers.update(headers)

    response = session.get(url, headers=req_headers, timeout=timeout, stream=True, allow_redirects=True)
    status_code = response.status_code
    final_url = str(response.url)

    # Read up to MAX_HTML_BYTES
    content_chunks = []
    total_bytes = 0
    for chunk in response.iter_content(chunk_size=8192):
        content_chunks.append(chunk)
        total_bytes += len(chunk)
        if total_bytes >= MAX_HTML_BYTES:
            break

    raw_data = b"".join(content_chunks)

    # Character encoding handling (EC11)
    encoding = response.encoding
    if not encoding or encoding.lower() == "iso-8859-1":
        encoding = response.apparent_encoding or "utf-8"

    try:
        text = raw_data.decode(encoding, errors="replace")
    except Exception:
        text = raw_data.decode("utf-8", errors="replace")

    return status_code, text, final_url


def evaluate_wmn_rule(
    status_code: int,
    text: str,
    final_url: str,
    site: dict,
) -> tuple[bool, str]:
    """
    Evaluate if a response matches the WhatsMyName existence criteria.
    Returns (is_match, evidence_description).

    Rules:
      - e_code: expected existence status code (usually 200)
      - e_string: expected existence string in response body (if specified)
      - m_code: missing status code (e.g. 404, 302)
      - m_string: missing string indicating user not found
      - redirect handling: redirects to login / home without e_string are considered missing (soft-404)
    """
    e_code = site.get("e_code")
    e_string = site.get("e_string", "")
    m_code = site.get("m_code")
    m_string = site.get("m_string", "")

    # 1. Missing string check: if m_string is defined and present in body -> definitely missing
    if m_string and m_string in text:
        return False, "m_string_present"

    # 2. Missing code check: if m_code is matched -> definitely missing
    if m_code and status_code == m_code:
        return False, "m_code_matched"

    # 3. Soft-404 / Auth redirect checks
    # If final URL was redirected to root, login, register, signin, or 404 page
    parsed_final = urllib.parse.urlparse(final_url)
    path_lower = parsed_final.path.lower()
    if any(auth_path in path_lower for auth_path in ("/login", "/signin", "/auth", "/register", "/signup", "/404")):
        if e_string and e_string not in text:
            return False, "auth_redirect_without_e_string"

    # 4. Existence status code check
    if status_code != e_code:
        return False, f"status_code_mismatch({status_code}!={e_code})"

    # 5. Existence string check
    if e_string:
        if e_string in text:
            return True, f"e_code={status_code},e_string_matched"
        else:
            return False, f"e_string_missing"

    # Status-only match
    return True, f"e_code={status_code}"


# ---------------------------------------------------------------------------
# Sweep Engine (EC9, EC20, EC21)
# ---------------------------------------------------------------------------

class UsernameSweepEngine:
    """
    Concurrent native Python sweep engine across WhatsMyName sites.
    """

    def __init__(
        self,
        sites: Optional[list[dict]] = None,
        max_workers: int = PIVOT_MAX_WORKERS,
        timeout: float = PIVOT_TIMEOUT,
        sweep_timeout: float = PIVOT_SWEEP_TIMEOUT,
        max_accounts: int = PIVOT_MAX_ACCOUNTS,
        max_sites: int = PIVOT_MAX_SITES,
        exhaustive: bool = PIVOT_EXHAUSTIVE,
        browser_fallback: bool = PIVOT_BROWSER_FALLBACK,
    ):
        self.sites = sites if sites is not None else load_wmn_sites()
        self.max_workers = max_workers
        self.timeout = timeout
        self.sweep_timeout = sweep_timeout
        self.max_accounts = max_accounts
        self.max_sites = max_sites
        self.exhaustive = exhaustive
        self.browser_fallback = browser_fallback

    def _filter_and_order_sites(self) -> list[dict]:
        """
        Filters and orders sites based on Tier 1 categories, protections, and caps.
        """
        filtered: list[dict] = []
        tier1_bucket: list[dict] = []
        other_bucket: list[dict] = []

        for s in self.sites:
            cat = s.get("cat", "misc")
            if cat == "xx NSFW xx" or cat == "archived":
                continue

            protections = set(s.get("protection") or [])
            # If site requires protection bypass and browser fallback is disabled, skip
            if not self.browser_fallback and (protections & BLOCKED_PROTECTIONS):
                continue

            if cat in TIER1_CATEGORIES:
                tier1_bucket.append(s)
            else:
                other_bucket.append(s)

        # Sort tier1 by category priority index
        def _cat_sort_key(site: dict) -> int:
            cat = site.get("cat", "misc")
            try:
                return TIER1_CATEGORIES.index(cat)
            except ValueError:
                return 999

        tier1_bucket.sort(key=_cat_sort_key)

        if self.exhaustive:
            filtered = tier1_bucket + other_bucket
        else:
            filtered = tier1_bucket

        return filtered[: self.max_sites]

    def _check_site(
        self,
        site: dict,
        handle: str,
        session: requests.Session,
    ) -> Optional[AccountHit]:
        """
        Worker checking a single site for username existence.
        """
        uri_check = site.get("uri_check", "")
        if not uri_check or "{account}" not in uri_check:
            return None

        # URL parameter encoding for account (EC5)
        encoded_handle = urllib.parse.quote(handle, safe="")
        target_url = uri_check.replace("{account}", encoded_handle)

        headers = site.get("headers") or {}

        try:
            status_code, text, final_url = _fetch_url_safe(
                session, target_url, timeout=self.timeout, headers=headers
            )
            is_match, evidence = evaluate_wmn_rule(status_code, text, final_url, site)
            if is_match:
                # Construct profile URL
                profile_url = site.get("uri_pretty", "")
                if profile_url and "{account}" in profile_url:
                    profile_url = profile_url.replace("{account}", encoded_handle)
                else:
                    profile_url = target_url

                return AccountHit(
                    site_name=site.get("name", "Unknown"),
                    category=site.get("cat", "misc"),
                    profile_url=profile_url,
                    evidence_type="wmn_direct",
                    raw_evidence=evidence,
                )
        except Exception as e:
            logger.debug("Site check failed for %s (%s): %s", site.get("name"), target_url, e)
            return None

        return None

    def sweep(
        self,
        handle: str,
        session: Optional[requests.Session] = None,
    ) -> list[AccountHit]:
        """
        Executes a concurrent sweep for the given handle.
        Returns up to PIVOT_MAX_ACCOUNTS AccountHits.
        """
        clean_handle = sanitize_handle(handle)
        if not clean_handle:
            logger.debug("Handle '%s' rejected by sanitization", handle)
            return []

        target_sites = self._filter_and_order_sites()
        if not target_sites:
            return []

        hits: list[AccountHit] = []
        owns_session = False
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=self.max_workers, pool_maxsize=self.max_workers)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            owns_session = True

        import time
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_site = {
                    executor.submit(self._check_site, site, clean_handle, session): site
                    for site in target_sites
                }

                for future in as_completed(future_to_site):
                    # Check global sweep timeout (EC21)
                    if (time.time() - start_time) > self.sweep_timeout:
                        logger.warning("Username sweep timed out after %.1fs; returning partial results", self.sweep_timeout)
                        for pending_fut in future_to_site:
                            pending_fut.cancel()
                        break

                    try:
                        hit = future.result()
                        if hit:
                            hits.append(hit)
                            if len(hits) >= self.max_accounts:
                                logger.info("Max account hits (%d) reached for handle @%s", self.max_accounts, clean_handle)
                                for pending_fut in future_to_site:
                                    pending_fut.cancel()
                                break
                    except Exception as e:
                        logger.debug("Error retrieving future result: %s", e)

        except KeyboardInterrupt:
            logger.warning("Username sweep interrupted by user; returning partial results")
        finally:
            if owns_session:
                try:
                    session.close()
                except Exception:
                    pass

        return hits
