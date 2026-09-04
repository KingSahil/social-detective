"""
FaceTrace — Face Search + Blockchain Verification CLI.

End-to-end pipeline:
    Face image → Face detection → Face encoding → Web search →
    Candidate matching → Content retrieval → SHA-256 fingerprint →
    Blockchain record → Verification

Usage:
    python -m app.main --image ./data/input/face.jpg
    python -m app.main --image ./data/input/face.jpg --threshold 0.60
    python -m app.main verify --record ./data/results/record.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging
logging.raiseExceptions = False
for _log_name in ["primp", "h2", "ddgs", "duckduckgo_search", "urllib3", "asyncio"]:
    logging.getLogger(_log_name).setLevel(logging.CRITICAL)

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    C_GREEN = Fore.GREEN
    C_RED = Fore.RED
    C_YELLOW = Fore.YELLOW
    C_CYAN = Fore.CYAN
    C_BOLD = Style.BRIGHT
    C_RESET = Style.RESET_ALL
except ImportError:
    C_GREEN = C_RED = C_YELLOW = C_CYAN = C_BOLD = C_RESET = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    print()
    print("=" * 60)
    print(f"{C_BOLD}               FACETRACE{C_RESET}")
    print(f"      Face Search + Blockchain Verification")
    print("=" * 60, flush=True)
    print()


def _step(num: int, total: int, title: str) -> None:
    print(f"  {C_BOLD}[{num}/{total}] {title}{C_RESET}", flush=True)


def _ok(msg: str) -> None:
    print(f"        {C_GREEN}✓{C_RESET} {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"        {C_RED}✗{C_RESET} {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"        {msg}", flush=True)


def _fatal(msg: str) -> None:
    print()
    _fail(msg)
    print()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    threshold: float = 0.70,
    platform: str | None = None,
    target: str | None = None,
    engine: str = "all",
    handle: str | None = None,
    lens_visible: bool = False,
) -> None:
    """Execute the full FaceTrace pipeline."""

    _banner()

    image_path_obj = Path(image_path).resolve()
    if not image_path_obj.exists():
        _fatal(f"Image not found: {image_path}")

    record: dict = {}
    total_steps = 7

    import time as _time
    _t0 = _time.perf_counter()
    _phase_marks: list[tuple[str, float]] = []

    def _mark(name: str) -> None:
        _phase_marks.append((name, _time.perf_counter() - _t0))

    # Pre-launch web search concurrently with Face Detection to maximize pipeline parallelism
    from concurrent.futures import ThreadPoolExecutor
    search_executor = ThreadPoolExecutor(max_workers=1)
    search_future = None
    prelaunched_search_provider = None

    if not target and not handle:
        from app.config import require_search_config, LENS_HEADLESS
        from app.search import (
            HeadlessLensProvider,
            DirectYandexProvider,
            FreeMultiEngineSearchProvider,
            SerpAPIProvider,
            YandexProvider,
        )
        api_key = require_search_config(optional=True)
        headless = (not lens_visible) and LENS_HEADLESS

        if engine == "serpapi":
            if not api_key:
                _fatal("SERPAPI_KEY is not configured but required for --engine serpapi.")
            prelaunched_search_provider = SerpAPIProvider(api_key=api_key)
        elif engine == "yandex":
            if api_key:
                prelaunched_search_provider = YandexProvider(api_key=api_key)
            else:
                prelaunched_search_provider = DirectYandexProvider()
        elif engine == "lens":
            if api_key:
                prelaunched_search_provider = SerpAPIProvider(api_key=api_key)
            else:
                prelaunched_search_provider = HeadlessLensProvider(headless=headless, fallback_on_captcha=True)
        else:  # "all" - default: SerpAPI Google Lens primary, free multi-engine fallback
            if api_key:
                prelaunched_search_provider = SerpAPIProvider(api_key=api_key)
            else:
                prelaunched_search_provider = FreeMultiEngineSearchProvider(headless=headless)

        search_future = search_executor.submit(prelaunched_search_provider.search, str(image_path_obj))

    # ==================================================================
    # [1/7] FACE DETECTION
    # ==================================================================
    _step(1, total_steps, "FACE DETECTION")

    from app.face import FaceProcessor, FaceProcessingError

    try:
        fp = FaceProcessor()
    except Exception as e:
        _fatal(f"Failed to initialize face processor: {e}")

    try:
        query_embedding = fp.get_embedding(str(image_path_obj))
    except FaceProcessingError as e:
        _fail(str(e))
        print()
        sys.exit(1)
    except Exception as e:
        _fatal(f"Face processing error: {e}")

    _ok("Face detected")
    _ok(f"Face embedding generated ({query_embedding.shape[0]}-d)")
    _mark("face_detect")
    print()

    record["query"] = {
        "image": str(image_path_obj),
        "face_detected": True,
        "embedding_dim": int(query_embedding.shape[0]),
    }

    # ==================================================================
    # [2/7] WEB SEARCH / TARGET MEDIA DISCOVERY
    # ==================================================================
    if target:
        _step(2, total_steps, "TARGET MEDIA DISCOVERY")
        from app.search import TargetURLProvider

        search_provider = TargetURLProvider(target_url=target)
        _info(f"Target: {C_CYAN}{target}{C_RESET}")
        _info("Extracting candidate media images...")

        try:
            search_result = search_provider.search(str(image_path_obj))
        except Exception as e:
            _fatal(f"Failed to extract media from target URL: {e}")

        candidate_count = len(search_result.candidates)
        if candidate_count == 0:
            _fail("No media images found at target URL")
            print()
            sys.exit(1)

        _ok("Target media extracted")
        _ok(f"{candidate_count} candidate images discovered from target")
        print()

        discovered_platforms = [search_result.candidates[0].domain] if search_result.candidates else []
        record["search"] = {
            "provider": search_provider.PROVIDER_NAME,
            "target_url": target,
            "searched_at": search_result.searched_at,
            "candidate_count": candidate_count,
            "platforms": discovered_platforms,
        }

    elif handle:
        _step(2, total_steps, "PROFILE & TIMELINE SEARCH")
        from app.search import TwitterProfileProvider, InstagramProfileProvider, SearchResult, Candidate
        from app.config import require_search_config
        from concurrent.futures import ThreadPoolExecutor

        clean_handle = handle.lstrip("@").strip()
        p_lower = (platform or "").lower().strip()
        do_ig = p_lower in {"instagram", "ig", "insta"} or p_lower in {"all", ""}
        do_tw = p_lower in {"twitter", "x"} or p_lower in {"all", ""}

        target_platforms = []
        if do_tw:
            target_platforms.append("X/Twitter")
        if do_ig:
            target_platforms.append("Instagram")

        _info(f"Searching {', '.join(target_platforms)} for: {C_CYAN}@{clean_handle}{C_RESET}")
        _info("Extracting candidate media...")

        all_candidates: list[Candidate] = []

        def _search_twitter() -> list[Candidate]:
            try:
                tw = TwitterProfileProvider(handle=clean_handle)
                res = tw.search(str(image_path_obj))
                return res.candidates if res else []
            except Exception:
                return []

        def _search_instagram() -> list[Candidate]:
            try:
                api_key = require_search_config(optional=True)
                ig = InstagramProfileProvider(api_key=api_key, allow_free=True)
                res = ig.search_handles([clean_handle])
                return res.candidates if res else []
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=2) as pool:
            tw_fut = pool.submit(_search_twitter) if do_tw else None
            ig_fut = pool.submit(_search_instagram) if do_ig else None

            if tw_fut:
                tw_cands = tw_fut.result()
                if tw_cands:
                    all_candidates.extend(tw_cands)
                    _ok(f"Extracted {len(tw_cands)} media candidate(s) from @{clean_handle} on X/Twitter")
            if ig_fut:
                ig_cands = ig_fut.result()
                if ig_cands:
                    all_candidates.extend(ig_cands)
                    _ok(f"Extracted {len(ig_cands)} media candidate(s) from @{clean_handle} on Instagram")

        search_result = SearchResult(
            candidates=all_candidates,
            provider="Multi-Platform Profile Discovery" if (do_tw and do_ig) else ("Instagram Profile Discovery" if do_ig else "Twitter Timeline Search"),
            searched_at=datetime.now(timezone.utc).isoformat(),
        )

        candidate_count = len(search_result.candidates)
        if candidate_count == 0:
            _fail(f"No media images found for @{clean_handle} across {', '.join(target_platforms)}")
            print()
            sys.exit(1)

        _ok(f"{candidate_count} total candidate media images discovered for @{clean_handle}")
        print()

        discovered_platforms = sorted({c.domain for c in search_result.candidates if c.domain})
        record["search"] = {
            "provider": search_result.provider,
            "handle": clean_handle,
            "searched_at": search_result.searched_at,
            "candidate_count": candidate_count,
            "platforms": discovered_platforms,
        }

    else:
        _step(2, total_steps, "WEB SEARCH")

        from app.config import require_search_config, LENS_HEADLESS
        from app.search import (
            HeadlessLensProvider,
            DirectYandexProvider,
            FreeMultiEngineSearchProvider,
            SerpAPIProvider,
            YandexProvider,
            TwitterProfileProvider,
            InstagramProfileProvider,
            SearchResult,
            extract_social_handles,
            find_social_handles_from_subject_memory,
            find_subject_memory_leads,
            search_web_leads,
        )
        from app.matcher import FaceMatcher
        import tempfile
        import cv2

        api_key = require_search_config(optional=True)
        headless = (not lens_visible) and LENS_HEADLESS

        # Primary Search Provider configuration
        search_provider = prelaunched_search_provider
        if search_provider is None:
            if engine == "serpapi":
                if not api_key:
                    _fatal("SERPAPI_KEY is not configured but required for --engine serpapi.")
                search_provider = SerpAPIProvider(api_key=api_key)
            elif engine == "yandex":
                if api_key:
                    search_provider = YandexProvider(api_key=api_key)
                else:
                    search_provider = DirectYandexProvider()
            elif engine == "lens":
                if api_key:
                    search_provider = SerpAPIProvider(api_key=api_key)
                else:
                    search_provider = HeadlessLensProvider(headless=headless, fallback_on_captcha=True)
            else:  # "all" - default: SerpAPI Google Lens primary, free multi-engine fallback
                if api_key:
                    search_provider = SerpAPIProvider(api_key=api_key)
                else:
                    search_provider = FreeMultiEngineSearchProvider(headless=headless)

        _info(f"Provider: {search_provider.PROVIDER_NAME}")
        _info("Searching...")

        try:
            if search_future is not None:
                search_result = search_future.result()
            else:
                search_result = search_provider.search(str(image_path_obj))
            candidate_count = len(search_result.candidates)
            _ok("Search completed")
            _ok(f"{candidate_count} candidates discovered across the web")
        except Exception as e:
            _info(f"Primary search engine notice: {e}")
            # Automatic fallback to Headless Lens / Direct Yandex
            if isinstance(search_provider, (SerpAPIProvider, YandexProvider)):
                _info("SerpAPI quota exhausted or unavailable. Activating free visual search fallback...")
                try:
                    fallback_provider = FreeMultiEngineSearchProvider(headless=headless)
                    _info(f"Fallback Provider: {fallback_provider.PROVIDER_NAME}")
                    search_result = fallback_provider.search(str(image_path_obj))
                    candidate_count = len(search_result.candidates)
                    _ok(f"{candidate_count} candidates discovered via free visual search fallback")
                except Exception as fb_err:
                    _info(f"Fallback visual search notice: {fb_err}")
                    search_result = SearchResult(
                        candidates=[],
                        provider=search_provider.PROVIDER_NAME,
                        searched_at=datetime.now(timezone.utc).isoformat(),
                    )
                    candidate_count = 0
            else:
                search_result = SearchResult(
                    candidates=[],
                    provider=search_provider.PROVIDER_NAME,
                    searched_at=datetime.now(timezone.utc).isoformat(),
                )
                candidate_count = 0

        # Automated Cross-Platform Social Pivoting (OSINT Discovery)
        # If visual reverse search yielded 0 candidates, immediately attempt social identity memory sweep
        if candidate_count == 0:
            _info("Scanning cross-platform social identity memory...")
            recalled_handles = set(find_social_handles_from_subject_memory(query_embedding, fp=fp))
            all_pivot_handles = sorted(recalled_handles)

            if all_pivot_handles:
                _info(f"Social Pivot: Correlating across {len(all_pivot_handles)} recalled handle(s): {', '.join(['@' + h for h in all_pivot_handles[:4]])}")
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _fetch_tw(h: str):
                    try:
                        tw_prov = TwitterProfileProvider(h)
                        return h, tw_prov.search()
                    except Exception:
                        return h, None

                def _fetch_ig():
                    try:
                        ig_prov = InstagramProfileProvider(api_key=api_key, allow_free=True)
                        from app.search import extract_associate_network_leads
                        _, contexts = extract_associate_network_leads()
                        return ig_prov.search_handles(all_pivot_handles, contexts=contexts)
                    except Exception:
                        return None

                with ThreadPoolExecutor(max_workers=min(len(all_pivot_handles) + 1, 6)) as pool:
                    futures = {pool.submit(_fetch_tw, h): h for h in all_pivot_handles}
                    ig_fut = pool.submit(_fetch_ig)

                    for fut in as_completed(futures):
                        h, tw_res = fut.result()
                        if tw_res and tw_res.candidates:
                            search_result.candidates.extend(tw_res.candidates)
                            _ok(f"Extracted {len(tw_res.candidates)} media candidate(s) from @{h} on X/Twitter")

                    ig_res = ig_fut.result()
                    if ig_res and ig_res.candidates:
                        search_result.candidates.extend(ig_res.candidates)
                        _ok(f"Extracted {len(ig_res.candidates)} candidate(s) from Instagram profile & post sweep")

                candidate_count = len(search_result.candidates)

        if candidate_count == 0:
            _fail("No candidates discovered across visual reverse search or social identity memory.")
            print()
            sys.exit(1)

        # Show platforms discovered across the web
        discovered_platforms = sorted({c.domain for c in search_result.candidates if c.domain})
        if discovered_platforms:
            _info(f"Sources found: {', '.join(discovered_platforms[:7])}" + (f" (+{len(discovered_platforms)-7} more)" if len(discovered_platforms) > 7 else ""))
        print()

        record["search"] = {
            "provider": search_result.provider,
            "searched_at": search_result.searched_at,
            "candidate_count": candidate_count,
            "platforms": discovered_platforms,
        }

    _mark("search")

    # ==================================================================
    # [3/7] FACE MATCHING
    # ==================================================================
    _step(3, total_steps, "FACE MATCHING")
    _info("Analyzing candidate face similarity...")
    print()

    from app.matcher import FaceMatcher

    # Filter candidates by platform if specified
    search_candidates = search_result.candidates
    if platform:
        p_lower = platform.lower()
        search_candidates = [
            c for c in search_candidates 
            if p_lower in c.domain.lower() or p_lower in c.source_url.lower() or p_lower in c.title.lower()
        ]
        if not search_candidates:
            _fail(f"No candidates found matching platform: {platform}")
            print()
            sys.exit(1)
        _info(f"Filtering to platform '{platform}': {len(search_candidates)} candidates")

    matcher = FaceMatcher(fp)
    all_matches = matcher.match_and_rank(query_embedding, search_candidates)
    matches = [m for m in all_matches if m.similarity >= threshold]

    # If no matches above threshold and we used open web search, try fallback to cropped face
    if not matches and not target and not handle:
        cropped = fp.get_face_crop(image_path_obj, margin=0.35)
        if cropped is not None:
            _info("No candidates above threshold with original image.")
            _info("Retrying web search with focused portrait face crop...")
            print()

            temp_crop_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix="_face_crop.jpg", delete=False)
                temp_crop_path = tmp.name
                tmp.close()
                cv2.imwrite(temp_crop_path, cropped)

                crop_search_result = search_provider.search(temp_crop_path)
                if crop_search_result.candidates:
                    _ok(f"{len(crop_search_result.candidates)} candidates discovered from cropped face search")
                    _info("Analyzing cropped face candidate similarity...")
                    print()
                    crop_matches = matcher.match_and_rank(query_embedding, crop_search_result.candidates)
                    if crop_matches:
                        all_matches = crop_matches
                        matches = [m for m in all_matches if m.similarity >= threshold]
            except Exception:
                pass
            finally:
                if temp_crop_path and Path(temp_crop_path).exists():
                    try:
                        Path(temp_crop_path).unlink()
                    except OSError:
                        pass

    # If still no matches above threshold and engine allows, try Yandex fallback (skip if already run in multi-engine)
    if not matches and not target and not handle and engine in ("all", "yandex") and not isinstance(search_provider, FreeMultiEngineSearchProvider):
        _info("No candidates above threshold with primary visual search.")
        _info("Searching Yandex Images (deep social/facial reverse search)...")
        print()
        from app.search import YandexProvider, DirectYandexProvider
        yandex_res = None
        if api_key:
            try:
                yandex_provider = YandexProvider(api_key=api_key)
                yandex_res = yandex_provider.search(str(image_path_obj))
            except Exception as ye:
                _info(f"SerpAPI Yandex notice ({ye}); delegating to Direct Yandex fallback...")
                try:
                    yandex_provider = DirectYandexProvider()
                    yandex_res = yandex_provider.search(str(image_path_obj))
                except Exception as dye:
                    _info(f"Direct Yandex search notice: {dye}")
        else:
            try:
                yandex_provider = DirectYandexProvider()
                yandex_res = yandex_provider.search(str(image_path_obj))
            except Exception as dye:
                _info(f"Direct Yandex search notice: {dye}")

        if yandex_res and yandex_res.candidates:
            _ok(f"{len(yandex_res.candidates)} candidates discovered from Yandex Images")
            _info("Analyzing Yandex candidate similarity...")
            print()
            y_matches = matcher.match_and_rank(query_embedding, yandex_res.candidates)
            if y_matches:
                all_matches = y_matches
                matches = [m for m in all_matches if m.similarity >= threshold]

    # If still no matches above threshold, activate Cross-Platform OSINT Social Pivot & Identity Memory
    if not matches and not target and not handle:
        _info("No candidates above threshold with visual search.")
        _info("Activating cross-platform OSINT social pivot & identity memory...")
        discovered_handles = set(extract_social_handles(search_result.candidates))
        recalled_handles_list, memory_candidates = find_subject_memory_leads(query_embedding, fp=fp)
        recalled_handles = set(recalled_handles_list)

        # 1. Immediately evaluate known verified appearances from subject identity memory
        if memory_candidates:
            _ok(f"Correlating with {len(memory_candidates)} verified appearance candidate(s) from subject memory")
            mem_matches = matcher.match_and_rank(query_embedding, memory_candidates)
            if mem_matches:
                all_matches = sorted(all_matches + mem_matches, key=lambda r: r.similarity, reverse=True)
                matches = [m for m in all_matches if m.similarity >= threshold]

        # 2. Correlate across discovered and recalled social/web handles
        if not matches:
            all_pivot_handles = sorted(recalled_handles) + [h for h in sorted(discovered_handles) if h not in recalled_handles][:3]

            if all_pivot_handles:
                _info(f"Social Pivot: Correlating across {len(all_pivot_handles)} handle(s): {', '.join(['@' + h for h in all_pivot_handles])}")
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _fetch_tw(h: str):
                    try:
                        tw_prov = TwitterProfileProvider(h)
                        return h, tw_prov.search()
                    except Exception:
                        return h, None

                def _fetch_ig():
                    try:
                        ig_prov = InstagramProfileProvider(api_key=api_key, allow_free=True)
                        from app.search import extract_associate_network_leads
                        _, contexts = extract_associate_network_leads()
                        return ig_prov.search_handles(all_pivot_handles, contexts=contexts)
                    except Exception:
                        return None

                def _fetch_web(h: str):
                    try:
                        return h, search_web_leads(h, max_results=4)
                    except Exception:
                        return h, []

                new_pivot_candidates = []
                with ThreadPoolExecutor(max_workers=min(len(all_pivot_handles) * 2 + 1, 8)) as pool:
                    futures_tw = {pool.submit(_fetch_tw, h): h for h in all_pivot_handles}
                    futures_web = {pool.submit(_fetch_web, h): h for h in all_pivot_handles}
                    ig_fut = pool.submit(_fetch_ig)

                    for fut in as_completed(futures_tw):
                        h, tw_res = fut.result()
                        if tw_res and tw_res.candidates:
                            new_pivot_candidates.extend(tw_res.candidates)
                            _ok(f"Extracted {len(tw_res.candidates)} media candidate(s) from @{h} on X/Twitter")

                    for fut in as_completed(futures_web):
                        h, web_res = fut.result()
                        if web_res:
                            new_pivot_candidates.extend(web_res)
                            _ok(f"Extracted {len(web_res)} candidate(s) from web OSINT pivot on @{h}")

                    ig_res = ig_fut.result()
                    if ig_res and ig_res.candidates:
                        new_pivot_candidates.extend(ig_res.candidates)
                        _ok(f"Extracted {len(ig_res.candidates)} candidate(s) from Instagram profile & post sweep")

                if new_pivot_candidates:
                    _info("Analyzing social pivot candidate similarity...")
                    pivot_matches = matcher.match_and_rank(query_embedding, new_pivot_candidates)
                    if pivot_matches:
                        all_matches = sorted(all_matches + pivot_matches, key=lambda r: r.similarity, reverse=True)
                        matches = [m for m in all_matches if m.similarity >= threshold]

    # If still no matches above threshold and not targeted, activate Associate Forensics Graph
    if not matches and not target and not handle:
        _info("Visual reverse search yielded 0 direct hits.")
        _info("Activating Associate Forensics Graph (Network Pivoting)...")
        print()
        from app.search import extract_associate_network_leads, LinkedInPostProvider
        assoc_names, assoc_contexts = extract_associate_network_leads()
        if assoc_names:
            assoc_display = ", ".join(assoc_names[:4]) + (f" (+{len(assoc_names)-4} more)" if len(assoc_names) > 4 else "")
            _info(f"Correlating with {len(assoc_names)} known network associate(s): {assoc_display}")
            if assoc_contexts:
                _info(f"Context tags: {', '.join(assoc_contexts[:3])}")
            try:
                li_provider = LinkedInPostProvider(api_key=api_key, allow_free=True)
                li_res = li_provider.search_leads(names=assoc_names, contexts=assoc_contexts)
                if li_res.candidates:
                    _ok(f"{len(li_res.candidates)} candidate post(s) discovered from LinkedIn associate sweep")
                    _info("Analyzing LinkedIn candidate similarity...")
                    print()
                    li_matches = matcher.match_and_rank(query_embedding, li_res.candidates)
                    if li_matches:
                        all_matches = li_matches
                        matches = [m for m in all_matches if m.similarity >= threshold]
            except Exception as e:
                _info(f"LinkedIn associate sweep skipped: {e}")

    # Display top results (show up to 10)
    display_matches = all_matches[:10]
    for i, m in enumerate(display_matches, 1):
        pct = m.similarity * 100
        color = C_GREEN if m.similarity >= 0.85 else (C_YELLOW if m.similarity >= threshold else C_RED)
        platform_label = m.candidate.domain or m.candidate.title[:30] or "Web"
        status_tag = "" if m.face_detected else " (no face detected)"
        print(f"        #{i:<3} {color}Similarity: {pct:.1f}%{C_RESET}  [{platform_label}]{status_tag}")

    if not matches:
        print()
        _fail(f"No candidates above threshold ({threshold:.0%})")
        if all_matches and all_matches[0].similarity > 0:
            top_sim = all_matches[0].similarity * 100
            print(f"  Highest candidate similarity was {top_sim:.1f}%")
        print("  Try lowering the threshold with --threshold 0.50")
        print()
        sys.exit(1)

    print()
    best = matches[0]
    _ok(f"Strongest candidate selected: {best.candidate.domain or 'Web'} (Similarity: {best.similarity*100:.1f}%)")
    print()

    record["match"] = {
        "source_url": best.candidate.source_url,
        "image_url": best.candidate.image_url,
        "similarity": round(best.similarity, 4),
        "domain": best.candidate.domain,
        "title": best.candidate.title,
    }
    _mark("matching")

    # ==================================================================
    # [4/7] CONTENT RETRIEVAL
    # ==================================================================
    _step(4, total_steps, "CONTENT RETRIEVAL")

    from app.content import ContentRetriever

    retriever = ContentRetriever()
    content = retriever.retrieve(best.candidate.source_url, best.candidate.image_url)

    _ok("Matching content retrieved")
    print()
    _info(f"Source:")
    _info(f"{C_CYAN}{content.source_url}{C_RESET}")
    if content.title:
        _info(f"Title: {content.title}")
    if content.platform:
        _info(f"Platform: {content.platform}")
    print()

    # Compute image hash for record
    image_hash = ""
    if content.image_bytes:
        image_hash = hashlib.sha256(content.image_bytes).hexdigest()
        _ok(f"Image downloaded ({len(content.image_bytes)} bytes)")
    else:
        _info("(Image bytes not available — metadata-only fingerprint)")
    print()

    record["content"] = {
        "source_url": content.source_url,
        "image_url": content.image_url,
        "platform": content.platform,
        "title": content.title,
        "text": content.text,
        "author": getattr(content, "author", "") or "",
        "image_hash": image_hash,
        "retrieved_at": content.retrieved_at,
    }
    if getattr(content, "author", None) and "match" in record:
        record["match"]["author"] = content.author
    _mark("content")

    # ==================================================================
    # [5/7] FINGERPRINT
    # ==================================================================
    _step(5, total_steps, "FINGERPRINT")

    from app.hashing import generate_fingerprint

    canonical = ContentRetriever.canonicalize(content)
    content_hash = generate_fingerprint(canonical)

    _info(f"Algorithm: SHA-256")
    print()
    _info(f"{C_CYAN}{content_hash}{C_RESET}")
    print()

    record["fingerprint"] = {
        "algorithm": "SHA-256",
        "hash": content_hash,
    }

    # ==================================================================
    # [6/7] BLOCKCHAIN
    # ==================================================================
    _step(6, total_steps, "BLOCKCHAIN")

    from app.config import require_blockchain_config
    from app.blockchain import BlockchainClient

    try:
        rpc, pk, ca = require_blockchain_config()
        bc = BlockchainClient(rpc, pk, ca)
    except SystemExit:
        raise
    except Exception as e:
        _fatal(f"Blockchain connection failed: {e}")

    _info(f"Network: {bc.network}")
    _info(f"Contract: {bc.contract_address}")
    _info("Submitting transaction...")

    source_id = content.platform or ""
    tx = bc.register_hash(content_hash, source_id)
    _mark("blockchain")

    if tx.status == "confirmed":
        _ok("Transaction confirmed")
        print()
        _info(f"TX:")
        _info(f"{C_CYAN}0x{tx.tx_hash}{C_RESET}")
        _info(f"Block: {tx.block_number}")
    elif tx.status == "error":
        _fail(f"Transaction failed: {tx.error}")
        # Continue to save record even if tx fails
    else:
        _fail(f"Transaction status: {tx.status}")

    print()

    record["blockchain"] = {
        "network": bc.network,
        "contract": bc.contract_address,
        "transaction": f"0x{tx.tx_hash}" if tx.tx_hash else "",
        "block": tx.block_number,
        "status": tx.status,
    }

    # ==================================================================
    # [7/7] VERIFICATION
    # ==================================================================
    _step(7, total_steps, "VERIFICATION")

    if tx.status == "confirmed":
        verify_result = bc.verify_hash(content_hash)

        _info(f"Local hash:")
        _info(f"{content_hash}")
        print()

        if verify_result.exists:
            _info(f"On-chain: ✓ Hash found")
            print()
            _ok("CONTENT VERIFIED")
            record["verification"] = {"verified": True}
        else:
            _info(f"On-chain: Hash not found")
            print()
            _fail("VERIFICATION FAILED")
            record["verification"] = {"verified": False, "error": "Hash not found on-chain"}
    else:
        _info("Blockchain transaction was not confirmed — skipping on-chain verification")
        record["verification"] = {"verified": False, "error": f"TX status: {tx.status}"}

    print()

    # ==================================================================
    # Save record
    # ==================================================================
    from app.config import RESULTS_DIR

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    record_filename = f"{timestamp_str}_record.json"
    record_path = RESULTS_DIR / record_filename
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Record saved:")
    print(f"  {C_CYAN}{record_path}{C_RESET}")
    print()
    print("=" * 60)
    print()

    total_s = _time.perf_counter() - _t0
    print("  Timing (cumulative):")
    _prev = 0.0
    for name, t in _phase_marks:
        print(f"    {name:<12} +{t - _prev:6.2f}s  (t={t:6.2f}s)")
        _prev = t
    print(f"    {'total':<12} {total_s:6.2f}s")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="facetrace",
        description="FaceTrace — Face Search + Blockchain Verification",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default / search command
    parser.add_argument(
        "--image",
        help="Path to the query face image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum face similarity threshold (default: 0.70).",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Optional: filter candidates to a specific platform (e.g. instagram, wikipedia, x.com).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional: direct URL of a specific post or page to verify against the query face (e.g. an X post, Reddit thread, or article).",
    )
    parser.add_argument(
        "--handle",
        "--user",
        type=str,
        default=None,
        help="Optional: search a specific user's public profile and posts on Instagram and/or X/Twitter (filter with --platform instagram or --platform twitter).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["all", "lens", "yandex", "serpapi"],
        default="all",
        help="Visual search engine: 'all' (primary SerpAPI with free fallback), 'lens' (Google Lens with fallback), 'yandex' (Yandex Images), or 'serpapi' (SerpAPI only). Default: 'all'.",
    )
    parser.add_argument(
        "--lens-visible",
        action="store_true",
        default=False,
        help="Launch Google Lens browser with visible window (useful for interactive anti-bot verification).",
    )

    # Verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify a saved record.")
    verify_parser.add_argument(
        "--record",
        required=True,
        help="Path to the saved record JSON file.",
    )

    args = parser.parse_args()

    if args.command == "verify":
        from app.verify import verify_record, _print_verification
        result = verify_record(args.record)
        _print_verification(result)
        sys.exit(0 if result.get("verified") else 1)

    elif args.image:
        run_pipeline(
            args.image,
            threshold=args.threshold,
            platform=args.platform,
            target=args.target,
            engine=args.engine,
            handle=args.handle,
            lens_visible=getattr(args, "lens_visible", False),
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
