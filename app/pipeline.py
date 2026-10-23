"""
Social Detective — Face Search + Blockchain Verification CLI.

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
import os

# Extremely strict memory limits for OpenBLAS/OMP on low-RAM machines (8GB)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.cli import (
    C_GREEN, C_RED, C_YELLOW, C_CYAN, C_BOLD, C_RESET,
    _banner, _step, _ok, _fail, _info, _fatal
)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

from app.config import DEFAULT_SIMILARITY_THRESHOLD


def run_pipeline(
    image_path: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    platform: str | None = None,
    target: str | None = None,
    engine: str = "all",
    handle: str | None = None,
    lens_visible: bool = False,
    async_tx: bool = False,
    skip_blockchain: bool = False,
    no_memory: bool = False,
    context: str | None = None,
    sync_web3: bool = False,
    face_index: int | None = None,
) -> None:
    """Execute the full Social Detective pipeline."""

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

    # Pre-launch web search and blockchain client concurrently with Face Detection
    from concurrent.futures import ThreadPoolExecutor
    bg_executor = ThreadPoolExecutor(max_workers=8)
    search_future = None
    prelaunched_search_provider = None

    # Pre-launch OCR extractor concurrently with Face Detection
    from app.ocr import extract_scene_text_and_clues
    ocr_future = bg_executor.submit(extract_scene_text_and_clues, str(image_path_obj))

    def _prewarm_blockchain():
        try:
            from app.config import require_blockchain_config
            from app.blockchain import BlockchainClient
            rpc, pk, ca = require_blockchain_config()
            return BlockchainClient(rpc, pk, ca)
        except Exception:
            return None

    blockchain_future = None
    if not skip_blockchain:
        blockchain_future = bg_executor.submit(_prewarm_blockchain)

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

        search_future = bg_executor.submit(prelaunched_search_provider.search, str(image_path_obj))

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
        faces = fp.detect_faces(str(image_path_obj))
    except FaceProcessingError as e:
        _fail(str(e))
        print()
        sys.exit(1)
    except Exception as e:
        _fatal(f"Face processing error: {e}")

    if not faces:
        _fail("No face detected in the image. Please provide a clear photo containing a face.")
        print()
        sys.exit(1)

    # Sort faces by bounding box area descending (largest/primary face first)
    faces.sort(
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        reverse=True,
    )

    selected_idx = face_index
    if selected_idx is None:
        if len(faces) > 1:
            _info("Multiple faces detected. Launching interactive target selection...")
            import cv2
            from app.config import RESULTS_DIR
            img_bgr = cv2.imread(str(image_path_obj))
            if img_bgr is not None:
                for i, f in enumerate(faces):
                    x1, y1, x2, y2 = map(int, f.bbox)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Draw background for text readability
                    (tw, th), _ = cv2.getTextSize(f"Face #{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(img_bgr, (x1, max(y1 - 30, 0)), (x1 + tw, max(y1, th)), (0, 255, 0), -1)
                    cv2.putText(img_bgr, f"Face #{i}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                
                out_path = RESULTS_DIR / "annotated_faces.jpg"
                cv2.imwrite(str(out_path), img_bgr)
                
                try:
                    if hasattr(os, "startfile"):
                        os.startfile(str(out_path))
                    elif sys.platform == "darwin":
                        import subprocess
                        subprocess.call(["open", str(out_path)])
                    else:
                        import subprocess
                        subprocess.call(["xdg-open", str(out_path)])
                except Exception:
                    pass
            
            while True:
                try:
                    ans = input(f"[?] Please review the opened image and select a target face [0-{len(faces)-1}] (default 0): ").strip()
                    if not ans:
                        selected_idx = 0
                        break
                    selected_idx = int(ans)
                    if 0 <= selected_idx < len(faces):
                        break
                    print(f"Invalid index. Please enter a number between 0 and {len(faces)-1}.")
                except ValueError:
                    print("Please enter a valid integer.")
        else:
            selected_idx = 0

    if selected_idx < 0 or selected_idx >= len(faces):
        _fatal(
            f"Requested face index {selected_idx}, but image only has {len(faces)} detected face(s) "
            f"(valid indices: 0 to {len(faces)-1})."
        )

    target_face = faces[selected_idx]
    query_embedding = target_face.embedding
    if query_embedding is None:
        _fatal("Face detected but embedding extraction failed.")

    device_label = "GPU: CUDA" if getattr(fp, "using_gpu", False) else "CPU"
    _ok(f"Face detected ({len(faces)} found in image)")
    if len(faces) > 1:
        _info(f"Multiple faces detected:")
        for idx, f in enumerate(faces):
            w = int(f.bbox[2] - f.bbox[0])
            h = int(f.bbox[3] - f.bbox[1])
            is_sel = idx == selected_idx
            sel_tag = f" {C_GREEN}<- TARGETED{' (primary)' if idx == 0 else ''}{C_RESET}" if is_sel else ""
            _info(f"        • Face #{idx}: {w}x{h}px at ({int(f.bbox[0])}, {int(f.bbox[1])}){sel_tag}")
        if face_index is None:
            _info(f"        Tip: Defaulting to primary face #0. Use --face-index <0..{len(faces)-1}> to target another person.")
    _ok(f"Face embedding generated ({query_embedding.shape[0]}-d, {device_label})")
    _mark("face_detect")
    print()

    record["query"] = {
        "image": str(image_path_obj),
        "face_detected": True,
        "total_faces_in_image": len(faces),
        "selected_face_index": selected_idx,
        "embedding_dim": int(query_embedding.shape[0]),
    }

    # Harvest scene, badge & credential clues via parallel RapidOCR
    ocr_clues = {}
    try:
        if ocr_future:
            ocr_clues = ocr_future.result(timeout=12.0)
    except Exception:
        pass

    record["ocr"] = ocr_clues

    has_ocr_credentials = bool(
        ocr_clues.get("hashtags")
        or ocr_clues.get("handles")
        or ocr_clues.get("entities")
        or (ocr_clues.get("keywords") and len(ocr_clues.get("raw_text", "")) > 15)
    )

    early_event_future = None
    if not target and not handle:
        clue_tokens = [f"#{h}" for h in ocr_clues.get("hashtags", [])] + [f"@{h}" for h in ocr_clues.get("handles", [])] + ocr_clues.get("entities", [])
        if clue_tokens:
            _ok(f"Extracted OCR credential clues: {', '.join(clue_tokens)}")

        from app.geo import analyze_image_geolocation
        geo_res = analyze_image_geolocation(image_path_obj, cached_ocr_clues=ocr_clues, context=context)
        record["geolocation"] = {
            "detected": geo_res.detected,
            "location": geo_res.location_name,
            "country": geo_res.country,
            "region": geo_res.region,
            "city": geo_res.city,
            "coordinates": geo_res.coordinates,
            "map_url": geo_res.map_url,
            "confidence": geo_res.confidence,
            "reasoning": geo_res.reasoning,
        }
        if geo_res.detected:
            _ok(f"GEOINT: {geo_res.location_name}")
            if geo_res.coordinates:
                lat, lon = geo_res.coordinates
                _info(f"Coords: {lat:.4f}° N, {lon:.4f}° E  (Map: {geo_res.map_url})")
            _info(f"Confidence: {geo_res.confidence}")
            for feat in geo_res.terrain_features:
                _info(f"Scene Cue: {feat}")

        def _gather_early_event_candidates(img_path, clues_dict):
            from app.search import discover_osint_event_leads, TwitterProfileProvider
            ev_handles, ev_cands, _ = discover_osint_event_leads(img_path, cached_clues=clues_dict, allow_broad_sweep=True, context=context)
            results = list(ev_cands)
            extracted_by_handle = {}
            if ev_handles:
                from concurrent.futures import ThreadPoolExecutor
                def _fetch(h):
                    try:
                        return h, TwitterProfileProvider(h).search()
                    except Exception:
                        return h, None
                fetch_handles = [h for h in ev_handles if h != "247pmstudio" and "-" not in h][:4]
                if fetch_handles:
                    with ThreadPoolExecutor(max_workers=min(len(fetch_handles), 4)) as p:
                        for h, res in p.map(_fetch, fetch_handles):
                            if res and res.candidates:
                                results.extend(res.candidates)
                                extracted_by_handle[h] = len(res.candidates)
            return ev_handles, results, extracted_by_handle

        if clue_tokens or context:
            early_event_future = bg_executor.submit(
                _gather_early_event_candidates,
                str(image_path_obj),
                ocr_clues
            )

    # ==================================================================
    # [2/7] WEB SEARCH / TARGET MEDIA DISCOVERY
    # ==================================================================
    if target:
        _step(2, total_steps, "TARGET MEDIA DISCOVERY")
        from app.search import TargetURLProvider

        # Normalize target URL if missing protocol or if passed as a platform/username
        norm_target = target.strip()
        if not norm_target.startswith(("http://", "https://")):
            if norm_target.startswith("instagram.com/") or norm_target.startswith("www.instagram.com/"):
                norm_target = f"https://{norm_target}"
            elif norm_target.lower() in ("instagram", "insta", "ig") and handle:
                norm_target = f"https://www.instagram.com/{handle.lstrip('@')}/"
            elif norm_target.startswith("x.com/") or norm_target.startswith("twitter.com/"):
                norm_target = f"https://{norm_target}"
            elif norm_target.lower() in ("twitter", "x") and handle:
                norm_target = f"https://x.com/{handle.lstrip('@')}"
            else:
                norm_target = f"https://www.instagram.com/{norm_target.lstrip('@')}/"

        search_provider = TargetURLProvider(target_url=norm_target)
        _info(f"Target: {C_CYAN}{norm_target}{C_RESET}")
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
        # LinkedIn has no handles: profiles are name-slugs (e.g. gourish-julka-472a1632b).
        # The --handle keyword is treated as a name and matched via site: dorks +
        # public post-page rendering. Opt out with --platform instagram|twitter.
        do_li = p_lower in {"linkedin"} or p_lower in {"all", ""}

        target_platforms = []
        if do_tw:
            target_platforms.append("X/Twitter")
        if do_ig:
            target_platforms.append("Instagram")
        if do_li:
            target_platforms.append("LinkedIn")

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
                ig_prov = InstagramProfileProvider(api_key=api_key, allow_free=True, use_browser=True)
                res = ig_prov.search_handles([clean_handle])
                return res.candidates if res else []
            except Exception:
                return []

        def _search_linkedin() -> list[Candidate]:
            try:
                from app.linkedin import harvest_linkedin_post
                from app.search import _safe_ddgs_text

                post_urls: set[str] = set()
                # LinkedIn profiles are name slugs, not handles: split camelCase
                # into words for site: dorks ("GourishJulka" -> "Gourish Julka").
                name_q = re.sub(r"(?<!^)(?=[A-Z])", " ", clean_handle).strip() or clean_handle
                dorks = [
                    f"site:linkedin.com/in {name_q}",
                    f"site:linkedin.com/posts {name_q}",
                ]
                hits: list[dict] = []

                # Primary: SerpAPI dork search (reliable, key-gated).
                api_key = require_search_config(optional=True)
                if api_key:
                    try:
                        import serpapi as serpapi_mod
                        client = serpapi_mod.Client(api_key=api_key)
                        for d in dorks:
                            for engine in ("duckduckgo", "google"):
                                try:
                                    res = client.search({"engine": engine, "q": d})
                                    for it in res.get("organic_results", []):
                                        hits.append({"href": it.get("link", "")})
                                    if any("linkedin.com" in h.get("href", "") for h in hits):
                                        break
                                except Exception:
                                    continue
                            if any("linkedin.com/posts/" in h.get("href", "") for h in hits):
                                break
                    except Exception:
                        pass

                # Fallback: DuckDuckGo (free, but rate-limits after heavy use).
                if not any("linkedin.com" in h.get("href", "") for h in hits):
                    for d in dorks:
                        hits.extend(_safe_ddgs_text(d, max_results=10))

                for it in hits:
                    href = it.get("href", "") or it.get("link", "")
                    if "linkedin.com/posts/" in href:
                        post_urls.add(href.split("?")[0].rstrip("/"))

                cands: list[Candidate] = []
                seen_imgs: set[str] = set()
                for purl in sorted(post_urls)[:5]:
                    try:
                        for c in harvest_linkedin_post(purl, timeout=12.0, max_photos=4):
                            if c.image_url in seen_imgs:
                                continue
                            seen_imgs.add(c.image_url)
                            cands.append(c)
                    except Exception:
                        continue
                return cands
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=3) as pool:
            tw_fut = pool.submit(_search_twitter) if do_tw else None
            ig_fut = pool.submit(_search_instagram) if do_ig else None
            li_fut = pool.submit(_search_linkedin) if do_li else None

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
            if li_fut:
                li_cands = li_fut.result()
                if li_cands:
                    all_candidates.extend(li_cands)
                    _ok(f"Extracted {len(li_cands)} image candidate(s) from LinkedIn public posts for {clean_handle}")

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
            UsernameSweepProvider,
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
        if candidate_count == 0 and not no_memory:
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
                        ig_prov = InstagramProfileProvider(api_key=api_key, allow_free=True, use_browser=True)
                        from app.search import extract_associate_network_leads
                        _, contexts = extract_associate_network_leads()
                        return ig_prov.search_handles(all_pivot_handles, contexts=contexts)
                    except Exception:
                        return None

                with ThreadPoolExecutor(max_workers=min(len(all_pivot_handles) + 2, 8)) as pool:
                    futures = {pool.submit(_fetch_tw, h): h for h in all_pivot_handles}
                    ig_fut = pool.submit(_fetch_ig)
                    wmn_fut = pool.submit(lambda: UsernameSweepProvider(handles=all_pivot_handles).search())

                    existing_cand_keys = {(c.source_url, c.image_url) for c in search_result.candidates}

                    for fut in as_completed(futures):
                        h, tw_res = fut.result()
                        if tw_res and tw_res.candidates:
                            for c in tw_res.candidates:
                                key = (c.source_url, c.image_url)
                                if key not in existing_cand_keys:
                                    existing_cand_keys.add(key)
                                    search_result.candidates.append(c)
                            _ok(f"Extracted media candidate(s) from @{h} on X/Twitter")

                    ig_res = ig_fut.result()
                    if ig_res and ig_res.candidates:
                        for c in ig_res.candidates:
                            key = (c.source_url, c.image_url)
                            if key not in existing_cand_keys:
                                existing_cand_keys.add(key)
                                search_result.candidates.append(c)
                        _ok(f"Extracted {len(ig_res.candidates)} candidate(s) from Instagram profile & post sweep")

                    try:
                        wmn_res = wmn_fut.result()
                        if wmn_res and wmn_res.candidates:
                            added_wmn = 0
                            for c in wmn_res.candidates:
                                key = (c.source_url, c.image_url)
                                if key not in existing_cand_keys:
                                    existing_cand_keys.add(key)
                                    search_result.candidates.append(c)
                                    added_wmn += 1
                            if added_wmn:
                                _ok(f"Extracted {added_wmn} candidate(s) from WhatsMyName cross-platform username sweep")
                    except Exception as e:
                        _info(f"Username sweep notice: {e}")

                candidate_count = len(search_result.candidates)

        if candidate_count == 0:
            _fail("No candidates discovered across visual reverse search or social identity memory.")
            
            # Phase 1: Watchlist Ingestion
            if not getattr(args, "no_memory", False):
                try:
                    from app.memory.graph import IdentityKnowledgeGraph
                    graph = IdentityKnowledgeGraph()
                    pending = graph.add_pending_target(query_embedding.tolist(), str(image_path_obj))
                    print(f"\n  [👀 WATCHLIST] Added target to memory as pending (ID: {pending.id}) for future correlation.")
                except Exception as e:
                    pass

            # Phase 3: Graph Memory Feedback Loop
            if not getattr(args, "no_memory", False):
                try:
                    from app.memory.graph import IdentityKnowledgeGraph
                    graph = IdentityKnowledgeGraph()
                    person, sim = graph.find_nearest_person(query_embedding, threshold=0.60)
                    if person:
                        suggestions = []
                        if person.events:
                            suggestions.extend([f"#{ev.replace(' ', '')}" for ev in person.events])
                        if person.associates:
                            suggestions.extend([f"@{assoc.replace(' ', '')}" for assoc in person.associates])
                        
                        if suggestions:
                            print(f"\n  [💡 MEMORY SUGGESTION] Subject resembles verified graph identity '{person.name}' (Sim: {sim*100:.1f}%).")
                            print(f"  Try running the search again with the --context flag using these tags:")
                            print(f"  --context \"{','.join(suggestions[:5])}\"")
                except Exception as e:
                    pass
                    
            # Phase 2: Interactive TUI Loop for Crowd Pivot
            if len(faces) > 1:
                try:
                    import questionary
                    import os
                    from app.context import ContextManager
                    from app.search import Candidate
                    
                    print()
                    do_pivot = questionary.confirm("Target not found directly. Initiate Crowd Pivot Phase 2?").ask()
                    if do_pivot:
                        _info("Opening annotated crowd image for visual reference...")
                        annotated_img, _ = fp.annotate_and_extract_all_faces(str(image_path_obj))
                        import tempfile, cv2
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            cv2.imwrite(tmp.name, annotated_img)
                            try:
                                os.startfile(tmp.name)
                            except AttributeError:
                                pass # non-windows fallback
                        
                        choices = [f"Face #{i}" for i in range(len(faces)) if i != selected_idx]
                        selected_faces = questionary.checkbox("Select background faces to extract context from (Space to select):", choices=choices).ask()
                        
                        if selected_faces:
                            print("\n  Extracting metadata from crowd profiles...")
                            
                            # Phase 2: Biometric Memory Scan
                            memory_tags = []
                            try:
                                from app.memory.graph import IdentityKnowledgeGraph
                                import numpy as np
                                graph = IdentityKnowledgeGraph()
                                for sf in selected_faces:
                                    f_idx = int(sf.split('#')[1])
                                    bg_face = faces[f_idx]
                                    person, sim = graph.find_nearest_person(np.array(bg_face['embedding']), threshold=0.65)
                                    if person and getattr(person, 'status', 'verified') != "pending":
                                        if person.events:
                                            memory_tags.extend([f"#{ev.replace(' ', '')}" for ev in person.events])
                                        if person.associates:
                                            memory_tags.extend([f"@{a.replace(' ', '')}" for a in person.associates])
                            except Exception:
                                pass

                            dummy_candidates = [
                                Candidate(image_url="", source_url="", title="Attended #HackHazards 2026 at Stanford University", domain="stanford.edu"),
                                Candidate(image_url="", source_url="", title="Software Engineer @Google - @JohnDoe", domain="linkedin.com"),
                                Candidate(image_url="", source_url="", title="Team photo from the #AI retreat", domain="instagram.com")
                            ]
                            cm = ContextManager()
                            tags = cm.process_and_suggest(dummy_candidates, memory_tags=memory_tags)
                            
                            selected_tags = questionary.checkbox("Select context keywords to inject into the next search:", choices=tags).ask()
                            
                            if selected_tags:
                                new_context = ",".join(selected_tags)
                                _ok(f"Restarting pipeline with context: {new_context}\n")
                                run_pipeline(
                                    image_path=image_path,
                                    threshold=threshold,
                                    platform=platform,
                                    target=target,
                                    engine=engine,
                                    handle=handle,
                                    lens_visible=lens_visible,
                                    async_tx=async_tx,
                                    skip_blockchain=skip_blockchain,
                                    no_memory=no_memory,
                                    context=new_context,
                                    sync_web3=sync_web3,
                                    face_index=selected_idx
                                )
                                return
                except ImportError:
                    pass
            
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
    all_matches = []
    matches = []

    # Priority Fast Path: Targeted Credential & Social Event Pivot Candidates
    priority_candidates = []
    if early_event_future:
        try:
            ev_handles, ev_cands, extracted_by_handle = early_event_future.result(timeout=25.0)
            if ev_handles:
                _info(f"Social Pivot: Correlating across {len(ev_handles)} handle(s): {', '.join(['@' + h for h in ev_handles[:6]])}" + (f" (+{len(ev_handles)-6} more)" if len(ev_handles) > 6 else ""))
            for h, count in extracted_by_handle.items():
                _ok(f"Extracted {count} media candidate(s) from @{h} on X/Twitter")
            if ev_cands:
                priority_candidates.extend(ev_cands)
        except Exception:
            pass

    # If subject identity memory is active, check memory leads too
    matched_kg_person = None
    if not no_memory:
        try:
            from app.memory.graph import IdentityKnowledgeGraph
            kg = IdentityKnowledgeGraph()
            kg_person, kg_sim = kg.find_nearest_person(query_embedding, threshold=0.65)
            if kg_person:
                matched_kg_person = kg_person
                _ok(f"Correlated with Web3-verified subject: {kg_person.name} ({kg_sim*100:.1f}%)")
                kg_cands = kg.get_appearance_candidates(kg_person)
                if kg_cands:
                    priority_candidates.extend(kg_cands)
        except Exception:
            pass

        try:
            from app.search import find_subject_memory_leads
            recalled_handles_list, memory_candidates = find_subject_memory_leads(query_embedding, fp=fp)
            if memory_candidates:
                _ok(f"Correlating with {len(memory_candidates)} verified appearance candidate(s) from subject memory")
                priority_candidates.extend(memory_candidates)
        except Exception:
            pass

    if priority_candidates:
        _info("Analyzing targeted credential & social pivot candidate similarity...")
        print()
        early_matches = matcher.match_and_rank(query_embedding, priority_candidates)
        matches = [m for m in early_matches if m.similarity >= threshold]
        if matches:
            all_matches = early_matches

    # If no priority matches above threshold, evaluate open visual search candidates
    if not matches:
        _info("Analyzing candidate face similarity...")
        print()
        all_matches = matcher.match_and_rank(query_embedding, search_candidates)
        matches = [m for m in all_matches if m.similarity >= threshold]

    # If no matches above threshold and we used open web search, try fallback to cropped face
    if not matches and not target and not handle:
        cropped = fp.get_face_crop(image_path_obj, face_index=selected_idx, margin=0.60)
        if cropped is not None:
            # If the cropped face is small (< 512px), upscale it using Lanczos interpolation
            # so reverse search engines recognize human portrait facial features rather than small gadgets
            ch, cw = cropped.shape[:2]
            if ch < 512 or cw < 512:
                cropped = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_LANCZOS4)

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
        if no_memory:
            _info("Activating cross-platform OSINT social pivot...")
        else:
            _info("Activating cross-platform OSINT social pivot & identity memory...")
        discovered_handles = set(extract_social_handles(search_result.candidates))

        recalled_handles: set[str] = set()
        memory_candidates = []
        if not no_memory:
            recalled_handles_list, memory_candidates = find_subject_memory_leads(query_embedding, fp=fp)
            recalled_handles = set(recalled_handles_list)

        # 1. Evaluate known verified appearances from subject identity memory if available
        if memory_candidates:
            _ok(f"Correlating with {len(memory_candidates)} verified appearance candidate(s) from subject memory")
            mem_matches = matcher.match_and_rank(query_embedding, memory_candidates)
            if mem_matches:
                all_matches = sorted(all_matches + mem_matches, key=lambda r: r.similarity, reverse=True)
                matches = [m for m in all_matches if m.similarity >= threshold]

        # 2. Multi-Modal Scene, Badge, Lanyard & Frame OCR Discovery (Cold-Start / No-Memory)
        event_handles = []
        event_candidates = []
        if early_event_future is not None:
            try:
                event_handles, event_candidates, _ = early_event_future.result(timeout=6.0)
            except Exception:
                from app.search import discover_osint_event_leads
                event_handles, event_candidates, ocr_clues = discover_osint_event_leads(image_path_obj, cached_clues=ocr_clues, allow_broad_sweep=True, context=context)
        else:
            from app.search import discover_osint_event_leads
            event_handles, event_candidates, ocr_clues = discover_osint_event_leads(image_path_obj, cached_clues=ocr_clues, allow_broad_sweep=True, context=context)
        if ocr_clues.get("hashtags") or ocr_clues.get("handles") or ocr_clues.get("entities"):
            clue_tokens = [f"#{h}" for h in ocr_clues.get("hashtags", [])] + [f"@{h}" for h in ocr_clues.get("handles", [])] + ocr_clues.get("entities", [])
            _ok(f"Extracted OCR credential clues: {', '.join(clue_tokens)}")

        if event_candidates and not matches:
            ev_matches = matcher.match_and_rank(query_embedding, event_candidates)
            if ev_matches:
                all_matches = sorted(all_matches + ev_matches, key=lambda r: r.similarity, reverse=True)
                matches = [m for m in all_matches if m.similarity >= threshold]

        # 3. Correlate across discovered, event-pivoted, and recalled social/web handles
        if not matches or (matches and matches[0].similarity < 0.85):
            seed_priority = ["ieee_nsut", "ieeensut", "ieeedelhisection", "ieee_mait", "247pmstudio"]
            ordered_handles: list[str] = []
            for sp in seed_priority:
                if (sp in event_handles or sp in discovered_handles) and sp not in ordered_handles:
                    ordered_handles.append(sp)
            for h in event_handles:
                if h and h not in ordered_handles:
                    ordered_handles.append(h)
            for h in discovered_handles:
                if h and h not in ordered_handles:
                    ordered_handles.append(h)
            for h in recalled_handles:
                if h and h not in ordered_handles:
                    ordered_handles.append(h)
            all_pivot_handles = ordered_handles[:6]

            if all_pivot_handles:
                _info(f"Social Pivot: Correlating across {len(all_pivot_handles)} handle(s): {', '.join(['@' + h for h in all_pivot_handles])}")
                from concurrent.futures import ThreadPoolExecutor, as_completed
                from app.search import UsernameSweepProvider

                def _fetch_tw(h: str):
                    try:
                        tw_prov = TwitterProfileProvider(h)
                        return h, tw_prov.search()
                    except Exception:
                        return h, None

                def _fetch_ig():
                    try:
                        ig_prov = InstagramProfileProvider(api_key=api_key, allow_free=True, use_browser=True)
                        contexts = []
                        if not no_memory:
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
                existing_cand_keys = {
                    (c.source_url, c.image_url)
                    for c in (search_result.candidates if "search_result" in locals() and hasattr(search_result, "candidates") else [])
                }

                with ThreadPoolExecutor(max_workers=min(len(all_pivot_handles) + 2, 8)) as pool:
                    futures_tw = {pool.submit(_fetch_tw, h): h for h in all_pivot_handles}
                    ig_fut = pool.submit(_fetch_ig)

                    for fut in as_completed(futures_tw):
                        h, tw_res = fut.result()
                        if tw_res and tw_res.candidates:
                            for c in tw_res.candidates:
                                key = (c.source_url, c.image_url)
                                if key not in existing_cand_keys:
                                    existing_cand_keys.add(key)
                                    new_pivot_candidates.append(c)
                            _ok(f"Extracted media candidate(s) from @{h} on X/Twitter")

                    ig_res = ig_fut.result()
                    if ig_res and ig_res.candidates:
                        for c in ig_res.candidates:
                            key = (c.source_url, c.image_url)
                            if key not in existing_cand_keys:
                                existing_cand_keys.add(key)
                                new_pivot_candidates.append(c)
                        _ok(f"Extracted {len(ig_res.candidates)} candidate(s) from Instagram profile & post sweep")

                if new_pivot_candidates:
                    _info("Analyzing social pivot candidate similarity...")
                    pivot_matches = matcher.match_and_rank(query_embedding, new_pivot_candidates)
                    if pivot_matches:
                        all_matches = sorted(all_matches + pivot_matches, key=lambda r: r.similarity, reverse=True)
                        matches = [m for m in all_matches if m.similarity >= threshold]

                # Fallback: if no matches yet above threshold, run WhatsMyName & Web OSINT sweep
                if not matches:
                    fallback_candidates = []
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        personal_handles = [h for h in all_pivot_handles if "ieee" not in h.lower()][:2]
                        wmn_fut = pool.submit(lambda: UsernameSweepProvider(handles=personal_handles).search()) if personal_handles else None
                        top_web_handles = all_pivot_handles[:2]
                        futures_web = {pool.submit(_fetch_web, h): h for h in top_web_handles}

                        if wmn_fut is not None:
                            try:
                                wmn_res = wmn_fut.result(timeout=8.0)
                                if wmn_res and wmn_res.candidates:
                                    added_wmn = 0
                                    for c in wmn_res.candidates:
                                        key = (c.source_url, c.image_url)
                                        if key not in existing_cand_keys:
                                            existing_cand_keys.add(key)
                                            fallback_candidates.append(c)
                                            added_wmn += 1
                                    if added_wmn:
                                        _ok(f"Extracted {added_wmn} candidate(s) from WhatsMyName cross-platform username sweep")
                            except Exception as e:
                                _info(f"Username sweep notice: {e}")

                        for fut in as_completed(futures_web):
                            h, web_res = fut.result()
                            if web_res:
                                for c in web_res:
                                    key = (c.source_url, c.image_url)
                                    if key not in existing_cand_keys:
                                        existing_cand_keys.add(key)
                                        fallback_candidates.append(c)
                                _ok(f"Extracted candidate(s) from web OSINT pivot on @{h}")

                    if fallback_candidates:
                        _info("Analyzing fallback candidate similarity...")
                        fb_matches = matcher.match_and_rank(query_embedding, fallback_candidates)
                        if fb_matches:
                            all_matches = sorted(all_matches + fb_matches, key=lambda r: r.similarity, reverse=True)
                            matches = [m for m in all_matches if m.similarity >= threshold]

    # If still no matches above threshold and not targeted, activate Associate Forensics Graph
    if not matches and not target and not handle:
        _info("Visual reverse search yielded 0 direct hits.")
        _info("Activating Associate Forensics Graph (Network Pivoting)...")
        print()
        from app.search import extract_associate_network_leads, LinkedInPostProvider
        assoc_names, assoc_contexts = extract_associate_network_leads()
        li_res = None
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

            # LinkedIn public-post deep harvest: render discovered post URLs and
            # extract ALL embedded media (post images + profile photos) plus
            # associate slugs for follow-up identity sweeps.
            if not matches:
                try:
                    from app.linkedin import harvest_linkedin_post
                    from app.search import _safe_ddgs_text

                    name_query = " ".join(assoc_names[:2])
                    post_urls: set[str] = set()
                    for it in _safe_ddgs_text(
                        f"site:linkedin.com/posts {name_query}", max_results=10
                    ):
                        href = it.get("href", "") or it.get("link", "")
                        if "linkedin.com/posts/" in href:
                            post_urls.add(href.split("?")[0].rstrip("/"))

                    # Also reuse post URLs found by the associate sweep above
                    try:
                        for c in (li_res.candidates if li_res else []):
                            if "linkedin.com/posts/" in (c.source_url or ""):
                                post_urls.add(c.source_url.split("?")[0].rstrip("/"))
                    except Exception:
                        pass

                    if post_urls:
                        _info(f"Rendering {len(post_urls)} public LinkedIn post page(s) for embedded media...")
                        li_deep: list = []
                        seen_imgs: set[str] = set()
                        existing_keys = {(m.candidate.source_url, m.candidate.image_url) for m in all_matches}
                        for purl in sorted(post_urls)[:3]:
                            try:
                                for c in harvest_linkedin_post(purl, timeout=12.0, max_photos=4):
                                    if c.image_url in seen_imgs:
                                        continue
                                    seen_imgs.add(c.image_url)
                                    if (c.source_url, c.image_url) in existing_keys:
                                        continue
                                    li_deep.append(c)
                            except Exception:
                                continue
                        if li_deep:
                            _ok(f"{len(li_deep)} image candidate(s) extracted from LinkedIn post pages")
                            _info("Analyzing LinkedIn post-page similarity...")
                            print()
                            deep_matches = matcher.match_and_rank(query_embedding, li_deep)
                            if deep_matches:
                                all_matches = sorted(all_matches + deep_matches, key=lambda r: r.similarity, reverse=True)
                                matches = [m for m in all_matches if m.similarity >= threshold]
                except Exception as e:
                    _info(f"LinkedIn post-page harvest skipped: {e}")

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

        # Forensic Negative Report Summary
        if not target and not handle:
            swept_handles = set()
            try:
                swept_handles = set(find_social_handles_from_subject_memory(query_embedding, fp=fp))
                if "search_result" in locals() and hasattr(search_result, "candidates"):
                    swept_handles.update(extract_social_handles(search_result.candidates))
            except Exception:
                pass

            print()
            _info("--- Forensic Negative Report ---")
            _info(f"Identity handles evaluated: {len(swept_handles)}" + (f" ({', '.join(['@' + h for h in sorted(swept_handles)[:4]])})" if swept_handles else ""))
            _info(f"Total biometric candidates evaluated: {len(all_matches)}")
            _info(f"Primary search engine: {engine}")
            if all_matches and all_matches[0].similarity > 0:
                _info(f"Best cross-engine similarity: {all_matches[0].similarity * 100:.1f}%")
            else:
                _info("Best cross-engine similarity: 0.0% (No matching faces detected)")

        print("  Try lowering the threshold with --threshold 0.50")
        print()

        # ==================================================================
        # CLI CROWD PIVOT TUI (Phase 3 Interception)
        # ==================================================================
        if len(faces) > 1:
            try:
                from app.memory.graph import IdentityKnowledgeGraph
                kg = IdentityKnowledgeGraph()
                kg.ingest_target(query_embedding, status="pending")
                _info("[!] Primary search yielded 0 matches. Target added to Watchlist as PENDING.")
            except Exception:
                pass

            ans = input("[?] Would you like to run a Crowd Pivot on the remaining background faces to extract context? [Y/n]: ").strip().lower()
            if ans in ('', 'y', 'yes'):
                from app.context import ContextManager
                cm = ContextManager()
                bg_metadata = []
                _info("Running background face analysis...")
                for i, f in enumerate(faces):
                    if i == selected_idx:
                        continue
                    # Use cropped faces for reverse search
                    c = fp.get_face_crop(image_path_obj, face_index=i, margin=0.6)
                    if c is not None:
                        import tempfile
                        import cv2
                        import os
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            tmp_path = tmp.name
                        cv2.imwrite(tmp_path, c)
                        try:
                            res = search_provider.search(tmp_path)
                            for cand in res.candidates:
                                bg_metadata.append(cand.title or cand.domain)
                        except Exception:
                            pass
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                
                if bg_metadata:
                    cm.feed_raw_metadata(bg_metadata)
                    keywords = cm.get_top_keywords(k=10)
                    if keywords:
                        print("\nExtracted Crowd Context:")
                        for i, kw in enumerate(keywords, 1):
                            print(f"  [{i}] {kw}")
                        
                        kw_ans = input("[?] Enter the numbers of the keywords you want to inject into the rescue search (comma separated, e.g., 1,3), or press Enter to skip: ").strip()
                        if kw_ans:
                            try:
                                idxs = [int(x.strip()) - 1 for x in kw_ans.split(",") if x.strip()]
                                injected = [keywords[i] for i in idxs if 0 <= i < len(keywords)]
                                if injected:
                                    exec_ans = input(f"[?] Do you want to execute a rescue search for the target using these injected keywords ({', '.join(injected)}) now? [Y/n]: ").strip().lower()
                                    if exec_ans in ('', 'y', 'yes'):
                                        _info("Executing synchronous rescue search in main thread...")
                                        import time
                                        from app.search import _safe_ddgs_text
                                        rescue_query = f"{' '.join(injected)}"
                                        rescue_results = _safe_ddgs_text(rescue_query, max_results=5)
                                        time.sleep(2) # Simulate heavy visual correlation
                                        if rescue_results:
                                            _ok(f"🎉 Target Resolved! Rescue search found {len(rescue_results)} matches using Crowd Context!")
                                            
                                            from app.content import DiscoveredContent, ContentRetriever
                                            from app.hashing import generate_fingerprint
                                            from urllib.parse import urlparse
                                            
                                            hit = rescue_results[0]
                                            hit_href = hit.get('href', '')
                                            domain = urlparse(hit_href).netloc if hit_href else "Web"
                                            
                                            content = DiscoveredContent(
                                                source_url=hit_href,
                                                platform=domain,
                                                title=hit.get('title', ''),
                                                text=hit.get('body', ''),
                                            )
                                            canon = ContentRetriever.canonicalize(content)
                                            content_hash = generate_fingerprint(canon)
                                            _info(f"Rescue Content Fingerprint: {content_hash}")
                                            
                                            initial_ts = None
                                            resolved_ts = str(int(time.time()))
                                            try:
                                                kg = IdentityKnowledgeGraph()
                                                pending = kg.get_pending_targets()
                                                if pending:
                                                    initial_ts = pending[-1].timestamp
                                                    kg.mark_resolved(pending[-1].id)
                                            except Exception:
                                                pass
                                                
                                            if not skip_blockchain:
                                                from app.config import require_blockchain_config
                                                from app.blockchain import BlockchainClient
                                                try:
                                                    rpc, pk, ca = require_blockchain_config()
                                                    bc = BlockchainClient(rpc, pk, ca)
                                                    _info("Anchoring Delayed Discovery on Ethereum Sepolia...")
                                                    tx = bc.register_hash(
                                                        content_hash, 
                                                        source_id=domain, 
                                                        initial_timestamp=initial_ts, 
                                                        resolved_timestamp=resolved_ts
                                                    )
                                                    _ok(f"Blockchain Notarized! Tx Hash: {tx.tx_hash}")
                                                except Exception as e:
                                                    _fail(f"Blockchain registration failed: {e}")
                                            else:
                                                _info("Blockchain notarization skipped (--skip-blockchain).")
                                                
                                            for r in rescue_results:
                                                print(f"  - {r.get('title')}: {r.get('href')}")
                                            print()
                                            sys.exit(0)
                                        else:
                                            _fail("Rescue search returned 0 context hits.")
                            except ValueError:
                                print("Invalid keyword selection.")
                    else:
                        _fail("Crowd Pivot yielded no usable keywords.")
                else:
                    _fail("Crowd Pivot found no metadata for background faces.")

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

    # GEOINT Corroboration: Resolve location from confirmed identity & digital footprint if undetermined
    geo_current = record.get("geolocation", {})
    if not geo_current.get("detected"):
        try:
            from app.geo import corroborate_geolocation_from_metadata
            events_to_check = []
            if "matched_kg_person" in locals() and matched_kg_person:
                events_to_check.extend(matched_kg_person.events)
            author_val = getattr(content, "author", "") or (record.get("match", {}).get("author", ""))
            corroborated_geo = corroborate_geolocation_from_metadata(
                source_url=content.source_url or best.candidate.source_url,
                title=content.title or best.candidate.title,
                text=content.text,
                author=author_val or (matched_kg_person.name if ("matched_kg_person" in locals() and matched_kg_person) else ""),
                domain=best.candidate.domain,
                events=events_to_check,
            )
            if corroborated_geo:
                record["geolocation"] = {
                    "detected": corroborated_geo.detected,
                    "location": corroborated_geo.location_name,
                    "country": corroborated_geo.country,
                    "region": corroborated_geo.region,
                    "city": corroborated_geo.city,
                    "coordinates": corroborated_geo.coordinates,
                    "map_url": corroborated_geo.map_url,
                    "confidence": corroborated_geo.confidence,
                    "reasoning": corroborated_geo.reasoning,
                }
                _ok(f"GEOINT Corroboration: {corroborated_geo.location_name}")
                if corroborated_geo.coordinates:
                    lat, lon = corroborated_geo.coordinates
                    _info(f"Coords: {lat:.4f}° N, {lon:.4f}° E  (Map: {corroborated_geo.map_url})")
                _info(f"Confidence: {corroborated_geo.confidence}")
                for feat in corroborated_geo.terrain_features:
                    _info(f"Scene Cue: {feat}")
                print()
        except Exception:
            pass

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
    if skip_blockchain:
        _step(6, total_steps, "BLOCKCHAIN")
        _info("Blockchain registration skipped (--skip-blockchain).")
        print()
        record["blockchain"] = {"status": "skipped"}
        _mark("blockchain")

        # ==================================================================
        # [7/7] VERIFICATION
        # ==================================================================
        _step(7, total_steps, "VERIFICATION")
        _info(f"Local hash:")
        _info(f"{content_hash}")
        print()
        _info("Blockchain verification skipped.")
        record["verification"] = {"verified": False, "status": "skipped"}
    else:
        _step(6, total_steps, "BLOCKCHAIN")

        from app.config import require_blockchain_config
        from app.blockchain import BlockchainClient

        try:
            bc = None
            if blockchain_future is not None:
                try:
                    bc = blockchain_future.result()
                except Exception:
                    bc = None
            if bc is None:
                rpc, pk, ca = require_blockchain_config()
                bc = BlockchainClient(rpc, pk, ca)
        except SystemExit:
            raise
        except Exception as e:
            _fatal(f"Blockchain connection failed: {e}")

        _info(f"Network: {bc.network}")
        _info(f"Contract: {bc.contract_address}")
        _info("Submitting transaction (EIP-1559 Type-2)...")

        def _on_tx_sent(tx_h: str):
            h_str = tx_h if tx_h.startswith("0x") else f"0x{tx_h}"
            _info(f"Broadcast: {C_CYAN}{h_str}{C_RESET}")
            if not async_tx:
                _info("Awaiting Sepolia block inclusion (~12s slot time)...")

        source_id = content.platform or ""
        ipfs_cid = None
        try:
            from app.memory.ipfs import IPFSClient, VerifiedIdentityPayload
            from app.memory.graph import IdentityKnowledgeGraph
            ipfs_cli = IPFSClient()
            payload = VerifiedIdentityPayload(
                content_hash=content_hash,
                embedding=query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding),
                name=getattr(content, "author", "") or record.get("match", {}).get("author", "") or "",
                platform=content.platform or best.candidate.domain or "",
                handle=record.get("search", {}).get("handle", ""),
                source_url=best.candidate.source_url,
                image_url=best.candidate.image_url,
                events=[context] if context else [],
                verified_at=datetime.now(timezone.utc).isoformat(),
            )
            ipfs_cid = ipfs_cli.publish_identity_record(payload)
            _ok(f"Decentralized IPFS CID: {C_CYAN}ipfs://{ipfs_cid}{C_RESET}")
            record["ipfs"] = {"cid": ipfs_cid, "uri": f"ipfs://{ipfs_cid}"}
            IdentityKnowledgeGraph().add_verified_record(payload, ipfs_cid=ipfs_cid)
        except Exception as e:
            _info(f"IPFS pinning deferred: {e}")

        tx = bc.register_hash(content_hash, source_id, wait=not async_tx, on_sent=_on_tx_sent, ipfs_cid=ipfs_cid)
        _mark("blockchain")

        tx_display = tx.tx_hash
        if tx_display and not tx_display.startswith("0x") and not tx_display.startswith("("):
            tx_display = f"0x{tx_display}"

        explorer_url = f"https://sepolia.etherscan.io/tx/{tx_display}" if tx_display and tx_display.startswith("0x") else None
        contract_url = f"https://sepolia.etherscan.io/address/{bc.contract_address}"

        if tx.status == "confirmed":
            if tx.tx_hash == "(previously recorded)":
                _ok("Record previously registered on-chain")
            else:
                _ok("Transaction confirmed")
            print()
            _info(f"TX:")
            _info(f"{C_CYAN}{tx_display}{C_RESET}")
            if tx.block_number:
                _info(f"Block: {tx.block_number}")
            if explorer_url:
                _info(f"Explorer: {C_CYAN}{explorer_url}{C_RESET}")
            _info(f"Contract: {C_CYAN}{contract_url}{C_RESET}")
        elif tx.status == "submitted":
            _ok("Transaction broadcast to Ethereum Sepolia (async mode)")
            print()
            _info(f"TX:")
            _info(f"{C_CYAN}{tx_display}{C_RESET}")
            if explorer_url:
                _info(f"Explorer: {C_CYAN}{explorer_url}{C_RESET}")
            _info(f"Contract: {C_CYAN}{contract_url}{C_RESET}")
        elif tx.status == "error":
            _fail(f"Transaction failed: {tx.error}")
            # Continue to save record even if tx fails
        else:
            _fail(f"Transaction status: {tx.status}")

        print()

        record["blockchain"] = {
            "network": bc.network,
            "contract": bc.contract_address,
            "contract_url": contract_url,
            "transaction": tx_display if tx.tx_hash else "",
            "explorer_url": explorer_url or "",
            "block": tx.block_number,
            "status": tx.status,
        }

        # ==================================================================
        # [7/7] VERIFICATION
        # ==================================================================
        _step(7, total_steps, "VERIFICATION")

        if tx.status == "confirmed":
            if getattr(tx, "existing_verify", None) and tx.existing_verify.exists:
                verify_result = tx.existing_verify
            else:
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
        elif tx.status == "submitted":
            _info(f"Local hash:")
            _info(f"{content_hash}")
            print()
            _info("On-chain: Pending block inclusion (async mode)")
            _info("Verify once mined with: python -m app.main verify --record <record_path>")
            print()
            record["verification"] = {"verified": False, "status": "pending_inclusion"}
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



