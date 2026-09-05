"""
Candidate face matching — download candidate images, extract embeddings,
compute cosine similarity against the query embedding, and rank results.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image

from app.face import FaceProcessor
from app.search import Candidate


@dataclass
class MatchResult:
    """One candidate's matching outcome."""
    candidate: Candidate
    similarity: float
    face_detected: bool
    error: str = ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class FaceMatcher:
    """Download candidate images, extract face embeddings, rank by similarity."""

    def __init__(self, face_processor: FaceProcessor, timeout: float = 3.5):
        self._fp = face_processor
        self._timeout = timeout
        # Shared keep-alive session: candidate images often come from the same
        # CDN hosts; reusing connections avoids a fresh TCP+TLS handshake each.
        self._http = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=35, pool_maxsize=35)
        self._http.mount("https://", adapter)
        self._http.mount("http://", adapter)
        self._http.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })

    def match_and_rank(
        self,
        query_embedding: np.ndarray,
        candidates: list[Candidate],
    ) -> list[MatchResult]:
        """
        For each candidate, download the image, extract a face embedding,
        and compute cosine similarity. Returns all results sorted by similarity (descending).

        Candidates sharing the same image URL are downloaded and embedded
        once; the result is fanned out to every duplicate.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Deduplicate by image URL (empty URLs get a unique key)
        unique: dict[str, Candidate] = {}
        keys: list[str] = []
        for c in candidates:
            k = (c.image_url or "").strip() or f"__idx_{id(c)}"
            unique[k] = c
            keys.append(k)

        def _worker(item) -> tuple[str, Optional[MatchResult]]:
            k, cand = item
            return k, self._process_candidate(query_embedding, cand)

        def _domain_priority(item):
            k, cand = item
            dom = (cand.domain or "").lower()
            src = (cand.source_url or "").lower()
            # Primary social / professional profile sources prioritized first
            if any(p in dom or p in src for p in ("linkedin.com", "x.com", "twitter.com", "github.com", "instagram.com")):
                return 0
            # High-reputation identity / media
            if any(p in dom or p in src for p in ("web3", "facebook.com", "youtube.com", "medium.com")):
                return 1
            return 2

        results_by_key: dict[str, MatchResult] = {}
        items = list(unique.items())
        items.sort(key=_domain_priority)

        batch_size = 24
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            max_workers = min(24, max(1, len(batch)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {executor.submit(_worker, item): item[0] for item in batch}
                for fut in as_completed(future_to_key):
                    try:
                        k, result = fut.result()
                        if result is not None:
                            results_by_key[k] = result
                    except Exception:
                        pass

            # Early exit: if we already found an unequivocal biometric match (similarity >= 85%),
            # skip downloading and evaluating the remaining low-relevance scraper sites.
            if any(r.similarity >= 0.85 for r in results_by_key.values()):
                break

        # Fan unique results back out to every candidate (duplicates share
        # the MatchResult outcome but keep their own candidate object).
        results: list[MatchResult] = []
        for cand, k in zip(candidates, keys):
            r = results_by_key.get(k)
            if r is None:
                results.append(MatchResult(
                    candidate=cand, similarity=0.0,
                    face_detected=False, error="not processed",
                ))
            elif r.candidate is cand:
                results.append(r)
            else:
                results.append(MatchResult(
                    candidate=cand, similarity=r.similarity,
                    face_detected=r.face_detected, error=r.error,
                ))

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)

        # Profile URL prioritization: if top result is a post URL but a direct
        # profile URL (/in/) is within 0.05 similarity, prioritize the profile.
        if len(results) > 1:
            top_sim = results[0].similarity
            for i in range(1, min(5, len(results))):
                if top_sim - results[i].similarity <= 0.05:
                    u = (results[i].candidate.source_url or "").lower()
                    top_u = (results[0].candidate.source_url or "").lower()
                    if "/in/" in u and "/posts/" in top_u:
                        best = results.pop(i)
                        results.insert(0, best)
                        break

        return results

    def match_above_threshold(
        self,
        query_embedding: np.ndarray,
        candidates: list[Candidate],
        threshold: float = 0.70,
    ) -> list[MatchResult]:
        """
        For each candidate, download the image, extract a face embedding,
        and compute similarity. Returns results sorted by similarity (desc),
        filtered by *threshold*.
        """
        results = self.match_and_rank(query_embedding, candidates)
        return [r for r in results if r.similarity >= threshold]

    # Alias for backwards compatibility
    match_candidates = match_above_threshold


    def _process_candidate(
        self, query_embedding: np.ndarray, candidate: Candidate
    ) -> Optional[MatchResult]:
        """Download one candidate image and compare."""
        try:
            img = self._download_image(candidate.image_url)
            if img is None:
                return MatchResult(
                    candidate=candidate,
                    similarity=0.0,
                    face_detected=False,
                    error="Failed to download image",
                )

            # Fast downscale large candidate images (e.g. 2000px, 4K) to max 640px.
            # Preserves ArcFace facial feature fidelity while matching native 640x640 detector.
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 640:
                scale = 640.0 / max_dim
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Multi-face inspection: inspect ALL detected faces in the candidate image
            # (e.g. event cards, builder badges, group photos)
            faces = self._fp._app.get(img)
            if not faces:
                return MatchResult(
                    candidate=candidate,
                    similarity=0.0,
                    face_detected=False,
                    error="No face detected in candidate image",
                )

            # Compute similarity against every face found and take the highest
            sims = [
                cosine_similarity(query_embedding, f.embedding)
                for f in faces
                if f.embedding is not None
            ]
            best_sim = max(sims) if sims else 0.0

            return MatchResult(
                candidate=candidate,
                similarity=float(best_sim),
                face_detected=True,
            )

        except Exception as e:
            return MatchResult(
                candidate=candidate,
                similarity=0.0,
                face_detected=False,
                error=str(e),
            )

    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Download an image URL and decode it as a BGR numpy array."""
        try:
            resp = self._http.get(url, timeout=min(5.0, self._timeout))
            resp.raise_for_status()
            data = resp.content
            # Decode with OpenCV
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
