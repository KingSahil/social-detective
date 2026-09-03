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

    def __init__(self, face_processor: FaceProcessor, timeout: int = 15):
        self._fp = face_processor
        self._timeout = timeout

    def match_and_rank(
        self,
        query_embedding: np.ndarray,
        candidates: list[Candidate],
    ) -> list[MatchResult]:
        """
        For each candidate, download the image, extract a face embedding,
        and compute cosine similarity. Returns all results sorted by similarity (descending).
        """
        from concurrent.futures import ThreadPoolExecutor

        results: list[MatchResult] = []

        def _worker(cand: Candidate) -> Optional[MatchResult]:
            return self._process_candidate(query_embedding, cand)

        max_workers = min(20, max(1, len(candidates)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_worker, candidates):
                if result is not None:
                    results.append(result)

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
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, timeout=min(8.0, self._timeout), headers=headers)
            resp.raise_for_status()
            data = resp.content
            # Decode with OpenCV
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
