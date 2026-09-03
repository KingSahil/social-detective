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
        results: list[MatchResult] = []

        for cand in candidates:
            result = self._process_candidate(query_embedding, cand)
            if result is not None:
                results.append(result)

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    def match_candidates(
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

            # Try to get an embedding (best face if multiple)
            embedding = self._fp.get_best_embedding_from_image(img)
            if embedding is None:
                return MatchResult(
                    candidate=candidate,
                    similarity=0.0,
                    face_detected=False,
                    error="No face detected in candidate image",
                )

            sim = cosine_similarity(query_embedding, embedding)
            return MatchResult(
                candidate=candidate,
                similarity=sim,
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
            resp = requests.get(url, timeout=self._timeout, stream=True)
            resp.raise_for_status()
            data = resp.content
            # Decode with OpenCV
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
