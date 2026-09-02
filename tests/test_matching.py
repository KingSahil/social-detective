"""Tests for face similarity matching."""

import numpy as np
import pytest

from app.matcher import cosine_similarity, MatchResult
from app.search import Candidate


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_similar_vectors(self):
        a = np.random.randn(512).astype(np.float32)
        b = a + np.random.randn(512).astype(np.float32) * 0.05
        sim = cosine_similarity(a, b)
        assert sim > 0.95  # very similar

    def test_random_vectors_low_similarity(self):
        a = np.random.randn(512).astype(np.float32)
        b = np.random.randn(512).astype(np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim) < 0.3  # random vectors should be near zero

    def test_zero_vector(self):
        a = np.zeros(512)
        b = np.ones(512)
        sim = cosine_similarity(a, b)
        assert sim == 0.0


class TestMatchResult:
    def test_sorting(self):
        results = [
            MatchResult(Candidate("a", "a"), similarity=0.5, face_detected=True),
            MatchResult(Candidate("b", "b"), similarity=0.9, face_detected=True),
            MatchResult(Candidate("c", "c"), similarity=0.7, face_detected=True),
        ]
        results.sort(key=lambda r: r.similarity, reverse=True)
        assert results[0].similarity == 0.9
        assert results[1].similarity == 0.7
        assert results[2].similarity == 0.5

    def test_threshold_filtering(self):
        results = [
            MatchResult(Candidate("a", "a"), similarity=0.9, face_detected=True),
            MatchResult(Candidate("b", "b"), similarity=0.5, face_detected=True),
            MatchResult(Candidate("c", "c"), similarity=0.8, face_detected=True),
        ]
        threshold = 0.70
        filtered = [r for r in results if r.similarity >= threshold]
        assert len(filtered) == 2
