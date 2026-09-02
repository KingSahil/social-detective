"""Tests for SHA-256 fingerprinting and canonicalization."""

import hashlib
import json
import pytest

from app.hashing import generate_fingerprint, hex_to_bytes32
from app.content import ContentRetriever, DiscoveredContent


class TestGenerateFingerprint:
    def test_deterministic(self):
        s = '{"key":"value"}'
        h1 = generate_fingerprint(s)
        h2 = generate_fingerprint(s)
        assert h1 == h2

    def test_known_hash(self):
        # SHA-256 of "test" is well-known
        h = generate_fingerprint("test")
        assert h == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

    def test_different_inputs(self):
        h1 = generate_fingerprint("hello")
        h2 = generate_fingerprint("world")
        assert h1 != h2

    def test_empty_string(self):
        h = generate_fingerprint("")
        assert len(h) == 64  # SHA-256 always 64 hex chars


class TestHexToBytes32:
    def test_conversion(self):
        h = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        b = hex_to_bytes32(h)
        assert isinstance(b, bytes)
        assert len(b) == 32

    def test_with_0x_prefix(self):
        h = "0x9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        b = hex_to_bytes32(h)
        assert len(b) == 32


class TestCanonicalization:
    def test_deterministic(self):
        c = DiscoveredContent(
            source_url="http://example.com",
            image_url="http://img.com/1.jpg",
            text="hello",
            image_bytes=b"fakeimage",
        )
        s1 = ContentRetriever.canonicalize(c)
        s2 = ContentRetriever.canonicalize(c)
        assert s1 == s2

    def test_sorted_keys(self):
        c = DiscoveredContent(
            source_url="http://example.com",
            image_url="http://img.com/1.jpg",
            text="hello",
            image_bytes=b"fakeimage",
        )
        s = ContentRetriever.canonicalize(c)
        parsed = json.loads(s)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_includes_image_hash(self):
        image_bytes = b"actual-image-content-here"
        c = DiscoveredContent(
            source_url="http://example.com",
            image_url="http://img.com/1.jpg",
            text="hello",
            image_bytes=image_bytes,
        )
        s = ContentRetriever.canonicalize(c)
        parsed = json.loads(s)
        expected_hash = hashlib.sha256(image_bytes).hexdigest()
        assert parsed["image_hash"] == expected_hash

    def test_different_image_bytes_different_hash(self):
        c1 = DiscoveredContent(
            source_url="http://example.com",
            image_url="http://img.com/1.jpg",
            text="hello",
            image_bytes=b"image_version_1",
        )
        c2 = DiscoveredContent(
            source_url="http://example.com",
            image_url="http://img.com/1.jpg",
            text="hello",
            image_bytes=b"image_version_2",
        )
        s1 = ContentRetriever.canonicalize(c1)
        s2 = ContentRetriever.canonicalize(c2)
        assert s1 != s2  # different image bytes → different canonical string

    def test_canonicalize_from_record(self):
        image_hash = hashlib.sha256(b"img").hexdigest()
        record = {
            "match": {
                "source_url": "http://example.com",
                "image_url": "http://img.com/1.jpg",
            },
            "content": {
                "text": "hello",
                "image_hash": image_hash,
            },
        }
        s = ContentRetriever.canonicalize_from_record(record)
        parsed = json.loads(s)
        assert parsed["image_hash"] == image_hash
        assert parsed["source_url"] == "http://example.com"
        assert parsed["text"] == "hello"


class TestVerificationRoundTrip:
    """Test that canonicalize → hash → save → reload → canonicalize_from_record → hash produces the same result."""

    def test_roundtrip(self):
        image_bytes = b"some-real-image-bytes-1234567890"
        content = DiscoveredContent(
            source_url="http://example.com/post/123",
            image_url="http://example.com/img.jpg",
            text="Test post content",
            image_bytes=image_bytes,
        )

        # Original canonicalization + hash
        canonical = ContentRetriever.canonicalize(content)
        original_hash = generate_fingerprint(canonical)

        # Build record as the main pipeline would
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        record = {
            "match": {
                "source_url": content.source_url,
                "image_url": content.image_url,
            },
            "content": {
                "text": content.text,
                "image_hash": image_hash,
                "source_url": content.source_url,
                "image_url": content.image_url,
            },
            "fingerprint": {
                "algorithm": "SHA-256",
                "hash": original_hash,
            },
        }

        # Reconstruct from record
        reconstructed = ContentRetriever.canonicalize_from_record(record)
        reconstructed_hash = generate_fingerprint(reconstructed)

        assert reconstructed_hash == original_hash

    def test_tamper_detected(self):
        image_bytes = b"original-image"
        content = DiscoveredContent(
            source_url="http://example.com",
            image_url="http://img.com/1.jpg",
            text="Original text",
            image_bytes=image_bytes,
        )

        canonical = ContentRetriever.canonicalize(content)
        original_hash = generate_fingerprint(canonical)

        # Build record and tamper with it
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        record = {
            "match": {
                "source_url": content.source_url,
                "image_url": content.image_url,
            },
            "content": {
                "text": "TAMPERED text",  # ← modified!
                "image_hash": image_hash,
            },
            "fingerprint": {
                "hash": original_hash,
            },
        }

        reconstructed = ContentRetriever.canonicalize_from_record(record)
        reconstructed_hash = generate_fingerprint(reconstructed)

        assert reconstructed_hash != original_hash  # tamper detected!
