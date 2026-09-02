"""Tests for blockchain data formatting (no credentials required)."""

import pytest

from app.hashing import hex_to_bytes32, generate_fingerprint


class TestBlockchainDataFormatting:
    def test_bytes32_from_sha256(self):
        """SHA-256 hex → bytes32 for Solidity."""
        h = generate_fingerprint("test content")
        b = hex_to_bytes32(h)
        assert len(b) == 32
        assert isinstance(b, bytes)

    def test_bytes32_roundtrip(self):
        """bytes32 → hex → bytes32 is lossless."""
        h = generate_fingerprint("roundtrip test")
        b = hex_to_bytes32(h)
        reconstructed_hex = b.hex()
        assert reconstructed_hex == h

    def test_abi_encoding_consistency(self):
        """Same content hash always produces the same bytes32."""
        canonical = '{"image_hash":"abc","image_url":"http://x","source_url":"http://y","text":"z"}'
        h1 = generate_fingerprint(canonical)
        h2 = generate_fingerprint(canonical)
        b1 = hex_to_bytes32(h1)
        b2 = hex_to_bytes32(h2)
        assert b1 == b2


class TestContractABI:
    def test_abi_loads(self):
        """ABI file should exist after compilation test."""
        from app.blockchain import load_abi
        abi = load_abi()
        assert isinstance(abi, list)
        assert len(abi) > 0

        # Check expected functions exist
        func_names = {e["name"] for e in abi if e.get("type") == "function"}
        assert "registerRecord" in func_names
        assert "verifyRecord" in func_names
