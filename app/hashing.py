"""
SHA-256 fingerprinting of canonical content.
"""

from __future__ import annotations

import hashlib


def generate_fingerprint(canonical_string: str) -> str:
    """
    Compute the SHA-256 hex digest of *canonical_string*.

    The same canonical string will always produce the same hash.
    """
    return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()


def hex_to_bytes32(hex_hash: str) -> bytes:
    """
    Convert a 64-char hex SHA-256 digest to a 32-byte ``bytes`` object
    suitable for passing to a Solidity ``bytes32`` parameter.
    """
    clean = hex_hash.removeprefix("0x")
    return bytes.fromhex(clean)
