"""
IPFS and decentralized content-addressing module for FaceTrace.

Handles:
- Building canonical verified identity payloads (embeddings, metadata, graph edges).
- Calculating deterministic IPFS CIDv1 (raw/sha256/base32) identifiers.
- Uploading/pinning payloads to IPFS (via public Pinata API if configured, with local content-addressed fallback).
- Fetching and resolving payloads from public decentralized IPFS gateways.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from app.config import DATA_DIR

IPFS_CACHE_DIR = DATA_DIR / "memory" / "ipfs_cache"
IPFS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_GATEWAYS = [
    "https://gateway.pinata.cloud/ipfs/",
    "https://dweb.link/ipfs/",
    "https://ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
]


@dataclass
class VerifiedIdentityPayload:
    """Canonical Web3 record of a verified facial identity and social edges."""
    version: str = "1.0"
    content_hash: str = ""
    embedding: list[float] = field(default_factory=list)
    name: str = ""
    platform: str = ""
    handle: str = ""
    source_url: str = ""
    image_url: str = ""
    associates: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    verified_at: str = ""
    blockchain_tx: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedIdentityPayload:
        valid_fields = {f for f in cls.__dataclass_fields__}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def calculate_ipfs_cid(data_bytes: bytes) -> str:
    """
    Calculate deterministic standard IPFS CIDv1 (raw-binary, sha2-256, base32).
    Format: 0x01 (CIDv1) + 0x55 (raw multicodec) + 0x12 (sha2-256) + 0x20 (32 bytes length) + digest.
    Prefix 'b' for RFC 4648 base32 lowercase without padding.
    """
    digest = hashlib.sha256(data_bytes).digest()
    multihash_prefix = bytes([0x01, 0x55, 0x12, 0x20])
    raw_cid = multihash_prefix + digest
    b32 = base64.b32encode(raw_cid).decode("ascii").rstrip("=").lower()
    return f"b{b32}"


class IPFSClient:
    """Client for decentralized content storage and resolution."""

    def __init__(self, pinata_jwt: Optional[str] = None, timeout: float = 5.0):
        self._jwt = pinata_jwt or os.getenv("PINATA_JWT") or ""
        self._timeout = timeout

    def publish_identity_record(self, payload: VerifiedIdentityPayload) -> str:
        """
        Canonicalize identity record, compute CIDv1, cache locally, and pin to IPFS if configured.
        Returns the IPFS CID string.
        """
        payload_dict = payload.to_dict()
        canonical_bytes = json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cid = calculate_ipfs_cid(canonical_bytes)

        # 1. Save to local content-addressed cache
        cache_path = IPFS_CACHE_DIR / f"{cid}.json"
        cache_path.write_bytes(canonical_bytes)

        # 2. Pin to IPFS via Pinata if JWT is configured
        if self._jwt and not self._jwt.startswith("your_"):
            try:
                headers = {
                    "Authorization": f"Bearer {self._jwt}",
                    "Content-Type": "application/json",
                }
                body = {
                    "pinataContent": payload_dict,
                    "pinataMetadata": {
                        "name": f"facetrace-identity-{cid[:12]}",
                        "keyvalues": {
                            "name": payload.name or "unknown",
                            "platform": payload.platform or "web",
                        },
                    },
                }
                resp = requests.post(
                    "https://api.pinata.cloud/pinning/pinJSONToIPFS",
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    remote_cid = resp.json().get("IpfsHash")
                    if remote_cid:
                        return remote_cid
            except Exception:
                pass

        return cid

    def resolve_identity_record(self, cid: str) -> Optional[VerifiedIdentityPayload]:
        """
        Fetch and parse a VerifiedIdentityPayload by its IPFS CID.
        Checks local cache first, then public decentralized IPFS gateways.
        """
        clean_cid = cid.strip().removeprefix("ipfs://").strip()
        if not clean_cid:
            return None

        # 1. Local cache hit
        cache_path = IPFS_CACHE_DIR / f"{clean_cid}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return VerifiedIdentityPayload.from_dict(data)
            except Exception:
                pass

        # 2. Resolve via public IPFS gateways
        headers = {
            "User-Agent": "FaceTrace-Decentralized-OSINT/1.0",
            "Accept": "application/json",
        }
        for gateway in PUBLIC_GATEWAYS:
            gateway_url = f"{gateway}{clean_cid}"
            try:
                resp = requests.get(gateway_url, headers=headers, timeout=self._timeout)
                if resp.status_code == 200 and resp.text:
                    data = json.loads(resp.text)
                    payload = VerifiedIdentityPayload.from_dict(data)
                    # Cache resolved payload locally
                    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    return payload
            except Exception:
                continue

        return None
