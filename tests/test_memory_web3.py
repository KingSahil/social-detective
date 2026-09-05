"""
Unit tests for Web3 Memory, IPFS CID calculation, and IdentityKnowledgeGraph.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from app.memory.ipfs import (
    IPFSClient,
    VerifiedIdentityPayload,
    calculate_ipfs_cid,
)
from app.memory.graph import IdentityKnowledgeGraph, GraphPerson


def test_calculate_ipfs_cid_deterministic():
    data = b"Hello FaceTrace Web3 Memory"
    cid1 = calculate_ipfs_cid(data)
    cid2 = calculate_ipfs_cid(data)
    assert cid1 == cid2
    # Standard raw-sha256 CIDv1 in base32 starts with 'b'
    assert cid1.startswith("b")
    assert len(cid1) > 40


def test_ipfs_publish_and_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr("app.memory.ipfs.IPFS_CACHE_DIR", tmp_path)
    client = IPFSClient()

    payload = VerifiedIdentityPayload(
        content_hash="abcdef1234567890",
        embedding=[0.1, 0.2, 0.3] + [0.0] * 509,
        name="Test Detective",
        platform="x.com",
        handle="test_detective",
        source_url="https://x.com/test_detective/status/123",
        image_url="https://pbs.twimg.com/media/sample.jpg",
        associates=["partner_1", "partner_2"],
        events=["Hackathon2026"],
        verified_at="2026-09-05T12:00:00Z",
    )

    cid = client.publish_identity_record(payload)
    assert cid.startswith("b")
    assert (tmp_path / f"{cid}.json").exists()

    # Resolve back from local cache
    resolved = client.resolve_identity_record(cid)
    assert resolved is not None
    assert resolved.name == "Test Detective"
    assert resolved.handle == "test_detective"
    assert len(resolved.embedding) == 512
    assert "partner_1" in resolved.associates


def test_identity_knowledge_graph_ingest_and_search(tmp_path):
    graph_file = tmp_path / "test_graph.json"
    kg = IdentityKnowledgeGraph(store_path=graph_file)

    # 1. Create fake embedding vectors
    vec_a = np.zeros(512, dtype=np.float32)
    vec_a[0] = 1.0  # Unit vector along axis 0

    payload = VerifiedIdentityPayload(
        content_hash="hash_a",
        embedding=vec_a.tolist(),
        name="Alice Builder",
        platform="linkedin.com",
        source_url="https://linkedin.com/in/alice",
        image_url="https://media.licdn.com/alice.jpg",
        associates=["Bob Hacker"],
        events=["HackHazards"],
    )

    kg.add_verified_record(payload, ipfs_cid="bafkreitest123")
    assert "alice_builder" in kg._persons
    assert (graph_file).exists()

    # Query with exact vector
    person, sim = kg.find_nearest_person(vec_a, threshold=0.90)
    assert person is not None
    assert person.name == "Alice Builder"
    assert pytest.approx(sim, 0.001) == 1.0

    # Query with orthogonal vector (similarity 0)
    vec_b = np.zeros(512, dtype=np.float32)
    vec_b[1] = 1.0
    person_b, sim_b = kg.find_nearest_person(vec_b, threshold=0.70)
    assert person_b is None
    assert pytest.approx(sim_b, 0.001) == 0.0

    # Test appearance candidates
    cands = kg.get_appearance_candidates(person)
    assert len(cands) == 1
    assert cands[0].image_url == "https://media.licdn.com/alice.jpg"
    assert cands[0].domain == "web3-memory"

    # Test event associate lookup
    assocs = kg.get_event_associates("hackhazards")
    assert "Alice Builder" in assocs


def test_web3_memory_syncer_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.memory.ipfs.IPFS_CACHE_DIR", tmp_path)
    client = IPFSClient()

    payload = VerifiedIdentityPayload(
        content_hash="test_hash_456",
        embedding=[0.5] * 512,
        name="OnChain Identity",
        platform="x.com",
        source_url="https://x.com/onchain/status/1",
        image_url="https://x.com/img.jpg",
    )
    cid = client.publish_identity_record(payload)

    # Mock blockchain client
    class MockBlockchain:
        def __init__(self):
            class MockW3:
                eth = type("Eth", (), {"block_number": 100})()
            self._w3 = MockW3()

        def get_registered_events(self, from_block=0, to_block="latest", chunk_size=9000):
            return [
                {"source_id": f"x.com|ipfs://{cid}", "content_hash": "test_hash_456", "block_number": 10},
                {"source_id": "legacy.com", "content_hash": "legacy_hash", "block_number": 5},
            ]

    from app.memory.web3_sync import Web3MemorySyncer
    kg = IdentityKnowledgeGraph(store_path=tmp_path / "sync_graph.json")
    syncer = Web3MemorySyncer(blockchain_client=MockBlockchain(), ipfs_client=client, graph=kg)

    stats = syncer.sync(lookback_blocks=100)
    assert stats["events_scanned"] == 2
    assert stats["cids_found"] == 1
    assert stats["records_ingested"] == 1
    assert "onchain_identity" in kg._persons
