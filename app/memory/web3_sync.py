"""
Web3 On-Chain Memory Synchronization module for FaceTrace.

Listens to `RecordRegistered` events emitted on Ethereum Sepolia,
extracts decentralized IPFS CIDs, downloads verified identity records,
and populates the shared local IdentityKnowledgeGraph.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.blockchain import BlockchainClient
from app.memory.graph import IdentityKnowledgeGraph
from app.memory.ipfs import IPFSClient


class Web3MemorySyncer:
    """Synchronizes collective memory from Ethereum Sepolia and IPFS."""

    def __init__(
        self,
        blockchain_client: BlockchainClient,
        ipfs_client: Optional[IPFSClient] = None,
        graph: Optional[IdentityKnowledgeGraph] = None,
    ):
        self._bc = blockchain_client
        self._ipfs = ipfs_client or IPFSClient()
        self._graph = graph or IdentityKnowledgeGraph()

    def sync(self, lookback_blocks: int = 15000) -> dict[str, Any]:
        """
        Scan recent Sepolia events, resolve IPFS payloads, and update knowledge graph.
        """
        stats = {
            "events_scanned": 0,
            "cids_found": 0,
            "records_ingested": 0,
            "identities_in_graph": len(self._graph._persons),
        }

        try:
            latest_block = self._bc._w3.eth.block_number
            from_block = max(0, latest_block - lookback_blocks)
            events = self._bc.get_registered_events(from_block=from_block)
            stats["events_scanned"] = len(events)
        except Exception:
            return stats

        discovered_cids: set[str] = set()
        for ev in events:
            source_id = ev.get("source_id", "")
            # Match ipfs://<cid> or /ipfs/<cid> or bare bafk... / Qm... CIDs
            m_ipfs = re.search(r"ipfs[/:]{1,2}([A-Za-z0-9_-]{40,})", source_id)
            if m_ipfs:
                discovered_cids.add(m_ipfs.group(1))
            else:
                m_bare = re.search(r"\b(baf[A-Za-z0-9]{40,}|Qm[A-Za-z0-9]{44})\b", source_id)
                if m_bare:
                    discovered_cids.add(m_bare.group(1))

        stats["cids_found"] = len(discovered_cids)

        for cid in discovered_cids:
            payload = self._ipfs.resolve_identity_record(cid)
            if payload:
                self._graph.add_verified_record(payload, ipfs_cid=cid)
                stats["records_ingested"] += 1

        stats["identities_in_graph"] = len(self._graph._persons)
        return stats
