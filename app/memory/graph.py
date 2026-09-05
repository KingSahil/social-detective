"""
Identity Knowledge Graph and Vector Memory module for FaceTrace.

Maintains:
- An in-memory and file-backed property graph (Persons, Accounts, Events).
- Relational edges (OWNS, ATTENDED, COLLABORATED_WITH).
- A 512-d ArcFace vector index for fast biometric nearest-neighbor queries.
- Ingestion of decentralized Web3/IPFS verified records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import DATA_DIR
from app.matcher import Candidate, cosine_similarity
from app.memory.ipfs import VerifiedIdentityPayload

GRAPH_STORE_FILE = DATA_DIR / "memory" / "knowledge_graph.json"
GRAPH_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class GraphPerson:
    id: str
    name: str
    embedding: list[float] = field(default_factory=list)
    accounts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    associates: list[str] = field(default_factory=list)
    verified_appearances: list[dict[str, str]] = field(default_factory=list)
    blockchain_hashes: list[str] = field(default_factory=list)
    ipfs_cids: list[str] = field(default_factory=list)


class IdentityKnowledgeGraph:
    """Relational knowledge graph and biometric vector memory."""

    def __init__(self, store_path: Path = GRAPH_STORE_FILE):
        self._store_path = store_path
        self._persons: dict[str, GraphPerson] = {}
        self._events: dict[str, set[str]] = {}  # event_name -> set of person_ids
        self.load()

    def load(self) -> None:
        """Load graph from JSON file."""
        if not self._store_path.exists():
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for p_dict in data.get("persons", []):
                p = GraphPerson(**p_dict)
                self._persons[p.id] = p
                for ev in p.events:
                    self._events.setdefault(ev.lower(), set()).add(p.id)
        except Exception:
            pass

    def save(self) -> None:
        """Persist graph to JSON file."""
        data = {
            "version": "1.0",
            "persons": [asdict(p) for p in self._persons.values()],
        }
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add_verified_record(self, payload: VerifiedIdentityPayload, ipfs_cid: str = "") -> GraphPerson:
        """Ingest a verified identity payload into the knowledge graph."""
        person_name = payload.name.strip() or "Unknown Identity"
        person_id = person_name.lower().replace(" ", "_")

        if person_id not in self._persons:
            self._persons[person_id] = GraphPerson(
                id=person_id,
                name=person_name,
                embedding=payload.embedding,
            )

        person = self._persons[person_id]
        if payload.embedding and not person.embedding:
            person.embedding = payload.embedding

        if payload.source_url:
            person.accounts = list(dict.fromkeys(person.accounts + [payload.source_url]))
        if payload.image_url and payload.source_url:
            app_entry = {"image_url": payload.image_url, "source_url": payload.source_url}
            if app_entry not in person.verified_appearances:
                person.verified_appearances.append(app_entry)

        for assoc in payload.associates:
            if assoc and assoc not in person.associates:
                person.associates.append(assoc)

        for ev in payload.events:
            if ev and ev not in person.events:
                person.events.append(ev)
                self._events.setdefault(ev.lower(), set()).add(person_id)

        if payload.content_hash and payload.content_hash not in person.blockchain_hashes:
            person.blockchain_hashes.append(payload.content_hash)

        if ipfs_cid and ipfs_cid not in person.ipfs_cids:
            person.ipfs_cids.append(ipfs_cid)

        self.save()
        return person

    def find_nearest_person(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.65,
    ) -> tuple[Optional[GraphPerson], float]:
        """Biometric nearest-neighbor search across all indexed faces in the graph."""
        best_person: Optional[GraphPerson] = None
        best_sim = -1.0

        for person in self._persons.values():
            if not person.embedding:
                continue
            emb = np.array(person.embedding, dtype=np.float32)
            sim = cosine_similarity(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                if sim >= threshold:
                    best_person = person

        return best_person, best_sim

    def get_appearance_candidates(self, person: GraphPerson) -> list[Candidate]:
        """Convert a person's verified appearances into search Candidate objects."""
        candidates = []
        for app in person.verified_appearances:
            candidates.append(Candidate(
                image_url=app["image_url"],
                source_url=app["source_url"],
                title=f"{person.name} (Verified Identity)",
                domain="web3-memory",
            ))
        return candidates

    def get_event_associates(self, event_name: str) -> list[str]:
        """Retrieve all known associate names for a given event."""
        p_ids = self._events.get(event_name.lower(), set())
        names = []
        for p_id in p_ids:
            p = self._persons.get(p_id)
            if p:
                names.append(p.name)
        return names
