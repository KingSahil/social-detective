"""
Migration script to transfer verified investigation records from data/results/*.json
into the decentralized Web3 IdentityKnowledgeGraph (data/memory/knowledge_graph.json).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from app.config import DATA_DIR, RESULTS_DIR
from app.face import FaceProcessor
from app.memory.graph import IdentityKnowledgeGraph
from app.memory.ipfs import IPFSClient, VerifiedIdentityPayload


NAME_MAPPINGS = {
    "sahil": "Sahil Gupta",
    "supreme__sahil on instagram:": "Sahil Gupta",
    "aryan gupta on x:": "Aryan Gupta",
    "tweet by @aryannn_6476476": "Aryan Gupta",
    "gndu": "GNDU Freshers",
}


def migrate_results_to_knowledge_graph() -> int:
    """Migrate and deduplicate past verified records into the knowledge graph."""
    if not RESULTS_DIR.exists():
        print("No data/results/ directory found.")
        return 0

    fp = FaceProcessor()
    ipfs_cli = IPFSClient()
    kg = IdentityKnowledgeGraph()

    processed = 0
    records = sorted(RESULTS_DIR.glob("*.json"))

    for rf in records:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)

            match = data.get("match")
            if not match or match.get("similarity", 0) < 0.70:
                continue

            img_path = data.get("query", {}).get("image")
            if not img_path or not Path(img_path).exists():
                continue

            author = match.get("author") or data.get("content", {}).get("author") or ""
            title = match.get("title") or ""
            src_url = match.get("source_url") or ""

            if not author:
                if "gndu" in src_url.lower():
                    author = "GNDU Freshers"
                elif "max ilgner" in title.lower():
                    author = "Max Ilgner"
                elif "resul" in title.lower():
                    author = "Resul Sarioglu"
                elif "|" in title:
                    author = title.split("|")[-1].strip()
                else:
                    author = title[:30]

            if not author or len(author) < 3 or author.lower() in ("unknown", "instagram", "yandex images"):
                continue

            # Standardize name
            for k, v in NAME_MAPPINGS.items():
                if k in author.lower():
                    author = v
                    break

            emb = fp.get_embedding(img_path)
            if emb is None:
                continue

            # Associate network context
            associates = []
            events = []
            if author in ("Sahil Gupta", "Gourish Julka", "Sparsh Khanna"):
                associates = ["Sahil Gupta", "Gourish Julka", "Sparsh Khanna"]
                events = ["HackHazards '26", "Hacker House Goa"]

            payload = VerifiedIdentityPayload(
                content_hash=data.get("fingerprint", {}).get("hash", ""),
                embedding=emb.tolist(),
                name=author,
                platform=match.get("domain") or data.get("content", {}).get("platform") or "",
                source_url=src_url,
                image_url=match.get("image_url", ""),
                associates=associates,
                events=events,
                verified_at=data.get("content", {}).get("retrieved_at", ""),
                blockchain_tx=data.get("blockchain", {}).get("tx_hash", ""),
            )

            cid = ipfs_cli.publish_identity_record(payload)
            kg.add_verified_record(payload, ipfs_cid=cid)
            processed += 1
        except Exception:
            continue

    # Clean and merge duplicate identities
    clean_persons = {}
    MERGE_MAP = {
        "sahil": "sahil_gupta",
        "supreme__sahil_on_instagram:_\"": "sahil_gupta",
        "aryan_gupta_on_x:_\"im_officia": "aryan_gupta",
        "tweet_by_@aryannn_6476476": "aryan_gupta",
    }
    NOISE = {"if_elon_sold_all_his_stocks/as", "yandex_images", "instagram", "unknown_identity"}

    for p_id, p in list(kg._persons.items()):
        if p_id in NOISE or p.name.lower() in ("unknown identity", "instagram", "yandex images"):
            continue
        target_id = MERGE_MAP.get(p_id, p_id)
        if target_id not in clean_persons:
            p.id = target_id
            if target_id == "sahil_gupta":
                p.name = "Sahil Gupta"
            elif target_id == "aryan_gupta":
                p.name = "Aryan Gupta"
            clean_persons[target_id] = p
        else:
            tgt = clean_persons[target_id]
            if not tgt.embedding and p.embedding:
                tgt.embedding = p.embedding
            tgt.accounts = list(dict.fromkeys(tgt.accounts + p.accounts))
            tgt.events = list(dict.fromkeys(tgt.events + p.events))
            tgt.associates = list(dict.fromkeys(tgt.associates + p.associates))
            tgt.blockchain_hashes = list(dict.fromkeys(tgt.blockchain_hashes + p.blockchain_hashes))
            tgt.ipfs_cids = list(dict.fromkeys(tgt.ipfs_cids + p.ipfs_cids))
            for app in p.verified_appearances:
                if app not in tgt.verified_appearances:
                    tgt.verified_appearances.append(app)

    kg._persons = clean_persons
    kg.save()
    return processed


if __name__ == "__main__":
    count = migrate_results_to_knowledge_graph()
    print(f"Successfully migrated {count} records into the Knowledge Graph!")
