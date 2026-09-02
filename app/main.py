"""
FaceTrace — Face Search + Blockchain Verification CLI.

End-to-end pipeline:
    Face image → Face detection → Face encoding → Web search →
    Candidate matching → Content retrieval → SHA-256 fingerprint →
    Blockchain record → Verification

Usage:
    python -m app.main --image ./data/input/face.jpg
    python -m app.main --image ./data/input/face.jpg --threshold 0.60
    python -m app.main verify --record ./data/results/record.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    C_GREEN = Fore.GREEN
    C_RED = Fore.RED
    C_YELLOW = Fore.YELLOW
    C_CYAN = Fore.CYAN
    C_BOLD = Style.BRIGHT
    C_RESET = Style.RESET_ALL
except ImportError:
    C_GREEN = C_RED = C_YELLOW = C_CYAN = C_BOLD = C_RESET = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    print()
    print("=" * 60)
    print(f"{C_BOLD}               FACETRACE{C_RESET}")
    print(f"      Face Search + Blockchain Verification")
    print("=" * 60, flush=True)
    print()


def _step(num: int, total: int, title: str) -> None:
    print(f"  {C_BOLD}[{num}/{total}] {title}{C_RESET}", flush=True)


def _ok(msg: str) -> None:
    print(f"        {C_GREEN}✓{C_RESET} {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"        {C_RED}✗{C_RESET} {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"        {msg}", flush=True)


def _fatal(msg: str) -> None:
    print()
    _fail(msg)
    print()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, threshold: float = 0.70, platform: str | None = None) -> None:
    """Execute the full FaceTrace pipeline."""

    _banner()

    image_path_obj = Path(image_path).resolve()
    if not image_path_obj.exists():
        _fatal(f"Image not found: {image_path}")

    record: dict = {}
    total_steps = 7

    # ==================================================================
    # [1/7] FACE DETECTION
    # ==================================================================
    _step(1, total_steps, "FACE DETECTION")

    from app.face import FaceProcessor, FaceProcessingError

    try:
        fp = FaceProcessor()
    except Exception as e:
        _fatal(f"Failed to initialize face processor: {e}")

    try:
        query_embedding = fp.get_embedding(str(image_path_obj))
    except FaceProcessingError as e:
        _fail(str(e))
        print()
        sys.exit(1)
    except Exception as e:
        _fatal(f"Face processing error: {e}")

    _ok("Face detected")
    _ok(f"Face embedding generated ({query_embedding.shape[0]}-d)")
    print()

    record["query"] = {
        "image": str(image_path_obj),
        "face_detected": True,
        "embedding_dim": int(query_embedding.shape[0]),
    }

    # ==================================================================
    # [2/7] WEB SEARCH
    # ==================================================================
    _step(2, total_steps, "WEB SEARCH")

    from app.config import require_search_config
    from app.search import SerpAPIProvider

    api_key = require_search_config()
    search_provider = SerpAPIProvider(api_key=api_key)
    _info(f"Provider: {search_provider.PROVIDER_NAME}")
    _info("Searching...")

    try:
        search_result = search_provider.search(str(image_path_obj))
    except Exception as e:
        _fatal(f"Search failed: {e}")

    candidate_count = len(search_result.candidates)
    if candidate_count == 0:
        _fail("No candidates found")
        print()
        print("  The search returned no visual matches.")
        print("  Try a different image or check your SerpAPI quota.")
        print()
        sys.exit(1)

    _ok("Search completed")
    _ok(f"{candidate_count} candidates discovered across the web")
    
    # Show platforms discovered across the web
    discovered_platforms = sorted({c.domain for c in search_result.candidates if c.domain})
    if discovered_platforms:
        _info(f"Sources found: {', '.join(discovered_platforms[:7])}" + (f" (+{len(discovered_platforms)-7} more)" if len(discovered_platforms) > 7 else ""))
    print()

    record["search"] = {
        "provider": search_result.provider,
        "searched_at": search_result.searched_at,
        "candidate_count": candidate_count,
        "platforms": discovered_platforms,
    }

    # ==================================================================
    # [3/7] FACE MATCHING
    # ==================================================================
    _step(3, total_steps, "FACE MATCHING")
    _info("Analyzing candidate face similarity...")
    print()

    from app.matcher import FaceMatcher

    # Filter candidates by platform if specified
    search_candidates = search_result.candidates
    if platform:
        p_lower = platform.lower()
        search_candidates = [
            c for c in search_candidates 
            if p_lower in c.domain.lower() or p_lower in c.source_url.lower() or p_lower in c.title.lower()
        ]
        if not search_candidates:
            _fail(f"No candidates found matching platform: {platform}")
            print()
            sys.exit(1)
        _info(f"Filtering to platform '{platform}': {len(search_candidates)} candidates")

    matcher = FaceMatcher(fp)
    matches = matcher.match_candidates(
        query_embedding, search_candidates, threshold=threshold
    )

    # Display top results (show up to 10)
    display_matches = matches[:10]
    for i, m in enumerate(display_matches, 1):
        pct = m.similarity * 100
        color = C_GREEN if m.similarity >= 0.85 else (C_YELLOW if m.similarity >= 0.70 else C_RED)
        platform_label = m.candidate.domain or m.candidate.title[:30] or "Web"
        print(f"        #{i:<3} {color}Similarity: {pct:.1f}%{C_RESET}  [{platform_label}]")

    if not matches:
        _fail(f"No candidates above threshold ({threshold:.0%})")
        print()
        print("  Try lowering the threshold with --threshold 0.50")
        print()
        sys.exit(1)

    print()
    best = matches[0]
    _ok(f"Strongest candidate selected: {best.candidate.domain or 'Web'} (Similarity: {best.similarity*100:.1f}%)")
    print()

    record["match"] = {
        "source_url": best.candidate.source_url,
        "image_url": best.candidate.image_url,
        "similarity": round(best.similarity, 4),
        "domain": best.candidate.domain,
        "title": best.candidate.title,
    }

    # ==================================================================
    # [4/7] CONTENT RETRIEVAL
    # ==================================================================
    _step(4, total_steps, "CONTENT RETRIEVAL")

    from app.content import ContentRetriever

    retriever = ContentRetriever()
    content = retriever.retrieve(best.candidate.source_url, best.candidate.image_url)

    _ok("Matching content retrieved")
    print()
    _info(f"Source:")
    _info(f"{C_CYAN}{content.source_url}{C_RESET}")
    if content.title:
        _info(f"Title: {content.title}")
    if content.platform:
        _info(f"Platform: {content.platform}")
    print()

    # Compute image hash for record
    image_hash = ""
    if content.image_bytes:
        image_hash = hashlib.sha256(content.image_bytes).hexdigest()
        _ok(f"Image downloaded ({len(content.image_bytes)} bytes)")
    else:
        _info("(Image bytes not available — metadata-only fingerprint)")
    print()

    record["content"] = {
        "source_url": content.source_url,
        "image_url": content.image_url,
        "platform": content.platform,
        "title": content.title,
        "text": content.text,
        "image_hash": image_hash,
        "retrieved_at": content.retrieved_at,
    }

    # ==================================================================
    # [5/7] FINGERPRINT
    # ==================================================================
    _step(5, total_steps, "FINGERPRINT")

    from app.hashing import generate_fingerprint

    canonical = ContentRetriever.canonicalize(content)
    content_hash = generate_fingerprint(canonical)

    _info(f"Algorithm: SHA-256")
    print()
    _info(f"{C_CYAN}{content_hash}{C_RESET}")
    print()

    record["fingerprint"] = {
        "algorithm": "SHA-256",
        "hash": content_hash,
    }

    # ==================================================================
    # [6/7] BLOCKCHAIN
    # ==================================================================
    _step(6, total_steps, "BLOCKCHAIN")

    from app.config import require_blockchain_config
    from app.blockchain import BlockchainClient

    try:
        rpc, pk, ca = require_blockchain_config()
        bc = BlockchainClient(rpc, pk, ca)
    except SystemExit:
        raise
    except Exception as e:
        _fatal(f"Blockchain connection failed: {e}")

    _info(f"Network: {bc.network}")
    _info(f"Contract: {bc.contract_address}")
    _info("Submitting transaction...")

    source_id = content.platform or ""
    tx = bc.register_hash(content_hash, source_id)

    if tx.status == "confirmed":
        _ok("Transaction confirmed")
        print()
        _info(f"TX:")
        _info(f"{C_CYAN}0x{tx.tx_hash}{C_RESET}")
        _info(f"Block: {tx.block_number}")
    elif tx.status == "error":
        _fail(f"Transaction failed: {tx.error}")
        # Continue to save record even if tx fails
    else:
        _fail(f"Transaction status: {tx.status}")

    print()

    record["blockchain"] = {
        "network": bc.network,
        "contract": bc.contract_address,
        "transaction": f"0x{tx.tx_hash}" if tx.tx_hash else "",
        "block": tx.block_number,
        "status": tx.status,
    }

    # ==================================================================
    # [7/7] VERIFICATION
    # ==================================================================
    _step(7, total_steps, "VERIFICATION")

    if tx.status == "confirmed":
        verify_result = bc.verify_hash(content_hash)

        _info(f"Local hash:")
        _info(f"{content_hash}")
        print()

        if verify_result.exists:
            _info(f"On-chain: ✓ Hash found")
            print()
            _ok("CONTENT VERIFIED")
            record["verification"] = {"verified": True}
        else:
            _info(f"On-chain: Hash not found")
            print()
            _fail("VERIFICATION FAILED")
            record["verification"] = {"verified": False, "error": "Hash not found on-chain"}
    else:
        _info("Blockchain transaction was not confirmed — skipping on-chain verification")
        record["verification"] = {"verified": False, "error": f"TX status: {tx.status}"}

    print()

    # ==================================================================
    # Save record
    # ==================================================================
    from app.config import RESULTS_DIR

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    record_filename = f"{timestamp_str}_record.json"
    record_path = RESULTS_DIR / record_filename
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Record saved:")
    print(f"  {C_CYAN}{record_path}{C_RESET}")
    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="facetrace",
        description="FaceTrace — Face Search + Blockchain Verification",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default / search command
    parser.add_argument(
        "--image",
        help="Path to the query face image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum face similarity threshold (default: 0.70).",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Optional: filter candidates to a specific platform (e.g. instagram, wikipedia, x.com).",
    )

    # Verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify a saved record.")
    verify_parser.add_argument(
        "--record",
        required=True,
        help="Path to the saved record JSON file.",
    )

    args = parser.parse_args()

    if args.command == "verify":
        from app.verify import verify_record, _print_verification
        result = verify_record(args.record)
        _print_verification(result)
        sys.exit(0 if result.get("verified") else 1)

    elif args.image:
        run_pipeline(args.image, threshold=args.threshold, platform=args.platform)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
