"""
Verification — reload a saved record, regenerate the fingerprint,
query the blockchain, and compare.

Usage:
    python -m app.verify --record ./data/results/record.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.content import ContentRetriever
from app.hashing import generate_fingerprint


def verify_record(record_path: str | Path, blockchain_client=None) -> dict:
    """
    Verify a saved investigation record against its on-chain fingerprint.

    Parameters
    ----------
    record_path : str | Path
        Path to the saved ``*_record.json`` file.
    blockchain_client : BlockchainClient, optional
        If not provided, one will be created from env vars.

    Returns
    -------
    dict
        Verification result with keys: local_hash, onchain_hash, verified, error.
    """
    record_path = Path(record_path)
    if not record_path.exists():
        return {"verified": False, "error": f"Record not found: {record_path}"}

    # Load the record
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"verified": False, "error": f"Invalid JSON: {e}"}

    # Reconstruct canonical content and regenerate hash
    canonical = ContentRetriever.canonicalize_from_record(record)
    local_hash = generate_fingerprint(canonical)

    # Original hash from the record
    original_hash = record.get("fingerprint", {}).get("hash", "")

    # Query blockchain
    onchain_hash = ""
    blockchain_info = record.get("blockchain", {})

    if blockchain_client is None:
        try:
            from app.config import require_blockchain_config
            rpc, pk, ca = require_blockchain_config()
            from app.blockchain import BlockchainClient
            blockchain_client = BlockchainClient(rpc, pk, ca)
        except (SystemExit, Exception) as e:
            # If blockchain is not configured, compare against original hash
            return {
                "local_hash": local_hash,
                "original_hash": original_hash,
                "onchain_hash": "",
                "verified": local_hash == original_hash,
                "blockchain_available": False,
                "error": "" if local_hash == original_hash else "Content has been modified (compared against original record hash)",
            }

    # Query the ORIGINAL hash on-chain (the one that was registered)
    result = blockchain_client.verify_hash(original_hash)

    if result.error:
        return {
            "local_hash": local_hash,
            "original_hash": original_hash,
            "onchain_hash": "",
            "verified": False,
            "blockchain_available": False,
            "error": f"Blockchain query failed: {result.error}",
        }

    if not result.exists:
        return {
            "local_hash": local_hash,
            "original_hash": original_hash,
            "onchain_hash": "",
            "verified": False,
            "blockchain_available": True,
            "error": "Hash not found on-chain",
        }

    # The on-chain hash exists — now check if local hash matches what was registered
    verified = local_hash == original_hash

    return {
        "local_hash": local_hash,
        "original_hash": original_hash,
        "onchain_exists": True,
        "onchain_timestamp": result.timestamp,
        "verified": verified,
        "blockchain_available": True,
        "error": "" if verified else "TAMPER DETECTED — content has been modified since registration",
    }


def _print_verification(result: dict) -> None:
    """Pretty-print verification result to terminal."""
    print()
    print("=" * 60)
    print("           VERIFICATION")
    print("=" * 60)
    print()

    local = result.get("local_hash", "")
    original = result.get("original_hash", "")
    verified = result.get("verified", False)

    print(f"  Local hash (current):")
    print(f"  {local}")
    print()
    print(f"  Original hash (registered):")
    print(f"  {original}")
    print()

    if result.get("blockchain_available"):
        if result.get("onchain_exists"):
            print(f"  On-chain:  ✓ Hash found")
            ts = result.get("onchain_timestamp", 0)
            if ts:
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                print(f"  Registered: {dt.isoformat()}")
        else:
            print(f"  On-chain:  ✗ Hash NOT found")
    else:
        print(f"  On-chain:  (blockchain not available)")

    print()

    if verified:
        print("  ✓ CONTENT VERIFIED")
        print("  The content has not been modified since registration.")
    else:
        print("  ✗ TAMPER DETECTED")
        error = result.get("error", "Content has been modified")
        print(f"  {error}")

    print()
    print("=" * 60)
    print()


def main() -> None:
    """CLI entry point for standalone verification."""
    parser = argparse.ArgumentParser(
        description="Verify a saved FaceTrace investigation record."
    )
    parser.add_argument(
        "--record",
        required=True,
        help="Path to the saved record JSON file.",
    )
    args = parser.parse_args()

    result = verify_record(args.record)
    _print_verification(result)

    sys.exit(0 if result.get("verified") else 1)


if __name__ == "__main__":
    main()
