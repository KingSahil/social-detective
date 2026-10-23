import argparse
import sys
from app.pipeline import run_pipeline, _banner, _ok, _fail
from app.config import DEFAULT_SIMILARITY_THRESHOLD

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="social-detective",
        description="Social Detective — Face Search + Blockchain Verification",
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
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Minimum face similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD:.2f}).",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Optional: filter candidates to a specific platform (e.g. instagram, wikipedia, x.com).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional: direct URL of a specific post or page to verify against the query face (e.g. an X post, Reddit thread, or article).",
    )
    parser.add_argument(
        "--handle",
        "--user",
        type=str,
        default=None,
        help="Optional: search a specific user's public profile and posts on Instagram and/or X/Twitter (filter with --platform instagram or --platform twitter).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["all", "lens", "yandex", "serpapi"],
        default="all",
        help="Visual search engine: 'all' (primary SerpAPI with free fallback), 'lens' (Google Lens with fallback), 'yandex' (Yandex Images), or 'serpapi' (SerpAPI only). Default: 'all'.",
    )
    parser.add_argument(
        "--lens-visible",
        action="store_true",
        default=False,
        help="Launch Google Lens browser with visible window (useful for interactive anti-bot verification).",
    )
    parser.add_argument(
        "--async-tx",
        "--no-wait-tx",
        dest="async_tx",
        action="store_true",
        default=False,
        help="Broadcast transaction to blockchain and return immediately without waiting ~12s for block confirmation.",
    )
    parser.add_argument(
        "--skip-blockchain",
        "--no-blockchain",
        dest="skip_blockchain",
        action="store_true",
        default=False,
        help="Skip blockchain registration and verification entirely (instant OSINT visual search & facial matching only).",
    )
    parser.add_argument(
        "--no-memory",
        dest="no_memory",
        action="store_true",
        default=False,
        help="Disable subject identity memory lookup to test cold-start discovery without past cases.",
    )
    parser.add_argument(
        "--context",
        dest="context",
        type=str,
        default=None,
        help="Event, organization, or campaign context keyword to guide dynamic OSINT search (e.g. 'HackHazards', 'Symbiosis').",
    )
    parser.add_argument(
        "--sync-web3",
        dest="sync_web3",
        action="store_true",
        default=False,
        help="Synchronize collective identity memory from Ethereum Sepolia smart contract and IPFS.",
    )
    parser.add_argument(
        "--face-index",
        "--face",
        dest="face_index",
        type=int,
        default=None,
        help="Optional: index of the target face when multiple faces are detected in query image (0 = largest/primary).",
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

    if getattr(args, "sync_web3", False):
        _banner()
        print("  [WEB3 MEMORY SYNC] Connecting to Ethereum Sepolia...")
        try:
            from app.config import require_blockchain_config
            from app.blockchain import BlockchainClient
            from app.memory.web3_sync import Web3MemorySyncer
            rpc, pk, ca = require_blockchain_config()
            bc = BlockchainClient(rpc, pk, ca)
            syncer = Web3MemorySyncer(bc)
            stats = syncer.sync(lookback_blocks=25000)
            _ok(f"Sepolia Events Scanned: {stats['events_scanned']}")
            _ok(f"IPFS CIDs Discovered: {stats['cids_found']}")
            _ok(f"Identities in Shared Knowledge Graph: {stats['identities_in_graph']}")
        except Exception as e:
            _fail(f"Web3 sync failed: {e}")
        print()
        if not args.image:
            sys.exit(0)

    if args.image:
        run_pipeline(
            args.image,
            threshold=args.threshold,
            platform=args.platform,
            target=args.target,
            engine=args.engine,
            handle=args.handle,
            lens_visible=getattr(args, "lens_visible", False),
            async_tx=getattr(args, "async_tx", False),
            skip_blockchain=getattr(args, "skip_blockchain", False),
            no_memory=getattr(args, "no_memory", False),
            context=getattr(args, "context", None),
            sync_web3=getattr(args, "sync_web3", False),
            face_index=getattr(args, "face_index", None),
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
