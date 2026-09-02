"""
Deploy the ContentRegistry smart contract to Ethereum Sepolia.

Usage:
    python scripts/deploy_contract.py

Requires RPC_URL and PRIVATE_KEY in .env (or exported).
Prints the deployed contract address on success.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import require_env, RPC_URL, PRIVATE_KEY, CONTRACTS_DIR
from app.blockchain import compile_contract


def main() -> None:
    rpc_url = require_env("RPC_URL", RPC_URL)
    private_key = require_env("PRIVATE_KEY", PRIVATE_KEY)

    print()
    print("=" * 60)
    print("  ContentRegistry — Contract Deployment")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Compile
    # ------------------------------------------------------------------
    print("  [1/3] Compiling ContentRegistry.sol ...")
    bytecode, abi = compile_contract()
    print("        ✓ Compilation successful")
    print()

    # ------------------------------------------------------------------
    # 2. Connect
    # ------------------------------------------------------------------
    print("  [2/3] Connecting to Ethereum Sepolia ...")
    try:
        from web3 import Web3  # type: ignore
    except ImportError:
        print("        ✗ web3 not installed. Run: pip install web3")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"        ✗ Cannot connect to RPC: {rpc_url}")
        sys.exit(1)
    print("        ✓ Connected")

    account = w3.eth.account.from_key(private_key)
    balance = w3.eth.get_balance(account.address)
    print(f"        Deployer: {account.address}")
    print(f"        Balance:  {w3.from_wei(balance, 'ether')} ETH")
    if balance == 0:
        print("        ✗ Insufficient balance. Get Sepolia ETH from a faucet.")
        sys.exit(1)
    print()

    # ------------------------------------------------------------------
    # 3. Deploy
    # ------------------------------------------------------------------
    print("  [3/3] Deploying contract ...")
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(account.address)

    # Estimate gas dynamically with a buffer
    try:
        est_gas = contract.constructor().estimate_gas({"from": account.address})
        gas_limit = int(est_gas * 1.5)
        print(f"        Estimated gas: {est_gas} (limit set to {gas_limit})")
    except Exception:
        gas_limit = 2_000_000
        print(f"        Gas limit set to: {gas_limit}")

    tx = contract.constructor().build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": gas_limit,
        "gasPrice": w3.eth.gas_price,
    })

    signed = w3.eth.account.sign_transaction(tx, account.key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"        TX: 0x{tx_hash.hex()}")
    print("        Waiting for confirmation ...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    if receipt.status != 1:
        print("        ✗ Deployment failed!")
        sys.exit(1)

    contract_address = receipt.contractAddress
    print(f"        ✓ Contract deployed!")
    print()
    print(f"  Contract deployed:")
    print(f"  {contract_address}")
    print()
    print(f"  Add to your .env file:")
    print(f"  CONTRACT_ADDRESS={contract_address}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
