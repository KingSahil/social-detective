"""
Blockchain interaction — compile, deploy, register, and verify content hashes
on Ethereum Sepolia via Web3.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import CONTRACTS_DIR


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TxResult:
    """Result of a blockchain transaction."""
    tx_hash: str = ""
    block_number: int = 0
    status: str = "unknown"  # "confirmed" | "failed" | "error"
    contract_address: str = ""
    network: str = ""
    error: str = ""


@dataclass
class VerifyResult:
    """Result of an on-chain verification query."""
    exists: bool = False
    timestamp: int = 0
    source_id: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Contract compilation helper
# ---------------------------------------------------------------------------

def compile_contract() -> tuple[str, list]:
    """
    Compile ``ContentRegistry.sol`` using py-solc-x.

    Returns (bytecode, abi).
    """
    try:
        from solcx import compile_standard, install_solc  # type: ignore
    except ImportError:
        print("\n  ✗ py-solc-x not installed. Run: pip install py-solc-x\n")
        sys.exit(1)

    sol_path = CONTRACTS_DIR / "ContentRegistry.sol"
    if not sol_path.exists():
        raise FileNotFoundError(f"Contract not found: {sol_path}")

    source = sol_path.read_text(encoding="utf-8")

    # Install solc version if needed
    solc_version = "0.8.19"
    install_solc(solc_version)

    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {
                "ContentRegistry.sol": {"content": source},
            },
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode.object"],
                    }
                }
            },
        },
        solc_version=solc_version,
    )

    contract_data = compiled["contracts"]["ContentRegistry.sol"]["ContentRegistry"]
    abi = contract_data["abi"]
    bytecode = contract_data["evm"]["bytecode"]["object"]

    # Save ABI for future use
    abi_path = CONTRACTS_DIR / "ContentRegistry.json"
    abi_path.write_text(json.dumps(abi, indent=2), encoding="utf-8")

    return bytecode, abi


def load_abi() -> list:
    """Load the saved ABI from ``contracts/ContentRegistry.json``."""
    abi_path = CONTRACTS_DIR / "ContentRegistry.json"
    if not abi_path.exists():
        # Compile to generate it
        _, abi = compile_contract()
        return abi
    return json.loads(abi_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Blockchain client
# ---------------------------------------------------------------------------

class BlockchainClient:
    """Interact with the ContentRegistry contract on Ethereum Sepolia."""

    def __init__(self, rpc_url: str, private_key: str, contract_address: str):
        try:
            from web3 import Web3  # type: ignore
        except ImportError:
            print("\n  ✗ web3 not installed. Run: pip install web3\n")
            sys.exit(1)

        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self._w3.is_connected():
            raise ConnectionError(f"Cannot connect to RPC: {rpc_url}")

        self._account = self._w3.eth.account.from_key(private_key)
        self._address = self._account.address

        abi = load_abi()
        self._contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi,
        )
        self._network = "Ethereum Sepolia"
        self._contract_address = contract_address

    @property
    def network(self) -> str:
        return self._network

    @property
    def contract_address(self) -> str:
        return self._contract_address

    def register_hash(self, content_hash_hex: str, source_id: str = "") -> TxResult:
        """
        Call ``registerRecord(bytes32, string)`` on-chain.

        Parameters
        ----------
        content_hash_hex : str
            64-char hex SHA-256 hash (with or without 0x prefix).
        source_id : str
            Non-sensitive identifier (e.g. domain name).
        """
        from app.hashing import hex_to_bytes32

        hash_bytes = hex_to_bytes32(content_hash_hex)

        # Check if already registered on-chain
        try:
            existing = self.verify_hash(content_hash_hex)
            if existing.exists:
                return TxResult(
                    tx_hash="(previously recorded)",
                    block_number=0,
                    status="confirmed",
                    contract_address=self._contract_address,
                    network=self._network,
                )
        except Exception:
            pass

        try:
            nonce = self._w3.eth.get_transaction_count(self._address, "pending")

            # Estimate gas dynamically with a buffer
            try:
                est_gas = self._contract.functions.registerRecord(
                    hash_bytes, source_id
                ).estimate_gas({"from": self._address})
                gas_limit = int(est_gas * 1.5)
            except Exception:
                gas_limit = 500_000

            # 25% buffer on gas price to avoid underpriced replacement
            current_gas_price = self._w3.eth.gas_price
            gas_price = int(current_gas_price * 1.25)

            tx = self._contract.functions.registerRecord(
                hash_bytes, source_id
            ).build_transaction({
                "from": self._address,
                "nonce": nonce,
                "gas": gas_limit,
                "gasPrice": gas_price,
            })

            signed = self._w3.eth.account.sign_transaction(tx, self._account.key)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            return TxResult(
                tx_hash=receipt.transactionHash.hex(),
                block_number=receipt.blockNumber,
                status="confirmed" if receipt.status == 1 else "failed",
                contract_address=self._contract_address,
                network=self._network,
            )

        except Exception as e:
            return TxResult(
                status="error",
                error=str(e),
                contract_address=self._contract_address,
                network=self._network,
            )

    def verify_hash(self, content_hash_hex: str) -> VerifyResult:
        """
        Call ``verifyRecord(bytes32)`` (view) on-chain.

        Returns whether the hash exists and its registration timestamp.
        """
        from app.hashing import hex_to_bytes32

        hash_bytes = hex_to_bytes32(content_hash_hex)

        try:
            exists, timestamp, source_id = self._contract.functions.verifyRecord(
                hash_bytes
            ).call()
            return VerifyResult(
                exists=exists,
                timestamp=timestamp,
                source_id=source_id,
            )
        except Exception as e:
            return VerifyResult(error=str(e))
