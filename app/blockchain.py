"""
Blockchain interaction — compile, deploy, register, and verify content hashes
on Ethereum Sepolia via Web3.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

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
    existing_verify: Optional[VerifyResult] = None


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
        try:
            self._chain_id = self._w3.eth.chain_id
        except Exception:
            self._chain_id = 11155111

    @property
    def network(self) -> str:
        return self._network

    @property
    def contract_address(self) -> str:
        return self._contract_address

    def register_hash(
        self,
        content_hash_hex: str,
        source_id: str = "",
        wait: bool = True,
        priority_fee_gwei: float = 2.5,
        on_sent: Optional[Callable[[str], None]] = None,
    ) -> TxResult:
        """
        Call ``registerRecord(bytes32, string)`` on-chain using EIP-1559 Type-2 transactions.

        Parameters
        ----------
        content_hash_hex : str
            64-char hex SHA-256 hash (with or without 0x prefix).
        source_id : str
            Non-sensitive identifier (e.g. domain name).
        wait : bool
            Whether to wait for the transaction to be mined into a block.
        priority_fee_gwei : float
            Validator priority fee (tip) in gwei (default: 2.5). Guarantees
            inclusion in the very next Ethereum Sepolia block slot (~12s).
        on_sent : Optional[Callable[[str], None]]
            Callback executed immediately after raw transaction broadcast.
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
                    existing_verify=existing,
                )
        except Exception:
            pass

        try:
            nonce = self._w3.eth.get_transaction_count(self._address, "pending")

            # EIP-1559 Type-2 dynamic fee calculation
            priority_fee = self._w3.to_wei(priority_fee_gwei, "gwei")
            try:
                latest_block = self._w3.eth.get_block("latest")
                base_fee = latest_block.get("baseFeePerGas", self._w3.to_wei(1, "gwei"))
            except Exception:
                base_fee = self._w3.to_wei(1, "gwei")

            # 2x base fee headroom + priority fee
            max_fee = int(base_fee * 2) + priority_fee

            # Safe fixed gas limit for registerRecord(bytes32, string)
            # avoids unnecessary estimate_gas RPC roundtrip (~400ms)
            gas_limit = 150_000

            tx = self._contract.functions.registerRecord(
                hash_bytes, source_id
            ).build_transaction({
                "from": self._address,
                "nonce": nonce,
                "gas": gas_limit,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": priority_fee,
                "chainId": getattr(self, "_chain_id", 11155111),
            })

            signed = self._w3.eth.account.sign_transaction(tx, self._account.key)
            raw_tx = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hash_hex = raw_tx.hex()

            if on_sent:
                try:
                    on_sent(tx_hash_hex)
                except Exception:
                    pass

            if not wait:
                return TxResult(
                    tx_hash=tx_hash_hex,
                    block_number=0,
                    status="submitted",
                    contract_address=self._contract_address,
                    network=self._network,
                )

            receipt = self._w3.eth.wait_for_transaction_receipt(raw_tx, timeout=120, poll_latency=0.5)

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
