"""
Configuration — loads environment variables and provides application settings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
RESULTS_DIR = DATA_DIR / "results"
CONTRACTS_DIR = _PROJECT_ROOT / "contracts"

# Ensure output dirs exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Search provider
# ---------------------------------------------------------------------------
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
LENS_HEADLESS: bool = os.getenv("LENS_HEADLESS", "true").lower() not in ("false", "0", "no")
LENS_TIMEOUT: float = float(os.getenv("LENS_TIMEOUT", "25.0"))

# ---------------------------------------------------------------------------
# Tor / Anonymity
# ---------------------------------------------------------------------------
USE_TOR: bool = os.getenv("USE_TOR", "false").lower() not in ("false", "0", "no", "")
TOR_PROXY_URL: str = os.getenv("TOR_PROXY_URL", "socks5://127.0.0.1:9050")
TOR_CONTROL_PORT: int = int(os.getenv("TOR_CONTROL_PORT", "9051"))
TOR_PASSWORD: str = os.getenv("TOR_PASSWORD", "")

# ---------------------------------------------------------------------------
# Blockchain
# ---------------------------------------------------------------------------
RPC_URL: str = os.getenv("RPC_URL", "")
PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")
NETWORK_NAME: str = "Ethereum Sepolia"

# ---------------------------------------------------------------------------
# Face matching
# ---------------------------------------------------------------------------
DEFAULT_SIMILARITY_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Identity pivot & username sweep
# ---------------------------------------------------------------------------
PIVOT_ENABLED: bool = os.getenv("PIVOT_ENABLED", "true").lower() not in ("false", "0", "no")
PIVOT_ENGINE: str = os.getenv("PIVOT_ENGINE", "wmn")
PIVOT_MAX_SITES: int = int(os.getenv("PIVOT_MAX_SITES", "300"))
PIVOT_TIMEOUT: float = float(os.getenv("PIVOT_TIMEOUT", "8.0"))
PIVOT_SWEEP_TIMEOUT: float = float(os.getenv("PIVOT_SWEEP_TIMEOUT", "30.0"))
PIVOT_MAX_WORKERS: int = int(os.getenv("PIVOT_MAX_WORKERS", "12"))
PIVOT_MAX_ACCOUNTS: int = int(os.getenv("PIVOT_MAX_ACCOUNTS", "25"))
PIVOT_MAX_CANDIDATES: int = int(os.getenv("PIVOT_MAX_CANDIDATES", "50"))
PIVOT_EXHAUSTIVE: bool = os.getenv("PIVOT_EXHAUSTIVE", "false").lower() not in ("false", "0", "no")
PIVOT_BROWSER_FALLBACK: bool = os.getenv("PIVOT_BROWSER_FALLBACK", "false").lower() not in ("false", "0", "no")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_env(name: str, value: str) -> str:
    """Return *value* if non-empty and not a placeholder, otherwise exit with a clear message."""
    placeholders = ("your_", "YOUR_", "0x_deployed", "_here")
    if not value or any(p in value for p in placeholders):
        print(f"\n  ✗ Missing or unconfigured environment variable: {name}")
        print(f"    Current value: '{value}'")
        print(f"    Set a valid value in your .env file or export it in your shell.")
        print(f"    See .env.example for details.\n")
        sys.exit(1)
    return value


def require_search_config(optional: bool = True) -> str:
    """
    Validate search-provider configuration.
    If optional=True, returns empty string when SERPAPI_KEY is not configured,
    allowing self-contained free headless search to run.
    """
    placeholders = ("your_", "YOUR_", "_here")
    if not SERPAPI_KEY or any(p in SERPAPI_KEY for p in placeholders):
        if optional:
            return ""
        return require_env("SERPAPI_KEY", SERPAPI_KEY)
    return SERPAPI_KEY


def require_blockchain_config() -> tuple[str, str, str]:
    """Validate blockchain configuration and return (rpc_url, private_key, contract_address)."""
    rpc = require_env("RPC_URL", RPC_URL)
    pk = require_env("PRIVATE_KEY", PRIVATE_KEY)
    ca = require_env("CONTRACT_ADDRESS", CONTRACT_ADDRESS)
    return rpc, pk, ca
