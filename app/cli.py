import sys

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
