# Social Detective (FaceTrace) — Comprehensive Architecture Document

## 1. System Overview

**Social Detective (FaceTrace)** is an autonomous forensic facial intelligence (OSINT) and decentralized tamper-evident evidence notarization system. It bridges deep facial biometrics, zero-CAPTCHA reverse visual search cascades, multimodal scene intelligence, collective context correlation, and immutable Ethereum blockchain notarization.

The system is designed to solve two core problems in digital forensics:
1. **Cold-Start Identification**: Identifying unindexed or obscure individuals in photos where traditional reverse search yields zero direct hits, by leveraging **Crowd Context**, **Associate Networks**, and **Scene OCR**.
2. **Mathematical Authenticity (Zero-PII On-Chain)**: Sealing public post evidence, author information, and raw media fingerprints onto the Ethereum blockchain without ever storing private biometric vectors or personal data on-chain.

---

## 2. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Runtime & Tooling** | Python 3.10 – 3.12 (Optimized for 3.11), `uv` package manager |
| **Biometrics & Inference** | InsightFace ArcFace (`buffalo_l` pack), ONNX Runtime (CUDA / CPU), OpenCV, NumPy |
| **Visual Search & Scraping** | SerpAPI (Google Lens), Playwright Stealth (Headless Google Lens), Direct Yandex Images, DuckDuckGo OSINT, Instaloader, BeautifulSoup4, `yt-dlp` |
| **Anonymity & Routing** | Tor Network (`stem`, SOCKS5 proxy via `PySocks`) for automated circuit / IP cycling |
| **Context & Correlation** | Collective Context Correlator (C3 - `app/context.py`), WhatsMyName dataset (300+ platforms) |
| **Scene Intelligence** | RapidOCR (Lanyards, badges, rotated text), OpenStreetMap Nominatim reverse geocoding, EXIF analysis |
| **Decentralized Storage & Graph** | IPFS (CIDv1 deterministic `bafkrei...`), `IdentityKnowledgeGraph` local property graph & vector memory |
| **Blockchain Notarization** | Web3.py, Solidity 0.8.19 smart contract (`ContentRegistry.sol`), Ethereum Sepolia Testnet |
| **User Interfaces** | Cyberpunk Streamlit Web Dashboard (`app/ui.py`), Rich Terminal CLI (`app/cli.py`, `questionary`, `colorama`) |

---

## 3. Repository & Module Structure

```
social-detective/
├── app/
│   ├── __init__.py
│   ├── main.py              # CLI entry point, argument parsing, command dispatch
│   ├── pipeline.py          # Central 7-phase forensic pipeline orchestrator
│   ├── cli.py               # Terminal banner, step indicators, and TUI formatting
│   ├── ui.py                # Streamlit cyber dashboard (Single Target & Crowd Context modes)
│   ├── config.py            # Environment validation (.env), thresholds, and timeouts
│   ├── face.py              # InsightFace ArcFace intake, 512-d embeddings, crowd face annotation
│   ├── matcher.py           # Multi-threaded cosine similarity ranking & connection pooling
│   ├── search.py            # Multi-engine search cascade (Lens, Yandex, social dorking)
│   ├── context.py           # Collective Context Correlator (C3) for crowd tag ranking
│   ├── tor_manager.py       # SOCKS5 Tor proxy session manager & stem circuit renewal
│   ├── harvest.py           # Social media asset harvesting, carousel & avatar extraction
│   ├── identity.py          # WhatsMyName (WMN) 300+ platform concurrent username sweeping
│   ├── linkedin.py          # Guest-level LinkedIn public post scraper & associate extractor
│   ├── geo.py               # EXIF GPS, Nominatim reverse geocoding, terrain/scene estimation
│   ├── ocr.py               # RapidOCR text/lanyard/hashtag parsing with multi-angle rotation
│   ├── content.py           # Canonical RFC key-sorted JSON packaging of evidence
│   ├── hashing.py           # Deterministic 32-byte SHA-256 fingerprinting for bytes32
│   ├── blockchain.py        # Web3 Sepolia transaction signing, contract queries & prewarming
│   ├── verify.py            # Standalone tamper-detection audit CLI
│   └── memory/
│       ├── graph.py         # Relational property graph, pending watchlist, and vector index
│       ├── ipfs.py          # CIDv1 generation and IPFS gateway synchronization
│       ├── web3_sync.py     # Pulls on-chain event logs to populate local knowledge graph
│       └── migrate.py       # Knowledge graph data migration and schema validation
├── contracts/
│   └── ContentRegistry.sol  # Solidity smart contract for immutable fingerprint timestamping
├── data/
│   ├── input/               # Query portraits and scene images
│   ├── results/             # Forensic dossier JSON records and cropped evidence
│   ├── memory/              # Local persistent knowledge graph (knowledge_graph.json)
│   └── wmn/                 # WhatsMyName dataset (wmn-data.json, 268KB)
├── scripts/
│   └── deploy_contract.py   # Solidity compilation (py-solc-x) & Sepolia contract deployment
└── tests/                   # 12 test modules covering matching, search, hashing, and blockchain
```

---

## 4. End-to-End Pipeline Execution Flow

```
                           [1] INTAKE & BIOMETRICS
               Query image ingested ➔ Face detection (buffalo_l)
                  5-point landmark alignment ➔ 512-d ArcFace vector
                                    │
                                    ├───► [Parallel] RapidOCR (Lanyards, badges, text)
                                    ├───► [Parallel] GEOINT (EXIF GPS + Nominatim)
                                    └───► [Parallel] Pre-warm Web3 Sepolia client
                                    │
                           [2] MEMORY LOOKUP
               Local Identity Knowledge Graph checked for past cases
                                    │
                                    ▼
                      [3] MULTI-ENGINE SEARCH CASCADE
           SerpAPI Google Lens ➔ Headless Stealth Lens (Protobuf Bounds)
                  ➔ Direct Yandex Images ➔ Social Sweeps (WMN)
                                    │
                                    ▼
                          [4] BIOMETRIC MATCHING
          Download candidates via keep-alive connection pool
          Extract candidate ArcFace embeddings ➔ Cosine similarity ranking
                                    │
                                    ▼
                         [5] CANONICAL PACKAGING
          Fetch matched post, author, source URL, and raw media bytes
          Key-sorted RFC-compliant JSON payload ➔ SHA-256 hash + IPFS CIDv1
                                    │
                                    ▼
                       [6] BLOCKCHAIN NOTARIZATION
          Sign transaction on Ethereum Sepolia calling ContentRegistry.sol:
                   registerRecord(bytes32 contentHash, string sourceId)
                                    │
                                    ▼
                        [7] INDEPENDENT AUDIT CLI
          Local Dossier Hash == On-Chain Contract Hash ➔ ✓ VERIFIED / ✗ TAMPER
```

---

## 5. Advanced Subsystems & Feature Enhancements

### 5.1 Collective Context Correlator (C3 - `app/context.py`)
When searching for an obscure or non-famous target in a crowd photo, direct facial search often returns 0 results (cold target). C3 implements a **Crowd Pivot**:
1. Scans background individuals detected in the same photograph.
2. Discovers social profiles for those background individuals.
3. Parses post titles, captions, handles, proper nouns (companies, universities), and hashtags (`#HackHazards`, `@stanford`, etc.).
4. Scores and ranks context tokens using frequency and regex weighting.
5. Re-injects the top-ranked context keywords into the primary target's dorking and visual queries to unlock cold-target discovery.

### 5.2 Pending Targets Watchlist & Retroactive Delayed Discovery
- **The Problem**: A target searched today may fail because their digital footprint has not yet been indexed or correlated.
- **The Solution**: Failed targets are stored in a `pending_targets` watchlist with their biometric embeddings.
- **Retroactive Trigger**: Whenever a future search successfully resolves new associate links, events, or faces, the `IdentityKnowledgeGraph` automatically runs background cosine similarity checks against all pending targets (capped at 50 targets to conserve CPU/RAM, with a manual override toggle in the UI). When a match is found, the system alerts the investigator that a previously failed target has been resolved.

### 5.3 Anonymity & Rate-Limit Resistance (`app/tor_manager.py`)
- OSINT scrapers routinely face IP rate-limits and 429/403 blocks from Google, Instagram, and LinkedIn.
- `TorManager` wraps `requests.Session` inside a SOCKS5 proxy (`127.0.0.1:9050`) and exposes `renew_ip()` via `stem.control.Controller`.
- When an engine signals a block, FaceTrace requests a `Signal.NEWNYM` from the Tor control port, obtaining a fresh circuit and IP address without requiring costly commercial proxy subscriptions.

### 5.4 Dual-Mode Web Dashboard (`app/ui.py`)
- Built with a high-contrast cyber-forensic glassmorphic design.
- **Single Target Hunt**: Upload a portrait, view detected bounding box and embedding dimensions, stream terminal logs live, and review extracted metadata.
- **Crowd Context Mode**: Upload a crowd photo and an optional reference photo. The dashboard calculates cosine similarity against all detected faces in the group, auto-selects the primary target, draws neon bounding boxes, and provides a Phase 2 Crowd Pivot UI if the primary hunt fails.

---

## 6. Security, Privacy & Integrity Guarantees

1. **Zero Biometrics On-Chain**: Facial coordinates, raw photos, and 512-d ArcFace embeddings are never transmitted to the blockchain. Only the SHA-256 fingerprint of the public canonical record is anchored.
2. **Deterministic Tamper Detection**: Because canonical JSON sorting enforces strictly sorted keys, any alteration to a scraped post's author, text, date, or image bytes alters the 32-byte digest, causing independent verification to immediately flag `✗ TAMPER DETECTED`.
3. **Guest-Only OSINT Harvesting**: The scraper relies strictly on publicly accessible endpoints (guest HTML, public post views, WhatsMyName rule checks) and never bypasses authentication walls.
