# FaceTrace (SocialDetective) Architecture Document

## 1. System Overview
**FaceTrace** is an autonomous forensic facial recognition and immutable evidence-notarization pipeline. The system bridges computer vision biometrics with decentralized ledger technology (Ethereum Sepolia) to prove content origin and detect post manipulation.

## 2. Technology Stack
- **Language**: Python 3.10+
- **Biometrics & Inference**: InsightFace (`buffalo_l` model for ArcFace), ONNX Runtime, OpenCV, NumPy
- **OSINT Search Cascade**: SerpAPI, Playwright (Headless Stealth Browser), Yandex Images, Instaloader, DuckDuckGo Search (DDGS), BeautifulSoup4
- **Blockchain / Cryptography**: Web3.py, py-solc-x, Solidity 0.8.19 (Ethereum Sepolia Testnet), SHA-256

## 3. Codebase Structure
The program is cleanly separated into distinct modules within the `app/` directory, mapping directly to the phases of the pipeline.

### Core Modules (`app/`)
* **`main.py`**: The CLI entry point and central orchestrator. It parses arguments, manages the pipeline flow, and outputs rich terminal formatting.
* **`face.py`**: Handles biometric intake. Uses InsightFace to detect facial bounding boxes, perform 5-point landmark alignment, and extract a normalized 512-dimensional ArcFace embedding vector. It also handles fallback facial cropping.
* **`search.py`**: The OSINT discovery engine. It executes a robust multi-tier search cascade:
  * Primary: SerpAPI Google Lens.
  * Fallback 1: Headless Playwright automation for zero-CAPTCHA direct Google Lens uploads.
  * Fallback 2: Direct Yandex visual search.
  * Also implements social platform pivoting (Instagram, X/Twitter, LinkedIn) based on target URLs or known handles.
* **`matcher.py`**: Handles biometric verification. It downloads candidate images from the search phase, extracts their face vectors, and computes the Cosine Similarity against the original query vector to rank the strongest match.
* **`content.py`**: Forensic content acquisition. Extracts post metadata (author, platform, text) and raw image bytes, packaging them into a deterministic, key-sorted canonical JSON representation.
* **`hashing.py`**: Cryptographic layer that computes a unique 32-byte SHA-256 hash of the canonical JSON payload.
* **`blockchain.py`**: Web3 integration. Signs and submits transactions containing the SHA-256 hash to the `ContentRegistry` smart contract on the Ethereum Sepolia network.
* **`verify.py`**: The independent verification module. Re-computes local hashes from a saved JSON record and compares them with the immutable on-chain record to detect tampering.
* **`config.py`**: Configuration manager that loads `.env` variables (API keys, RPC URLs, Private Keys).

### Smart Contracts (`contracts/`)
* **`ContentRegistry.sol`**: A Solidity smart contract mapping `bytes32 contentHash` to `Record` structs containing block timestamps and source identifiers. This acts as the immutable evidence ledger.

## 4. Pipeline Workflow

1. **Phase 1: Biometric Extraction**: An image is ingested, and a 512-d ArcFace vector is computed.
2. **Phase 2: Multi-Engine Search Cascade**: The system autonomously searches the open web and social media for visual matches, degrading gracefully from API searches to headless scraping if necessary.
3. **Phase 3: Biometric Verification**: Candidate images are scored mathematically (Cosine Similarity) against the query.
4. **Phase 4: Content Acquisition**: Metadata and media bytes from the best match are collected and packaged into a canonical format.
5. **Phase 5: Blockchain Notarization**: The canonical data is hashed (SHA-256) and submitted to the Ethereum blockchain as an immutable timestamped record. The local metadata is saved to a `data/results/*_record.json` dossier.
6. **Phase 6: Verification**: The `verify` command allows recalculation of the hash from the local dossier and cross-referencing with the blockchain to ensure zero tampering occurred.
