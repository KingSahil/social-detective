<p align="center">
  <h1 align="center">🔍 FaceTrace (SocialDetective)</h1>
  <p align="center">
    <strong>Autonomous Biometric OSINT Facial Recognition & Immutable Blockchain Notarization Pipeline</strong>
  </p>
  <p align="center">
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?logo=python&logoColor=white" alt="Python Version" /></a>
    <a href="https://github.com/deepinsight/insightface"><img src="https://img.shields.io/badge/Biometrics-InsightFace%20ArcFace-7952B3?logo=opencv&logoColor=white" alt="InsightFace" /></a>
    <a href="https://onnxruntime.ai"><img src="https://img.shields.io/badge/Inference-ONNX%20Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime" /></a>
    <a href="https://sepolia.etherscan.io/address/0xe25BfF359d31b3E2B3fF99692E6cE025f273BC21"><img src="https://img.shields.io/badge/Ethereum-Sepolia%20Testnet-627EEA?logo=ethereum&logoColor=white" alt="Sepolia Contract" /></a>
    <a href="https://soliditylang.org"><img src="https://img.shields.io/badge/Smart%20Contract-Solidity%200.8.19-363636?logo=solidity&logoColor=white" alt="Solidity" /></a>
    <a href="https://web3py.readthedocs.io"><img src="https://img.shields.io/badge/Web3-Web3.py-F16822?logo=ethereum&logoColor=white" alt="Web3.py" /></a>
  </p>
  <p align="center">
    <a href="https://serpapi.com"><img src="https://img.shields.io/badge/Search-SerpAPI%20Google%20Lens-4285F4?logo=google&logoColor=white" alt="SerpAPI Google Lens" /></a>
    <a href="https://yandex.com/images"><img src="https://img.shields.io/badge/Visual%20Search-Yandex%20Images-FC3F1D?logo=yandex&logoColor=white" alt="Yandex Images" /></a>
    <a href="https://duckduckgo.com"><img src="https://img.shields.io/badge/Fallback-DuckDuckGo%20Search-DE5833?logo=duckduckgo&logoColor=white" alt="DuckDuckGo" /></a>
    <a href="https://instagram.com"><img src="https://img.shields.io/badge/OSINT-Instagram%20Reels%20%26%20Carousels-E4405F?logo=instagram&logoColor=white" alt="Instagram" /></a>
    <a href="https://x.com"><img src="https://img.shields.io/badge/OSINT-X%20%2F%20Twitter-000000?logo=x&logoColor=white" alt="X/Twitter" /></a>
    <a href="https://linkedin.com"><img src="https://img.shields.io/badge/OSINT-LinkedIn%20Network-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  </p>
  <p align="center">
    <a href="https://en.wikipedia.org/wiki/SHA-2"><img src="https://img.shields.io/badge/Fingerprint-SHA--256-555555" alt="SHA-256" /></a>
    <a href="https://pytest.org"><img src="https://img.shields.io/badge/Test%20Suite-49%20Passed-2ea44f?logo=pytest&logoColor=white" alt="Pytest Suite" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  </p>
</p>

---

## 📑 Executive Overview

**FaceTrace** is an enterprise-grade forensic intelligence and integrity auditing tool designed for open-source intelligence (OSINT) investigators, security researchers, and digital forensics professionals. 

Given an arbitrary query face portrait, the system:
1. **Extracts** high-dimensional normalized biometric embeddings (512-d ArcFace vectors) using multi-landmark facial geometry alignment.
2. **Executes** an automated multi-engine reverse visual search cascade (Google Lens $\rightarrow$ Focused Crop $\rightarrow$ Yandex Images).
3. **Pivots** across cross-platform social identity memory, extracting candidates from Instagram profiles, video reels, multi-photo carousels (`GraphSidecar`), X/Twitter timelines, and LinkedIn associate graphs.
4. **Calculates** cosine similarity against discovered candidate media to isolate the strongest visual match.
5. **Acquires** public post metadata, downloads media bytes, formats deterministic canonical representations, and derives a cryptographic SHA-256 fingerprint.
6. **Notarizes** the evidence hash immutably on the **Ethereum Sepolia** blockchain via a custom Solidity smart contract (`ContentRegistry.sol`).
7. **Verifies** proof-of-authenticity at any future time, detecting any post alteration, deleted media, or metadata tampering.

```
Query Image ➔ Face Detection ➔ ArcFace 512-d Vector ➔ Multi-Engine Web & Social Search
     ➔ Biometric Similarity Ranking ➔ Canonical Serialization ➔ SHA-256 Fingerprint
     ➔ Ethereum Sepolia Notarization ➔ Cryptographic Verification & Tamper Detection
```

<p align="center">
  <img src="docs/assets/face_embedding_concept.jpg" alt="FaceTrace AI Face Embedding & Biometric Verification Pipeline" width="100%" style="border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);" />
  <br>
  <em>Figure 1: Mathematical transformation of facial geometry into normalized 512-dimensional ArcFace embeddings, cosine similarity matching, and cryptographic tamper-proof evidence sealing.</em>
</p>

---

## 🏗 System Architecture

```mermaid
flowchart TD
    %% ─────────────────────────────────────────────────────────────
    %% STYLING DEFINITIONS
    %% ─────────────────────────────────────────────────────────────
    classDef default fill:#1f2937,stroke:#475569,color:#f8fafc,stroke-width:1.5px
    classDef inputNode fill:#0f172a,stroke:#38bdf8,color:#f8fafc,stroke-width:2px
    classDef aiNode fill:#1e1b4b,stroke:#818cf8,color:#e0e7ff,stroke-width:2px
    classDef searchNode fill:#064e3b,stroke:#34d399,color:#ecfdf5,stroke-width:2px
    classDef matchNode fill:#3b0764,stroke:#a855f7,color:#faf5ff,stroke-width:2px
    classDef cryptoNode fill:#701a75,stroke:#f472b6,color:#fdf2f8,stroke-width:2px
    classDef chainNode fill:#431407,stroke:#fb923c,color:#fff7ed,stroke-width:2px
    classDef verifyNode fill:#082f49,stroke:#38bdf8,color:#f0f9ff,stroke-width:2px
    classDef verifiedNode fill:#14532d,stroke:#22c55e,color:#f0fdf4,stroke-width:2.5px
    classDef tamperedNode fill:#7f1d1d,stroke:#ef4444,color:#fef2f2,stroke-width:2.5px

    %% ─────────────────────────────────────────────────────────────
    %% PHASE 1: BIOMETRIC INGESTION & FEATURE EXTRACTION
    %% ─────────────────────────────────────────────────────────────
    subgraph P1 ["Phase 1: Biometric Intake & Feature Extraction"]
        IN(["Query Portrait Image<br/>e.g. test_face_11.jpg"]):::inputNode
        CLI["CLI Options & Parameters<br/>--threshold, --platform, --target, --engine, --handle"]:::inputNode
        DET["InsightFace Detector<br/>buffalo_l Model (5-point Landmark Alignment)"]:::aiNode
        EMB["ArcFace Embedding Engine<br/>512-Dimensional Normalized Vector"]:::aiNode
        CROP["Portrait Face Cropper<br/>Tight Bounding Box + 35% Margin (Fallback)"]:::aiNode

        IN --> DET
        CLI -.->|"Configures thresholds & flags"| DET
        DET --> EMB
        DET --> CROP
    end

    %% ─────────────────────────────────────────────────────────────
    %% PHASE 2: SEARCH CASCADE & MULTI-PLATFORM OSINT
    %% ─────────────────────────────────────────────────────────────
    subgraph P2 ["Phase 2: Multi-Engine Search Cascade & OSINT Discovery"]
        ROUTER{"Execution Mode<br/>Target vs Handle vs Cascade"}:::searchNode

        T_URL["Target URL Inspector<br/>Direct Instagram Posts/Reels, Carousels, X/Twitter, Web"]:::searchNode
        T_HANDLE["Multi-Platform Handle Sweeper<br/>Concurrent Instagram & Twitter Profile Discovery"]:::searchNode

        LENS["Primary: Google Lens Visual Search<br/>SerpAPI Reverse Image Discovery"]:::searchNode
        LENS_CROP["Fallback 1: Cropped Face Search<br/>Focused Facial Geometry Query"]:::searchNode
        YANDEX["Fallback 2: Yandex Images<br/>Deep Biometric Facial Search"]:::searchNode

        subgraph PIVOT_SYS ["OSINT Identity Memory & Network Pivoting"]
            MEMORY["Subject Identity Memory<br/>Correlates 512-d Face Vector, Author & Title Tags"]:::searchNode
            TW_PIVOT["Twitter Profile Provider<br/>Concurrent Media Timeline Sweep"]:::searchNode
            IG_PIVOT["Instagram Profile Provider<br/>Google Redirect Unwrapping, Carousels & Silent DDGS"]:::searchNode
            LI_PIVOT["LinkedIn Post Provider<br/>Associate Forensics & Open Graph Post Discovery"]:::searchNode
            UNPACK["Media & Carousel Unpacker<br/>Extracts Multi-Slide Carousels & Video Cover Frames"]:::searchNode
        end

        MEDIA_POOL[("Candidate Media Pool<br/>Post URLs, Carousel Slides, Video Covers, Image URLs")]:::searchNode

        EMB --> ROUTER
        ROUTER -->|"--target URL"| T_URL
        ROUTER -->|"--handle USER"| T_HANDLE
        ROUTER -->|"Default: Open Web Cascade"| LENS

        LENS -->|"Below threshold"| LENS_CROP
        LENS_CROP -->|"Below threshold"| YANDEX
        YANDEX -->|"0 direct hits"| LI_PIVOT

        LENS -.->|"Discovered Handles"| MEMORY
        EMB -.->|"Biometric Correlation"| MEMORY
        MEMORY --> TW_PIVOT
        MEMORY --> IG_PIVOT
        IG_PIVOT --> UNPACK

        T_URL --> MEDIA_POOL
        T_HANDLE --> MEDIA_POOL
        LENS --> MEDIA_POOL
        LENS_CROP --> MEDIA_POOL
        YANDEX --> MEDIA_POOL
        TW_PIVOT --> MEDIA_POOL
        IG_PIVOT --> MEDIA_POOL
        LI_PIVOT --> MEDIA_POOL
        UNPACK --> MEDIA_POOL
    end

    %% ─────────────────────────────────────────────────────────────
    %% PHASE 3: CANDIDATE MATCHING & RANKING
    %% ─────────────────────────────────────────────────────────────
    subgraph P3 ["Phase 3: Biometric Verification & Candidate Ranking"]
        CAND_EMB["Candidate Face Processor<br/>Extract 512-d ArcFace Vector per Candidate"]:::aiNode
        MATCHER["FaceMatcher Engine<br/>Cosine Similarity = dot(q, c) / (||q|| * ||c||)"]:::matchNode
        FILTER["Ranking & Threshold Filter<br/>Platform Filtering & Score Meets Threshold (e.g. 70%)"]:::matchNode
        WINNER(["Rank #1 Strongest Match Selected<br/>Highest Facial Similarity Score"]):::matchNode

        MEDIA_POOL --> CAND_EMB
        EMB -.->|"Query Vector (512-d)"| MATCHER
        CAND_EMB --> MATCHER
        MATCHER --> FILTER
        FILTER --> WINNER
    end

    %% ─────────────────────────────────────────────────────────────
    %% PHASE 4: FORENSIC ACQUISITION & CANONICALIZATION
    %% ─────────────────────────────────────────────────────────────
    subgraph P4 ["Phase 4: Forensic Content Acquisition & Canonical Packaging"]
        ACQUIRE["Content Retriever<br/>Fetch Post HTML, Text, Author, Timestamp"]:::cryptoNode
        IMG_DOWNLOAD["Media Ingestion<br/>Download Raw Image / Thumbnail Bytes"]:::cryptoNode
        IMG_HASH["Image Cryptographic Hash<br/>Compute SHA-256 of Raw Image Bytes"]:::cryptoNode
        CANON["Forensic Canonicalizer<br/>Deterministic Sorted JSON Key-Value Map"]:::cryptoNode

        WINNER --> ACQUIRE
        WINNER --> IMG_DOWNLOAD
        IMG_DOWNLOAD --> IMG_HASH
        ACQUIRE --> CANON
        IMG_HASH --> CANON
    end

    %% ─────────────────────────────────────────────────────────────
    %% PHASE 5: CRYPTOGRAPHIC SEAL & BLOCKCHAIN NOTARIZATION
    %% ─────────────────────────────────────────────────────────────
    subgraph P5 ["Phase 5: Cryptographic Sealing & Blockchain Notarization"]
        SHA["SHA-256 Fingerprint Generator<br/>Produces Unique 32-Byte Content Hash"]:::cryptoNode
        WEB3["Web3.py Client<br/>Sign & Submit Transaction to Ethereum Sepolia"]:::chainNode
        CONTRACT[("ContentRegistry.sol Smart Contract<br/>Address: 0xe25BfF359d31b3E2B3fF99692E6cE025f273BC21<br/>Ethereum Sepolia Testnet")]:::chainNode
        DOSSIER[("Local Forensic Dossier<br/>Saved to data/results/*_record.json")]:::chainNode

        CANON --> SHA
        SHA -->|"bytes32 contentHash"| WEB3
        WEB3 -->|"registerContent(hash, sourceId)"| CONTRACT
        CONTRACT -.->|"Tx Hash & Block Confirmation"| DOSSIER
    end

    %% ─────────────────────────────────────────────────────────────
    %% PHASE 6: INDEPENDENT VERIFICATION & TAMPER DETECTION
    %% ─────────────────────────────────────────────────────────────
    subgraph P6 ["Phase 6: Independent Verification & Tamper Detection"]
        V_CLI["facetrace verify --record record.json<br/>CLI Verification Tool"]:::verifyNode
        V_LOCAL["Recompute Canonical SHA-256 Hash<br/>From Local Record Fields"]:::verifyNode
        V_QUERY["Query Smart Contract<br/>Check ContentRegistry.records(hash)"]:::verifyNode
        V_CHECK{"Integrity Check:<br/>Local Hash == On-Chain Hash?"}:::verifyNode

        V_PASS(["✓ CONTENT VERIFIED<br/>Proof Intact: 100% Authentic & Untampered"]):::verifiedNode
        V_FAIL(["✗ TAMPER DETECTED<br/>Hash Mismatch: Text, Image, or Meta Altered"]):::tamperedNode

        DOSSIER -.->|"Audit Target"| V_CLI
        V_CLI --> V_LOCAL
        V_CLI --> V_QUERY
        V_LOCAL --> V_CHECK
        V_QUERY --> V_CHECK
        V_CHECK -->|"Identical Hash Found On-Chain"| V_PASS
        V_CHECK -->|"Hash Differs or Not Registered"| V_FAIL
    end
```

---

## ⚡ Key Capabilities & Technical Innovations

### 1. Biometric Precision & Multi-Face Landmark Alignment
* Powered by InsightFace and ArcFace with the `buffalo_l` deep convolutional neural network pack.
* Computes normalized 512-dimensional feature representations invariant to variable illumination, camera focal lengths, and pose angles up to $\pm 45^\circ$.
* Incorporates automated face cropping with configurable safety margins (default 35%) for focused secondary search cascades.

### 2. Multi-Engine Visual Reverse Search Cascade
* **Primary**: Google Lens reverse image discovery via SerpAPI to capture mass indexed web appearances.
* **Secondary**: Portrait-cropped visual query targeting isolated facial structures.
* **Tertiary**: Yandex Images facial biometric index for deep Russian/Eastern European and secondary social platform discovery.

### 3. Cross-Platform Social Pivoting & Forensic Identity Memory
* **Persistent Subject Memory**: Automatically scans previous case files in `data/results/` and matches query face vectors against historical targets to recall known social handles and creator usernames.
* **Structured Author Correlation**: Extracts handles from URLs, title headers (`<user> on Instagram`), and investigation records to seed cross-platform investigative hops.
* **Associate Network Graph**: Correlates co-occurring tagged associates and event contexts across LinkedIn and Instagram posts.

### 4. Deep Instagram & Video Reels Extraction
* **Instagram Reel & Post Covers**: Dynamically extracts high-resolution cover frames from video posts and reels.
* **Carousel Unpacking**: Automatically decomposes multi-photo carousel posts (`GraphSidecar`) into discrete slide candidates (`?img_index=N`) to identify tagged associates in background slides.
* **Search Engine Redirect Unwrapping**: Resolves search wrapper redirects (`google.com/goto`, `google.com/url`) so reel shortcodes and video anchors are never lost.
* **Silent Resilient DuckDuckGo Fallback**: Employs low-level C file descriptor redirection (`os.dup2`) and concurrency locks to eliminate Rust `rustls`/`h2` TLS disconnect warnings when querying DuckDuckGo.

### 5. Multi-Platform Handle Profiling (`--handle`)
* Execute targeted sweeps across a suspected identity without manual URL scraping:
  * `--handle USER --platform instagram`: Sweeps public Instagram posts and reels.
  * `--handle USER --platform twitter`: Extracts media tweets from the user's timeline.
  * `--handle USER`: Concurrently sweeps **both** platforms, merging all discovered media into the comparison pool.

### 6. Cryptographic Fingerprinting & Smart Contract Notarization
* Normalizes discovered content (source URL, image URL, clean text/caption, image bytes) into a deterministic canonical key-sorted structure.
* Generates a 32-byte SHA-256 fingerprint registered onto the Ethereum Sepolia blockchain via `ContentRegistry.sol`.
* Zero biometric vectors or personal identity data are written on-chain—only the irreversible cryptographic seal.

---

## 🛠 Technology Stack

| Layer | Component | Implementation |
|:---|:---|:---|
| **Biometrics** | Face Detection & Alignment | InsightFace (`buffalo_l`, ONNX Runtime) |
| **Embeddings** | Feature Extraction | 512-dimensional ArcFace normalized vector |
| **Search Engines** | Visual Reverse Search | SerpAPI Google Lens, Yandex Images |
| **OSINT Pivoting** | Handle & Associate Sweeps | Twitter/X, Instagram, LinkedIn, DuckDuckGo Search |
| **Media Unpacking** | Instagram & Social Media | Instaloader, BeautifulSoup4, Regex redirect resolvers |
| **Hashing** | Canonical Fingerprint | SHA-256 (RFC 6234 via Python `hashlib`) |
| **Smart Contract** | On-Chain Notarization | Solidity 0.8.19 (`ContentRegistry.sol`) |
| **Blockchain** | Client & Node RPC | Web3.py, Infura / Alchemy (Ethereum Sepolia) |
| **CLI & Testing** | Interface & Test Suite | Argparse, Colorama, Pytest (49 unit tests) |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/social-detective.git
cd social-detective
```

### 2. Configure Python Environment
Python 3.10+ is recommended:
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> On the first execution, InsightFace automatically downloads the `buffalo_l` model pack (~300MB) to `~/.insightface/models/`. No manual download is required.

### 3. Environment Variables

Create a `.env` configuration file from the provided template:
```bash
cp .env.example .env
```

Configure your API credentials in `.env`:
```ini
# SerpAPI Key for Google Lens & Reverse Search (https://serpapi.com)
SERPAPI_KEY=your_serpapi_key_here

# Ethereum Sepolia RPC Endpoint (Infura, Alchemy, or public node)
RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID

# Ethereum Account Private Key (Used to sign notarization transactions)
PRIVATE_KEY=your_wallet_private_key_without_0x

# Deployed ContentRegistry Contract Address on Sepolia
CONTRACT_ADDRESS=0xe25BfF359d31b3E2B3fF99692E6cE025f273BC21
```

---

## 📜 Smart Contract Reference

The canonical notarization registry is deployed on the **Ethereum Sepolia** testnet:

* **Contract Address:** [`0xe25BfF359d31b3E2B3fF99692E6cE025f273BC21`](https://sepolia.etherscan.io/address/0xe25BfF359d31b3E2B3fF99692E6cE025f273BC21)
* **Creation Transaction:** [`0xc96cff185e7d482fc64a76a2fa2a4a907fa0dcd2bc0b760d76447d9789c90eb5`](https://sepolia.etherscan.io/tx/0xc96cff185e7d482fc64a76a2fa2a4a907fa0dcd2bc0b760d76447d9789c90eb5)
* **Solidity Version:** `0.8.19`

To deploy your own private instance:
```bash
python scripts/deploy_contract.py
```

---

## 💻 CLI Usage Guide

### 1. Autonomous Web & Social Reverse Search
Search across Google Lens, Yandex, and Subject Memory to locate public web/social appearances:
```bash
python -m app.main --image ./data/input/test_face.jpg
```

With custom similarity threshold (default: `0.70`):
```bash
python -m app.main --image ./data/input/test_face.jpg --threshold 0.80
```

### 2. Targeted Social Post & Reel Verification (`--target`)
Inspect a suspected Instagram reel, multi-photo carousel post, Twitter/X post, or web article directly:
```bash
# Verify against an Instagram Reel or Carousel
python -m app.main --image ./data/input/test_face_11.jpg --target https://www.instagram.com/p/DbvdVHXOLSG/

# Verify against an X/Twitter Status
python -m app.main --image ./data/input/test_face_4.png --target https://x.com/supreme__sahil/status/2087906598962524208
```

### 3. Multi-Platform Handle Profiling (`--handle`)
Sweep all media published by a specific user across social networks:
```bash
# Concurrently search both Instagram and X/Twitter
python -m app.main --image ./data/input/test_face_11.jpg --handle supreme__sahil

# Restrict sweep specifically to Instagram
python -m app.main --image ./data/input/test_face_11.jpg --handle supreme__sahil --platform instagram

# Restrict sweep specifically to X/Twitter
python -m app.main --image ./data/input/test_face_11.jpg --handle supreme__sahil --platform twitter
```

### 4. Search Engine Selection (`--engine`)
```bash
# Cascade: Google Lens -> Cropped Query -> Yandex (Default)
python -m app.main --image ./data/input/test_face.jpg --engine all

# Force Google Lens only
python -m app.main --image ./data/input/test_face.jpg --engine lens

# Force Yandex Images only
python -m app.main --image ./data/input/test_face.jpg --engine yandex
```

### 5. Verify & Audit a Record
Verify an existing investigation dossier against the blockchain:
```bash
python -m app.main verify --record ./data/results/20260904_064037_record.json
```

---

## 📋 Comprehensive Execution Output

```
============================================================
               FACETRACE
      Face Search + Blockchain Verification
============================================================

  [1/7] FACE DETECTION
        ✓ Face detected
        ✓ Face embedding generated (512-d)

  [2/7] TARGET MEDIA DISCOVERY
        Target: https://www.instagram.com/p/DbvdVHXOLSG/
        Extracting candidate media images...
        ✓ Target media extracted
        ✓ 1 candidate images discovered from target

  [3/7] FACE MATCHING
        Analyzing candidate face similarity...

        #1   Similarity: 97.5%  [www.instagram.com]

        ✓ Strongest candidate selected: www.instagram.com (Similarity: 97.5%)

  [4/7] CONTENT RETRIEVAL
        ✓ Matching content retrieved

        Source:
        https://www.instagram.com/p/DbvdVHXOLSG/
        Title: supreme__sahil on Instagram: "My custom implementation of snapchat lens studio using machine learning Lol..."
        Platform: www.instagram.com

        ✓ Image downloaded (49217 bytes)

  [5/7] FINGERPRINT
        Algorithm: SHA-256

        ad7b1828546b2e5e7465c7b39cfcda99b47a0bce997666666698d230a82fb92b

  [6/7] BLOCKCHAIN
        Network: Ethereum Sepolia
        Contract: 0xe25BfF359d31b3E2B3fF99692E6cE025f273BC21
        Submitting transaction...
        ✓ Transaction confirmed

        TX:
        0x5973722a9fe742e8690581794dc77a47328ad980bfe39bdd3e31f2646d19cd35
        Block: 11631794

  [7/7] VERIFICATION
        Local hash:
        ad7b1828546b2e5e7465c7b39cfcda99b47a0bce997666666698d230a82fb92b

        On-chain: ✓ Hash found

        ✓ CONTENT VERIFIED

  Record saved:
  data/results/20260904_064037_record.json
============================================================
```

---

## 🛡️ Demonstrating Tamper Detection

1. **Run the pipeline to record evidence:**
   ```bash
   python -m app.main --image ./data/input/test_face_11.jpg --target https://www.instagram.com/p/DbvdVHXOLSG/
   ```

2. **Verify the unaltered record:**
   ```bash
   python -m app.main verify --record ./data/results/20260904_064037_record.json
   # Result: ✓ CONTENT VERIFIED
   ```

3. **Simulate tampering:**
   Open the saved `.json` record and alter any string in `"text"`, or change a single character in `"image_hash"`.

4. **Audit the tampered record:**
   ```bash
   python -m app.main verify --record ./data/results/20260904_064037_record.json
   ```

   **Output:**
   ```
   ============================================================
              VERIFICATION
   ============================================================

     Local hash (current):
     03fba1995818dae92bc491176bfa2549d44cba366914cf227918a245598ba994

     Original hash (registered):
     ad7b1828546b2e5e7465c7b39cfcda99b47a0bce997666666698d230a82fb92b

     On-chain:  ✓ Hash found

     ✗ TAMPER DETECTED
     TAMPER DETECTED — content has been modified since registration

   ============================================================
   ```

---

## 📁 Repository Structure

```
social-detective/
├── app/
│   ├── __init__.py          # Module initialization
│   ├── main.py              # CLI entry point and pipeline orchestrator
│   ├── config.py            # Environment configuration and validation
│   ├── face.py              # InsightFace ArcFace detection and embedding engine
│   ├── search.py            # Search providers (Lens, Yandex, IG, Twitter, LinkedIn, Memory)
│   ├── matcher.py           # Cosine similarity ranking and candidate matching
│   ├── content.py           # Content retrieval, author capture, and canonicalization
│   ├── hashing.py           # Cryptographic SHA-256 fingerprint generator
│   ├── blockchain.py        # Web3.py client for Solidity contract interaction
│   └── verify.py            # Standalone integrity and blockchain verification logic
├── contracts/
│   └── ContentRegistry.sol  # Solidity 0.8.19 smart contract
├── scripts/
│   └── deploy_contract.py   # Compilation and deployment automation script
├── data/
│   ├── input/               # Query face portrait images
│   └── results/             # Forensic JSON dossiers and embeddings cache
├── tests/
│   ├── test_face.py         # Unit tests for face detection and embedding extraction
│   ├── test_search.py       # Unit tests for multi-platform search and redirect resolution
│   ├── test_matching.py     # Unit tests for cosine similarity and ranking
│   ├── test_hashing.py      # Unit tests for canonicalization and hashing
│   └── test_blockchain.py   # Unit tests for ABI loading and smart contract helpers
├── requirements.txt         # Production dependencies
├── pyproject.toml           # Packaging and tool configurations
└── README.md                # Project documentation
```

---

## 🧪 Testing & Validation

FaceTrace includes a comprehensive unit test suite covering all modules without requiring active API keys or live blockchain transactions:

```bash
pytest
```

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Projects\social-detective
configfile: pyproject.toml
testpaths: tests
collected 49 items

tests\test_blockchain.py ....                                            [  8%]
tests\test_face.py ....                                                  [ 16%]
tests\test_hashing.py .............                                      [ 42%]
tests\test_matching.py ........                                          [ 59%]
tests\test_search.py ....................                                [100%]

============================= 49 passed in 4.60s ==============================
```

---

## ⚖️ Privacy, Ethics & Responsible Disclosure

> [!IMPORTANT]
> **Facial similarity is a statistical metric, not legal proof of personal identity.**
> A high similarity score (e.g., 95%+) indicates strong geometrical resemblance between two photographic representations. It should serve as an investigative lead, not conclusive proof of identity.

> [!IMPORTANT]
> **The blockchain notarizes content authenticity, not human truth.**
> Registering a content hash on Ethereum proves cryptographically that a specific digital artifact (text, URL, and media bytes) was discovered in that exact form at that specific timestamp. It does not certify the veracity of statements made in the post.

> [!WARNING]
> **Zero biometric data is stored on-chain.**
> Facial embeddings, coordinates, crops, and private identity records are never transmitted to the blockchain. Only the irreversible SHA-256 fingerprint of the public post content is immutably recorded.

> [!CAUTION]
> **Strictly for lawful OSINT and academic research.**
> Always obtain appropriate authorizations and adhere to local privacy regulations (e.g. GDPR, CCPA, BIPA). Do not use this software for harassment, stalking, or unauthorized surveillance.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
