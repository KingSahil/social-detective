# FaceTrace — Face Search + Blockchain Verification

> **SocialDetective** CLI application: an end-to-end pipeline that detects faces,
> searches the web for matching images, fingerprints discovered content, records
> it on the Ethereum blockchain, and verifies integrity.

```
Face image → Face detection → Face encoding → Web search →
Candidate matching → Content retrieval → SHA-256 fingerprint →
Blockchain record → Verification
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Input Image │────▶│  InsightFace  │────▶│  512-d Embedding │
└─────────────┘     │  (ArcFace)   │     └────────┬────────┘
                    └──────────────┘              │
                                                  ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  SerpAPI     │────▶│  Candidate URLs  │
                    │  Google Lens │     └────────┬────────┘
                    └──────────────┘              │
                                                  ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  Face Matcher │────▶│  Ranked Results  │
                    │  (Cosine Sim)│     └────────┬────────┘
                    └──────────────┘              │
                                                  ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  Content     │────▶│  Canonical JSON  │
                    │  Retriever   │     │  + Image Bytes   │
                    └──────────────┘     └────────┬────────┘
                                                  │
                                                  ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  SHA-256     │────▶│  Content Hash    │
                    └──────────────┘     └────────┬────────┘
                                                  │
                                                  ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  Ethereum    │────▶│  On-chain Record │
                    │  Sepolia     │     └────────┬────────┘
                    └──────────────┘              │
                                                  ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  Verifier    │────▶│  ✓ VERIFIED      │
                    └──────────────┘     └─────────────────┘
```

---

## Features

- **Face Detection & Encoding** — InsightFace with ArcFace (`buffalo_l` model, 512-d embeddings)
- **Genuine Web Search** — SerpAPI Google Lens reverse-image search at runtime (no hardcoded results)
- **Face Similarity Matching** — Cosine similarity ranking with configurable threshold
- **Content Fingerprinting** — SHA-256 hash of canonical content + actual image bytes
- **Blockchain Recording** — Ethereum Sepolia smart contract for tamper-evident storage
- **Integrity Verification** — Compare local content hash against on-chain record
- **Tamper Detection** — Modify any field and verification catches it

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Face Detection | InsightFace / ArcFace (buffalo_l) |
| Face Embeddings | 512-d ArcFace via ONNX Runtime |
| Web Search | SerpAPI Google Lens |
| Content Hashing | SHA-256 (hashlib) |
| Blockchain | Ethereum Sepolia + Solidity 0.8.19 |
| Smart Contract | Web3.py + py-solc-x |
| CLI | Python argparse + colorama |

---

## Installation

```bash
# Clone
git clone <repo-url>
cd social-detective

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your credentials (see below)
```

### InsightFace Models

On first run, InsightFace automatically downloads the `buffalo_l` model pack (~300MB) to `~/.insightface/models/`. No manual setup needed.

---

## Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `SERPAPI_KEY` | ✓ | API key from [serpapi.com](https://serpapi.com/manage-api-key) |
| `RPC_URL` | ✓ | Ethereum Sepolia RPC endpoint (Infura/Alchemy) |
| `PRIVATE_KEY` | ✓ | Wallet private key (NEVER share or commit) |
| `CONTRACT_ADDRESS` | ✓ | Deployed ContentRegistry address |

### SerpAPI Setup

1. Sign up at [serpapi.com](https://serpapi.com)
2. Get your API key from [Manage API Key](https://serpapi.com/manage-api-key)
3. Add to `.env`: `SERPAPI_KEY=your_key_here`

### Blockchain Setup

1. Create a wallet (e.g., MetaMask) and export the private key
2. Get an RPC endpoint:
   - [Infura](https://infura.io) — create a project, copy Sepolia endpoint
   - [Alchemy](https://alchemy.com) — create an app, copy Sepolia endpoint
3. Get free Sepolia ETH:
   - [Google Cloud Faucet](https://cloud.google.com/application/web3/faucet/ethereum/sepolia)
   - [Alchemy Faucet](https://sepoliafaucet.com)
4. Deploy the contract (see below)

---

## Smart Contract Deployment

```bash
python scripts/deploy_contract.py
```

Output:
```
============================================================
  ContentRegistry — Contract Deployment
============================================================

  [1/3] Compiling ContentRegistry.sol ...
        ✓ Compilation successful

  [2/3] Connecting to Ethereum Sepolia ...
        ✓ Connected
        Deployer: 0x...
        Balance:  0.1 ETH

  [3/3] Deploying contract ...
        TX: 0x...
        ✓ Contract deployed!

  Contract deployed:
  0x1234567890abcdef...

  Add to your .env file:
  CONTRACT_ADDRESS=0x1234567890abcdef...
```

Copy the contract address to your `.env` file.

---

## How to Run

### Full Pipeline

```bash
python -m app.main --image ./data/input/face.jpg
```

With custom threshold:
```bash
python -m app.main --image ./data/input/face.jpg --threshold 0.60
```

### Example Output

```
============================================================
               FACETRACE
      Face Search + Blockchain Verification
============================================================

  [1/7] FACE DETECTION
        ✓ Face detected
        ✓ Face embedding generated (512-d)

  [2/7] WEB SEARCH
        Provider: SerpAPI Google Lens
        Searching...
        ✓ Search completed
        ✓ 18 candidates discovered

  [3/7] FACE MATCHING
        Analyzing candidates...

        #1  Similarity: 91.4%  instagram.com
        #2  Similarity: 87.2%  twitter.com
        #3  Similarity: 73.6%  linkedin.com

        ✓ Strongest candidate selected (Similarity: 91.4%)

  [4/7] CONTENT RETRIEVAL
        ✓ Matching content retrieved

        Source:
        https://www.instagram.com/p/...

  [5/7] FINGERPRINT
        Algorithm: SHA-256

        8f91c2f3d91a8c7e4b2a...

  [6/7] BLOCKCHAIN
        Network: Ethereum Sepolia
        ✓ Transaction confirmed

        TX:
        0xabc123...

  [7/7] VERIFICATION
        Local hash:
        8f91c2f3d91a8c7e4b2a...

        On-chain: ✓ Hash found

        ✓ CONTENT VERIFIED

  Record saved:
  data/results/20260902_100000_record.json

============================================================
```

---

## How to Verify a Record

```bash
python -m app.main verify --record ./data/results/20260902_100000_record.json
```

Or directly:
```bash
python -m app.verify --record ./data/results/20260902_100000_record.json
```

---

## How to Demonstrate Tampering

1. **Run the pipeline** and save a record:
   ```bash
   python -m app.main --image ./data/input/face.jpg
   ```

2. **Verify it** (should show ✓ VERIFIED):
   ```bash
   python -m app.main verify --record ./data/results/<timestamp>_record.json
   ```

3. **Tamper with the record** — open the JSON file and change any field in the
   `content` section. For example, change `"text"` or `"image_hash"`:
   ```json
   {
     "content": {
       "text": "TAMPERED content here"
     }
   }
   ```

4. **Verify again** (should show ✗ TAMPER DETECTED):
   ```bash
   python -m app.main verify --record ./data/results/<timestamp>_record.json
   ```

   Output:
   ```
   ============================================================
              VERIFICATION
   ============================================================

     Local hash (current):
     abc123...

     Original hash (registered):
     xyz789...

     On-chain:  ✓ Hash found

     ✗ TAMPER DETECTED
     TAMPER DETECTED — content has been modified since registration

   ============================================================
   ```

---

## Saved Record Format

Each investigation is saved as `data/results/<timestamp>_record.json`:

```json
{
  "query": {
    "image": "/path/to/face.jpg",
    "face_detected": true,
    "embedding_dim": 512
  },
  "search": {
    "provider": "SerpAPI Google Lens",
    "searched_at": "2026-09-02T10:00:00+00:00",
    "candidate_count": 18
  },
  "match": {
    "source_url": "https://...",
    "image_url": "https://...",
    "similarity": 0.914,
    "domain": "instagram.com",
    "title": "..."
  },
  "content": {
    "source_url": "https://...",
    "image_url": "https://...",
    "platform": "www.instagram.com",
    "title": "...",
    "text": "...",
    "image_hash": "a1b2c3...",
    "retrieved_at": "2026-09-02T10:00:01+00:00"
  },
  "fingerprint": {
    "algorithm": "SHA-256",
    "hash": "8f91c2f3..."
  },
  "blockchain": {
    "network": "Ethereum Sepolia",
    "contract": "0x...",
    "transaction": "0x...",
    "block": 123456,
    "status": "confirmed"
  },
  "verification": {
    "verified": true
  }
}
```

---

## Testing

```bash
# Run all unit tests (no credentials needed)
python -m pytest tests/ -v
```

Tests cover:
- Face detection (invalid images, no face, embedding extraction)
- Search providers (mock provider, API key validation)
- Cosine similarity (identical, orthogonal, random vectors)
- SHA-256 hashing (determinism, known values)
- Canonicalization (sorted keys, image hash inclusion, round-trip)
- Tamper detection (modified content → different hash)
- Blockchain data formatting (bytes32 conversion, ABI loading)

---

## Project Structure

```
social-detective/
├── app/
│   ├── __init__.py        # Package init
│   ├── main.py            # CLI entry point + pipeline orchestration
│   ├── config.py          # Environment variables + settings
│   ├── face.py            # Face detection + ArcFace embeddings
│   ├── search.py          # Search provider (SerpAPI Google Lens)
│   ├── matcher.py         # Face similarity matching
│   ├── content.py         # Content retrieval + canonicalization
│   ├── hashing.py         # SHA-256 fingerprinting
│   ├── blockchain.py      # Web3.py + Solidity interaction
│   └── verify.py          # Verification logic
├── contracts/
│   └── ContentRegistry.sol
├── scripts/
│   └── deploy_contract.py
├── data/
│   ├── input/             # Place query images here
│   └── results/           # Saved investigation records
├── tests/
│   ├── test_face.py
│   ├── test_search.py
│   ├── test_matching.py
│   ├── test_hashing.py
│   └── test_blockchain.py
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Known Limitations

- **SerpAPI quota**: Free tier has limited searches per month
- **Google Lens results**: Not all images produce face-containing results
- **Candidate images**: Some thumbnails may be too small for reliable face detection
- **Blockchain costs**: Transactions require Sepolia test ETH (free from faucets)
- **Model download**: First run downloads ~300MB of InsightFace models
- **Image size**: SerpAPI upload has a 500KB file size limit; large images should be resized

---

## Privacy & Responsible Use

> [!IMPORTANT]
> **Face matching determines visual similarity between images.**
> It does NOT prove identity. A high similarity score means two face images
> look alike — it does not confirm they are the same person.

> [!IMPORTANT]
> **The blockchain does NOT prove a person's identity.**
> It provides a tamper-evident fingerprint of discovered web content.
> The on-chain record proves that specific content existed at a specific time
> and has not been modified since.

> [!WARNING]
> **No biometric data is stored on-chain.**
> Face embeddings, face images, and private information are NEVER sent
> to the blockchain. Only a SHA-256 hash of the discovered content metadata
> and image bytes is stored.

> [!CAUTION]
> **Use responsibly.** This tool is designed for authorized investigations
> and research purposes. Always obtain proper consent and follow applicable
> laws regarding facial recognition, data privacy, and blockchain usage.
> Do not use this tool to stalk, harass, or invade anyone's privacy.

---

## License

MIT
