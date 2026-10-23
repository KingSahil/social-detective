# Crowd Context & C3 (Collective Context Correlator) — Technical Guide

## 1. The Core Forensic Problem: Cold-Start Targets

When conducting facial OSINT on unindexed or non-public figures (e.g., a student, an obscure hackathon participant, or a background spectator), traditional reverse image search (Google Lens, Yandex, PimEyes) frequently produces **0 candidate matches**. The target individual does not have enough high-ranking indexed web media under their own name.

However, in real-world investigations, individuals are rarely isolated:
- They appear in **crowd photos**, **group pictures**, and **event stages**.
- The people standing next to them (associates, teammates, organizers) often **do** have high-profile, indexed social profiles.
- Event stages, lanyards, and photo captions carry contextual tokens (event names, dates, university tags, organization handles).

**C3 (Collective Context Correlator)** transforms these background associates into a dynamic forensic bridge to uncover the primary target.

---

## 2. Architecture & The Multi-Phase Crowd Workflow

```
               [Query Image with Multiple Individuals]
                                  │
                                  ▼
                     [Face Detection & Annotation]
       All faces isolated & numbered: Face #1, Face #2, ... Face #N
                                  │
            ┌─────────────────────┴─────────────────────┐
            ▼                                           ▼
   [Manual Target Choice]                    [Reference Auto-Match]
Select numbered bounding box           Upload known reference selfie/portrait
                                       ➔ Cosine similarity auto-selects target
            │                                           │
            └─────────────────────┬─────────────────────┘
                                  ▼
                     [Phase 1: Primary Target Hunt]
              Direct visual search & biometric verification
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
              [MATCH FOUND]               [0 MATCHES FOUND]
         Notarize on Blockchain        Initiate Phase 2: Crowd Pivot!
                                                 │
                                                 ▼
                                     [Phase 2: Crowd Pivot]
                        1. Select background faces (Auto-top 3 / Manual)
                        2. Query background faces against Knowledge Graph & Web
                        3. Extract OSINT tags from background hits (C3)
                                                 │
                                                 ▼
                                   [Context Token Scoring & Ranking]
                        - #hashtags (2x boost)
                        - @handles (2x boost)
                        - Capitalized Proper Nouns (universities, orgs)
                        - Domain identifiers (e.g. stanford.edu)
                        - [MEMORY] Verified Knowledge Graph events
                                                 │
                                                 ▼
                               [Phase 3: Targeted Context Re-Injection]
                     Re-launch search on primary target augmented with
                           C3-extracted context clues & dorks
```

---

## 3. How Reference Auto-Matching Works (`app/ui.py`)

When an investigator uploads a crowded event photo:
1. `fp.annotate_and_extract_all_faces(crowd_photo)` generates bounding boxes and 512-d ArcFace embeddings for every face detected in the frame.
2. If an optional **Reference Photo** (e.g., a solo ID photo, selfie, or passport crop) is supplied, `fp.get_embedding(reference_photo)` computes the reference vector $\mathbf{v}_{\text{ref}}$.
3. FaceTrace computes pairwise cosine similarity between $\mathbf{v}_{\text{ref}}$ and each crowd face vector $\mathbf{u}_i$:
   $$\text{Sim}(\mathbf{v}_{\text{ref}}, \mathbf{u}_i) = \frac{\mathbf{v}_{\text{ref}} \cdot \mathbf{u}_i}{\|\mathbf{v}_{\text{ref}}\|_2 \|\mathbf{u}_i\|_2}$$
4. The highest-scoring face in the crowd is automatically identified and highlighted (`Auto-Matched to Face #K (89.4% similarity)`). The investigator can verify or manually override the selection.

---

## 4. The C3 Tag Extraction & Ranking Engine (`app/context.py`)

`ContextManager` processes candidates and social hits gathered from background associates:

### 4.1 Token Extraction
- **Hashtags**: `re.findall(r'#\w+', title)` (e.g., `#HackHazards`, `#IEEE`, `#Symbiosis`).
- **Handles**: `re.findall(r'@\w+', title)` (e.g., `@stanford`, `@KingSahil`).
- **Proper Nouns**: Capitalized words >3 characters after filtering against an extensive English and web stop-word dictionary (`photo`, `instagram`, `linkedin`, `user`, etc.).
- **Domain Names**: High-reputation root domains (e.g. `stanford.edu` $\rightarrow$ `stanford`).
- **Biometric Memory**: Past event tags associated with any background person already recognized in `IdentityKnowledgeGraph`.

### 4.2 Scoring Formula
$$\text{Score}(t) = \text{Freq}(t) \times W_{\text{type}}(t) \times W_{\text{length}}(t)$$

Where:
- $W_{\text{type}} = 2.0$ for explicit `#hashtags` and `@handles`.
- $W_{\text{type}} = 1.0$ for proper nouns and domains.
- $W_{\text{length}} = 0.5$ if $\text{length}(t) < 4$.
- $W_{\text{length}} = 0.8$ if $\text{length}(t) > 15$.

### 4.3 Auto-Selection
`auto_select(candidates, top_k=5)` ranks all extracted tags and prioritizes verified `[MEMORY]` tags from the local graph, returning the 5 most statistically potent clues to feed back into the search engine.

---

## 5. Watchlist & Retroactive Delayed Discovery

If a target remains unverified even after C3 re-injection:
1. The target is moved to the **Watchlist** (`pending_targets.json` in the Knowledge Graph) with their timestamp, crop, and 512-d vector.
2. In subsequent days or investigations, whenever another search resolves a related post or teammate, FaceTrace triggers a background scan of pending targets.
3. If similarity exceeds `0.65`, the system alerts the investigator that a cold case has been retroactively cracked.
