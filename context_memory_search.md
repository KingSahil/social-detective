# Implementation Plan: Watchlist & Asynchronous Detection Daemon

## Understanding Your Request
You are describing a **continuous correlation engine**. When a user hunts for a primary target and the search *fails*, that target shouldn't just be discarded. It should be saved as a "Pending" or "Watched" target. 
Later, if the user performs a **Crowd Pivot** (scanning a completely different background face), the metadata extracted from that *new* face might contain the missing link (e.g., a shared event, an @handle, or a location) that perfectly correlates back to the original *failed* target. 
When this happens, the system needs to **detect** the correlation asynchronously and trigger a **notification** to alert the user that their previously lost target has been found through secondary association.

## Proposed Architecture

### 1. The Watchlist Storage (`app/watchlist.py`)
We will create a local database or JSON store (`pending_targets.json`) that holds failed searches. 
Each entry will store:
- The biometric embedding of the failed target.
- The original timestamp and image reference.
- Any partial metadata we *did* have.

### 2. The Correlation Engine
During **Phase 3** (Graph Memory Sync) or at the end of any new search pipeline, we will hook into the `IdentityKnowledgeGraph`.
Every time *new* metadata (events, aliases, embeddings) is ingested into the graph from a new target, the system will run a cross-check:
`Did this new metadata just complete the profile for anyone on the Watchlist?`

### 3. The Notification Daemon
We will implement a lightweight background process that handles these alerts natively.
If a correlation threshold is met (e.g., Target B attended an event that matches the location of Pending Target A), the daemon will fire an OS-level desktop notification alerting the user:
*“🚨 Alert: Pending Target #123 has been correlated via a secondary search!”*

## Verification Plan
1. Manually run a search that is designed to fail (saving the embedding to the Watchlist).
2. Run a secondary Crowd Pivot search that injects metadata known to correlate with the first target.
3. Ensure the daemon detects the link and fires a desktop notification.

## User Review Required

> [!IMPORTANT]
> This requires adding a dependency for OS notifications (like `plyer` or `win11toast` for Windows). 

## Open Questions

> [!WARNING]
> How would you like the notifications to be delivered? Should it be a native Windows popup (Toast notification), a Streamlit UI alert in the dashboard, or both?
