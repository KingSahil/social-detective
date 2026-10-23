# Implementation Plan: Retroactive Search & Resource Protection Toggle

## Goal
Implement the active background trigger for **Retroactive Search (Delayed Discovery)** so that new entries automatically scan the `pending` watchlist to rescue previously failed searches. To prevent performance explosion, an automatic resource limit will be implemented alongside a manual override toggle switch in the UI's sidebar.

## Proposed Changes

### 1. Update `IdentityKnowledgeGraph` Backend
#### [MODIFY] `app/memory/graph.py`
- Update `add_verified_record()` to accept a new parameter: `force_retroactive: bool = False`.
- **Resource Logic**: Check `len(self.get_pending_targets())`. 
  - Since biometric NumPy comparisons are computationally heavy in the main thread, the threshold will be strictly set to **50**.
  - If pending targets exceed **50** and `force_retroactive` is `False`, skip the scan and return a `retro_disabled_warning=True` flag.
  - Else, loop through pending targets, compute biometric `cosine_similarity(new_embedding, pending_embedding)`.
  - If similarity > 0.65, call `self.mark_resolved(pending_id)`.
- Change the return type from a single object to a tuple: `return person, newly_resolved_ids, retro_disabled_warning`.

### 2. Update Sync and Pipeline Connectors
#### [MODIFY] `app/pipeline.py`, `app/memory/web3_sync.py`, & `app/memory/migrate.py`
- *Explanation*: Because we changed the backend function `add_verified_record()` to return three things instead of one, we must update every other script that uses this function. If we don't, they will crash expecting 1 item but receiving 3.
- *Change*: We will update those files to correctly unpack the tuple:
  `person, newly_resolved_ids, retro_disabled_warning = graph.add_verified_record(...)`
- In `pipeline.py` and `ui.py`, we will use those new variables to trigger the UI notifications.

### 3. Add Control Toggle & Notifications to UI
#### [MODIFY] `app/ui.py`
- In the left control panel (sidebar), at the bottom side, add a new section:
  ```python
  st.sidebar.divider()
  st.sidebar.header("🧠 Graph Memory")
  force_retroactive = st.sidebar.toggle("Force Retroactive Search", value=False, help="Forces biometric scanning of all pending targets against new data, overriding the 50-target limit.")
  ```
- Pass `force_retroactive` to the pipeline execution.
- **Dynamic Notifications**:
  - Display `st.warning("Retroactive search was disabled to conserve resources (>50 pending targets). Enable 'Force Retroactive Search' in the sidebar to run it anyway.")` if the warning flag triggers.
  - Display `st.success("🎉 Retroactive Search Hit! X previously failed targets were resolved!")` if new matches are found.

## Verification Plan
1. Add a dummy pending target to the graph.
2. Ensure the UI toggle switch successfully appears in the bottom left sidebar.
3. Run a successful search that adds a new verified record.
4. Confirm the background retroactive search correctly identifies the biometric match and resolves the pending target, flashing the success notification.
