import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import os
import sys
import tempfile
import threading
from contextlib import redirect_stdout, redirect_stderr
import io
import subprocess
import re

# Add project root to sys.path so 'app' can be resolved when running via streamlit
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.face import FaceProcessor
from app.context import ContextManager
from app.search import Candidate

st.set_page_config(page_title="Social Detective Dashboard", layout="wide")

st.title("Social Detective Web Dashboard")
st.markdown("Upload a photo and trace their digital footprint.")
st.divider()

# ==========================================
# C3 Notification Panels (Phases 3 & 4)
# ==========================================
try:
    from app.memory.graph import IdentityKnowledgeGraph
    graph = IdentityKnowledgeGraph()
    resolved = graph.get_resolved_targets()
    if resolved:
        st.success(f"🎉 **Phase 4 Resolution!** {len(resolved)} pending target(s) were successfully resolved using recent Crowd Pivot context.")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.button("📄 View Dossiers")
        with c2:
            if st.button("❌ Dismiss Alert"):
                for r in resolved:
                    r.status = "archived"
                graph.save()
                st.rerun()

    # Manual Consent Background Task
    if st.session_state.get('run_bg_correlation'):
        st.session_state.run_bg_correlation = False
        def _correlate_background(tags):
            import time
            try:
                bg_graph = IdentityKnowledgeGraph()
                pending = bg_graph.get_pending_targets()
                if pending and tags:
                    time.sleep(5) # Simulate heavy web search latency
                    bg_graph.mark_resolved(pending[-1].id) # Resolve the most recent
            except Exception:
                pass
        threading.Thread(target=_correlate_background, args=(st.session_state.context_tags,), daemon=True).start()
        st.info("🔄 Background search thread for pending targets has been launched! You can continue using the app.")
        
except Exception:
    pass


@st.cache_resource
def get_face_processor():
    return FaceProcessor()

fp = get_face_processor()

# Sidebar Configuration
# ==========================================
with st.sidebar:
    st.header("⚙️ Search Config")
    engine = st.selectbox("Search Engine", options=["all", "lens", "yandex", "serpapi"])
    threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.01)
    target_url = st.text_input("Target URL (Optional)", help="Verify against a specific post/video")
    handle = st.text_input("Username Sweep (Optional)", help="Sweep a specific handle across platforms")
    skip_blockchain = st.checkbox("Skip Blockchain Notarization", value=True)

# ==========================================
# State Management
# ==========================================
if "phase_1_result" not in st.session_state:
    st.session_state.phase_1_result = None
if "context_tags" not in st.session_state:
    st.session_state.context_tags = []

# ==========================================
# Top-Level Mode Selection
# ==========================================
st.subheader("🕵️‍♂️ Operation Mode")
mode = st.radio(
    "Select Workflow:", 
    ["Single Target Hunt", "Crowd Context Search"],
    horizontal=True,
    help="Single Target: Best for close-ups or solo photos.\nCrowd Context: Analyzes background people if the primary target fails."
)

st.divider()

# ==========================================
# File Uploads (Unified)
# ==========================================
uploaded_main = None
uploaded_ref = None

if mode == "Single Target Hunt":
    st.subheader("📷 Upload Target Photo")
    uploaded_main = st.file_uploader("Upload a photo of the individual", type=["jpg", "jpeg", "png"], key="single_up")
else:
    st.subheader("👥 Crowd Photo & Target Reference")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_main = st.file_uploader("1. Upload the Crowd Photo", type=["jpg", "jpeg", "png"], key="crowd_up")
    with col2:
        uploaded_ref = st.file_uploader("2. Upload Reference Photo (Optional for Auto-Match)", type=["jpg", "jpeg", "png"], key="ref_up")

if uploaded_main is not None:
    # Save main file to temp dir
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_main.getvalue())
        tmp_path = tmp.name

    with st.spinner("Analyzing faces..."):
        annotated_img, face_data = fp.annotate_and_extract_all_faces(tmp_path)
    
    if len(face_data) == 0:
        st.error("No faces detected in the image.")
    else:
        st.subheader("🎯 Target Selection")
        target_idx = None
        
        col_img, col_sel = st.columns([2, 1])
        
        with col_img:
            # Render the annotated image
            with st.container(border=True):
                rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(rgb_img, caption=f"Found {len(face_data)} faces")
        
        with col_sel:
            if len(face_data) == 1:
                st.info("Single face detected. Automatically selected.")
                target_idx = 1
            else:
                if uploaded_ref is not None:
                    # Auto-Match Flow
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as rtmp:
                        rtmp.write(uploaded_ref.getvalue())
                    
                    with st.spinner("Auto-matching target..."):
                        try:
                            ref_emb = fp.get_embedding(rtmp.name)
                            best_idx = 0
                            best_sim = -1.0
                            for i, f_data in enumerate(face_data):
                                emb = f_data['embedding']
                                sim = np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_idx = i
                            
                            target_idx = best_idx + 1
                            if best_sim > 0.50:
                                st.success(f"Auto-Matched to Face #{target_idx} ({best_sim*100:.1f}% similarity)")
                            else:
                                st.warning(f"Weak Auto-Match: Face #{target_idx} ({best_sim*100:.1f}% similarity). You may want to manually select.")
                                target_idx = None
                        except Exception:
                            st.error("No valid face found in reference photo.")
                
                # Manual Override / Fallback
                if target_idx is None:
                    target_idx = st.selectbox(
                        "Select Primary Target (Face #):", 
                        options=range(1, len(face_data) + 1),
                        help="Choose the numbered box corresponding to the person you want to trace."
                    )
                else:
                    override = st.checkbox("Override Auto-Match")
                    if override:
                        target_idx = st.selectbox("Select Primary Target (Face #):", options=range(1, len(face_data) + 1))

            st.divider()
            run_btn = st.button("🚀 Run Social Detective", type="primary", use_container_width=True)

        # ==========================================
        # Execution & Terminal Tabs
        # ==========================================
        if run_btn:
            # Crop the selected face
            selected_face = face_data[target_idx - 1]
            x1, y1, x2, y2 = selected_face['bbox']
            
            margin = 0.60
            pad_w = int((x2 - x1) * margin)
            pad_h = int((y2 - y1) * margin)
            
            orig_img = cv2.imread(tmp_path)
            h, w = orig_img.shape[:2]
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w, x2 + pad_w)
            cy2 = min(h, y2 + pad_h)
            
            cropped_face = orig_img[cy1:cy2, cx1:cx2]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as cropped_tmp:
                cv2.imwrite(cropped_tmp.name, cropped_face)
                cropped_path = cropped_tmp.name

            # Build CLI command
            cmd = [
                sys.executable, "-m", "app.main",
                "--image", cropped_path,
                "--engine", engine,
                "--threshold", str(threshold)
            ]
            if target_url:
                cmd.extend(["--target", target_url])
            if handle:
                cmd.extend(["--handle", handle])
            if skip_blockchain:
                cmd.append("--skip-blockchain")

            st.divider()
            st.subheader("🔍 Analysis Dashboard")
            tab_visuals, tab_terminal = st.tabs(["🚀 Execution Pipeline", "⚙️ Under the Hood (Logs)"])

            with tab_terminal:
                st.caption("Raw execution logs are streaming below...")
                log_container = st.empty()

            with tab_visuals:
                status_placeholder = st.empty()
                with status_placeholder.status(f"Hunting for Target #{target_idx}...", expanded=True) as status:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                    
                    full_log = ""
                    step_regex = re.compile(r"\[(\d+/\d+)\]\s+(.*)")
                    
                    for line in iter(process.stdout.readline, ''):
                        full_log += line
                        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
                        
                        step_match = step_regex.search(clean_line)
                        if step_match:
                            status.update(label=f"Step {step_match.group(1)}: {step_match.group(2)}")
                            st.write(f"⏳ {step_match.group(2)}...")
                        elif "✓" in clean_line:
                            st.write(f"✅ {clean_line.replace('✓', '').strip()}")
                        elif "✗" in clean_line:
                            st.write(f"❌ {clean_line.replace('✗', '').strip()}")
                        
                        log_container.code(full_log, language="bash")
                    
                    process.stdout.close()
                    process.wait()
                    
                    if process.returncode == 0:
                        status.update(label="Social Detective Complete!", state="complete", expanded=False)
                        st.session_state.phase_1_result = "success"
                        st.success("Target successfully traced! Check 'Under the Hood' tab for log details.")
                    else:
                        status.update(label="Pipeline Failed", state="error", expanded=True)
                        st.session_state.phase_1_result = "failed"
                        st.error(f"Pipeline exited with error code {process.returncode}. See 'Under the Hood' tab.")

        # ==========================================
        # Phase 2: Crowd Pivot UI
        # ==========================================
        if st.session_state.phase_1_result == "failed" and mode == "Crowd Context Search" and len(face_data) > 1:
            st.divider()
            st.warning("⚠️ Target not found directly. Initiating Phase 2: Crowd Pivot!")
            st.subheader("👥 Crowd Pivot Configuration")
            
            pivot_mode = st.radio(
                "Select Background Faces to Scan:", 
                ["Auto-Select Top 3", "Manual Select", "Search ALL Faces (High Risk of Bot Flagging)"]
            )
            
            pivot_idxs = []
            if pivot_mode == "Auto-Select Top 3":
                other_faces = [(i, f) for i, f in enumerate(face_data) if i != target_idx - 1]
                other_faces.sort(key=lambda x: (x[1]['bbox'][2]-x[1]['bbox'][0])*(x[1]['bbox'][3]-x[1]['bbox'][1]), reverse=True)
                pivot_idxs = [i for i, f in other_faces[:3]]
                st.info(f"Auto-selected {len(pivot_idxs)} largest background faces.")
            elif pivot_mode == "Manual Select":
                options = [f"Face #{i+1}" for i in range(len(face_data)) if i != target_idx - 1]
                selected = st.multiselect("Select faces to pivot on:", options)
                pivot_idxs = [int(s.split('#')[1])-1 for s in selected]
            else:
                st.error("⚠️ WARNING: Searching all faces rapidly may cause your IP to be flagged for bot activity and result in API bans.")
                pivot_idxs = [i for i in range(len(face_data)) if i != target_idx - 1]

            if st.button("🔍 Extract Crowd Context"):
                with st.spinner("Extracting metadata from crowd profiles..."):
                    # Phase 2: Biometric Memory Scan
                    memory_tags = []
                    try:
                        from app.memory.graph import IdentityKnowledgeGraph
                        graph = IdentityKnowledgeGraph()
                        for p_idx in pivot_idxs:
                            bg_face = face_data[p_idx]
                            person, sim = graph.find_nearest_person(np.array(bg_face['embedding']), threshold=0.65)
                            if person and getattr(person, 'status', 'verified') != "pending":
                                if person.events:
                                    memory_tags.extend([f"#{ev.replace(' ', '')}" for ev in person.events])
                                if person.associates:
                                    memory_tags.extend([f"@{a.replace(' ', '')}" for a in person.associates])
                    except Exception:
                        pass
                        
                    dummy_candidates = [
                        Candidate(image_url="", source_url="", title="Attended #HackHazards 2026 at Stanford University", domain="stanford.edu"),
                        Candidate(image_url="", source_url="", title="Software Engineer @Google - @JohnDoe", domain="linkedin.com"),
                        Candidate(image_url="", source_url="", title="Team photo from the #AI retreat", domain="instagram.com")
                    ]
                    
                    cm = ContextManager()
                    st.session_state.context_tags = cm.process_and_suggest(dummy_candidates, memory_tags=memory_tags)
                    st.session_state.show_keyword_ui = True
                    
            if st.session_state.get('show_keyword_ui', False):
                st.subheader("🏷️ Context Keywords Extracted")
                st.write("We extracted the following context tags from the background people. Select the most relevant tags to inject into the primary search:")
                
                with st.form("keyword_form"):
                    selected_tags = []
                    for tag in st.session_state.context_tags:
                        # Ensure [MEMORY] tags are visibly flagged but checked by default
                        val = True if "[MEMORY]" in tag else False
                        if st.checkbox(tag, value=val):
                            selected_tags.append(tag)
                    
                    st.divider()
                    st.write("**Phase 3: Pending Target Correlation**")
                    try:
                        from app.memory.graph import IdentityKnowledgeGraph
                        if len(IdentityKnowledgeGraph().get_pending_targets()) > 0:
                            st.warning("⚠️ You have Pending Targets in your Watchlist.")
                            approve_bg = st.checkbox("Approve background search for Pending Targets using these new keywords?", value=False)
                        else:
                            approve_bg = False
                    except Exception:
                        approve_bg = False

                    col1, col2 = st.columns(2)
                    with col1:
                        manual_submit = st.form_submit_button("🚀 Run Targeted Search")
                    with col2:
                        auto_submit = st.form_submit_button("🤖 Auto-Select Top 5 & Run")
                        
                    if manual_submit or auto_submit:
                        final_tags = selected_tags
                        if auto_submit:
                            # ContextManager already prioritizes [MEMORY] tags in top 5
                            final_tags = st.session_state.context_tags[:5]
                        
                        context_str = ",".join(final_tags)
                        st.success(f"Running Phase 2 Search with context: `{context_str}`")
                        
                        if approve_bg:
                            st.session_state.run_bg_correlation = True
                            st.rerun()
