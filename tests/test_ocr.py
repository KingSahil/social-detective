"""
Unit tests for app.ocr and cold-start event discovery.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from app.ocr import extract_scene_text_and_clues
from app.search import discover_osint_event_leads


def test_extract_scene_text_test_face_16():
    img_path = Path("data/input/test_face_16.jpg")
    if not img_path.exists():
        return
    clues = extract_scene_text_and_clues(img_path)
    assert isinstance(clues, dict)
    assert "hashtags" in clues
    assert "FrameInGoa" in clues["hashtags"]


def test_discover_osint_event_leads_test_face_16():
    img_path = Path("data/input/test_face_16.jpg")
    if not img_path.exists():
        return
    handles, candidates, clues = discover_osint_event_leads(img_path)
    assert isinstance(handles, list)
    assert len(handles) > 0
    # 247pmstudio is extracted from the frame branding '2:47PM STUDIO'
    assert "247pmstudio" in handles
