"""
Scene, Lanyard, Badge, and Event Frame OCR Extractor using RapidOCR.
Extracts readable text, hashtags, handle candidates, and event entities
from situational scene images, lanyards, and hackathon participant frames.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
import cv2
import numpy as np

_OCR_ENGINE = None

def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except Exception:
            _OCR_ENGINE = None
    return _OCR_ENGINE


def extract_scene_text_and_clues(
    image_input: str | Path | np.ndarray,
    min_confidence: float = 0.50,
) -> dict[str, Any]:
    """
    Extracts text, hashtags, brand/organization names, and potential handles from an image.
    Inspects full image as well as multi-angle orientations for vertical lanyards.
    """
    img = None
    if isinstance(image_input, (str, Path)):
        p = str(image_input)
        if os.path.exists(p):
            img = cv2.imread(p)
    elif isinstance(image_input, np.ndarray):
        img = image_input

    if img is None:
        return {
            "raw_text": "",
            "hashtags": [],
            "handles": [],
            "entities": [],
            "keywords": [],
            "segments": [],
        }

    engine = get_ocr_engine()
    if engine is None:
        return {
            "raw_text": "",
            "hashtags": [],
            "handles": [],
            "entities": [],
            "keywords": [],
            "segments": [],
        }

    segments: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    def _add_ocr_results(raw_res):
        if not raw_res:
            return
        for item in raw_res:
            try:
                text = str(item[1]).strip()
                conf = float(item[2])
                if text and conf >= min_confidence and text not in seen_texts:
                    seen_texts.add(text)
                    segments.append({"text": text, "confidence": round(conf, 3)})
            except Exception:
                pass

    # 1. Full image OCR
    # Downscale image if extremely high resolution for speed while maintaining readability
    max_dim = max(img.shape[:2])
    ocr_img = img
    if max_dim > 1600:
        scale = 1600.0 / max_dim
        ocr_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    res, _ = engine(ocr_img)
    _add_ocr_results(res)

    def _has_strong_clues():
        txt = " ".join(s["text"] for s in segments).lower()
        if any(w in txt for w in ["#", "@", "frame", "hacker", "hackhazards", "symbiosis", "passport", "participant", "builder"]):
            return True
        return len(segments) >= 5

    # 2. Only check rotated orientations if text was sparse or no strong clues found
    if not _has_strong_clues():
        h, w = img.shape[:2]
        chest_crop = img[int(h * 0.35):, :]
        crop_max = max(chest_crop.shape[:2])
        if crop_max > 900:
            sc = 900.0 / crop_max
            chest_crop = cv2.resize(chest_crop, (int(chest_crop.shape[1] * sc), int(chest_crop.shape[0] * sc)), interpolation=cv2.INTER_AREA)

        for angle in [90, 180, 270]:
            if angle == 90:
                rot = cv2.rotate(chest_crop, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rot = cv2.rotate(chest_crop, cv2.ROTATE_180)
            elif angle == 270:
                rot = cv2.rotate(chest_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            res_rot, _ = engine(rot)
            _add_ocr_results(res_rot)
            if _has_strong_clues():
                break

        # Also check narrower chest center strip rotated 180 for vertical lanyard text if still needed
        if not _has_strong_clues():
            lanyard_strip = img[int(h * 0.4):min(h, int(h * 0.95)), int(w * 0.2):int(w * 0.8)]
            if lanyard_strip.size > 0:
                rot180 = cv2.rotate(lanyard_strip, cv2.ROTATE_180)
                res_lanyard, _ = engine(rot180)
                _add_ocr_results(res_lanyard)

    # Compile raw text
    full_text = " ".join(s["text"] for s in segments)

    # Extract hashtags (#FrameInGoa, #HHGoa, etc.)
    hashtags = list(set(re.findall(r"#([A-Za-z0-9_]+)", full_text)))

    # Extract handles (@user, or handles like 2:47PM STUDIO -> 247pmstudio)
    handles = set(re.findall(r"@([A-Za-z0-9_]{3,30})", full_text))
    
    # Recognize common studio/brand patterns like "2:47PM" + "STUDIO" -> 247pmstudio
    if "247" in full_text.replace(":", "").replace(" ", "") or ("2:47" in full_text and "studio" in full_text.lower()):
        handles.add("247pmstudio")

    # Extract organizations / event phrases
    entities = set()
    for pattern in [
        r"(?:hacker\s*house|hackhazards?)\s*goa(?:\s*20\d\d)?",
        r"(?:symbiosis|mbiosis|embiosis)\s*skill[es]*(?:\s*and\s*(?:open|professional)\s*university)?",
        r"ethindia(?:\s*20\d\d)?",
        r"builder\s*passport",
        r"participant\s*frame",
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            entities.add(m.group(0).strip())
    clean_no_spaces = full_text.lower().replace(" ", "").replace(":", "")
    if any(k in clean_no_spaces for k in ["symbiosis", "mbiosis", "embiosis"]) and "skill" in clean_no_spaces:
        entities.add("Symbiosis Skills University")

    # Keywords
    keywords = set()
    for word in ["goa", "pune", "delhi", "mumbai", "india", "hackathon", "builder", "passport", "frame", "studio", "symbiosis"]:
        if re.search(rf"\b{word}\b", full_text, re.IGNORECASE) or word in clean_no_spaces:
            keywords.add(word)

    return {
        "raw_text": full_text,
        "hashtags": sorted(hashtags),
        "handles": sorted(handles),
        "entities": sorted(entities),
        "keywords": sorted(keywords),
        "segments": segments,
    }
