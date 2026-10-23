"""
Face detection and embedding using InsightFace / ArcFace.

Provides:
    FaceProcessor — wraps InsightFace's FaceAnalysis for detection + embedding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np


class FaceProcessingError(Exception):
    """Raised when face processing fails."""


class FaceProcessor:
    """Detect faces and compute ArcFace embeddings via InsightFace."""

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        """
        Parameters
        ----------
        model_name : str
            InsightFace model pack name (default ``buffalo_l``).
        ctx_id : int
            Execution context.  0 = GPU:0 (with CPU fallback), -1 = CPU only.
        """
        import os
        import warnings
        import logging

        # Suppress insightface internal logging and future warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("insightface").setLevel(logging.ERROR)

        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError:
            print("\n  ✗ insightface is not installed.")
            print("    Run: pip install insightface onnxruntime opencv-python\n")
            sys.exit(1)

        # The pipeline only uses detection + recognition (512-d embeddings).
        # genderage/landmark3d/2d106det models are never consumed — skip them.
        allowed = ["detection", "recognition"]

        # Prefer GPU (CUDA EP) for ~9x faster embedding; silently fall back
        # to CPU-only when onnxruntime-gpu is absent or the GPU is unusable.
        # FACE_DEVICE=cpu|auto (default auto) forces the CPU path.
        import os as _os
        import sys as _sys

        if _sys.platform == "win32":
            # On Windows, pip-installed NVIDIA runtime wheels (cublas, cudnn, etc.)
            # place DLLs inside site-packages/nvidia/<pkg>/bin. Inject them into PATH
            # and os.add_dll_directory so ONNX Runtime's CUDA EP finds them cleanly.
            import importlib.util as _util
            for _pkg in ("nvidia.cu13", "nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cu12"):
                try:
                    _spec = _util.find_spec(_pkg)
                    if _spec and _spec.submodule_search_locations:
                        for _loc in _spec.submodule_search_locations:
                            for _root, _, _files in _os.walk(_loc):
                                if any(_f.endswith(".dll") for _f in _files):
                                    if _root not in _os.environ["PATH"]:
                                        _os.environ["PATH"] = _root + _os.pathsep + _os.environ["PATH"]
                                    try:
                                        _os.add_dll_directory(_root)
                                    except (AttributeError, OSError):
                                        pass
                except Exception:
                    pass

        device_pref = _os.getenv("FACE_DEVICE", "auto").strip().lower()
        providers = None
        try:
            import onnxruntime as _ort
            if (
                device_pref != "cpu"
                and "CUDAExecutionProvider" in _ort.get_available_providers()
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            providers = None

        import contextlib
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                if providers is not None:
                    try:
                        self._app = FaceAnalysis(
                            name=model_name,
                            allowed_modules=allowed,
                            providers=providers,
                        )
                        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                    except Exception:
                        # GPU session creation failed (driver/VRAM) — degrade to CPU
                        providers = None
                if providers is None:
                    self._app = FaceAnalysis(
                        name=model_name,
                        allowed_modules=allowed,
                        providers=["CPUExecutionProvider"],
                    )
                    self._app.prepare(ctx_id=-1, det_size=(640, 640))
        self.using_gpu = providers is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image_path: str | Path) -> list:
        """
        Detect all faces in *image_path*.

        Returns a list of InsightFace ``Face`` objects (may be empty).

        Raises
        ------
        FaceProcessingError
            If the image cannot be loaded.
        """
        image_path = str(image_path)
        img = cv2.imread(image_path)
        if img is None:
            raise FaceProcessingError(f"Cannot load image: {image_path}")
        faces = self._app.get(img)
        return faces

    def get_embedding(
        self,
        image_path: str | Path,
        face_index: Optional[int] = None,
        auto_select_largest: bool = True,
    ) -> np.ndarray:
        """
        Detect face(s) in *image_path* and return a 512-d ArcFace embedding.

        Parameters
        ----------
        image_path : str | Path
            Path to the image file.
        face_index : Optional[int]
            When multiple faces are detected, specifies which face to use (0 = largest/primary).
        auto_select_largest : bool
            If True (default) and face_index is None, automatically selects the largest face.

        Raises
        ------
        FaceProcessingError
            If no face is detected, or the requested face_index is out of range.
        """
        faces = self.detect_faces(image_path)

        if len(faces) == 0:
            raise FaceProcessingError(
                "No face detected in the image. "
                "Please provide a clear photo containing a face."
            )

        # Sort detected faces by bounding-box area descending (largest first)
        def _area(f):
            bbox = f.bbox
            return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        faces.sort(key=_area, reverse=True)

        if len(faces) > 1 and not auto_select_largest and face_index is None:
            raise FaceProcessingError(
                f"Multiple faces detected ({len(faces)}). "
                f"Please specify --face-index <0..{len(faces)-1}> to select a face."
            )

        idx = face_index if face_index is not None else 0
        if idx < 0 or idx >= len(faces):
            raise FaceProcessingError(
                f"Invalid face_index {idx}. Image has {len(faces)} detected face(s) "
                f"(valid indices: 0 to {len(faces)-1})."
            )

        face = faces[idx]
        embedding = face.embedding  # 512-d float32 vector
        if embedding is None:
            raise FaceProcessingError("Face detected but embedding extraction failed.")

        return embedding

    def get_embedding_from_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempt to extract a single-face embedding from an already-loaded image
        (numpy BGR array).  Returns ``None`` if no single face is found.
        """
        faces = self._app.get(img)
        if len(faces) != 1:
            return None
        return faces[0].embedding

    def get_best_embedding_from_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the embedding of the largest face in the image.
        Returns ``None`` if no face is found.
        """
        faces = self._app.get(img)
        if not faces:
            return None
        # Pick the face with the largest bounding-box area
        def _area(f):
            bbox = f.bbox  # [x1, y1, x2, y2]
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        best = max(faces, key=_area)
        return best.embedding

    def get_face_crop(
        self,
        image_path: str | Path,
        face_index: Optional[int] = None,
        margin: float = 0.35,
    ) -> Optional[np.ndarray]:
        """
        Detect faces in *image_path* and return a cropped BGR image of the target face
        (0 = largest face) with an added margin/padding (default 35% around the bounding box).
        Returns None if no face is detected or image cannot be read.
        """
        image_path = str(image_path)
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = self._app.get(img)
        if not faces:
            return None

        def _area(f):
            bbox = f.bbox
            return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        faces.sort(key=_area, reverse=True)
        idx = face_index if (face_index is not None and 0 <= face_index < len(faces)) else 0
        target = faces[idx]

        x1, y1, x2, y2 = [int(v) for v in target.bbox]
        h, w = img.shape[:2]

        pad_w = int((x2 - x1) * margin)
        pad_h = int((y2 - y1) * margin)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)

        crop = img[cy1:cy2, cx1:cx2]
        return crop

    def annotate_and_extract_all_faces(self, image_path: str | Path) -> tuple[Optional[np.ndarray], list[dict]]:
        """
        Detects all faces in a crowd photo.
        Returns:
            - annotated_img: BGR numpy array with numbered bounding boxes drawn around each face.
            - face_data: A list of dicts {'embedding': np.ndarray, 'bbox': list} corresponding to the numbers (1-indexed).
        Returns (None, []) if the image fails to load or no faces are found.
        """
        image_path = str(image_path)
        img = cv2.imread(image_path)
        if img is None:
            return None, []
        
        faces = self._app.get(img)
        if not faces:
            return img, []

        annotated_img = img.copy()
        face_data = []

        # Sort faces from left-to-right based on x-coordinate
        faces = sorted(faces, key=lambda f: f.bbox[0])

        img_h, img_w = img.shape[:2]
        font_scale = max(0.4, img_w / 1500.0)
        thickness = max(1, int(font_scale * 2))
        box_thickness = max(1, int(font_scale * 3))

        for idx, face in enumerate(faces):
            # 1-indexed numbering for user selection
            face_id = idx + 1
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            
            face_data.append({
                'embedding': face.embedding,
                'bbox': [x1, y1, x2, y2]
            })
            
            # Draw bold rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
            
            # Draw label background
            label = f"#{face_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(annotated_img, (x1, y1 - h - 10), (x1 + w + 10, y1), (0, 255, 0), -1)
            cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        return annotated_img, face_data


