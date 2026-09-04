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

    def get_embedding(self, image_path: str | Path) -> np.ndarray:
        """
        Detect exactly one face and return its 512-d ArcFace embedding.

        Raises
        ------
        FaceProcessingError
            If zero or more than one face is detected, or the image is invalid.
        """
        faces = self.detect_faces(image_path)

        if len(faces) == 0:
            raise FaceProcessingError(
                "No face detected in the image. "
                "Please provide a clear photo containing exactly one face."
            )

        if len(faces) > 1:
            raise FaceProcessingError(
                f"Multiple faces detected ({len(faces)}). "
                "Please provide an image containing exactly one face."
            )

        face = faces[0]
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
        self, image_path: str | Path, margin: float = 0.35
    ) -> Optional[np.ndarray]:
        """
        Detect faces in *image_path* and return a cropped BGR image of the largest face
        with an added margin/padding (default 35% around the bounding box).
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
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        best = max(faces, key=_area)
        x1, y1, x2, y2 = [int(v) for v in best.bbox]
        h, w = img.shape[:2]

        pad_w = int((x2 - x1) * margin)
        pad_h = int((y2 - y1) * margin)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)

        crop = img[cy1:cy2, cx1:cx2]
        return crop

