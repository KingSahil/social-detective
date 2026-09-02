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

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1):
        """
        Parameters
        ----------
        model_name : str
            InsightFace model pack name (default ``buffalo_l``).
        ctx_id : int
            ONNX Runtime execution provider.  -1 = CPU, 0 = GPU:0.
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

        # Suppress ONNX runtime/model load logs during prepare
        import contextlib
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
                self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

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
