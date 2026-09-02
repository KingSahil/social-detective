"""Tests for face detection and embedding."""

import numpy as np
import cv2
import pytest
from pathlib import Path

from app.face import FaceProcessor, FaceProcessingError


@pytest.fixture(scope="module")
def processor():
    """Create a shared FaceProcessor (model loading is expensive)."""
    return FaceProcessor()


@pytest.fixture
def no_face_image(tmp_path):
    """Create a blank image with no face."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    path = tmp_path / "no_face.jpg"
    cv2.imwrite(str(path), img)
    return path


class TestFaceProcessor:
    def test_invalid_image_path(self, processor):
        with pytest.raises(FaceProcessingError, match="Cannot load image"):
            processor.get_embedding("nonexistent_image.jpg")

    def test_no_face_detected(self, processor, no_face_image):
        with pytest.raises(FaceProcessingError, match="No face detected"):
            processor.get_embedding(str(no_face_image))

    def test_detect_faces_returns_list(self, processor, no_face_image):
        faces = processor.detect_faces(str(no_face_image))
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_get_best_embedding_no_face(self, processor):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = processor.get_best_embedding_from_image(img)
        assert result is None
