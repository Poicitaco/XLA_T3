from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """InsightFace-based embedding extractor for detected face boxes."""

    def __init__(self, model: str = "buffalo_l", det_size: tuple[int, int] = (256, 256)) -> None:
        self.model = model
        self.app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return -1.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def extract_embeddings_from_frame(self, frame: np.ndarray, boxes: List[List[int]]) -> List[Optional[np.ndarray]]:
        """Extract one embedding per input box; returns None if extraction fails for a box."""
        embeddings: List[Optional[np.ndarray]] = []
        h, w = frame.shape[:2]

        for box in boxes:
            x, y, bw, bh = box
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            if x2 <= x1 or y2 <= y1:
                embeddings.append(None)
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                embeddings.append(None)
                continue

            faces = self.app.get(roi)
            if not faces:
                embeddings.append(None)
                continue

            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = getattr(best_face, "embedding", None)
            embeddings.append(np.asarray(emb, dtype=np.float32) if emb is not None else None)

        return embeddings

    def extract_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single embedding from an image (largest detected face)."""
        if image is None or image.size == 0:
            return None
        faces = self.app.get(image)
        if not faces:
            return None
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(best_face, "embedding", None)
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)
