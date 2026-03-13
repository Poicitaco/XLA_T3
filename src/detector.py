from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class FaceDetector:
    """Face detector backed by Ultralytics YOLOv11-face."""

    def __init__(
        self,
        model_path: str = "models/yolov11n-face.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the detector and load YOLO model.

        Args:
            model_path: Path/name of YOLOv11-face weights.
            conf_threshold: Confidence threshold for face detections.
            iou_threshold: IoU threshold for NMS.
            imgsz: Inference image size.
            device: Inference device, e.g. "cpu", "cuda:0". None for auto.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.model: Optional[Any] = None
        self.model_path = model_path
        self._model_unavailable_logged = False

        try:
            yolo_cls = self._resolve_yolo_class()
            if yolo_cls is None:
                return
            self.model = yolo_cls(model_path)
        except Exception as exc:  # pragma: no cover - defensive runtime safety
            LOGGER.exception("Failed to load model '%s': %s", model_path, exc)

    @property
    def is_ready(self) -> bool:
        """Return True when the model is available for inference."""
        return self.model is not None

    @property
    def model_hint(self) -> str:
        """Return a user-facing hint about the expected model location."""
        model_file = Path(self.model_path)
        if model_file.exists():
            return str(model_file)
        return (
            f"Model file not found: {self.model_path}. "
            "Place the weight file in project root or pass --model with full path."
        )

    @staticmethod
    def _resolve_yolo_class() -> Optional[type]:
        """Resolve ultralytics.YOLO lazily to avoid hard import failures."""
        try:
            ultralytics_module = import_module("ultralytics")
            return getattr(ultralytics_module, "YOLO", None)
        except Exception as exc:  # pragma: no cover - defensive runtime safety
            LOGGER.exception("Ultralytics import failed: %s", exc)
            return None

    def detect_faces(self, frame: np.ndarray) -> List[List[int]]:
        """
        Detect faces from an OpenCV frame.

        Args:
            frame: Image frame as numpy array (H, W, C) from OpenCV.

        Returns:
            List of face boxes as [x, y, w, h] (integers), suitable for
            cropping with NumPy: face = frame[y:y+h, x:x+w].
        """
        return [box for box, _ in self.detect_faces_with_scores(frame)]

    def detect_faces_with_scores(self, frame: np.ndarray) -> List[Tuple[List[int], float]]:
        """
        Detect faces and return both bounding boxes and confidence scores.

        Args:
            frame: Image frame as numpy array (H, W, C) from OpenCV.

        Returns:
            List of tuples: ([x, y, w, h], confidence).
        """
        if self.model is None:
            if not self._model_unavailable_logged:
                LOGGER.error("Face detector model is not available. %s", self.model_hint)
                self._model_unavailable_logged = True
            return []

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            LOGGER.warning("Invalid frame input for face detection.")
            return []

        height, width = frame.shape[:2]
        detections: List[Tuple[List[int], float]] = []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

            if not results:
                return []

            boxes = results[0].boxes
            if boxes is None or boxes.xyxy is None:
                return []

            conf_tensor = boxes.conf if boxes.conf is not None else None

            for idx, xyxy in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = xyxy.tolist()

                x1_i = max(0, min(width - 1, int(round(x1))))
                y1_i = max(0, min(height - 1, int(round(y1))))
                x2_i = max(0, min(width, int(round(x2))))
                y2_i = max(0, min(height, int(round(y2))))

                w_i = x2_i - x1_i
                h_i = y2_i - y1_i

                if w_i > 0 and h_i > 0:
                    conf = float(conf_tensor[idx].item()) if conf_tensor is not None else 0.0
                    detections.append(([x1_i, y1_i, w_i, h_i], conf))

            return detections
        except Exception as exc:  # pragma: no cover - defensive runtime safety
            LOGGER.exception("Face detection failed: %s", exc)
            return []
