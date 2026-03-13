from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from src.detector import FaceDetector
    from src.privacy import anonymize_faces
except ImportError:
    from detector import FaceDetector
    from privacy import anonymize_faces

AnonymizeMode = Literal[
    "none",
    "blur",
    "pixelate",
    "solid",
    "obliterate",
    "headcloak",
    "silhouette",
]

PresetName = Literal["fast", "balanced", "strong", "strict"]


@dataclass
class AnonymizationConfig:
    """Backend config designed to be easy for frontend integration."""

    mode: AnonymizeMode = "headcloak"
    conf_threshold: float = 0.35
    imgsz: int = 512
    detect_every: int = 2
    proc_width: int = 640

    blur_scale: float = 0.2
    blur_kernel: int = 31
    pixel_block: int = 16
    face_padding: float = 0.16

    obliterate_scale: float = 0.08
    scramble_block: int = 12
    palette_levels: int = 10
    scramble_seed: int = 1337

    head_ratio: float = 0.7
    silhouette_ratio: float = 0.8


PRESETS: Dict[PresetName, AnonymizationConfig] = {
    "fast": AnonymizationConfig(
        mode="pixelate",
        imgsz=416,
        detect_every=3,
        proc_width=512,
        pixel_block=18,
        face_padding=0.12,
    ),
    "balanced": AnonymizationConfig(
        mode="headcloak",
        imgsz=512,
        detect_every=2,
        proc_width=640,
        head_ratio=0.65,
        face_padding=0.16,
    ),
    "strong": AnonymizationConfig(
        mode="obliterate",
        imgsz=512,
        detect_every=2,
        proc_width=640,
        obliterate_scale=0.06,
        scramble_block=10,
        palette_levels=8,
        face_padding=0.2,
    ),
    "strict": AnonymizationConfig(
        mode="headcloak",
        imgsz=416,
        detect_every=2,
        proc_width=512,
        head_ratio=0.95,
        face_padding=0.24,
        blur_kernel=39,
    ),
}


class FaceAnonymizationEngine:
    """
    Reusable backend engine.

    Frontend can call `process_frame(...)` for each frame and consume:
    - output frame
    - detected boxes
    - latency/FPS-friendly stats
    """

    def __init__(
        self,
        model_path: str = "models/yolov11n-face.pt",
        config: Optional[AnonymizationConfig] = None,
    ) -> None:
        self.config = config or PRESETS["balanced"]
        self.detector = FaceDetector(
            model_path=model_path,
            conf_threshold=self.config.conf_threshold,
            imgsz=self.config.imgsz,
        )
        self._frame_index = 0
        self._last_detections: List[Tuple[List[int], float]] = []

    @staticmethod
    def from_preset(
        preset: PresetName,
        model_path: str = "models/yolov11n-face.pt",
    ) -> "FaceAnonymizationEngine":
        return FaceAnonymizationEngine(model_path=model_path, config=PRESETS[preset])

    @property
    def is_ready(self) -> bool:
        return self.detector.is_ready

    @staticmethod
    def _prepare_inference_frame(frame: np.ndarray, proc_width: int) -> Tuple[np.ndarray, float, float]:
        if proc_width <= 0:
            return frame, 1.0, 1.0

        h, w = frame.shape[:2]
        if w <= proc_width:
            return frame, 1.0, 1.0

        scale = proc_width / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        import cv2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_x = w / float(new_w)
        scale_y = h / float(new_h)
        return resized, scale_x, scale_y

    @staticmethod
    def _scale_detections(
        detections: List[Tuple[List[int], float]],
        scale_x: float,
        scale_y: float,
    ) -> List[Tuple[List[int], float]]:
        if scale_x == 1.0 and scale_y == 1.0:
            return detections

        output: List[Tuple[List[int], float]] = []
        for box, score in detections:
            x, y, w, h = box
            output.append(
                (
                    [
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                        int(round(w * scale_x)),
                        int(round(h * scale_y)),
                    ],
                    score,
                )
            )
        return output

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Main backend API for frontend integration."""
        t0 = time.perf_counter()
        self._frame_index += 1

        if not self.is_ready:
            return {
                "ok": False,
                "error": "detector_not_ready",
                "model_hint": self.detector.model_hint,
                "frame_index": self._frame_index,
            }

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "ok": False,
                "error": "invalid_frame",
                "frame_index": self._frame_index,
            }

        detect_latency_ms = 0.0
        anonymize_latency_ms = 0.0

        if self._frame_index % max(self.config.detect_every, 1) == 0:
            td = time.perf_counter()
            infer_frame, sx, sy = self._prepare_inference_frame(frame, self.config.proc_width)
            detections = self.detector.detect_faces_with_scores(infer_frame)
            self._last_detections = self._scale_detections(detections, sx, sy)
            detect_latency_ms = (time.perf_counter() - td) * 1000.0

        boxes = [box for box, _ in self._last_detections]

        ta = time.perf_counter()
        anonymized = frame.copy()
        anonymize_faces(
            anonymized,
            boxes,
            mode=self.config.mode,
            blur_scale=self.config.blur_scale,
            blur_kernel=self.config.blur_kernel,
            pixel_block=self.config.pixel_block,
            padding_ratio=self.config.face_padding,
            obliterate_scale=self.config.obliterate_scale,
            scramble_block=self.config.scramble_block,
            palette_levels=self.config.palette_levels,
            scramble_seed=self.config.scramble_seed,
            head_ratio=self.config.head_ratio,
            silhouette_ratio=self.config.silhouette_ratio,
        )
        anonymize_latency_ms = (time.perf_counter() - ta) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "ok": True,
            "frame_index": self._frame_index,
            "boxes": boxes,
            "scores": [float(score) for _, score in self._last_detections],
            "stats": {
                "faces": len(boxes),
                "detect_latency_ms": round(detect_latency_ms, 3),
                "anonymize_latency_ms": round(anonymize_latency_ms, 3),
                "total_latency_ms": round(total_ms, 3),
            },
            "frame": anonymized,
        }
