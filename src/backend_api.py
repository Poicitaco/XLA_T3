from __future__ import annotations

import base64
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src.anonymizer_backend import AnonymizationConfig, FaceAnonymizationEngine, PRESETS
    from src.face_lock import FaceRegionLocker
except ImportError:
    from anonymizer_backend import AnonymizationConfig, FaceAnonymizationEngine, PRESETS
    from face_lock import FaceRegionLocker


app = FastAPI(title="Face Privacy Backend", version="0.1.0")


class ConfigOverride(BaseModel):
    mode: Optional[str] = None
    conf_threshold: Optional[float] = None
    imgsz: Optional[int] = None
    detect_every: Optional[int] = None
    proc_width: Optional[int] = None
    blur_scale: Optional[float] = None
    blur_kernel: Optional[int] = None
    pixel_block: Optional[int] = None
    face_padding: Optional[float] = None
    obliterate_scale: Optional[float] = None
    scramble_block: Optional[int] = None
    palette_levels: Optional[int] = None
    scramble_seed: Optional[int] = None
    head_ratio: Optional[float] = None
    silhouette_ratio: Optional[float] = None


class FrameRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    preset: str = Field(default="balanced")
    model_path: str = Field(default="models/yolov11n-face.pt")
    config: Optional[ConfigOverride] = None
    image_format: str = Field(default="jpg")
    image_quality: int = Field(default=90, ge=50, le=100)


class LockRequest(FrameRequest):
    lock_key: str
    lock_overlay: str = Field(default="rps")
    lock_head_ratio: float = Field(default=0.45)
    lock_pixel_block: int = Field(default=12)
    lock_noise_intensity: int = Field(default=70)
    lock_noise_mix: float = Field(default=0.9)
    rps_tile_size: int = Field(default=8)
    rps_rounds: int = Field(default=2)


class UnlockRequest(BaseModel):
    image_b64: str
    payloads: List[Dict[str, Any]]
    unlock_key: str
    image_format: str = Field(default="jpg")
    image_quality: int = Field(default=90, ge=50, le=100)


_ENGINE_CACHE: Dict[str, FaceAnonymizationEngine] = {}


def _decode_image(image_b64: str) -> np.ndarray:
    try:
        binary = base64.b64decode(image_b64)
        buffer = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as exc:  # pragma: no cover - API input guard
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Image decode failed.")
    return frame


def _encode_image(frame: np.ndarray, image_format: str, image_quality: int) -> str:
    ext = ".png" if image_format.lower() == "png" else ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, image_quality] if ext == ".jpg" else []
    ok, encoded = cv2.imencode(ext, frame, params)
    if not ok:
        raise HTTPException(status_code=500, detail="Image encode failed.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _resolve_config(preset: str, override: Optional[ConfigOverride]) -> AnonymizationConfig:
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")

    config = replace(PRESETS[preset])
    if override is None:
        return config

    updates = override.model_dump(exclude_none=True)
    for key, value in updates.items():
        setattr(config, key, value)
    return config


def _engine_cache_key(model_path: str, preset: str, config: AnonymizationConfig) -> str:
    return f"{model_path}|{preset}|{asdict(config)}"


def _get_engine(model_path: str, preset: str, override: Optional[ConfigOverride]) -> FaceAnonymizationEngine:
    config = _resolve_config(preset, override)
    cache_key = _engine_cache_key(model_path, preset, config)
    engine = _ENGINE_CACHE.get(cache_key)
    if engine is None:
        engine = FaceAnonymizationEngine(model_path=model_path, config=config)
        _ENGINE_CACHE[cache_key] = engine
    return engine


def _detect_boxes(engine: FaceAnonymizationEngine, frame: np.ndarray) -> List[List[int]]:
    infer_frame, sx, sy = engine._prepare_inference_frame(frame, engine.config.proc_width)
    detections = engine.detector.detect_faces_with_scores(infer_frame)
    scaled = engine._scale_detections(detections, sx, sy)
    return [box for box, _ in scaled]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "face-privacy-backend"}


@app.get("/presets")
def presets() -> Dict[str, Any]:
    return {"ok": True, "presets": {name: asdict(config) for name, config in PRESETS.items()}}


@app.post("/anonymize")
def anonymize(request: FrameRequest) -> Dict[str, Any]:
    frame = _decode_image(request.image_b64)
    engine = _get_engine(request.model_path, request.preset, request.config)
    result = engine.process_frame(frame)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)

    encoded = _encode_image(result["frame"], request.image_format, request.image_quality)
    return {
        "ok": True,
        "frame_b64": encoded,
        "frame_index": result["frame_index"],
        "boxes": result["boxes"],
        "scores": result["scores"],
        "stats": result["stats"],
    }


@app.post("/lock")
def lock(request: LockRequest) -> Dict[str, Any]:
    frame = _decode_image(request.image_b64)
    engine = _get_engine(request.model_path, request.preset, request.config)
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail={"error": "detector_not_ready", "model_hint": engine.detector.model_hint})

    boxes = _detect_boxes(engine, frame)
    locker = FaceRegionLocker(request.lock_key)
    locked_frame, payloads = locker.lock_faces(
        frame,
        boxes,
        overlay_mode=request.lock_overlay,
        overlay_pixel_block=request.lock_pixel_block,
        overlay_noise_intensity=request.lock_noise_intensity,
        overlay_noise_mix=request.lock_noise_mix,
        head_ratio=request.lock_head_ratio,
        rps_tile_size=request.rps_tile_size,
        rps_rounds=request.rps_rounds,
    )

    encoded = _encode_image(locked_frame, request.image_format, request.image_quality)
    return {
        "ok": True,
        "frame_b64": encoded,
        "boxes": boxes,
        "payloads": FaceRegionLocker.payloads_to_jsonable(payloads),
        "stats": {"faces": len(boxes)},
    }


@app.post("/unlock")
def unlock(request: UnlockRequest) -> Dict[str, Any]:
    frame = _decode_image(request.image_b64)
    locker = FaceRegionLocker(request.unlock_key)
    payloads = FaceRegionLocker.payloads_from_jsonable(request.payloads)

    try:
        restored = locker.unlock_faces(frame, payloads)
    except Exception as exc:  # pragma: no cover - wrong key/corrupt payload path
        raise HTTPException(status_code=400, detail=f"Unlock failed: {exc}") from exc

    encoded = _encode_image(restored, request.image_format, request.image_quality)
    return {
        "ok": True,
        "frame_b64": encoded,
        "restored_regions": len(payloads),
    }
