"""
Face Privacy Pipeline – Comprehensive REST + WebSocket API
===========================================================
Designed for frontend integration.

Start the server:
    uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --reload

All Endpoints
─────────────────────────────────────────────────────────────────────
Health / Info
  GET  /health                   Service health check
  GET  /info                     Version, capabilities, defaults
  GET  /presets                  Anonymization preset catalogue
  GET  /pipeline/modes           Available anonymize + selective modes

Stateless Image Processing
  POST /anonymize                Anonymize faces in a single frame
  POST /lock                     Encrypt (lock) faces in a frame
  POST /unlock                   Decrypt (unlock) faces in a frame

Recognition (stateless, per-image)
  POST /recognize                Detect + identify all faces in a frame

Cameras
  GET  /cameras                  List enumerated system cameras

Members (face database CRUD)
  GET    /members                List all registered members
  GET    /members/{member_id}    Get one member by ID
  POST   /members                Register / update member (multipart image upload)
  PATCH  /members/{member_id}    Rename or change matching threshold
  DELETE /members/{member_id}    Remove member from database

Live Pipeline (singleton)
  POST   /pipeline/start         Start background camera pipeline
  DELETE /pipeline/stop          Stop background pipeline
  GET    /pipeline/status        Running status + live stats
  PATCH  /pipeline/config        Hot-update settings without restart

Streaming
  GET  /stream/frame             Latest processed frame as base64 JSON (polling)
  GET  /stream/mjpeg             MJPEG stream  – use as <img src="/stream/mjpeg">
  GET  /stream/sse               Server-Sent Events – use with EventSource
  WS   /stream/ws                WebSocket stream (?format=binary|json)
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, replace as dc_replace
from datetime import datetime, timedelta, timezone
from queue import Empty
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from src.anonymizer_backend import AnonymizationConfig, FaceAnonymizationEngine, PRESETS
    from src.admin_auth import is_admin_setup, setup_admin, verify_admin, change_password as admin_change_password
    from src.clip_recorder import (
        ClipRecorder, decrypt_clip, delete_clip as delete_clip_files,
        frames_to_mp4_bytes, list_clips,
    )
    from src.detector import FaceDetector
    from src.face_lock import FaceRegionLocker
    from src.face_recognition import FaceRecognizer
    from src.family_database import FamilyDatabase
    from src.privacy import anonymize_faces
    from src.run_phone_cam import (
        FrameReader,
        RecognitionWorker,
        SimpleTracker,
        backend_name,
        get_enumerated_cameras,
        is_usable_frame,
        list_readable_cameras,
        open_capture,
        parse_backend,
        prepare_inference_frame,
        scale_boxes,
    )
except ImportError:
    from anonymizer_backend import AnonymizationConfig, FaceAnonymizationEngine, PRESETS
    from admin_auth import is_admin_setup, setup_admin, verify_admin, change_password as admin_change_password
    from clip_recorder import (
        ClipRecorder, decrypt_clip, delete_clip as delete_clip_files,
        frames_to_mp4_bytes, list_clips,
    )
    from detector import FaceDetector
    from face_lock import FaceRegionLocker
    from face_recognition import FaceRecognizer
    from family_database import FamilyDatabase
    from privacy import anonymize_faces
    from run_phone_cam import (
        FrameReader,
        RecognitionWorker,
        SimpleTracker,
        backend_name,
        get_enumerated_cameras,
        is_usable_frame,
        list_readable_cameras,
        open_capture,
        parse_backend,
        prepare_inference_frame,
        scale_boxes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# App + CORS
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Face Privacy Pipeline API",
    version="1.0.0",
    description=(
        "REST + WebSocket API for real-time face detection, recognition, and anonymization. "
        "Use /pipeline/start to launch the camera pipeline, then connect to /stream/mjpeg or "
        "/stream/ws for live frames."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


@app.get("/", include_in_schema=False)
def root_redirect() -> FileResponse:
    """Serve the dashboard at / for quick browser access."""
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/yolov11n-face.pt"
DEFAULT_DB_PATH = "data/family_members.json"
DEFAULT_RECOGNITION_MODEL = "buffalo_l"

ANONYMIZE_MODES = [
    "none", "blur", "neckup", "pixelate", "solid",
    "obliterate", "headcloak", "silhouette", "lock",
]
SELECTIVE_MODES = ["disabled", "family", "stranger", "registered_only"]


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic – request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ConfigOverride(BaseModel):
    """Partial override of an anonymization preset."""
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
    """Base request for single-frame processing endpoints."""
    image_b64: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    preset: str = Field(default="balanced", description="Preset name: fast | balanced | strong | strict")
    model_path: str = Field(default=DEFAULT_MODEL_PATH)
    config: Optional[ConfigOverride] = Field(default=None, description="Optional field-level overrides on top of the preset")
    image_format: str = Field(default="jpg", description="Output format: jpg | png")
    image_quality: int = Field(default=90, ge=50, le=100, description="JPEG output quality (50–100)")


class LockRequest(FrameRequest):
    """Request for the /lock endpoint (reversible face encryption)."""
    lock_key: str = Field(..., description="Passphrase used for AES encryption")
    lock_overlay: str = Field(default="rps", description="Overlay style after encryption: solid | noise | ciphernoise | rps")
    lock_head_ratio: float = Field(default=0.45)
    lock_pixel_block: int = Field(default=12)
    lock_noise_intensity: int = Field(default=70)
    lock_noise_mix: float = Field(default=0.9)
    rps_tile_size: int = Field(default=8)
    rps_rounds: int = Field(default=2)


class UnlockRequest(BaseModel):
    """Request for the /unlock endpoint."""
    image_b64: str = Field(..., description="Base64-encoded locked image")
    payloads: List[Dict[str, Any]] = Field(..., description="Encrypted payload list returned by /lock")
    unlock_key: str = Field(..., description="Passphrase used to decrypt")
    image_format: str = Field(default="jpg")
    image_quality: int = Field(default=90, ge=50, le=100)


class RecognizeRequest(BaseModel):
    """Request for the stateless /recognize endpoint."""
    image_b64: str = Field(..., description="Base64-encoded image")
    model_path: str = Field(default=DEFAULT_MODEL_PATH, description="YOLO face detector weights")
    face_db: str = Field(default=DEFAULT_DB_PATH, description="Path to family_members.json")
    recognition_model: str = Field(default=DEFAULT_RECOGNITION_MODEL, description="InsightFace model pack, e.g. buffalo_l")
    threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum cosine similarity to accept a match")
    conf_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Face detector confidence threshold")
    image_format: str = Field(default="jpg")
    image_quality: int = Field(default=90, ge=50, le=100)
    return_annotated_frame: bool = Field(default=True, description="Draw boxes + labels on the output frame")


class FaceResult(BaseModel):
    box: List[int] = Field(description="Bounding box [x, y, w, h]")
    detection_score: float = Field(description="YOLO detection confidence")
    member_id: Optional[str] = None
    name: Optional[str] = None
    similarity: float = Field(default=0.0, description="Cosine similarity to best matched embedding")


class RecognizeResponse(BaseModel):
    ok: bool = True
    faces: List[FaceResult]
    total_faces: int
    frame_b64: Optional[str] = Field(default=None, description="Annotated frame, present when return_annotated_frame=true")
    latency_ms: float = 0.0


class MemberInfo(BaseModel):
    member_id: str
    name: str
    threshold: float
    embedding_count: int


class MemberListResponse(BaseModel):
    ok: bool = True
    members: List[MemberInfo]
    total: int
    model_name: str
    embedding_dim: int


class MemberPatchRequest(BaseModel):
    """Update a member's display name or per-member match threshold."""
    name: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineStartRequest(BaseModel):
    """Configuration to start the live camera pipeline."""
    # Camera source (choose one)
    source_choice: Optional[int] = Field(default=None, description="Index from GET /cameras (recommended)")
    source: Optional[str] = Field(default=None, description="Camera index (as string) or stream URL")
    backend: str = Field(default="dshow", description="Windows camera backend: dshow | msmf | any")
    camera_width: int = Field(default=1280)
    camera_height: int = Field(default=720)
    camera_fps: int = Field(default=30)

    # Detector
    model_path: str = Field(default=DEFAULT_MODEL_PATH)
    conf_threshold: float = Field(default=0.35)
    imgsz: int = Field(default=512)
    detect_every: int = Field(default=4, description="Run detector every N frames")
    proc_width: int = Field(default=640, description="Downscale width before detection (0 = disabled)")

    # Anonymization
    anonymize_mode: str = Field(default="blur", description="One of: " + " | ".join(ANONYMIZE_MODES))
    blur_scale: float = Field(default=0.2)
    blur_kernel: int = Field(default=31)
    pixel_block: int = Field(default=16)
    face_padding: float = Field(default=0.12)
    head_ratio: float = Field(default=0.65)
    neck_ratio: float = Field(default=0.6)
    silhouette_ratio: float = Field(default=0.8)

    # Selective recognition
    selective_mode: str = Field(default="disabled", description="One of: " + " | ".join(SELECTIVE_MODES))
    face_db: str = Field(default=DEFAULT_DB_PATH)
    recognition_model: str = Field(default=DEFAULT_RECOGNITION_MODEL)
    recognition_threshold: float = Field(default=0.35)
    reid_min_interval: int = Field(default=12, description="Minimum frames between re-identification attempts")
    reid_max_interval: int = Field(default=90, description="Force re-identification after N frames regardless of movement")
    reid_center_shift_ratio: float = Field(default=0.14, description="Center shift ratio to trigger early re-ID")
    reid_size_change_ratio: float = Field(default=0.22, description="Bounding box size change ratio to trigger early re-ID")
    track_max_missed: int = Field(default=12, description="Drop a track after N consecutive missed detections")

    # Stream output
    stream_jpeg_quality: int = Field(default=75, ge=30, le=100, description="JPEG quality for streamed frames")
    stream_max_fps: float = Field(default=20.0, description="Cap streaming rate (frames per second)")

    # Clip recording
    enable_clip_recording: bool = Field(default=False, description="Save face-triggered 5-10s clips (blurred MP4 + encrypted raw)")
    admin_password: Optional[str] = Field(default=None, description="Admin password – required when enable_clip_recording=true")
    clip_duration_s: float = Field(default=8.0, ge=2.0, le=60.0, description="Duration of each recorded clip in seconds")
    clip_pre_buffer_s: float = Field(default=3.0, ge=0.0, le=15.0, description="Pre-trigger rolling buffer in seconds")
    clip_cooldown_s: float = Field(default=15.0, ge=5.0, le=300.0, description="Minimum gap between clips")


class PipelineConfigPatch(BaseModel):
    """Settings that can be hot-updated while the pipeline is running."""
    anonymize_mode: Optional[str] = None
    selective_mode: Optional[str] = None
    detect_every: Optional[int] = None
    proc_width: Optional[int] = None
    recognition_threshold: Optional[float] = None
    reid_min_interval: Optional[int] = None
    reid_max_interval: Optional[int] = None
    reid_center_shift_ratio: Optional[float] = None
    reid_size_change_ratio: Optional[float] = None
    blur_scale: Optional[float] = None
    blur_kernel: Optional[int] = None
    pixel_block: Optional[int] = None
    face_padding: Optional[float] = None
    head_ratio: Optional[float] = None
    neck_ratio: Optional[float] = None
    silhouette_ratio: Optional[float] = None
    stream_jpeg_quality: Optional[int] = Field(default=None, ge=30, le=100)
    stream_max_fps: Optional[float] = None


class PipelineStatusResponse(BaseModel):
    ok: bool = True
    running: bool
    uptime_s: float = 0.0
    frame_idx: int = 0
    fps: float = 0.0
    faces_detected: int = 0
    registered_count: int = 0
    unknown_count: int = 0
    anonymize_mode: str = "none"
    selective_mode: str = "disabled"
    source_info: str = ""
    has_frame: bool = False
    clip_recording: bool = False
    clips_saved: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Module-level lazy singletons (recognizer + DB)
# ─────────────────────────────────────────────────────────────────────────────

_singleton_lock = threading.Lock()
_recognizer_cache: Dict[str, FaceRecognizer] = {}
_db_cache: Dict[str, FamilyDatabase] = {}
_ENGINE_CACHE: Dict[str, FaceAnonymizationEngine] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Admin session tokens (in-memory, localhost/LAN only)
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_TTL = timedelta(hours=2)
_admin_tokens: Dict[str, datetime] = {}   # token → UTC expiry
_tokens_lock = threading.Lock()


def _issue_admin_token() -> tuple:
    """Create a new session token valid for TOKEN_TTL and return (token, expires_at)."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + _TOKEN_TTL
    with _tokens_lock:
        # Opportunistically purge expired tokens to keep the dict small
        now = datetime.now(timezone.utc)
        expired = [t for t, exp in list(_admin_tokens.items()) if exp <= now]
        for t in expired:
            del _admin_tokens[t]
        _admin_tokens[token] = expires_at
    return token, expires_at


def _verify_admin_token(token: str) -> bool:
    """Return True if the token exists and has not expired."""
    with _tokens_lock:
        exp = _admin_tokens.get(token)
        if exp is None:
            return False
        if datetime.now(timezone.utc) > exp:
            del _admin_tokens[token]
            return False
        return True


def _revoke_admin_token(token: str) -> None:
    """Invalidate a session token immediately."""
    with _tokens_lock:
        _admin_tokens.pop(token, None)


def _require_admin_auth(password: Optional[str], token: Optional[str]) -> None:
    """Raise HTTP 401 unless a valid session token or the admin password is provided."""
    if token and _verify_admin_token(token):
        return
    if password and verify_admin(password):
        return
    raise HTTPException(
        status_code=401,
        detail="Admin authentication required: provide a valid token or password.",
    )


def _get_recognizer(model: str) -> FaceRecognizer:
    with _singleton_lock:
        if model not in _recognizer_cache:
            _recognizer_cache[model] = FaceRecognizer(model=model)
        return _recognizer_cache[model]


def _get_db(db_path: str) -> FamilyDatabase:
    with _singleton_lock:
        if db_path not in _db_cache:
            _db_cache[db_path] = FamilyDatabase(db_path=db_path)
        return _db_cache[db_path]


def _reload_db(db_path: str) -> FamilyDatabase:
    """Force reload from disk and refresh the in-memory cache."""
    db = FamilyDatabase(db_path=db_path)
    with _singleton_lock:
        _db_cache[db_path] = db
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _LiveConfig:
    """Mutable pipeline settings, protected by PipelineSession.cfg_lock."""
    anonymize_mode: str = "blur"
    selective_mode: str = "disabled"
    detect_every: int = 4
    proc_width: int = 640
    recognition_threshold: float = 0.35
    reid_min_interval: int = 12
    reid_max_interval: int = 90
    reid_center_shift_ratio: float = 0.14
    reid_size_change_ratio: float = 0.22
    blur_scale: float = 0.2
    blur_kernel: int = 31
    pixel_block: int = 16
    face_padding: float = 0.12
    head_ratio: float = 0.65
    neck_ratio: float = 0.6
    silhouette_ratio: float = 0.8
    stream_jpeg_quality: int = 75
    stream_max_fps: float = 20.0


class _FrameBuffer:
    """Thread-safe store for the latest processed JPEG frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._seq: int = 0
        self._stats: Dict[str, Any] = {}

    def put(self, jpeg: bytes, stats: Dict[str, Any]) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1
            self._stats = stats

    def get(self, last_seq: int = -1) -> Tuple[Optional[bytes], int, Dict[str, Any]]:
        with self._lock:
            return self._jpeg, self._seq, dict(self._stats)

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._jpeg is not None


class PipelineSession:
    """Singleton background camera + detection + anonymization pipeline."""

    def __init__(self, req: PipelineStartRequest) -> None:
        self._req = req
        self.live_cfg = _LiveConfig(
            anonymize_mode=req.anonymize_mode,
            selective_mode=req.selective_mode,
            detect_every=req.detect_every,
            proc_width=req.proc_width,
            recognition_threshold=req.recognition_threshold,
            reid_min_interval=req.reid_min_interval,
            reid_max_interval=req.reid_max_interval,
            reid_center_shift_ratio=req.reid_center_shift_ratio,
            reid_size_change_ratio=req.reid_size_change_ratio,
            blur_scale=req.blur_scale,
            blur_kernel=req.blur_kernel,
            pixel_block=req.pixel_block,
            face_padding=req.face_padding,
            head_ratio=req.head_ratio,
            neck_ratio=req.neck_ratio,
            silhouette_ratio=req.silhouette_ratio,
            stream_jpeg_quality=req.stream_jpeg_quality,
            stream_max_fps=req.stream_max_fps,
        )
        self.cfg_lock = threading.Lock()
        self.frame_buffer = _FrameBuffer()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at: float = 0.0
        self.source_info: str = ""
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "frame_idx": 0,
            "fps": 0.0,
            "faces_detected": 0,
            "registered_count": 0,
            "unknown_count": 0,
        }
        # Face DB reference for hot-reload after member registration
        self._db: Optional[FamilyDatabase] = None
        self._db_lock = threading.Lock()
        # Clip recorder (None when not enabled)
        self._clip_recorder: Optional[ClipRecorder] = None
        if req.enable_clip_recording:
            if not req.admin_password:
                raise ValueError("admin_password is required when enable_clip_recording=true")
            if not is_admin_setup():
                raise ValueError("Admin is not set up yet. Call POST /admin/setup first.")
            if not verify_admin(req.admin_password):
                raise ValueError("Invalid admin password.")
            self._clip_recorder = ClipRecorder(
                admin_password=req.admin_password,
                fps=float(req.camera_fps),
                pre_buffer_s=req.clip_pre_buffer_s,
                clip_duration_s=req.clip_duration_s,
                cooldown_s=req.clip_cooldown_s,
            )

    def start(self) -> None:
        self._started_at = time.time()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10.0)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def patch_config(self, patch: PipelineConfigPatch) -> None:
        with self.cfg_lock:
            for key, value in patch.model_dump(exclude_none=True).items():
                if hasattr(self.live_cfg, key):
                    setattr(self.live_cfg, key, value)

    def reload_db(self) -> int:
        """Reload the face DB from disk so the running pipeline picks up new members."""
        with self._db_lock:
            if self._db is not None:
                self._db.load()
                print(f"[Pipeline] DB reloaded: {len(self._db.members)} member(s)")
                return len(self._db.members)
        return 0

    def get_db_path(self) -> str:
        return self._req.face_db

    def get_status(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        with self.cfg_lock:
            mode = self.live_cfg.anonymize_mode
            sel = self.live_cfg.selective_mode
        clips_saved = self._clip_recorder.clips_saved if self._clip_recorder else 0
        return {
            "running": self.is_alive,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            "frame_idx": stats["frame_idx"],
            "fps": stats["fps"],
            "faces_detected": stats["faces_detected"],
            "registered_count": stats["registered_count"],
            "unknown_count": stats["unknown_count"],
            "anonymize_mode": mode,
            "selective_mode": sel,
            "source_info": self.source_info,
            "has_frame": self.frame_buffer.has_data,
            "clip_recording": self._clip_recorder is not None,
            "clips_saved": clips_saved,
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _snapshot_cfg(self) -> _LiveConfig:
        with self.cfg_lock:
            return dc_replace(self.live_cfg)

    def _write_stats(self, **kwargs: Any) -> None:
        with self._stats_lock:
            self._stats.update(kwargs)

    # ── background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        req = self._req

        # Resolve camera source
        backend_flag = parse_backend(req.backend)
        enumerated_cameras = get_enumerated_cameras()

        if req.source_choice is not None:
            # Use enumerated cameras (faster, same as CLI)
            if not enumerated_cameras:
                print(f"[Pipeline] enumerated cameras unavailable, falling back to source 0")
                source = 0
                backend_flag = parse_backend(req.backend)
            elif req.source_choice < 0 or req.source_choice >= len(enumerated_cameras):
                print(f"[Pipeline] Invalid source_choice {req.source_choice}, falling back to source 0")
                source = 0
                backend_flag = parse_backend(req.backend)
            else:
                # Use enumerated camera (same as CLI choose_named_camera function)
                camera = enumerated_cameras[req.source_choice]
                source = int(camera.index)
                backend_flag = int(camera.backend) if int(camera.backend) != 0 else cv2.CAP_ANY
                print(f"[Pipeline] Using named camera choice {req.source_choice}: source={source} backend={backend_name(backend_flag)}")
            
            self.source_info = f"choice={req.source_choice} source={source} backend={backend_name(backend_flag)}"
        elif req.source is not None:
            source = int(req.source) if req.source.isdigit() else req.source
            self.source_info = f"source={req.source} backend={backend_name(backend_flag)}"
        else:
            source = 0   # first available
            self.source_info = "auto:0"

        cap = open_capture(source, backend_flag, req.camera_width, req.camera_height, req.camera_fps)
        if not cap.isOpened():
            return

        detector = FaceDetector(
            model_path=req.model_path,
            conf_threshold=req.conf_threshold,
            imgsz=req.imgsz,
        )
        if not detector.is_ready:
            cap.release()
            return

        # Always initialize recognition so selective_mode can be changed via hot-patch
        tracker: Optional[SimpleTracker] = None
        recognition_worker: Optional[RecognitionWorker] = None

        try:
            _db = FamilyDatabase(db_path=req.face_db)
            if _db.members:
                with self._db_lock:
                    self._db = _db
                _rec = FaceRecognizer(model=req.recognition_model)
                tracker = SimpleTracker(max_missed=req.track_max_missed)
                recognition_worker = RecognitionWorker(
                    recognizer=_rec,
                    database=_db,
                    threshold=req.recognition_threshold,
                )
                recognition_worker.start()
                print(f"[Pipeline] Recognition ready: {len(_db.members)} member(s), model={req.recognition_model}")
            else:
                print(f"[Pipeline] DB empty ({req.face_db}), recognition will activate after member registration")
        except Exception as _exc:
            import traceback as _tb
            print(f"[Pipeline] Recognition init error: {_exc}")
            _tb.print_exc()
            tracker = None
            recognition_worker = None

        # Skip camera warmup -let FrameReader handle it asynchronously
        print(f"[Pipeline] Starting FrameReader thread...")
        
        frame_reader = FrameReader(cap)
        frame_reader.start()
        
        # Give FrameReader 2 seconds to start producing frames
        print(f"[Pipeline] Waiting for first frame...")
        warmup_wait = 0
        while warmup_wait < 20:  # 2 seconds max (20 * 0.1s)
            if frame_reader.frames_read > 0:
                print(f"[Pipeline] First frame received after {warmup_wait * 0.1:.1f}s")
                break
            time.sleep(0.1)
            warmup_wait += 1
        
        if frame_reader.frames_read == 0:
            print(f"[Pipeline] Warning: No frames after 2s, but continuing anyway")
        
        print(f"[Pipeline] Pipeline thread started successfully")

        frame_idx = 0
        last_boxes: List[List[int]] = []
        last_detections: List[Tuple[List[int], float]] = []
        fps_ema = 0.0
        t_prev = time.perf_counter()
        
        # Camera reconnection tracking
        consecutive_failures = 0
        max_consecutive_failures = 30  # Reduced from 50 (faster reconnection)
        last_successful_read = time.time()
        total_failures = 0

        try:
            while not self._stop_event.is_set():
                ok, frame = frame_reader.read(timeout=1.0)  # Increased to 1.0s for virtual cameras
                if not ok or frame is None:
                    consecutive_failures += 1
                    total_failures += 1
                    
                    # Debug logging
                    if consecutive_failures == 1:
                        print(f"[Pipeline] First frame read failure (total failures: {total_failures})")
                    elif consecutive_failures % 10 == 0:
                        print(f"[Pipeline] {consecutive_failures} consecutive failures")
                    
                    # Camera reconnection logic
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"[Pipeline] Camera disconnected after {consecutive_failures} failures. Attempting reconnection...")
                        frame_reader.stop()
                        cap.release()
                        time.sleep(1.0)  # Longer pause before reconnection
                        
                        # Reconnect camera
                        cap = open_capture(source, backend_flag, req.camera_width, req.camera_height, req.camera_fps)
                        if not cap.isOpened():
                            print(f"[Pipeline] Camera reconnection failed. Stopping pipeline.")
                            break
                        
                        # Warmup after reconnect
                        warmup_ok = False
                        for _ in range(20):
                            ok_warm, _ = cap.read()
                            if ok_warm:
                                warmup_ok = True
                                break
                        
                        if not warmup_ok:
                            print(f"[Pipeline] Camera warmup after reconnect failed.")
                            cap.release()
                            break
                        
                        frame_reader = FrameReader(cap)
                        frame_reader.start()
                        consecutive_failures = 0
                        print(f"[Pipeline] Camera reconnected successfully.")
                    
                    time.sleep(0.01)  # Brief sleep to avoid busy loop
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                last_successful_read = time.time()

                cfg = self._snapshot_cfg()
                detect_every = max(1, cfg.detect_every)
                frame_idx += 1

                # ── Detection ────────────────────────────────────────────────
                if frame_idx % detect_every == 0:
                    infer, sx, sy = prepare_inference_frame(frame, cfg.proc_width)
                    last_detections = scale_boxes(
                        detector.detect_faces_with_scores(infer), sx, sy
                    )
                    if cfg.selective_mode != "disabled" and tracker is not None:
                        tracker.update([box for box, _ in last_detections], frame_idx)
                    else:
                        last_boxes = [box for box, _ in last_detections]

                # ── Recognition submit + drain ────────────────────────────────
                reg_count = 0
                unk_count = 0
                if cfg.selective_mode != "disabled" and tracker is not None and recognition_worker is not None:
                    for _tid, state in tracker.tracks.items():
                        if tracker.should_reidentify(
                            state=state,
                            frame_idx=frame_idx,
                            min_interval=max(1, cfg.reid_min_interval),
                            max_interval=max(1, cfg.reid_max_interval),
                            center_shift_ratio=max(0.01, cfg.reid_center_shift_ratio),
                            size_change_ratio=max(0.01, cfg.reid_size_change_ratio),
                        ):
                            recognition_worker.submit(_tid, frame, state["bbox"])
                            tracker.mark_reid_submitted(state, frame_idx)

                    while True:
                        result = recognition_worker.get_result_nowait()
                        if result is None:
                            break
                        tracker.apply_recognition(
                            track_id=int(result["track_id"]),
                            member_id=result["member_id"],
                            name=str(result["name"]),
                            score=float(result["score"]),
                        )

                    last_boxes = []
                    for _tid, state in tracker.tracks.items():
                        identity = str(state.get("identity", "pending"))
                        is_reg = identity not in ("pending", "unknown")
                        should_anon = (
                            (cfg.selective_mode == "family" and is_reg)
                            or (cfg.selective_mode == "stranger" and not is_reg)
                            or (cfg.selective_mode == "registered_only" and is_reg)
                        )
                        if should_anon:
                            last_boxes.append(state["bbox"])
                        reg_count += int(is_reg)
                        unk_count += int(not is_reg)

                    self._write_stats(registered_count=reg_count, unknown_count=unk_count)

                # ── Anonymize ────────────────────────────────────────────────
                display = frame.copy()
                if last_boxes and cfg.anonymize_mode != "none":
                    anonymize_faces(
                        display,
                        last_boxes,
                        mode=cfg.anonymize_mode,
                        blur_scale=cfg.blur_scale,
                        blur_kernel=cfg.blur_kernel,
                        pixel_block=cfg.pixel_block,
                        padding_ratio=cfg.face_padding,
                        head_ratio=cfg.head_ratio,
                        silhouette_ratio=cfg.silhouette_ratio,
                        neck_ratio=cfg.neck_ratio,
                    )

                # ── Clip recording ───────────────────────────────────────────
                if self._clip_recorder is not None:
                    face_detected_this_frame = len(last_boxes) > 0 or len(last_detections) > 0
                    self._clip_recorder.push_frame(
                        raw_frame=frame,
                        display_frame=display,
                        face_detected=face_detected_this_frame,
                    )

                # ── FPS ──────────────────────────────────────────────────────
                t_now = time.perf_counter()
                fps_inst = 1.0 / max(t_now - t_prev, 1e-9)
                t_prev = t_now
                fps_ema = 0.9 * fps_ema + 0.1 * fps_inst if fps_ema > 0 else fps_inst

                # ── Encode to JPEG → frame buffer ────────────────────────────
                quality = max(30, min(100, cfg.stream_jpeg_quality))
                ok_enc, jpeg_buf = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if ok_enc:
                    self.frame_buffer.put(
                        jpeg_buf.tobytes(),
                        {
                            "frame_idx": frame_idx,
                            "fps": round(fps_ema, 1),
                            "faces_detected": len(last_boxes),
                        },
                    )
                    self._write_stats(
                        frame_idx=frame_idx,
                        fps=round(fps_ema, 1),
                        faces_detected=len(last_boxes),
                    )

                # ── Stream rate cap ──────────────────────────────────────────
                min_period = 1.0 / max(0.5, cfg.stream_max_fps)
                elapsed = time.perf_counter() - t_now
                if elapsed < min_period:
                    time.sleep(min_period - elapsed)

        finally:
            frame_reader.stop()
            cap.release()
            if recognition_worker is not None:
                recognition_worker.stop()


# Module-level singleton
_pipeline_lock = threading.Lock()
_pipeline_session: Optional[PipelineSession] = None


# ─────────────────────────────────────────────────────────────────────────────
# Shared image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image(image_b64: str) -> np.ndarray:
    try:
        binary = base64.b64decode(image_b64)
        buf = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Image decode failed.")
    return frame


def _encode_image(frame: np.ndarray, fmt: str, quality: int) -> str:
    ext = ".png" if fmt.lower() == "png" else ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, encoded = cv2.imencode(ext, frame, params)
    if not ok:
        raise HTTPException(status_code=500, detail="Image encode failed.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _resolve_config(preset: str, override: Optional[ConfigOverride]) -> AnonymizationConfig:
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset}'. Valid: {list(PRESETS)}")
    cfg = dc_replace(PRESETS[preset])
    if override:
        for key, val in override.model_dump(exclude_none=True).items():
            setattr(cfg, key, val)
    return cfg


def _get_engine(model_path: str, preset: str, override: Optional[ConfigOverride]) -> FaceAnonymizationEngine:
    cfg = _resolve_config(preset, override)
    key = f"{model_path}|{preset}|{asdict(cfg)}"
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        engine = FaceAnonymizationEngine(model_path=model_path, config=cfg)
        _ENGINE_CACHE[key] = engine
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Health / Info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Info"], summary="Service health check")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "face-privacy-pipeline", "version": "1.0.0"}


@app.get("/info", tags=["Info"], summary="Service capabilities and defaults")
def info() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": "1.0.0",
        "anonymize_modes": ANONYMIZE_MODES,
        "selective_modes": SELECTIVE_MODES,
        "default_model": DEFAULT_MODEL_PATH,
        "default_db": DEFAULT_DB_PATH,
        "default_recognition_model": DEFAULT_RECOGNITION_MODEL,
        "presets": list(PRESETS.keys()),
    }


@app.get("/presets", tags=["Info"], summary="List all anonymization presets with full configs")
def get_presets() -> Dict[str, Any]:
    return {"ok": True, "presets": {name: asdict(cfg) for name, cfg in PRESETS.items()}}


@app.get("/pipeline/modes", tags=["Info"], summary="Enumerate valid mode strings")
def pipeline_modes() -> Dict[str, Any]:
    return {
        "ok": True,
        "anonymize_modes": ANONYMIZE_MODES,
        "selective_modes": SELECTIVE_MODES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Stateless image processing
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/anonymize",
    tags=["Image Processing"],
    summary="Detect and anonymize faces in a single uploaded frame",
)
def anonymize(request: FrameRequest) -> Dict[str, Any]:
    """
    Detect all faces in the image and apply the selected anonymization mode.
    Returns the processed frame (base64) along with detected bounding boxes and stats.
    """
    frame = _decode_image(request.image_b64)
    engine = _get_engine(request.model_path, request.preset, request.config)
    result = engine.process_frame(frame)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return {
        "ok": True,
        "frame_b64": _encode_image(result["frame"], request.image_format, request.image_quality),
        "frame_index": result["frame_index"],
        "boxes": result["boxes"],
        "scores": result["scores"],
        "stats": result["stats"],
    }


@app.post(
    "/lock",
    tags=["Image Processing"],
    summary="Encrypt face regions with AES-GCM (reversible lock)",
)
def lock(request: LockRequest) -> Dict[str, Any]:
    """
    Detect faces and AES-GCM encrypt each face ROI. The returned `payloads` list
    must be stored by the client and passed to /unlock to restore the original faces.
    """
    frame = _decode_image(request.image_b64)
    engine = _get_engine(request.model_path, request.preset, request.config)
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="Detector not ready.")
    infer, sx, sy = engine._prepare_inference_frame(frame, engine.config.proc_width)
    dets = engine.detector.detect_faces_with_scores(infer)
    boxes = [box for box, _ in engine._scale_detections(dets, sx, sy)]

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
    return {
        "ok": True,
        "frame_b64": _encode_image(locked_frame, request.image_format, request.image_quality),
        "boxes": boxes,
        "payloads": FaceRegionLocker.payloads_to_jsonable(payloads),
        "stats": {"faces_locked": len(boxes)},
    }


@app.post(
    "/unlock",
    tags=["Image Processing"],
    summary="Decrypt previously locked face regions",
)
def unlock(request: UnlockRequest) -> Dict[str, Any]:
    """
    Restore original face ROIs from an AES-GCM locked frame.
    Requires the same key used during /lock and the payloads that were returned.
    """
    frame = _decode_image(request.image_b64)
    locker = FaceRegionLocker(request.unlock_key)
    payloads = FaceRegionLocker.payloads_from_jsonable(request.payloads)
    try:
        restored = locker.unlock_faces(frame, payloads)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unlock failed: {exc}") from exc
    return {
        "ok": True,
        "frame_b64": _encode_image(restored, request.image_format, request.image_quality),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Recognition (stateless, per-image)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/recognize",
    response_model=RecognizeResponse,
    tags=["Recognition"],
    summary="Detect and identify all faces in a single image",
)
def recognize(request: RecognizeRequest) -> RecognizeResponse:
    """
    Detect faces with YOLO, extract InsightFace embeddings, and match against the
    registered members database. Returns per-face identity + similarity score, and
    optionally the frame with annotations drawn.
    """
    t0 = time.perf_counter()
    frame = _decode_image(request.image_b64)

    # Detect faces
    engine = _get_engine(
        request.model_path,
        "balanced",
        ConfigOverride(conf_threshold=request.conf_threshold),
    )
    infer, sx, sy = engine._prepare_inference_frame(frame, 640)
    dets = engine.detector.detect_faces_with_scores(infer)
    scaled = engine._scale_detections(dets, sx, sy)
    boxes = [box for box, _ in scaled]
    det_scores = [float(s) for _, s in scaled]

    if not boxes:
        return RecognizeResponse(
            faces=[],
            total_faces=0,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    recognizer = _get_recognizer(request.recognition_model)
    db = _get_db(request.face_db)
    embeddings = recognizer.extract_embeddings_from_frame(frame, boxes)

    faces: List[FaceResult] = []
    annotated = frame.copy() if request.return_annotated_frame else frame

    for i, (box, det_score) in enumerate(zip(boxes, det_scores)):
        emb = embeddings[i] if i < len(embeddings) else None
        member_id: Optional[str] = None
        name: Optional[str] = None
        similarity = 0.0

        if emb is not None:
            match = db.match(emb, threshold=request.threshold)
            if match is not None:
                member_id, name, similarity = match

        faces.append(FaceResult(
            box=box,
            detection_score=round(det_score, 4),
            member_id=member_id,
            name=name,
            similarity=round(float(similarity), 4),
        ))

        if request.return_annotated_frame:
            x, y, w, h = box
            color = (0, 220, 0) if member_id else (30, 180, 255)
            label = f"{name} {similarity:.2f}" if name else "unknown"
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated, label, (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

    frame_b64 = (
        _encode_image(annotated, request.image_format, request.image_quality)
        if request.return_annotated_frame
        else None
    )

    return RecognizeResponse(
        faces=faces,
        total_faces=len(faces),
        frame_b64=frame_b64,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Cameras
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/cameras",
    tags=["Cameras"],
    summary="List all enumerated system cameras",
)
def list_cameras() -> Dict[str, Any]:
    """
    Returns the list of detected cameras with their name, `choice` index
    (to use with /pipeline/start's `source_choice`), and raw OpenCV index.

    Uses get_enumerated_cameras() to match CLI behavior - this will find
    cameras like Camo with index 1400, not just 0-9.
    """
    enumerated = get_enumerated_cameras()
    
    if not enumerated:
        # Fallback: scan index 0-9 if enumeration fails
        print("[API] enumerate_cameras unavailable, falling back to manual scan")
        dshow_backend = parse_backend("dshow")
        msmf_backend  = parse_backend("msmf")

        dshow_cams = list_readable_cameras(max_index=9, backend=dshow_backend)
        msmf_cams  = list_readable_cameras(max_index=9, backend=msmf_backend)

        # Merge: keep DSHOW entries and fill in any indices only found via MSMF
        seen: set[int] = {int(c["index"]) for c in dshow_cams}
        merged = list(dshow_cams)
        for c in msmf_cams:
            if int(c["index"]) not in seen:
                merged.append(c)
                seen.add(int(c["index"]))

        # Sort by OpenCV index so the order is stable
        merged.sort(key=lambda c: int(c["index"]))

        return {
            "ok": True,
            "cameras": [
                {
                    "choice": i,
                    "name": f"Camera {cam['index']}",
                    "index": int(cam["index"]),
                    "backend": backend_name(int(cam["backend"])),
                    "resolution": f"{cam['width']}x{cam['height']}",
                }
                for i, cam in enumerate(merged)
            ],
            "total": len(merged),
        }
    
    # Use enumerated cameras (matches CLI behavior)
    return {
        "ok": True,
        "cameras": [
            {
                "choice": i,
                "name": cam.name,
                "index": int(cam.index),
                "backend": backend_name(int(cam.backend)),
                "resolution": "unknown",  # enumerate_cameras doesn't provide resolution
            }
            for i, cam in enumerate(enumerated)
        ],
        "total": len(enumerated),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Members
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/members",
    response_model=MemberListResponse,
    tags=["Members"],
    summary="List all registered members in the face database",
)
def list_members(db_path: str = DEFAULT_DB_PATH) -> MemberListResponse:
    db = _reload_db(db_path)
    return MemberListResponse(
        members=[
            MemberInfo(
                member_id=m.member_id,
                name=m.name,
                threshold=m.threshold,
                embedding_count=len(m.embeddings),
            )
            for m in db.members
        ],
        total=len(db.members),
        model_name=db.model_name,
        embedding_dim=db.embedding_dim,
    )


@app.get(
    "/members/{member_id}",
    tags=["Members"],
    summary="Get a single member by ID",
)
def get_member(member_id: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    db = _get_db(db_path)
    member = db.get_member_by_id(member_id)
    if member is None:
        raise HTTPException(status_code=404, detail=f"Member '{member_id}' not found.")
    return {
        "ok": True,
        "member": {
            "member_id": member.member_id,
            "name": member.name,
            "threshold": member.threshold,
            "embedding_count": len(member.embeddings),
        },
    }


@app.post(
    "/members",
    tags=["Members"],
    summary="Register a new member or update an existing one from uploaded images",
)
async def register_member(
    member_id: str = Form(..., description="Stable unique identifier, e.g. 'khoa'"),
    name: str = Form(..., description="Display name, e.g. 'Khoa'"),
    threshold: Optional[float] = Form(default=None, description="Per-member cosine threshold (default 0.35)"),
    recognition_model: str = Form(default=DEFAULT_RECOGNITION_MODEL),
    max_samples: int = Form(default=25, description="Maximum number of valid embeddings to keep"),
    db_path: str = Form(default=DEFAULT_DB_PATH),
    images: List[UploadFile] = File(..., description="One or more face images (JPEG/PNG)"),
) -> Dict[str, Any]:
    """
    Upload face images and register the member in the database.
    If the member already exists, their embeddings are replaced.
    Accepts `multipart/form-data` with both fields and file uploads.
    """
    recognizer = _get_recognizer(recognition_model)
    db = _reload_db(db_path)

    valid_embeddings: List[np.ndarray] = []
    failed = 0

    for upload in images:
        if len(valid_embeddings) >= max_samples:
            break
        content = await upload.read()
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            failed += 1
            continue
        emb = recognizer.extract_embedding_from_image(img)
        if emb is None:
            failed += 1
            continue
        valid_embeddings.append(emb)

    if not valid_embeddings:
        raise HTTPException(
            status_code=422,
            detail="No valid face embedding could be extracted. Ensure images contain a clear, visible face.",
        )

    db.model_name = recognition_model
    db.embedding_dim = int(valid_embeddings[0].shape[0])
    db.upsert_member(member_id, name, valid_embeddings, threshold=threshold)
    db.save()
    _reload_db(db_path)   # refresh shared API cache

    # Hot-reload running pipeline so it recognizes the new member immediately
    with _pipeline_lock:
        _sess = _pipeline_session
    if _sess is not None and _sess.is_alive and _sess.get_db_path() == db_path:
        _sess.reload_db()

    return {
        "ok": True,
        "member_id": member_id,
        "name": name,
        "embeddings_registered": len(valid_embeddings),
        "images_failed": failed,
        "threshold": float(threshold) if threshold is not None else 0.35,
        "model": recognition_model,
        "db_path": db_path,
    }


@app.patch(
    "/members/{member_id}",
    tags=["Members"],
    summary="Rename a member or change their match threshold",
)
def update_member(
    member_id: str,
    payload: MemberPatchRequest,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    db = _reload_db(db_path)
    member = db.get_member_by_id(member_id)
    if member is None:
        raise HTTPException(status_code=404, detail=f"Member '{member_id}' not found.")
    if payload.name is not None:
        member.name = payload.name
    if payload.threshold is not None:
        member.threshold = float(payload.threshold)
    db._rebuild_index()
    db.save()
    _reload_db(db_path)
    with _pipeline_lock:
        _sess = _pipeline_session
    if _sess is not None and _sess.is_alive and _sess.get_db_path() == db_path:
        _sess.reload_db()
    return {
        "ok": True,
        "member_id": member_id,
        "name": member.name,
        "threshold": member.threshold,
    }


@app.delete(
    "/members/{member_id}",
    tags=["Members"],
    summary="Remove a member from the database",
)
def delete_member(member_id: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    db = _reload_db(db_path)
    if db.get_member_by_id(member_id) is None:
        raise HTTPException(status_code=404, detail=f"Member '{member_id}' not found.")
    db.members = [m for m in db.members if m.member_id != member_id]
    db._rebuild_index()
    db.save()
    _reload_db(db_path)
    with _pipeline_lock:
        _sess = _pipeline_session
    if _sess is not None and _sess.is_alive and _sess.get_db_path() == db_path:
        _sess.reload_db()
    return {"ok": True, "deleted": member_id}


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Live Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/pipeline/start",
    tags=["Pipeline"],
    summary="Start the background camera pipeline",
)
def pipeline_start(req: PipelineStartRequest) -> Dict[str, Any]:
    """
    Launches a background thread that reads the camera, detects and optionally
    recognizes faces, applies anonymization, and stores the latest JPEG frame
    in a shared buffer (available via /stream/* endpoints).

    Only one pipeline can run at a time. Call /pipeline/stop first if one is already active.
    """
    global _pipeline_session
    with _pipeline_lock:
        if _pipeline_session is not None and _pipeline_session.is_alive:
            raise HTTPException(
                status_code=409,
                detail="A pipeline is already running. Call DELETE /pipeline/stop first.",
            )
        session = PipelineSession(req)
        session.start()
        _pipeline_session = session
    return {"ok": True, "message": "Pipeline started."}


@app.delete(
    "/pipeline/stop",
    tags=["Pipeline"],
    summary="Stop the running pipeline and release the camera",
)
def pipeline_stop() -> Dict[str, Any]:
    global _pipeline_session
    with _pipeline_lock:
        if _pipeline_session is None or not _pipeline_session.is_alive:
            raise HTTPException(status_code=404, detail="No active pipeline.")
        _pipeline_session.stop()
        _pipeline_session = None
    return {"ok": True, "message": "Pipeline stopped."}


@app.get(
    "/pipeline/status",
    response_model=PipelineStatusResponse,
    tags=["Pipeline"],
    summary="Check pipeline running status and live stats",
)
def pipeline_status() -> PipelineStatusResponse:
    with _pipeline_lock:
        session = _pipeline_session
    if session is None:
        return PipelineStatusResponse(ok=True, running=False)
    s = session.get_status()
    return PipelineStatusResponse(ok=True, **s)


@app.patch(
    "/pipeline/config",
    tags=["Pipeline"],
    summary="Hot-update pipeline settings without restarting",
)
def pipeline_update_config(patch: PipelineConfigPatch) -> Dict[str, Any]:
    """
    Update anonymization mode, selective mode, detection cadence, and recognition
    parameters while the pipeline is running. Changes take effect on the next frame cycle.

    Note: Changing the camera source, YOLO model, or InsightFace model requires a full restart.
    """
    with _pipeline_lock:
        session = _pipeline_session
    if session is None or not session.is_alive:
        raise HTTPException(status_code=404, detail="No active pipeline.")
    session.patch_config(patch)
    applied = patch.model_dump(exclude_none=True)
    return {"ok": True, "message": "Config updated.", "applied": applied}


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Streaming
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/stream/frame",
    tags=["Streaming"],
    summary="Fetch the latest processed frame as base64 JSON (polling)",
)
def stream_frame() -> Dict[str, Any]:
    """
    Returns the most recent processed frame from the pipeline as a base64-encoded JPEG
    along with a sequence number and live stats. Suitable for simple polling clients.

    The `seq` field can be used to detect if a new frame is available since the last poll.
    """
    with _pipeline_lock:
        session = _pipeline_session
    if session is None or not session.is_alive:
        raise HTTPException(status_code=404, detail="Pipeline is not running.")
    jpeg, seq, stats = session.frame_buffer.get()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="No frame available yet. Try again shortly.")
    return {
        "ok": True,
        "frame_b64": base64.b64encode(jpeg).decode("ascii"),
        "seq": seq,
        "stats": stats,
    }


@app.get(
    "/stream/mjpeg",
    tags=["Streaming"],
    summary="MJPEG stream — use directly as <img src=/stream/mjpeg> in the browser",
)
async def stream_mjpeg() -> StreamingResponse:
    """
    Serves a `multipart/x-mixed-replace` MJPEG stream. The simplest way to display
    the live camera feed in a browser:

        <img src="http://localhost:8000/stream/mjpeg" />
    """
    with _pipeline_lock:
        session = _pipeline_session
    if session is None or not session.is_alive:
        raise HTTPException(status_code=404, detail="Pipeline is not running.")

    async def _generate() -> AsyncGenerator[bytes, None]:
        last_seq = -1
        while True:
            with _pipeline_lock:
                s = _pipeline_session
            if s is None or not s.is_alive:
                break
            jpeg, seq, _ = s.frame_buffer.get()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.04)   # max ~25 fps polling

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get(
    "/stream/sse",
    tags=["Streaming"],
    summary="Server-Sent Events stream with frame + stats",
)
async def stream_sse() -> StreamingResponse:
    """
    Streams processed frames via Server-Sent Events. Each event's `data` field is a
    JSON object with `seq`, `frame_b64` (base64 JPEG), and `stats`.

    JavaScript usage:
        const es = new EventSource('/stream/sse');
        es.onmessage = e => {
            const { frame_b64, stats } = JSON.parse(e.data);
            document.getElementById('feed').src = 'data:image/jpeg;base64,' + frame_b64;
        };
    """
    with _pipeline_lock:
        session = _pipeline_session
    if session is None or not session.is_alive:
        raise HTTPException(status_code=404, detail="Pipeline is not running.")

    async def _generate() -> AsyncGenerator[str, None]:
        last_seq = -1
        while True:
            with _pipeline_lock:
                s = _pipeline_session
            if s is None or not s.is_alive:
                yield "event: stopped\ndata: {}\n\n"
                break
            jpeg, seq, stats = s.frame_buffer.get()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                payload = json.dumps({
                    "seq": seq,
                    "frame_b64": base64.b64encode(jpeg).decode("ascii"),
                    "stats": stats,
                })
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.websocket("/stream/ws")
async def stream_ws(websocket: WebSocket, format: str = "binary") -> None:
    """
    WebSocket stream of processed frames.

    Query params:
      - `format=binary` (default) — sends raw JPEG bytes per message
      - `format=json` — sends JSON `{"seq":…, "frame_b64":…, "stats":{…}}`

    The client can send any text message to receive a one-time status JSON reply:
        { "running": true, "fps": 28.3, "faces_detected": 2, ... }

    Connects: ws://localhost:8000/stream/ws?format=json
    """
    await websocket.accept()
    with _pipeline_lock:
        session = _pipeline_session
    if session is None or not session.is_alive:
        await websocket.close(code=1011, reason="Pipeline is not running.")
        return

    last_seq = -1
    try:
        while True:
            # Non-blocking check for client messages (status queries)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                status = session.get_status()
                await websocket.send_text(json.dumps(status))
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            with _pipeline_lock:
                s = _pipeline_session
            if s is None or not s.is_alive:
                await websocket.send_text(json.dumps({"event": "stopped"}))
                break

            jpeg, seq, stats = s.frame_buffer.get()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                if format == "binary":
                    await websocket.send_bytes(jpeg)
                else:
                    await websocket.send_text(json.dumps({
                        "seq": seq,
                        "frame_b64": base64.b64encode(jpeg).decode("ascii"),
                        "stats": stats,
                    }))
            await asyncio.sleep(0.04)

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Admin management
# ─────────────────────────────────────────────────────────────────────────────

class AdminSetupRequest(BaseModel):
    password: str = Field(..., min_length=8, description="Admin password (min 8 chars)")


class AdminVerifyRequest(BaseModel):
    password: str


class AdminAuthRequest(BaseModel):
    """Used by endpoints that require admin privileges but do NOT need the plaintext password.
    Accepts either a session token (issued by POST /admin/verify) or the password directly.
    """
    password: Optional[str] = Field(default=None, description="Admin password (fallback when no token)")
    token: Optional[str] = Field(default=None, description="Session token from POST /admin/verify")


class AdminLogoutRequest(BaseModel):
    token: str = Field(..., description="Session token to invalidate")


class AdminChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=8)


@app.post("/admin/setup", tags=["Admin"], summary="First-time admin password setup")
def admin_setup(req: AdminSetupRequest) -> Dict[str, Any]:
    """
    Set up the admin password for the first time.
    Only succeeds once — use /admin/change-password to update afterwards.
    """
    try:
        setup_admin(req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "message": "Admin set up successfully."}


@app.get("/admin/status", tags=["Admin"], summary="Check whether admin is configured")
def admin_status() -> Dict[str, Any]:
    return {"ok": True, "admin_setup": is_admin_setup()}


@app.post("/admin/verify", tags=["Admin"], summary="Verify admin password and receive a session token")
def admin_verify(req: AdminVerifyRequest) -> Dict[str, Any]:
    """
    Verify the admin password. On success returns a **session token** valid for 2 hours.
    Store this token client-side (sessionStorage) and send it in subsequent admin requests
    via the `token` field instead of the password.
    """
    if not verify_admin(req.password):
        raise HTTPException(status_code=401, detail="Invalid admin password.")
    token, expires_at = _issue_admin_token()
    return {
        "ok": True,
        "token": token,
        "expires_at": expires_at.isoformat(),
        "ttl_hours": _TOKEN_TTL.total_seconds() / 3600,
    }


@app.post("/admin/logout", tags=["Admin"], summary="Invalidate a session token")
def admin_logout(req: AdminLogoutRequest) -> Dict[str, Any]:
    """Revoke a session token immediately. The token becomes invalid right away."""
    _revoke_admin_token(req.token)
    return {"ok": True, "message": "Token revoked."}


@app.post("/admin/change-password", tags=["Admin"], summary="Change admin password")
def admin_change_pwd(req: AdminChangePasswordRequest) -> Dict[str, Any]:
    try:
        admin_change_password(req.old_password, req.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "message": "Password updated successfully."}


# ─────────────────────────────────────────────────────────────────────────────
# Routes: Clips (member view + admin decrypt)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/clips", tags=["Clips"], summary="List all recorded clips")
def get_clips() -> Dict[str, Any]:
    """
    Returns metadata for all recorded clips.
    Everyone (member + admin) can list clips and stream the blurred version.
    Only admin can decrypt the original via POST /clips/{id}/decrypt.
    """
    clips = list_clips()
    # Normalize data: add clip_id, name, and created_at for frontend compatibility
    normalized = []
    for c in clips:
        clip_id = c.get("id", "")
        normalized.append({
            "clip_id": clip_id,
            "id": clip_id,  # Keep original for backward compat
            "name": clip_id or "Unknown Clip",
            "created_at": c.get("created", ""),
            "timestamp": c.get("timestamp", 0),
            "frames": c.get("frames", 0),
            "fps": c.get("fps", 0),
            "duration_s": c.get("duration_s", 0),
        })
    return {"ok": True, "clips": normalized, "total": len(normalized)}


@app.get(
    "/clips/{clip_id}/video",
    tags=["Clips"],
    summary="Stream blurred (anonymized) video — accessible to all",
)
def get_clip_video(clip_id: str) -> StreamingResponse:
    """Download the blurred (publicly accessible) MP4 for a clip."""
    from pathlib import Path as _P
    path = _P("data/clips/blurred") / f"{clip_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' blurred video not found.")
    return StreamingResponse(
        open(path, "rb"),
        media_type="video/mp4",
        headers={"Content-Disposition": f"inline; filename=\"{clip_id}_blurred.mp4\""},
    )


@app.get(
    "/clips/{clip_id}/thumbnail",
    tags=["Clips"],
    summary="First frame of a clip as JPEG (blurred)",
)
def get_clip_thumbnail(clip_id: str) -> StreamingResponse:
    """Returns the first frame of the blurred video as a JPEG thumbnail."""
    from pathlib import Path as _P
    path = _P("data/clips/blurred") / f"{clip_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' not found.")
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="Could not read clip thumbnail.")
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")


class ClipDecryptRequest(BaseModel):
    password: str = Field(..., description="Admin password — required both for auth and AES key derivation")
    token: Optional[str] = Field(default=None, description="Session token (optional; skips password auth check when valid)")


@app.post(
    "/clips/{clip_id}/decrypt",
    tags=["Clips"],
    summary="Decrypt and download original (unblurred) video — admin only",
)
def decrypt_clip_endpoint(clip_id: str, req: ClipDecryptRequest) -> StreamingResponse:
    """
    Decrypts the AES-256-GCM encrypted clip and returns the original unblurred video.

    The `password` field is **always required** because it is used as the AES key material
    (not just for authentication).  If you hold a valid session token you may also pass it
    in `token`; the server will then skip the password-as-auth check while still using the
    password for AES key derivation.

    Wrong password → 401.  Missing clip → 404.
    """
    # 1. Verify admin — accept token (faster) or fall back to password check
    if not (req.token and _verify_admin_token(req.token)) and not verify_admin(req.password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials.")

    # 2. Load encrypted file
    from pathlib import Path as _P
    enc_path = _P("data/clips/secure") / f"{clip_id}.enc"
    if not enc_path.exists():
        raise HTTPException(status_code=404, detail=f"Encrypted clip '{clip_id}' not found.")

    # 3. Decrypt
    try:
        frames, fps = decrypt_clip(enc_path.read_bytes(), req.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    # 4. Re-encode to MP4 in memory and stream
    mp4_bytes = frames_to_mp4_bytes(frames, fps)
    return StreamingResponse(
        io.BytesIO(mp4_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=\"{clip_id}_original.mp4\""},
    )


@app.delete(
    "/clips/{clip_id}",
    tags=["Clips"],
    summary="Delete a clip — admin only",
)
def delete_clip_endpoint(clip_id: str, req: AdminAuthRequest) -> Dict[str, Any]:
    _require_admin_auth(req.password, req.token)
    removed = delete_clip_files(clip_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' not found.")
    return {"ok": True, "deleted": clip_id}


@app.delete("/clips", tags=["Clips"], summary="Delete ALL clips — admin only")
def delete_all_clips(req: AdminAuthRequest) -> Dict[str, Any]:
    _require_admin_auth(req.password, req.token)
    clips = list_clips()
    count = 0
    for c in clips:
        if delete_clip_files(c["id"]):
            count += 1
    return {"ok": True, "deleted": count}

