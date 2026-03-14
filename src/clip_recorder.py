"""
Clip recorder: records face-triggered video clips with dual storage.

──────────────────────────────────────────────────────────────────────
Architecture
──────────────────────────────────────────────────────────────────────
ClipRecorder.push_frame(raw, display, face_detected)
  ├── pre-buffer (rolling deque, configurable seconds)
  ├── on trigger → start recording for clip_duration_s seconds
  └── on finish  → background thread saves:
        • data/clips/blurred/<clip_id>.mp4  – display (anonymized) frames
        • data/clips/secure/<clip_id>.enc   – raw frames AES-256-GCM encrypted

Encrypted file format:
  [ 32 B salt ][ 12 B nonce ][ AES-256-GCM ciphertext ]

Ciphertext decrypts to:
  [ 4 B n_frames ][ 4 B fps*1000 ]
  for each frame: [ 4 B jpeg_len ][ jpeg_bytes (quality=90) ]

Key derivation:
  key = PBKDF2-SHA256(admin_password, per-clip salt, ITERATIONS=390000)

Only admin knows the password → only admin can decrypt.
──────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import io
import json
import secrets
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

try:
    from src.admin_auth import derive_clip_key
except ImportError:
    from admin_auth import derive_clip_key


# ─── directory layout ────────────────────────────────────────────────────────
_BASE_DIR    = Path("data/clips")
_BLURRED_DIR = _BASE_DIR / "blurred"
_SECURE_DIR  = _BASE_DIR / "secure"
_META_FILE   = _BASE_DIR / "clips.json"

_PRE_BUFFER_S  = 3.0    # seconds of pre-event rolling buffer
_CLIP_DURATION = 8.0    # seconds to record after trigger
_COOLDOWN_S    = 15.0   # min gap between clips to avoid flooding


def _ensure_dirs(base: Optional[Path] = None) -> None:
    if base:
        blurred = base / "blurred"
        secure  = base / "secure"
    else:
        blurred = _BLURRED_DIR
        secure  = _SECURE_DIR
    blurred.mkdir(parents=True, exist_ok=True)
    secure.mkdir(parents=True, exist_ok=True)


# ─── serialisation helpers ────────────────────────────────────────────────────

def _pack_frames(frames: List[np.ndarray], fps: float) -> bytes:
    """Serialise frames as length-prefixed JPEG stream."""
    buf = io.BytesIO()
    buf.write(struct.pack(">I", len(frames)))
    buf.write(struct.pack(">I", int(fps * 1000)))
    for f in frames:
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        data = enc.tobytes()
        buf.write(struct.pack(">I", len(data)))
        buf.write(data)
    return buf.getvalue()


def _unpack_frames(raw: bytes) -> Tuple[List[np.ndarray], float]:
    """Deserialise frames from length-prefixed JPEG stream."""
    buf = io.BytesIO(raw)
    n_frames = struct.unpack(">I", buf.read(4))[0]
    fps = struct.unpack(">I", buf.read(4))[0] / 1000.0
    frames: List[np.ndarray] = []
    for _ in range(n_frames):
        size_bytes = buf.read(4)
        if len(size_bytes) < 4:
            break
        size = struct.unpack(">I", size_bytes)[0]
        jpeg = buf.read(size)
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames, fps


# ─── encryption / decryption ─────────────────────────────────────────────────

def encrypt_clip(frames: List[np.ndarray], fps: float, password: str) -> bytes:
    """
    Encrypt frames with AES-256-GCM.
    Returns bytes: [ 32B salt ][ 12B nonce ][ ciphertext+tag ]
    """
    plaintext = _pack_frames(frames, fps)
    salt  = secrets.token_bytes(32)
    nonce = secrets.token_bytes(12)
    key   = derive_clip_key(password, salt)
    ct    = AESGCM(key).encrypt(nonce, plaintext, None)
    return salt + nonce + ct


def decrypt_clip(data: bytes, password: str) -> Tuple[List[np.ndarray], float]:
    """
    Decrypt an encrypted clip blob.
    Raises ValueError on wrong password or corrupted data.
    """
    if len(data) < 44:
        raise ValueError("Encrypted clip file is too short / corrupted.")
    salt      = data[:32]
    nonce     = data[32:44]
    ciphertext = data[44:]
    key = derive_clip_key(password, salt)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception:
        raise ValueError("Decryption failed – wrong password or corrupted clip.")
    return _unpack_frames(plaintext)


# ─── MP4 writer ──────────────────────────────────────────────────────────────

def _transcode_h264(path: Path) -> None:
    """Re-encode an mp4v file to H.264 using ffmpeg for browser playback compatibility."""
    import subprocess, shutil
    tmp = path.with_suffix(".h264tmp.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                str(tmp),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            shutil.move(str(tmp), str(path))
    except Exception:
        pass  # leave original mp4v in place if ffmpeg fails
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_mp4(frames: List[np.ndarray], fps: float, path: Path) -> None:
    """Write a list of BGR frames to an H.264 MP4 file (browser-compatible)."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(1.0, fps), (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    _transcode_h264(path)


def frames_to_mp4_bytes(frames: List[np.ndarray], fps: float) -> bytes:
    """Encode frames to MP4 in memory (via temp file) and return raw bytes."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        write_mp4(frames, fps, tmp)
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


# ─── metadata helpers ────────────────────────────────────────────────────────

def _read_meta(meta_file: Path) -> dict:
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"clips": []}


def _write_meta(meta_file: Path, meta: dict) -> None:
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_clip_meta(
    clip_id: str,
    n_frames: int,
    fps: float,
    meta_file: Path = _META_FILE,
) -> None:
    meta = _read_meta(meta_file)
    meta["clips"].append({
        "id": clip_id,
        "timestamp": time.time(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames": n_frames,
        "fps": fps,
        "duration_s": round(n_frames / fps, 2) if fps > 0 else 0,
    })
    _write_meta(meta_file, meta)


def list_clips(meta_file: Path = _META_FILE) -> List[dict]:
    return _read_meta(meta_file)["clips"]


def delete_clip(clip_id: str, base: Path = _BASE_DIR) -> bool:
    blurred_dir = base / "blurred"
    secure_dir  = base / "secure"
    meta_file   = base / "clips.json"
    removed = False
    for p in [blurred_dir / f"{clip_id}.mp4", secure_dir / f"{clip_id}.enc"]:
        if p.exists():
            p.unlink()
            removed = True
    meta = _read_meta(meta_file)
    before = len(meta["clips"])
    meta["clips"] = [c for c in meta["clips"] if c["id"] != clip_id]
    if len(meta["clips"]) < before:
        _write_meta(meta_file, meta)
    return removed


# ─── ClipRecorder ─────────────────────────────────────────────────────────────

class ClipRecorder:
    """
    Thread-safe clip recorder for the live pipeline.

    Usage (from pipeline loop):
        recorder.push_frame(raw_frame, display_frame, face_detected=bool)

    raw_frame    → stored encrypted (admin-only)
    display_frame → stored as plain MP4 (already anonymized by pipeline)
    """

    def __init__(
        self,
        admin_password: str,
        fps: float = 15.0,
        pre_buffer_s: float = _PRE_BUFFER_S,
        clip_duration_s: float = _CLIP_DURATION,
        cooldown_s: float = _COOLDOWN_S,
        clips_dir: Optional[str] = None,
    ) -> None:
        self._password = admin_password
        self._fps = fps

        # Output directories
        if clips_dir:
            self._base       = Path(clips_dir)
            self._blurred    = self._base / "blurred"
            self._secure     = self._base / "secure"
            self._meta_file  = self._base / "clips.json"
        else:
            self._base       = _BASE_DIR
            self._blurred    = _BLURRED_DIR
            self._secure     = _SECURE_DIR
            self._meta_file  = _META_FILE
        _ensure_dirs(self._base)

        self._pre_buf_size  = max(1, int(pre_buffer_s * fps))
        self._record_target = max(1, int(clip_duration_s * fps))
        self._cooldown_s    = cooldown_s

        # Rolling pre-buffer stores (raw, display) tuples
        self._pre_raw:  deque[np.ndarray] = deque(maxlen=self._pre_buf_size)
        self._pre_disp: deque[np.ndarray] = deque(maxlen=self._pre_buf_size)

        self._lock             = threading.Lock()
        self._recording        = False
        self._raw_buf:  List[np.ndarray] = []
        self._disp_buf: List[np.ndarray] = []
        self._frames_left      = 0
        self._last_clip_time   = 0.0
        self._current_clip_id: Optional[str] = None

        # Expose for status endpoint
        self.clips_saved = 0

    # ── public API ────────────────────────────────────────────────────────────

    def push_frame(
        self,
        raw_frame: np.ndarray,
        display_frame: np.ndarray,
        face_detected: bool = False,
    ) -> None:
        """
        Feed a frame pair into the recorder.
        Must be called from the pipeline loop for every frame.
        """
        with self._lock:
            if self._recording:
                self._raw_buf.append(raw_frame.copy())
                self._disp_buf.append(display_frame.copy())
                self._frames_left -= 1
                if self._frames_left <= 0:
                    self._finish_locked()
            else:
                self._pre_raw.append(raw_frame.copy())
                self._pre_disp.append(display_frame.copy())
                if face_detected and (time.time() - self._last_clip_time) >= self._cooldown_s:
                    self._start_locked()

    def update_fps(self, fps: float) -> None:
        """Update FPS hint (affects new clips only; called from pipeline on each frame)."""
        with self._lock:
            self._fps = fps

    # ── internal ─────────────────────────────────────────────────────────────

    def _start_locked(self) -> None:
        self._current_clip_id = time.strftime("%Y%m%d_%H%M%S")
        # Seed recording buffers with pre-buffer content
        self._raw_buf  = list(self._pre_raw)
        self._disp_buf = list(self._pre_disp)
        self._frames_left = self._record_target
        self._recording   = True

    def _finish_locked(self) -> None:
        self._recording = False
        self._last_clip_time = time.time()
        clip_id   = self._current_clip_id
        raw_buf   = self._raw_buf[:]
        disp_buf  = self._disp_buf[:]
        fps       = self._fps
        password  = self._password
        blurred   = self._blurred
        secure    = self._secure
        meta_file = self._meta_file
        self._raw_buf  = []
        self._disp_buf = []
        self._current_clip_id = None
        # Save asynchronously so pipeline loop is not blocked
        threading.Thread(
            target=_save_clip_worker,
            args=(clip_id, raw_buf, disp_buf, fps, password, blurred, secure, meta_file),
            daemon=True,
        ).start()
        self.clips_saved += 1


# ── save worker (runs in background thread) ────────────────────────────────────

def _save_clip_worker(
    clip_id: str,
    raw_frames: List[np.ndarray],
    disp_frames: List[np.ndarray],
    fps: float,
    password: str,
    blurred_dir: Path,
    secure_dir: Path,
    meta_file: Path,
) -> None:
    try:
        # 1. Blurred MP4 (display frames already anonymized by pipeline)
        blurred_path = blurred_dir / f"{clip_id}.mp4"
        write_mp4(disp_frames, fps, blurred_path)

        # 2. Encrypted raw frames
        enc_path = secure_dir / f"{clip_id}.enc"
        enc_data = encrypt_clip(raw_frames, fps, password)
        enc_path.write_bytes(enc_data)

        # 3. Metadata
        _append_clip_meta(clip_id, len(raw_frames), fps, meta_file)

        print(f"[ClipRecorder] Saved {clip_id}: {len(raw_frames)} frames "
              f"| blurred={blurred_path.stat().st_size//1024}KB "
              f"| enc={enc_path.stat().st_size//1024}KB")
    except Exception as exc:
        import traceback
        print(f"[ClipRecorder] Error saving {clip_id}: {exc}")
        traceback.print_exc()
