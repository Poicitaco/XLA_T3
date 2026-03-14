from __future__ import annotations

import argparse
import json
import time
import threading
from collections import deque
from queue import Empty, Full, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cryptography.exceptions import InvalidTag
from cv2_enumerate_cameras import enumerate_cameras

try:
    from src.detector import FaceDetector
    from src.face_lock import EncryptedFaceRegion, FaceRegionLocker
    from src.privacy import anonymize_faces
    from src.bbox_smoother import BBoxSmoother, KalmanBBoxSmoother
    from src.face_recognition import FaceRecognizer
    from src.family_database import FamilyDatabase
except ImportError:
    from detector import FaceDetector
    from face_lock import EncryptedFaceRegion, FaceRegionLocker
    from privacy import anonymize_faces
    from bbox_smoother import BBoxSmoother, KalmanBBoxSmoother
    from face_recognition import FaceRecognizer
    from family_database import FamilyDatabase


def parse_source(source: str) -> int | str:
    """Allow numeric camera index or URL/path sources."""
    return int(source) if source.isdigit() else source


def parse_backend(backend: str) -> int:
    """Map backend name to OpenCV backend flag."""
    name = backend.strip().lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def backend_name(backend: int) -> str:
    """Convert OpenCV backend flag to a readable name."""
    if backend == cv2.CAP_DSHOW:
        return "dshow"
    if backend == cv2.CAP_MSMF:
        return "msmf"
    if backend == cv2.CAP_ANY:
        return "any"
    return str(backend)


def get_enumerated_cameras() -> list:
    """Return best-effort camera list with names on supported platforms."""
    try:
        return list(enumerate_cameras())
    except Exception:
        return []


def print_camera_list() -> list:
    """Print enumerated cameras with names and return the list."""
    cameras = get_enumerated_cameras()
    if not cameras:
        print("No named camera list available. Falling back to OpenCV index scan only.")
        return []

    print("Available cameras:")
    for choice, camera in enumerate(cameras):
        print(
            f"[{choice}] name={camera.name} open_index={camera.index} backend={backend_name(camera.backend)}"
        )
    return cameras


def is_usable_frame(frame: Optional[np.ndarray], min_mean: float = 8.0, min_std: float = 5.0) -> bool:
    """Reject empty or nearly black frames during camera probing."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(gray.mean()) >= min_mean and float(gray.std()) >= min_std


def find_first_available_camera(max_index: int, backend: int) -> Optional[int]:
    """Find the first readable local camera index for a selected backend."""
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = False
        frame = None
        for _ in range(8):
            ok, frame = cap.read()
            if ok and is_usable_frame(frame):
                break
        cap.release()
        if ok and is_usable_frame(frame):
            return idx
    return None


def list_readable_cameras(max_index: int, backend: int) -> list[dict]:
    """Return local camera indexes that OpenCV can actually open and read.

    Virtual cameras (Camo, OBS, etc.) often output black frames before the
    phone/source is actively streaming, so we accept any readable frame here
    and only use is_usable_frame for the find_first_available_camera helper.
    """
    cameras: list[dict] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = False
        frame = None
        if cap.isOpened():
            for _ in range(12):
                ret, frm = cap.read()
                if ret and frm is not None and frm.size > 0:
                    ok = True
                    frame = frm
                    break
        # Fallback: use cap.get() for resolution if frame is all-black / tiny
        w = int(frame.shape[1]) if frame is not None else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(frame.shape[0]) if frame is not None else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        cap.release()
        if ok and w > 0 and h > 0:
            cameras.append({
                "index": idx,
                "backend": backend,
                "width": w,
                "height": h,
            })
    return cameras


def choose_camera_interactively(
    max_index: int,
    backend: int,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> Optional[int]:
    """Preview candidate cameras and let the user choose one manually."""
    window_name = "Choose Camera: Y=select N=next Q=quit"

    for idx in range(max_index + 1):
        cap = open_capture(idx, backend, camera_width, camera_height, camera_fps)
        selected = False
        try:
            if not cap.isOpened():
                continue

            frame = None
            usable = False
            for _ in range(12):
                ok, frame = cap.read()
                if ok and is_usable_frame(frame):
                    usable = True
                    break

            if not usable or frame is None:
                continue

            while True:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"source={idx}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    "Y=select  N=next  Q=quit",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("y"), ord("Y")):
                    selected = True
                    break
                if key in (ord("n"), ord("N")):
                    break
                if key in (ord("q"), ord("Q"), 27):
                    cv2.destroyWindow(window_name)
                    return None

                ok, next_frame = cap.read()
                if ok and is_usable_frame(next_frame):
                    frame = next_frame

            if selected:
                cv2.destroyWindow(window_name)
                return idx
        finally:
            cap.release()

    cv2.destroyWindow(window_name)
    return None


def choose_named_camera(cameras: list, choice: int) -> tuple[int, int]:
    """Resolve a manual choice from enumerated cameras."""
    if choice < 0 or choice >= len(cameras):
        raise RuntimeError(f"Invalid camera choice {choice}. Use --list-sources first.")
    camera = cameras[choice]
    selected_backend = camera.backend if int(camera.backend) != 0 else cv2.CAP_ANY
    return int(camera.index), int(selected_backend)


def open_capture(
    source: int | str,
    backend: int,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> cv2.VideoCapture:
    """Open capture with low-latency buffer settings."""
    cap = cv2.VideoCapture(source, backend) if isinstance(source, int) else cv2.VideoCapture(source)
    if isinstance(source, int):
        if camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        if camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        if camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Flush initial frames to ensure fresh camera stream
    if cap.isOpened():
        for _ in range(5):
            cap.grab()
    
    return cap


def prepare_inference_frame(frame: np.ndarray, proc_width: int) -> Tuple[np.ndarray, float, float]:
    """Resize frame for faster inference and provide scale factors to map boxes back."""
    if proc_width <= 0:
        return frame, 1.0, 1.0

    h, w = frame.shape[:2]
    if w <= proc_width:
        return frame, 1.0, 1.0

    scale = proc_width / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_x = w / float(new_w)
    scale_y = h / float(new_h)
    return resized, scale_x, scale_y


def scale_boxes(detections: List[Tuple[List[int], float]], scale_x: float, scale_y: float) -> List[Tuple[List[int], float]]:
    """Scale detections from inference frame back to original frame coordinates."""
    if scale_x == 1.0 and scale_y == 1.0:
        return detections

    scaled: List[Tuple[List[int], float]] = []
    for box, conf in detections:
        x, y, w, h = box
        scaled_box = [
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        ]
        scaled.append((scaled_box, conf))
    return scaled


def draw_debug_overlay(frame, detections: List[Tuple[List[int], float]]) -> None:
    """Draw face boxes and confidence labels for validation."""
    for idx, (box, conf) in enumerate(detections, start=1):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 80), 2)
        label = f"face#{idx} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 80),
            2,
            cv2.LINE_AA,
        )


def draw_key_overlay(frame: np.ndarray, anonymize_mode: str, unlock_on: bool) -> None:
    """Draw keyboard controls on the preview frame for easier live demos."""
    lines = ["Q: quit"]
    if anonymize_mode == "lock":
        lines.append(f"U: toggle unlock ({'ON' if unlock_on else 'OFF'})")

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26


class FrameReader(threading.Thread):
    """Read frames on a dedicated thread and keep only the latest frame."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self.cap = cap
        self.queue: Queue = Queue(maxsize=1)
        self.running = True
        self.frames_read = 0
        self.frames_failed = 0

    def run(self) -> None:
        consecutive_read_failures = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.frames_failed += 1
                consecutive_read_failures += 1
                
                # Log first failure and every 50 failures
                if consecutive_read_failures == 1:
                    print(f"[FrameReader] First read failure")
                elif consecutive_read_failures % 50 == 0:
                    print(f"[FrameReader] {consecutive_read_failures} consecutive failures (total failed: {self.frames_failed}, succeeded: {self.frames_read})")
                
                # Adaptive sleep: longer sleep after repeated failures to reduce CPU usage
                sleep_time = min(0.1, 0.005 * (consecutive_read_failures // 10))
                time.sleep(sleep_time)
                continue
            
            # Reset failure counter on successful read
            consecutive_read_failures = 0
            self.frames_read += 1
            
            # Log progress every 100 frames
            if self.frames_read % 100 == 0:
                print(f"[FrameReader] Read {self.frames_read} frames (failed: {self.frames_failed})")

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass

            try:
                self.queue.put_nowait((ok, frame))
            except Full:
                pass

    def read(self, timeout: float = 0.6) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return False, None

    def stop(self) -> None:
        self.running = False
        print(f"[FrameReader] Stopped. Total frames read: {self.frames_read}, failed: {self.frames_failed}")


class SimpleTracker:
    """Lightweight centroid tracker with per-track recognition state."""

    def __init__(self, max_missed: int = 12, max_distance: float = 130.0) -> None:
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    @staticmethod
    def _center(box: List[int]) -> Tuple[float, float]:
        x, y, w, h = box
        return (x + w * 0.5, y + h * 0.5)

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    @staticmethod
    def _bbox_change_metrics(old_box: List[int], new_box: List[int]) -> Tuple[float, float]:
        ox, oy, ow, oh = old_box
        nx, ny, nw, nh = new_box

        old_cx, old_cy = ox + ow * 0.5, oy + oh * 0.5
        new_cx, new_cy = nx + nw * 0.5, ny + nh * 0.5
        shift_px = float(((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) ** 0.5)
        ref_size = max(1.0, float((ow + oh) * 0.5))
        shift_ratio = shift_px / ref_size

        old_area = max(1.0, float(ow * oh))
        new_area = max(1.0, float(nw * nh))
        size_ratio = abs(new_area - old_area) / old_area
        return shift_ratio, size_ratio

    def should_reidentify(
        self,
        state: dict,
        frame_idx: int,
        min_interval: int,
        max_interval: int,
        center_shift_ratio: float,
        size_change_ratio: float,
    ) -> bool:
        last_reid = int(state.get("last_reid", -99999))
        if last_reid < 0:
            return True

        elapsed = frame_idx - last_reid
        if elapsed < max(1, min_interval):
            return False

        prev_box = state.get("last_reid_bbox")
        if prev_box is None:
            return elapsed >= max(1, min_interval)

        shift_ratio, scale_ratio = self._bbox_change_metrics(prev_box, state["bbox"])
        significant_change = (shift_ratio >= center_shift_ratio) or (scale_ratio >= size_change_ratio)
        return significant_change or (elapsed >= max(1, max_interval))

    @staticmethod
    def mark_reid_submitted(state: dict, frame_idx: int) -> None:
        state["last_reid"] = frame_idx
        state["last_reid_bbox"] = list(state["bbox"])

    def update(self, detections: List[List[int]], frame_idx: int) -> dict[int, dict]:
        assigned_tracks: set[int] = set()

        for box in detections:
            c = self._center(box)
            best_id = None
            best_dist = 1e9

            for track_id, state in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                d = self._distance(c, state["center"])
                if d < best_dist:
                    best_dist = d
                    best_id = track_id

            if best_id is not None and best_dist <= self.max_distance:
                state = self.tracks[best_id]
                state["bbox"] = box
                state["center"] = c
                state["missed"] = 0
                state["last_seen"] = frame_idx
                assigned_tracks.add(best_id)
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "bbox": box,
                    "center": c,
                    "missed": 0,
                    "last_seen": frame_idx,
                    "member_id": None,
                    "identity": "pending",
                    "score": 0.0,
                    "votes": deque(maxlen=5),
                    "last_reid": -99999,
                    "last_reid_bbox": None,
                }
                assigned_tracks.add(track_id)

        to_remove: List[int] = []
        for track_id, state in self.tracks.items():
            if track_id not in assigned_tracks:
                state["missed"] += 1
                if state["missed"] > self.max_missed:
                    to_remove.append(track_id)

        for track_id in to_remove:
            self.tracks.pop(track_id, None)

        return self.tracks

    def apply_recognition(self, track_id: int, member_id: Optional[str], name: str, score: float) -> None:
        state = self.tracks.get(track_id)
        if state is None:
            return

        if member_id is None:
            state["votes"].append("unknown")
        else:
            state["votes"].append(name)

        # Majority vote to stabilize identity labels between frames.
        if state["votes"]:
            values = list(state["votes"])
            top = max(set(values), key=values.count)
            state["identity"] = top
        else:
            state["identity"] = "pending"

        state["member_id"] = member_id
        state["score"] = score


class RecognitionWorker(threading.Thread):
    """Async face recognition worker to keep main loop responsive."""

    def __init__(self, recognizer: FaceRecognizer, database: FamilyDatabase, threshold: float) -> None:
        super().__init__(daemon=True)
        self.recognizer = recognizer
        self.database = database
        self.threshold = threshold
        self.in_q: Queue = Queue(maxsize=8)
        self.out_q: Queue = Queue()
        self.running = True
        self._pending_track_ids: set[int] = set()
        self._pending_lock = threading.Lock()

    def submit(self, track_id: int, frame: np.ndarray, bbox: List[int]) -> None:
        if not self.running:
            return

        x, y, w, h = bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        with self._pending_lock:
            if track_id in self._pending_track_ids:
                return
            self._pending_track_ids.add(track_id)

        try:
            self.in_q.put_nowait((track_id, roi.copy()))
        except Full:
            with self._pending_lock:
                self._pending_track_ids.discard(track_id)
            return

    def get_result_nowait(self) -> Optional[dict]:
        try:
            return self.out_q.get_nowait()
        except Empty:
            return None

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            try:
                track_id, roi = self.in_q.get(timeout=0.1)
            except Empty:
                continue

            try:
                emb = self.recognizer.extract_embedding_from_image(roi)
                if emb is None:
                    self.out_q.put({
                        "track_id": track_id,
                        "member_id": None,
                        "name": "unknown",
                        "score": 0.0,
                    })
                    continue

                match = self.database.match(emb, threshold=self.threshold)
                if match is None:
                    self.out_q.put({
                        "track_id": track_id,
                        "member_id": None,
                        "name": "unknown",
                        "score": 0.0,
                    })
                    continue

                member_id, name, score = match
                self.out_q.put({
                    "track_id": track_id,
                    "member_id": member_id,
                    "name": name,
                    "score": float(score),
                })
            except Exception:
                self.out_q.put({
                    "track_id": track_id,
                    "member_id": None,
                    "name": "unknown",
                    "score": 0.0,
                })
            finally:
                with self._pending_lock:
                    self._pending_track_ids.discard(track_id)


def draw_tracking_overlay(frame: np.ndarray, tracks: dict[int, dict], show_registered_only: bool) -> None:
    for track_id, state in tracks.items():
        identity = state.get("identity", "pending")
        if show_registered_only and identity in ("pending", "unknown"):
            continue

        x, y, w, h = state["bbox"]
        score = state.get("score", 0.0)
        color = (0, 255, 0) if identity not in ("pending", "unknown") else (30, 200, 255)
        label = f"T{track_id} {identity} {score:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run face detection from phone camera stream or local camera."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        help="Camera source, e.g. 0, http://192.168.1.10:8080/video, rtsp://...",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print detected camera names and indexes, then exit.",
    )
    parser.add_argument(
        "--source-choice",
        type=int,
        default=None,
        help="Choose a camera by number from --list-sources output.",
    )
    parser.add_argument(
        "--choose-source",
        action="store_true",
        help="Preview local camera indexes and choose one manually.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dshow",
        choices=["dshow", "msmf", "any"],
        help="OpenCV backend for local camera indexes on Windows.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov11n-face.pt",
        help="Path/name to YOLOv11-face model weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Inference image size.",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=2,
        help="Run detector every N frames to reduce CPU/GPU load.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Preview stream window (without drawing boxes).",
    )
    parser.add_argument(
        "--max-cam-index",
        type=int,
        default=10,
        help="Maximum local camera index to scan when --source auto.",
    )
    parser.add_argument(
        "--reconnect",
        type=int,
        default=3,
        help="Number of reconnect attempts when stream read fails.",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between reconnect attempts.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print status every N frames to reduce terminal overhead.",
    )
    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help="Draw face boxes and confidence on preview window for validation.",
    )
    parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Show keyboard shortcuts overlay on preview window.",
    )
    parser.add_argument(
        "--proc-width",
        type=int,
        default=640,
        help="Resize frame to this width before detection for higher FPS. Use 0 to disable.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested capture width for local camera source.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested capture height for local camera source.",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Requested capture FPS for local camera source.",
    )
    parser.add_argument(
        "--fps-ema",
        type=float,
        default=0.9,
        help="EMA factor for smoother FPS display (0..1, higher is smoother).",
    )
    parser.add_argument(
        "--anonymize-mode",
        type=str,
        default="blur",
        choices=["none", "blur", "neckup", "pixelate", "solid", "obliterate", "headcloak", "silhouette", "lock"],
        help="Face anonymization mode for detected boxes.",
    )
    parser.add_argument(
        "--blur-scale",
        type=float,
        default=0.2,
        help="Downscale factor before blur (lower is faster/stronger).",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=31,
        help="Gaussian blur kernel size (odd value preferred).",
    )
    parser.add_argument(
        "--pixel-block",
        type=int,
        default=16,
        help="Pixel block size for pixelate mode.",
    )
    parser.add_argument(
        "--face-padding",
        type=float,
        default=0.12,
        help="Padding ratio around each detected face box.",
    )
    parser.add_argument(
        "--obliterate-scale",
        type=float,
        default=0.08,
        help="Aggressive downscale factor for obliterate mode (lower = stronger destruction).",
    )
    parser.add_argument(
        "--scramble-block",
        type=int,
        default=12,
        help="Block size used to scramble local facial structure in obliterate mode.",
    )
    parser.add_argument(
        "--palette-levels",
        type=int,
        default=10,
        help="Color quantization levels in obliterate mode (lower = stronger anonymization).",
    )
    parser.add_argument(
        "--scramble-seed",
        type=int,
        default=1337,
        help="Seed for deterministic scrambling pattern in obliterate mode.",
    )
    parser.add_argument(
        "--head-ratio",
        type=float,
        default=0.6,
        help="Expansion ratio for headcloak mode to include hair/jaw/ears.",
    )
    parser.add_argument(
        "--neck-ratio",
        type=float,
        default=0.6,
        help="Expansion ratio for neckup mode (head + neck only).",
    )
    parser.add_argument(
        "--silhouette-ratio",
        type=float,
        default=0.8,
        help="Expansion ratio for silhouette mode to include shoulders and upper body cues.",
    )
    parser.add_argument(
        "--lock-key",
        type=str,
        default="",
        help="Passphrase for reversible face lock mode.",
    )
    parser.add_argument(
        "--unlock-key",
        type=str,
        default="",
        help="Optional passphrase used for unlock (defaults to --lock-key).",
    )
    parser.add_argument(
        "--lock-overlay",
        type=str,
        default="rps",
        choices=["solid", "noise", "ciphernoise", "rps"],
        help="Preview style used after face ROI is encrypted in lock mode.",
    )
    parser.add_argument(
        "--lock-pixel-block",
        type=int,
        default=12,
        help="Pixel block size for lock overlay (larger gives stronger mosaic).",
    )
    parser.add_argument(
        "--lock-noise-intensity",
        type=int,
        default=70,
        help="Noise intensity for lock overlay noise/ciphernoise modes.",
    )
    parser.add_argument(
        "--lock-noise-mix",
        type=float,
        default=0.9,
        help="Noise blend factor for lock overlay noise/ciphernoise modes (0..1).",
    )
    parser.add_argument(
        "--lock-head-ratio",
        type=float,
        default=0.45,
        help="Expand lock region to full head (including hair) in lock mode.",
    )
    parser.add_argument(
        "--rps-tile-size",
        type=int,
        default=8,
        help="Tile size for Reversible Pixel Shuffling lock overlay.",
    )
    parser.add_argument(
        "--rps-rounds",
        type=int,
        default=2,
        help="Shuffle rounds for Reversible Pixel Shuffling lock overlay.",
    )
    parser.add_argument(
        "--save-lock-payload",
        type=str,
        default="",
        help="Optional JSONL path to save encrypted face payloads per frame.",
    )
    parser.add_argument(
        "--smooth-boxes",
        action="store_true",
        help="Enable temporal smoothing for bounding boxes to reduce jitter.",
    )
    parser.add_argument(
        "--smooth-method",
        type=str,
        default="ema",
        choices=["ema", "kalman"],
        help="Smoothing method: ema (exponential moving average) or kalman (Kalman filter).",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.7,
        help="EMA smoothing factor (0-1). Higher = more responsive, lower = smoother.",
    )
    parser.add_argument(
        "--smooth-history",
        type=int,
        default=5,
        help="Number of frames in smoothing history.",
    )
    parser.add_argument(
        "--smooth-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for matching boxes across frames during smoothing.",
    )
    parser.add_argument(
        "--selective-mode",
        type=str,
        default="disabled",
        choices=["disabled", "family", "stranger", "registered_only"],
        help="ID filter mode: disabled, family(blur registered), stranger(blur unknown), registered_only(show only registered).",
    )
    parser.add_argument(
        "--face-db",
        type=str,
        default="data/family_members.json",
        help="Family database JSON path.",
    )
    parser.add_argument(
        "--recognition-model",
        type=str,
        default="buffalo_l",
        help="InsightFace model used for registration and runtime recognition.",
    )
    parser.add_argument(
        "--recognition-threshold",
        type=float,
        default=0.35,
        help="Cosine similarity threshold for registered identity matching.",
    )
    parser.add_argument(
        "--reidentify-interval",
        type=int,
        default=30,
        help="Run recognition every N frames per active track.",
    )
    parser.add_argument(
        "--reid-min-interval",
        type=int,
        default=10,
        help="Minimum frames between recognition submits for the same track.",
    )
    parser.add_argument(
        "--reid-max-interval",
        type=int,
        default=90,
        help="Force re-recognition after this many frames even without major bbox change.",
    )
    parser.add_argument(
        "--reid-center-shift-ratio",
        type=float,
        default=0.12,
        help="Re-recognize when center shift exceeds this ratio of face size.",
    )
    parser.add_argument(
        "--reid-size-change-ratio",
        type=float,
        default=0.20,
        help="Re-recognize when face area change ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=12,
        help="Remove track if unmatched for this many cycles.",
    )
    args = parser.parse_args()
    backend_flag = parse_backend(args.backend)
    enumerated_cameras = get_enumerated_cameras()

    if args.list_sources:
        print_camera_list()
        return

    detector = FaceDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
    )
    if not detector.is_ready:
        raise RuntimeError(
            "Face model failed to load. "
            "Provide a valid weight file with --model, for example: "
            "--model models/yolov11n-face.pt"
        )

    locker: Optional[FaceRegionLocker] = None
    unlocker: Optional[FaceRegionLocker] = None
    last_payloads: List[EncryptedFaceRegion] = []
    show_unlocked = False
    decrypt_error_logged = False
    if args.anonymize_mode == "lock":
        if not args.lock_key:
            raise RuntimeError("--lock-key is required when --anonymize-mode lock is used.")
        locker = FaceRegionLocker(args.lock_key)
        unlocker = FaceRegionLocker(args.unlock_key if args.unlock_key else args.lock_key)
        print("Lock mode controls: press U to toggle unlock preview.")

    # Initialize bbox smoother if enabled
    bbox_smoother: Optional[BBoxSmoother | KalmanBBoxSmoother] = None
    if args.smooth_boxes:
        if args.smooth_method == "ema":
            bbox_smoother = BBoxSmoother(
                alpha=args.smooth_alpha,
                history_size=args.smooth_history,
                iou_threshold=args.smooth_iou_threshold,
            )
            print(f"BBox smoothing enabled: EMA (alpha={args.smooth_alpha})")
        else:
            bbox_smoother = KalmanBBoxSmoother(
                iou_threshold=args.smooth_iou_threshold,
            )
            print("BBox smoothing enabled: Kalman filter")

    source: int | str
    if args.source_choice is not None:
        if not enumerated_cameras:
            raise RuntimeError("Named camera list is unavailable. Use --source 0/1/2 or --choose-source.")
        source, backend_flag = choose_named_camera(enumerated_cameras, args.source_choice)
        print(f"Using named camera choice {args.source_choice}: source={source} backend={backend_name(backend_flag)}")
    elif args.choose_source:
        chosen_idx = choose_camera_interactively(
            args.max_cam_index,
            backend_flag,
            args.camera_width,
            args.camera_height,
            args.camera_fps,
        )
        if chosen_idx is None:
            raise RuntimeError("No camera selected.")
        source = chosen_idx
        print(f"Using manually selected camera source: {source} (backend={args.backend})")
    elif args.source.lower() == "auto":
        auto_idx = find_first_available_camera(args.max_cam_index, backend_flag)
        if auto_idx is None:
            raise RuntimeError(
                "No readable camera found. Keep Camo Studio open and try --backend dshow or --backend msmf with --source 0/1/2/3."
            )
        source = auto_idx
        print(f"Using auto-detected camera source: {source} (backend={args.backend})")
    else:
        source = parse_source(args.source)

    cap = open_capture(
        source,
        backend_flag,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )

    if not cap.isOpened():
        if isinstance(source, int):
            raise RuntimeError(
                f"Cannot open camera index {source}. "
                "Try --backend dshow or --backend msmf, keep Camo Studio open, and test --source 0/1/2/3."
            )
        raise RuntimeError(
            "Cannot open stream URL. Verify the URL and network connectivity."
        )

    frame_idx = 0
    last_boxes: List[List[int]] = []
    last_detections: List[Tuple[List[int], float]] = []
    t_prev = time.perf_counter()
    fps_ema = 0.0
    anonymize_ms_ema = 0.0
    reconnect_attempts = max(args.reconnect, 0)
    detect_every = max(args.detect_every, 1)
    print_every = max(args.print_every, 1)
    fps_alpha = min(max(args.fps_ema, 0.0), 1.0)

    selective_enabled = args.selective_mode != "disabled"
    tracker: Optional[SimpleTracker] = None
    recognition_worker: Optional[RecognitionWorker] = None

    if selective_enabled:
        db = FamilyDatabase(db_path=args.face_db)
        if not db.members:
            raise RuntimeError(f"No registered members found in database: {args.face_db}")
        recognizer = FaceRecognizer(model=args.recognition_model)
        tracker = SimpleTracker(max_missed=max(1, args.track_max_missed))
        recognition_worker = RecognitionWorker(
            recognizer=recognizer,
            database=db,
            threshold=args.recognition_threshold,
        )
        recognition_worker.start()
        print(f"Selective mode: {args.selective_mode} | members={len(db.members)} | model={args.recognition_model}")

    frame_reader = FrameReader(cap)
    frame_reader.start()

    try:
        while True:
            ok, frame = frame_reader.read(timeout=0.6)
            if ok and not is_usable_frame(frame):
                ok = False
            if not ok:
                recovered = False
                for attempt in range(1, reconnect_attempts + 1):
                    print(
                        f"Frame read failed. Reconnecting {attempt}/{reconnect_attempts}..."
                    )
                    frame_reader.stop()
                    cap.release()
                    time.sleep(max(args.reconnect_delay, 0.0))
                    cap = open_capture(
                        source,
                        backend_flag,
                        args.camera_width,
                        args.camera_height,
                        args.camera_fps,
                    )
                    for _ in range(8):
                        ok, frame = cap.read()
                        if ok and is_usable_frame(frame):
                            break
                    if ok and is_usable_frame(frame):
                        frame_reader = FrameReader(cap)
                        frame_reader.start()
                        recovered = True
                        print("Reconnect successful.")
                        break
                if not recovered:
                    print("Stream disconnected and reconnect failed.")
                    break

            frame_idx += 1
            if frame_idx % detect_every == 0:
                infer_frame, scale_x, scale_y = prepare_inference_frame(frame, args.proc_width)
                current_detections = detector.detect_faces_with_scores(infer_frame)
                last_detections = scale_boxes(current_detections, scale_x, scale_y)

                if selective_enabled and tracker is not None:
                    detected_boxes = [box for box, _ in last_detections]
                    tracker.update(detected_boxes, frame_idx)
                else:
                    # Apply bbox smoothing if enabled
                    if bbox_smoother is not None:
                        smoothed_boxes = bbox_smoother.update(last_detections)
                        last_boxes = smoothed_boxes
                    else:
                        last_boxes = [box for box, _ in last_detections]

            if selective_enabled and tracker is not None and recognition_worker is not None:
                # Submit recognition only when track changes significantly, with a fallback interval.
                for track_id, state in tracker.tracks.items():
                    should_reid = tracker.should_reidentify(
                        state=state,
                        frame_idx=frame_idx,
                        min_interval=max(1, args.reid_min_interval),
                        max_interval=max(1, max(args.reid_max_interval, args.reidentify_interval)),
                        center_shift_ratio=max(0.01, args.reid_center_shift_ratio),
                        size_change_ratio=max(0.01, args.reid_size_change_ratio),
                    )
                    if should_reid:
                        recognition_worker.submit(track_id, frame, state["bbox"])
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
                for _track_id, state in tracker.tracks.items():
                    identity = str(state.get("identity", "pending"))
                    is_registered = identity not in ("pending", "unknown")
                    should_blur = False

                    if args.selective_mode == "family":
                        should_blur = is_registered
                    elif args.selective_mode == "stranger":
                        should_blur = not is_registered

                    if should_blur:
                        last_boxes.append(state["bbox"])

            t_anonymize_start = time.perf_counter()
            if args.anonymize_mode == "lock" and locker is not None:
                locked_frame, last_payloads = locker.lock_faces(
                    frame,
                    last_boxes,
                    overlay_mode=args.lock_overlay,
                    overlay_pixel_block=args.lock_pixel_block,
                    overlay_noise_intensity=args.lock_noise_intensity,
                    overlay_noise_mix=args.lock_noise_mix,
                    head_ratio=args.lock_head_ratio,
                    rps_tile_size=args.rps_tile_size,
                    rps_rounds=args.rps_rounds,
                )
                if show_unlocked:
                    try:
                        frame = (unlocker or locker).unlock_faces(locked_frame, last_payloads)
                        decrypt_error_logged = False
                    except InvalidTag:
                        if not decrypt_error_logged:
                            print("\nUnlock failed: wrong key or corrupted payload.")
                            decrypt_error_logged = True
                        frame = locked_frame
                else:
                    frame = locked_frame

                if args.save_lock_payload and last_payloads:
                    record = {
                        "frame_idx": frame_idx,
                        "payloads": FaceRegionLocker.payloads_to_jsonable(last_payloads),
                    }
                    with open(args.save_lock_payload, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(record, ensure_ascii=True) + "\n")
            else:
                anonymize_faces(
                    frame,
                    last_boxes,
                    mode=args.anonymize_mode,
                    blur_scale=args.blur_scale,
                    blur_kernel=args.blur_kernel,
                    pixel_block=args.pixel_block,
                    padding_ratio=args.face_padding,
                    obliterate_scale=args.obliterate_scale,
                    scramble_block=args.scramble_block,
                    palette_levels=args.palette_levels,
                    scramble_seed=args.scramble_seed,
                    head_ratio=args.head_ratio,
                    silhouette_ratio=args.silhouette_ratio,
                    neck_ratio=args.neck_ratio,
                )
            anonymize_ms = (time.perf_counter() - t_anonymize_start) * 1000.0
            if anonymize_ms_ema <= 0.0:
                anonymize_ms_ema = anonymize_ms
            else:
                anonymize_ms_ema = (fps_alpha * anonymize_ms_ema) + ((1.0 - fps_alpha) * anonymize_ms)

            t_now = time.perf_counter()
            fps_instant = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            if fps_ema <= 0.0:
                fps_ema = fps_instant
            else:
                fps_ema = (fps_alpha * fps_ema) + ((1.0 - fps_alpha) * fps_instant)

            if frame_idx % print_every == 0:
                if selective_enabled and tracker is not None:
                    registered_count = sum(1 for s in tracker.tracks.values() if str(s.get("identity", "pending")) not in ("pending", "unknown"))
                    unknown_count = sum(1 for s in tracker.tracks.values() if str(s.get("identity", "pending")) == "unknown")
                else:
                    registered_count = 0
                    unknown_count = 0
                print(
                    (
                        f"faces={len(last_boxes)} boxes={last_boxes} "
                        f"fps={fps_ema:.1f} anonymize_ms={anonymize_ms_ema:.2f} mode={args.anonymize_mode} "
                        f"unlock={'on' if show_unlocked else 'off'} reg={registered_count} unk={unknown_count}"
                    ),
                    end="\r",
                )

            if args.show:
                if args.debug_draw:
                    if selective_enabled and tracker is not None:
                        draw_tracking_overlay(
                            frame,
                            tracker.tracks,
                            show_registered_only=(args.selective_mode == "registered_only"),
                        )
                    else:
                        draw_debug_overlay(frame, last_detections)
                if args.show_keys:
                    draw_key_overlay(frame, args.anonymize_mode, show_unlocked)
                cv2.imshow("Phone Camera Stream", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if args.anonymize_mode == "lock" and key in (ord("u"), ord("U")):
                    show_unlocked = not show_unlocked
    finally:
        frame_reader.stop()
        if recognition_worker is not None:
            recognition_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
