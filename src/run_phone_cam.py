from __future__ import annotations

import argparse
import json
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cryptography.exceptions import InvalidTag
from cv2_enumerate_cameras import enumerate_cameras

try:
    from src.detector import FaceDetector
    from src.face_lock import EncryptedFaceRegion, FaceRegionLocker
    from src.privacy import anonymize_faces
except ImportError:
    from detector import FaceDetector
    from face_lock import EncryptedFaceRegion, FaceRegionLocker
    from privacy import anonymize_faces


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
        choices=["none", "blur", "pixelate", "solid", "obliterate", "headcloak", "silhouette", "lock"],
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
        default="pixelate",
        choices=["pixelate", "solid", "noise"],
        help="Preview style used after face ROI is encrypted in lock mode.",
    )
    parser.add_argument(
        "--save-lock-payload",
        type=str,
        default="",
        help="Optional JSONL path to save encrypted face payloads per frame.",
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

    try:
        while True:
            ok, frame = cap.read()
            if ok and not is_usable_frame(frame):
                ok = False
            if not ok:
                recovered = False
                for attempt in range(1, reconnect_attempts + 1):
                    print(
                        f"Frame read failed. Reconnecting {attempt}/{reconnect_attempts}..."
                    )
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
                last_boxes = [box for box, _ in last_detections]

            t_anonymize_start = time.perf_counter()
            if args.anonymize_mode == "lock" and locker is not None:
                locked_frame, last_payloads = locker.lock_faces(
                    frame,
                    last_boxes,
                    overlay_mode=args.lock_overlay,
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
                print(
                    (
                        f"faces={len(last_boxes)} boxes={last_boxes} "
                        f"fps={fps_ema:.1f} anonymize_ms={anonymize_ms_ema:.2f} mode={args.anonymize_mode} "
                        f"unlock={'on' if show_unlocked else 'off'}"
                    ),
                    end="\r",
                )

            if args.show:
                if args.debug_draw:
                    draw_debug_overlay(frame, last_detections)
                cv2.imshow("Phone Camera Stream", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if args.anonymize_mode == "lock" and key in (ord("u"), ord("U")):
                    show_unlocked = not show_unlocked
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
