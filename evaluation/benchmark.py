"""
Performance Benchmark Script
Compares FPS, latency, and SSIM across different overlay modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate

from src.detector import YOLOv11FaceDetector
from src.face_lock import FaceRegionLocker


@dataclass
class BenchmarkResult:
    mode: str
    fps: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    ssim_score: float
    frames_processed: int


class FaceAnonymizerBenchmark:
    """Benchmark different anonymization overlay modes."""

    def __init__(self, video_path: str, passphrase: str = "benchmark_key_2026"):
        self.video_path = video_path
        self.detector = YOLOv11FaceDetector(model_path="models/yolov11n-face.pt")
        self.locker = FaceRegionLocker(passphrase=passphrase)
        self.modes = ["solid", "noise", "ciphernoise", "rps"]

    def _compute_ssim(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Compute SSIM between original and processed frames."""
        if original.shape != processed.shape:
            return 0.0
        
        # Convert to grayscale for SSIM computation
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        score, _ = ssim(gray_orig, gray_proc, full=True)
        return float(score)

    def benchmark_mode(
        self,
        mode: str,
        max_frames: int = 100,
        conf_threshold: float = 0.6,
    ) -> BenchmarkResult:
        """Benchmark a single overlay mode."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        latencies: List[float] = []
        ssim_scores: List[float] = []
        frames_processed = 0

        start_time = time.perf_counter()

        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                ret, frame = cap.read()
                if not ret:
                    break

            frame_start = time.perf_counter()

            # Detect faces
            results = self.detector.detect(frame, conf_threshold=conf_threshold)
            boxes = [[int(r.box.x), int(r.box.y), int(r.box.w), int(r.box.h)] for r in results]

            # Lock faces
            locked_frame, payloads = self.locker.lock_faces(
                frame,
                boxes,
                overlay_mode=mode,
                head_ratio=0.7,
            )

            # Unlock faces (for roundtrip verification)
            unlocked_frame = self.locker.unlock_faces(locked_frame, payloads)

            frame_end = time.perf_counter()
            latencies.append((frame_end - frame_start) * 1000)  # ms

            # Compute SSIM between original and unlocked
            ssim_score = self._compute_ssim(frame, unlocked_frame)
            ssim_scores.append(ssim_score)

            frames_processed += 1

        end_time = time.perf_counter()
        cap.release()

        total_time = end_time - start_time
        fps = frames_processed / total_time if total_time > 0 else 0.0

        return BenchmarkResult(
            mode=mode,
            fps=fps,
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            median_latency_ms=float(np.median(latencies)) if latencies else 0.0,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            ssim_score=float(np.mean(ssim_scores)) if ssim_scores else 0.0,
            frames_processed=frames_processed,
        )

    def run_full_benchmark(self, max_frames: int = 100) -> List[BenchmarkResult]:
        """Run benchmark for all modes."""
        print(f"🚀 Starting benchmark with {max_frames} frames per mode...")
        print(f"📹 Video: {self.video_path}\n")

        results: List[BenchmarkResult] = []

        for mode in self.modes:
            print(f"⏳ Benchmarking mode: {mode}")
            result = self.benchmark_mode(mode, max_frames=max_frames)
            results.append(result)
            print(f"   ✓ FPS: {result.fps:.2f} | Latency: {result.avg_latency_ms:.2f}ms | SSIM: {result.ssim_score:.4f}\n")

        return results

    def print_comparison_table(self, results: List[BenchmarkResult]) -> None:
        """Print formatted comparison table."""
        headers = ["Mode", "FPS", "Avg Latency (ms)", "Median Latency (ms)", "P95 Latency (ms)", "SSIM", "Frames"]
        table_data = [
            [
                r.mode,
                f"{r.fps:.2f}",
                f"{r.avg_latency_ms:.2f}",
                f"{r.median_latency_ms:.2f}",
                f"{r.p95_latency_ms:.2f}",
                f"{r.ssim_score:.4f}",
                r.frames_processed,
            ]
            for r in results
        ]

        print("\n" + "=" * 100)
        print("📊 BENCHMARK RESULTS")
        print("=" * 100)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("=" * 100)

        # Find best performers
        best_fps = max(results, key=lambda r: r.fps)
        best_latency = min(results, key=lambda r: r.avg_latency_ms)
        best_ssim = max(results, key=lambda r: r.ssim_score)

        print(f"\n🏆 Best FPS: {best_fps.mode} ({best_fps.fps:.2f} FPS)")
        print(f"⚡ Lowest Latency: {best_latency.mode} ({best_latency.avg_latency_ms:.2f}ms)")
        print(f"🎯 Best SSIM: {best_ssim.mode} ({best_ssim.ssim_score:.4f})")
        print()


def main():
    """Run benchmark with sample video or webcam."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark face anonymization modes")
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Video path or camera index (default: 0 for webcam)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process per mode (default: 100)",
    )
    parser.add_argument(
        "--passphrase",
        type=str,
        default="benchmark_key_2026",
        help="Encryption passphrase",
    )

    args = parser.parse_args()

    # Convert camera index if needed
    video_source = int(args.video) if args.video.isdigit() else args.video

    benchmark = FaceAnonymizerBenchmark(video_path=video_source, passphrase=args.passphrase)
    results = benchmark.run_full_benchmark(max_frames=args.frames)
    benchmark.print_comparison_table(results)


if __name__ == "__main__":
    main()
