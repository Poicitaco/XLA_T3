"""
Benchmark script for face anonymization performance evaluation.

Measures FPS, latency, and SSIM across different anonymization modes and presets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

try:
    from src.anonymizer_backend import FaceAnonymizationEngine, PRESETS
    from src.detector import FaceDetector
except ImportError:
    from anonymizer_backend import FaceAnonymizationEngine, PRESETS
    from detector import FaceDetector


@dataclass
class BenchmarkResult:
    mode: str
    preset: str
    avg_fps: float
    avg_latency_ms: float
    ssim_score: float
    frame_count: int


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate structural similarity index between two images."""
    if img1.shape != img2.shape:
        return 0.0
    
    # Convert to grayscale for SSIM calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)


def benchmark_mode(
    engine: FaceAnonymizationEngine,
    test_frames: List[np.ndarray],
    mode: str,
    preset: str,
    warmup_runs: int = 3,
) -> BenchmarkResult:
    """Benchmark a specific anonymization mode."""
    print(f"  Testing mode: {mode}")
    
    # Warmup
    for i in range(min(warmup_runs, len(test_frames))):
        _ = engine.process_frame(test_frames[i])
    
    # Actual benchmark
    latencies = []
    ssim_scores = []
    
    for frame in test_frames:
        start_time = time.perf_counter()
        result = engine.process_frame(frame)
        end_time = time.perf_counter()
        
        if not result["ok"]:
            continue
            
        anonymized = result["frame"]
        
        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)
        
        # Calculate SSIM
        ssim_score = calculate_ssim(frame, anonymized)
        ssim_scores.append(ssim_score)
    
    avg_latency = sum(latencies) / len(latencies)
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    
    return BenchmarkResult(
        mode=mode,
        preset=preset,
        avg_fps=avg_fps,
        avg_latency_ms=avg_latency,
        ssim_score=avg_ssim,
        frame_count=len(test_frames),
    )


def load_test_frames(
    video_path: str | None = None,
    num_frames: int = 100,
) -> List[np.ndarray]:
    """Load test frames from video or webcam."""
    frames = []
    
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        print(f"Loading {num_frames} frames from {video_path}...")
    else:
        cap = cv2.VideoCapture(0)
        print(f"Capturing {num_frames} frames from webcam...")
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    
    frame_skip = 2  # Skip frames for faster loading
    count = 0
    captured = 0
    
    while captured < num_frames:
        ret, frame = cap.read()
        if not ret:
            if video_path:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        count += 1
        if count % frame_skip != 0:
            continue
        
        frames.append(frame.copy())
        captured += 1
    
    cap.release()
    print(f"Loaded {len(frames)} frames")
    return frames


def print_results_table(results: List[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Mode':<20} {'Preset':<15} {'FPS':>10} {'Latency (ms)':>15} {'SSIM':>10} {'Frames':>8}")
    print("-" * 90)
    
    for result in results:
        print(
            f"{result.mode:<20} {result.preset:<15} "
            f"{result.avg_fps:>10.2f} {result.avg_latency_ms:>15.2f} "
            f"{result.ssim_score:>10.4f} {result.frame_count:>8}"
        )
    
    print("=" * 90)


def save_results_csv(results: List[BenchmarkResult], output_path: str = "benchmark_results.csv") -> None:
    """Save benchmark results to CSV file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Mode,Preset,FPS,Latency_ms,SSIM,Frame_Count\n")
        for result in results:
            f.write(
                f"{result.mode},{result.preset},{result.avg_fps:.2f},"
                f"{result.avg_latency_ms:.2f},{result.ssim_score:.4f},{result.frame_count}\n"
            )
    print(f"\nResults saved to: {output_path}")


def main():
    """Run comprehensive benchmark across all modes and presets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark face anonymization performance")
    parser.add_argument("--video", type=str, help="Path to test video file (optional)")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to test")
    parser.add_argument("--model", type=str, default="models/yolov11n-face.pt", help="Model path")
    parser.add_argument("--preset", type=str, default="all", help="Preset to test (default: all)")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    # Load test frames
    try:
        test_frames = load_test_frames(args.video, args.frames)
    except Exception as e:
        print(f"Error loading test frames: {e}")
        return
    
    if not test_frames:
        print("No frames loaded. Exiting.")
        return
    
    # Define modes to test
    modes_to_test = [
        "blur",
        "pixelate",
        "obliterate",
        "scramble",
        "palette",
        "silhouette",
    ]
    
    # Define presets to test
    if args.preset.lower() == "all":
        presets_to_test = list(PRESETS.keys())
    else:
        if args.preset not in PRESETS:
            print(f"Unknown preset: {args.preset}")
            return
        presets_to_test = [args.preset]
    
    print(f"\nBenchmarking {len(modes_to_test)} modes × {len(presets_to_test)} presets")
    print(f"Test frames: {len(test_frames)}")
    print(f"Model: {args.model}\n")
    
    results: List[BenchmarkResult] = []
    
    for preset in presets_to_test:
        print(f"\nPreset: {preset}")
        print("-" * 50)
        
        for mode in modes_to_test:
            # Create engine with specific mode
            from dataclasses import replace
            config = replace(PRESETS[preset], mode=mode)
            
            engine = FaceAnonymizationEngine(
                config=config,
                model_path=args.model,
            )
            
            if not engine.detector.is_ready:
                print(f"Detector not ready. Hint: {engine.detector.model_hint}")
                return
            
            try:
                result = benchmark_mode(engine, test_frames, mode, preset)
                results.append(result)
            except Exception as e:
                print(f"  Error benchmarking {mode}: {e}")
    
    # Print and save results
    print_results_table(results)
    save_results_csv(results, args.output)
    
    # Print summary statistics
    if results:
        print("\nSUMMARY STATISTICS")
        print("-" * 90)
        
        # Best FPS
        best_fps = max(results, key=lambda r: r.avg_fps)
        print(f"Highest FPS: {best_fps.avg_fps:.2f} ({best_fps.mode} @ {best_fps.preset})")
        
        # Lowest latency
        best_latency = min(results, key=lambda r: r.avg_latency_ms)
        print(f"Lowest latency: {best_latency.avg_latency_ms:.2f}ms ({best_latency.mode} @ {best_latency.preset})")
        
        # Highest SSIM (most similar to original)
        best_ssim = max(results, key=lambda r: r.ssim_score)
        print(f"Highest SSIM: {best_ssim.ssim_score:.4f} ({best_ssim.mode} @ {best_ssim.preset})")
        
        # Lowest SSIM (most different from original)
        worst_ssim = min(results, key=lambda r: r.ssim_score)
        print(f"Lowest SSIM: {worst_ssim.ssim_score:.4f} ({worst_ssim.mode} @ {worst_ssim.preset})")
        print("-" * 90)


if __name__ == "__main__":
    main()
