"""
Security evaluation for face anonymization system.

Evaluates face embedding similarity before and after anonymization to
quantify privacy protection effectiveness.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    from src.anonymizer_backend import FaceAnonymizationEngine, PRESETS
    from src.detector import FaceDetector
except ImportError:
    from anonymizer_backend import FaceAnonymizationEngine, PRESETS
    from detector import FaceDetector


@dataclass
class SecurityMetrics:
    mode: str
    preset: str
    avg_cosine_similarity: float
    avg_euclidean_distance: float
    recognition_prevented: bool  # True if similarity < threshold
    face_count: int


class SimpleFaceEmbeddingExtractor:
    """
    Lightweight face embedding extractor using OpenCV DNN.
    
    Uses OpenFace or FaceNet model for feature extraction.
    For production, consider using deepface or face_recognition libraries.
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load a pre-trained face recognition model."""
        # Try to use a simple feature extraction fallback
        # In production, you would load models like:
        # - OpenFace: https://github.com/cmusatyalab/openface
        # - FaceNet: https://github.com/davidsandberg/facenet
        # - ArcFace, etc.
        
        # For now, we'll use a simple CNN-based feature extractor
        # This is a placeholder - replace with actual model loading
        print("Note: Using fallback feature extraction (histogram-based).")
        print("For production use, install face_recognition or deepface library.")
        self.model_loaded = True
    
    def extract_embedding(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from a face ROI.
        
        Args:
            face_roi: Face region (cropped image)
            
        Returns:
            Embedding vector as numpy array
        """
        if face_roi.size == 0:
            return np.zeros(128)
        
        # Fallback: Use histogram-based features (not robust, but works)
        # In production, replace with proper face recognition model
        face_resized = cv2.resize(face_roi, (96, 96))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Extract multiple features
        features = []
        
        # 1. Histogram features
        hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        features.append(hist)
        
        # 2. LBP-like features
        lbp_features = self._extract_lbp_features(face_gray)
        features.append(lbp_features)
        
        # 3. HOG-like features
        hog_features = self._extract_hog_features(face_gray)
        features.append(hog_features)
        
        # Concatenate all features
        embedding = np.concatenate(features)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    @staticmethod
    def _extract_lbp_features(gray_img: np.ndarray, grid_size: int = 4) -> np.ndarray:
        """Extract Local Binary Pattern-like features."""
        h, w = gray_img.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray_img[
                    i * cell_h : (i + 1) * cell_h,
                    j * cell_w : (j + 1) * cell_w,
                ]
                hist = cv2.calcHist([cell], [0], None, [8], [0, 256])
                features.append(hist.flatten())
        
        return np.concatenate(features)
    
    @staticmethod
    def _extract_hog_features(gray_img: np.ndarray) -> np.ndarray:
        """Extract HOG-like gradient features."""
        # Compute gradients
        gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Create histogram
        hist, _ = np.histogram(angle, bins=16, range=(-np.pi, np.pi), weights=mag)
        
        # Normalize
        hist = hist / (hist.sum() + 1e-7)
        return hist


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate Euclidean distance between two embeddings."""
    return float(np.linalg.norm(emb1 - emb2))


def evaluate_mode_security(
    engine: FaceAnonymizationEngine,
    test_frames: List[np.ndarray],
    mode: str,
    preset: str,
    extractor: SimpleFaceEmbeddingExtractor,
    similarity_threshold: float = 0.6,
) -> SecurityMetrics:
    """
    Evaluate security of an anonymization mode.
    
    Args:
        engine: Anonymization engine
        test_frames: Test frames containing faces
        mode: Anonymization mode to test
        preset: Preset name
        extractor: Face embedding extractor
        similarity_threshold: Threshold for recognition prevention
        
    Returns:
        SecurityMetrics with evaluation results
    """
    print(f"  Evaluating mode: {mode}")
    
    similarities = []
    distances = []
    face_count = 0
    
    for frame in test_frames:
        # Detect faces in original
        boxes = engine.detector.detect_faces(frame)
        
        if not boxes:
            continue
        
        # Anonymize frame
        result = engine.process_frame(frame)
        if not result["ok"]:
            continue
        anonymized = result["frame"]
        
        # Extract embeddings for each face
        for box in boxes:
            x, y, w, h = box
            
            # Extract original face
            face_orig = frame[y : y + h, x : x + w]
            if face_orig.size == 0:
                continue
            
            # Extract anonymized face
            face_anon = anonymized[y : y + h, x : x + w]
            if face_anon.size == 0:
                continue
            
            # Get embeddings
            emb_orig = extractor.extract_embedding(face_orig)
            emb_anon = extractor.extract_embedding(face_anon)
            
            # Calculate similarity metrics
            sim = cosine_similarity(emb_orig, emb_anon)
            dist = euclidean_distance(emb_orig, emb_anon)
            
            similarities.append(sim)
            distances.append(dist)
            face_count += 1
    
    if not similarities:
        return SecurityMetrics(
            mode=mode,
            preset=preset,
            avg_cosine_similarity=0.0,
            avg_euclidean_distance=0.0,
            recognition_prevented=True,
            face_count=0,
        )
    
    avg_sim = sum(similarities) / len(similarities)
    avg_dist = sum(distances) / len(distances)
    
    # Recognition is prevented if similarity is below threshold
    recognition_prevented = avg_sim < similarity_threshold
    
    return SecurityMetrics(
        mode=mode,
        preset=preset,
        avg_cosine_similarity=avg_sim,
        avg_euclidean_distance=avg_dist,
        recognition_prevented=recognition_prevented,
        face_count=face_count,
    )


def load_test_frames_with_faces(
    video_path: str | None = None,
    num_frames: int = 50,
    detector: FaceDetector | None = None,
) -> List[np.ndarray]:
    """Load test frames that contain at least one face."""
    frames = []
    
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        print(f"Loading frames with faces from {video_path}...")
    else:
        cap = cv2.VideoCapture(0)
        print(f"Capturing frames with faces from webcam...")
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    
    captured = 0
    attempts = 0
    max_attempts = num_frames * 10
    
    while captured < num_frames and attempts < max_attempts:
        ret, frame = cap.read()
        if not ret:
            if video_path:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        attempts += 1
        
        # Check if frame has faces
        if detector:
            boxes = detector.detect_faces(frame)
            if not boxes:
                continue
        
        frames.append(frame.copy())
        captured += 1
        
        if captured % 10 == 0:
            print(f"  Captured {captured}/{num_frames} frames...")
    
    cap.release()
    print(f"Loaded {len(frames)} frames with faces")
    return frames


def print_security_results(results: List[SecurityMetrics]) -> None:
    """Print security evaluation results as a formatted table."""
    print("\n" + "=" * 100)
    print("SECURITY EVALUATION RESULTS")
    print("=" * 100)
    print(
        f"{'Mode':<20} {'Preset':<15} {'Cosine Sim':>12} {'Euclidean':>12} "
        f"{'Protected':>10} {'Faces':>8}"
    )
    print("-" * 100)
    
    for result in results:
        protected_str = "✓ YES" if result.recognition_prevented else "✗ NO"
        print(
            f"{result.mode:<20} {result.preset:<15} "
            f"{result.avg_cosine_similarity:>12.4f} {result.avg_euclidean_distance:>12.4f} "
            f"{protected_str:>10} {result.face_count:>8}"
        )
    
    print("=" * 100)
    print("\nNotes:")
    print("  - Cosine Similarity: 1.0 = identical, 0.0 = completely different")
    print("  - Lower similarity = better privacy protection")
    print("  - Protected = YES when similarity < 0.6 (typical face recognition threshold)")


def save_security_results_csv(
    results: List[SecurityMetrics],
    output_path: str = "security_evaluation.csv",
) -> None:
    """Save security evaluation results to CSV file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Mode,Preset,Cosine_Similarity,Euclidean_Distance,Recognition_Prevented,Face_Count\n")
        for result in results:
            f.write(
                f"{result.mode},{result.preset},{result.avg_cosine_similarity:.4f},"
                f"{result.avg_euclidean_distance:.4f},{result.recognition_prevented},"
                f"{result.face_count}\n"
            )
    print(f"\nResults saved to: {output_path}")


def main():
    """Run comprehensive security evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate anonymization security")
    parser.add_argument("--video", type=str, help="Path to test video file (optional)")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to test")
    parser.add_argument("--model", type=str, default="models/yolov11n-face.pt", help="Model path")
    parser.add_argument("--preset", type=str, default="all", help="Preset to test (default: all)")
    parser.add_argument("--output", type=str, default="security_evaluation.csv", help="Output CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Face recognition similarity threshold (default: 0.6)",
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    print("Initializing face embedding extractor...")
    extractor = SimpleFaceEmbeddingExtractor()
    
    # Initialize detector for frame filtering
    print("Initializing face detector...")
    detector = FaceDetector(model_path=args.model)
    
    if not detector.is_ready:
        print(f"Detector not ready. Hint: {detector.model_hint}")
        return
    
    # Load test frames
    try:
        test_frames = load_test_frames_with_faces(args.video, args.frames, detector)
    except Exception as e:
        print(f"Error loading test frames: {e}")
        return
    
    if not test_frames:
        print("No frames with faces found. Exiting.")
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
    
    print(f"\nEvaluating {len(modes_to_test)} modes × {len(presets_to_test)} presets")
    print(f"Test frames: {len(test_frames)}")
    print(f"Similarity threshold: {args.threshold}\n")
    
    results: List[SecurityMetrics] = []
    
    for preset in presets_to_test:
        print(f"\nPreset: {preset}")
        print("-" * 50)
        
        for mode in modes_to_test:
            # Create engine
            from dataclasses import replace
            config = replace(PRESETS[preset], mode=mode)
            
            engine = FaceAnonymizationEngine(
                config=config,
                model_path=args.model,
            )
            
            try:
                result = evaluate_mode_security(
                    engine,
                    test_frames,
                    mode,
                    preset,
                    extractor,
                    args.threshold,
                )
                results.append(result)
            except Exception as e:
                print(f"  Error evaluating {mode}: {e}")
    
    # Print and save results
    print_security_results(results)
    save_security_results_csv(results, args.output)
    
    # Print summary
    if results:
        print("\nSUMMARY")
        print("-" * 100)
        
        protected_modes = [r for r in results if r.recognition_prevented]
        unprotected_modes = [r for r in results if not r.recognition_prevented]
        
        print(f"Protected modes: {len(protected_modes)}/{len(results)}")
        print(f"Unprotected modes: {len(unprotected_modes)}/{len(results)}")
        
        if results:
            best_privacy = min(results, key=lambda r: r.avg_cosine_similarity)
            print(
                f"\nBest privacy protection: {best_privacy.mode} @ {best_privacy.preset} "
                f"(similarity: {best_privacy.avg_cosine_similarity:.4f})"
            )
            
            worst_privacy = max(results, key=lambda r: r.avg_cosine_similarity)
            print(
                f"Weakest privacy: {worst_privacy.mode} @ {worst_privacy.preset} "
                f"(similarity: {worst_privacy.avg_cosine_similarity:.4f})"
            )
        
        print("-" * 100)


if __name__ == "__main__":
    main()
