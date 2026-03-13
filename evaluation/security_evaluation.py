"""
Security Evaluation: Face Embedding Similarity Analysis
Evaluates anonymization effectiveness by comparing face embeddings before/after locking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from tabulate import tabulate

from src.detector import YOLOv11FaceDetector
from src.face_lock import FaceRegionLocker


@dataclass
class SecurityMetrics:
    mode: str
    avg_similarity: float
    median_similarity: float
    max_similarity: float
    min_similarity: float
    faces_evaluated: int
    detection_rate_after_lock: float  # % of faces still detected after locking


class FaceEmbeddingSecurityEvaluator:
    """Evaluate anonymization security via face embedding similarity."""

    def __init__(
        self,
        passphrase: str = "security_eval_2026",
        embedding_model: str = "Facenet512",
    ):
        """
        Initialize evaluator.
        
        Args:
            passphrase: Encryption passphrase
            embedding_model: DeepFace model - options: VGG-Face, Facenet, Facenet512, 
                           OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
        """
        self.locker = FaceRegionLocker(passphrase=passphrase)
        self.detector = YOLOv11FaceDetector(model_path="models/yolov11n-face.pt")
        self.embedding_model = embedding_model

    def _extract_face_embedding(self, face_roi: np.ndarray) -> np.ndarray | None:
        """Extract face embedding using DeepFace."""
        try:
            # DeepFace expects BGR format
            embedding_obj = DeepFace.represent(
                img_path=face_roi,
                model_name=self.embedding_model,
                enforce_detection=False,
            )
            
            # Handle both single and multiple face results
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                return np.array(embedding_obj[0]["embedding"])
            return None
        except Exception as e:
            print(f"⚠️  Embedding extraction failed: {e}")
            return None

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

    def evaluate_mode(
        self,
        video_path: str,
        mode: str,
        max_frames: int = 50,
        conf_threshold: float = 0.6,
    ) -> SecurityMetrics:
        """Evaluate security for a specific overlay mode."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        similarities: List[float] = []
        detected_after_lock = 0
        total_faces = 0

        frames_processed = 0

        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            # Detect faces in original
            results = self.detector.detect(frame, conf_threshold=conf_threshold)
            if not results:
                frames_processed += 1
                continue

            boxes = [[int(r.box.x), int(r.box.y), int(r.box.w), int(r.box.h)] for r in results]
            total_faces += len(boxes)

            # Lock faces
            locked_frame, payloads = self.locker.lock_faces(
                frame,
                boxes,
                overlay_mode=mode,
                head_ratio=0.7,
            )

            # Try to detect faces in locked frame
            locked_results = self.detector.detect(locked_frame, conf_threshold=conf_threshold)
            detected_after_lock += len(locked_results)

            # Extract embeddings from original vs locked faces
            for box in boxes:
                x, y, w, h = box
                
                # Expand box slightly for better face extraction
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                original_face = frame[y1:y2, x1:x2]
                locked_face = locked_frame[y1:y2, x1:x2]

                if original_face.size == 0 or locked_face.size == 0:
                    continue

                # Extract embeddings
                emb_original = self._extract_face_embedding(original_face)
                emb_locked = self._extract_face_embedding(locked_face)

                if emb_original is not None and emb_locked is not None:
                    similarity = self._cosine_similarity(emb_original, emb_locked)
                    similarities.append(similarity)

            frames_processed += 1

        cap.release()

        detection_rate = (detected_after_lock / total_faces * 100) if total_faces > 0 else 0.0

        return SecurityMetrics(
            mode=mode,
            avg_similarity=float(np.mean(similarities)) if similarities else 0.0,
            median_similarity=float(np.median(similarities)) if similarities else 0.0,
            max_similarity=float(np.max(similarities)) if similarities else 0.0,
            min_similarity=float(np.min(similarities)) if similarities else 0.0,
            faces_evaluated=len(similarities),
            detection_rate_after_lock=detection_rate,
        )

    def run_full_evaluation(
        self,
        video_path: str,
        max_frames: int = 50,
    ) -> List[SecurityMetrics]:
        """Run security evaluation for all modes."""
        modes = ["solid", "noise", "ciphernoise", "rps"]
        
        print(f"🔒 Starting security evaluation with {max_frames} frames per mode...")
        print(f"📹 Video: {video_path}")
        print(f"🧠 Embedding model: {self.embedding_model}\n")

        results: List[SecurityMetrics] = []

        for mode in modes:
            print(f"⏳ Evaluating mode: {mode}")
            result = self.evaluate_mode(video_path, mode, max_frames=max_frames)
            results.append(result)
            print(f"   ✓ Avg Similarity: {result.avg_similarity:.4f} | Detection Rate: {result.detection_rate_after_lock:.1f}%\n")

        return results

    def print_security_report(self, results: List[SecurityMetrics]) -> None:
        """Print formatted security evaluation report."""
        headers = [
            "Mode",
            "Avg Similarity",
            "Median Similarity",
            "Max Similarity",
            "Min Similarity",
            "Faces Evaluated",
            "Detection Rate (%)",
        ]
        
        table_data = [
            [
                r.mode,
                f"{r.avg_similarity:.4f}",
                f"{r.median_similarity:.4f}",
                f"{r.max_similarity:.4f}",
                f"{r.min_similarity:.4f}",
                r.faces_evaluated,
                f"{r.detection_rate_after_lock:.1f}%",
            ]
            for r in results
        ]

        print("\n" + "=" * 120)
        print("🔐 SECURITY EVALUATION REPORT")
        print("=" * 120)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("=" * 120)

        print("\n📊 Interpretation:")
        print("• Lower similarity scores = Better anonymization")
        print("• Lower detection rates = Better privacy protection")
        print("• Similarity < 0.3 = Strong anonymization")
        print("• Similarity 0.3-0.6 = Moderate anonymization")
        print("• Similarity > 0.6 = Weak anonymization")
        print()

        # Find best performer
        best_anonymization = min(results, key=lambda r: r.avg_similarity)
        lowest_detection = min(results, key=lambda r: r.detection_rate_after_lock)

        print(f"🏆 Best Anonymization: {best_anonymization.mode} (similarity: {best_anonymization.avg_similarity:.4f})")
        print(f"👻 Lowest Detection Rate: {lowest_detection.mode} ({lowest_detection.detection_rate_after_lock:.1f}%)")
        print()


def main():
    """Run security evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate face anonymization security")
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Video path or camera index",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames to evaluate per mode (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Facenet512",
        choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"],
        help="Face recognition model for embeddings",
    )

    args = parser.parse_args()

    video_source = int(args.video) if args.video.isdigit() else args.video

    evaluator = FaceEmbeddingSecurityEvaluator(embedding_model=args.model)
    results = evaluator.run_full_evaluation(video_source, max_frames=args.frames)
    evaluator.print_security_report(results)


if __name__ == "__main__":
    main()
