from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.face_recognition import FaceRecognizer
from src.family_database import FamilyDatabase


def collect_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [str(p) for p in Path(folder).glob("**/*") if p.suffix.lower() in exts]
    paths.sort()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Register family member embeddings from image folder")
    parser.add_argument("--name", required=True, help="Display name, e.g. Khoa")
    parser.add_argument("--member-id", required=True, help="Stable member id, e.g. khoa")
    parser.add_argument("--image-dir", required=True, help="Folder containing face images")
    parser.add_argument("--db-path", default="data/family_members.json", help="Database json path")
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    parser.add_argument("--max-samples", type=int, default=25, help="Max valid samples to keep")
    parser.add_argument("--member-threshold", type=float, default=None, help="Optional per-member cosine threshold")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise RuntimeError(f"Image folder not found: {args.image_dir}")

    recognizer = FaceRecognizer(model=args.model)
    db = FamilyDatabase(db_path=args.db_path)

    image_paths = collect_images(args.image_dir)
    if not image_paths:
        raise RuntimeError("No images found for registration")

    valid_embeddings: List[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        emb = recognizer.extract_embedding_from_image(image)
        if emb is None:
            continue

        valid_embeddings.append(emb)
        if len(valid_embeddings) >= max(1, args.max_samples):
            break

    if not valid_embeddings:
        raise RuntimeError("Could not extract any valid face embedding from provided images")

    db.model_name = args.model
    db.embedding_dim = int(valid_embeddings[0].shape[0])
    db.upsert_member(args.member_id, args.name, valid_embeddings, threshold=args.member_threshold)
    db.save()

    member = db.get_member_by_id(args.member_id)
    saved_threshold = float(member.threshold) if member is not None else 0.35

    print(f"Registered: {args.name} ({args.member_id})")
    print(f"Embeddings: {len(valid_embeddings)}")
    print(f"Embedding dim: {db.embedding_dim}")
    print(f"Member threshold: {saved_threshold:.3f}")
    print(f"DB path: {args.db_path}")


if __name__ == "__main__":
    main()
