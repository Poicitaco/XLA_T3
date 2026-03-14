from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np

try:
    from src.face_recognition import FaceRecognizer
except ImportError:
    from face_recognition import FaceRecognizer


@dataclass
class FamilyMember:
    member_id: str
    name: str
    embeddings: List[List[float]]
    threshold: float = 0.35


class FamilyDatabase:
    def __init__(self, db_path: str = "data/family_members.json") -> None:
        self.db_path = db_path
        self.members: List[FamilyMember] = []
        self.model_name = "unknown"
        self.embedding_dim = 0
        self._emb_matrix = np.empty((0, 0), dtype=np.float32)
        self._member_ids: List[str] = []
        self._member_names: List[str] = []
        self._member_thresholds = np.empty((0,), dtype=np.float32)
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.db_path):
            self.members = []
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.model_name = payload.get("model_name", "unknown")
        self.embedding_dim = int(payload.get("embedding_dim", 0))

        members = payload.get("members", [])
        self.members = [
            FamilyMember(
                member_id=m["member_id"],
                name=m["name"],
                embeddings=m.get("embeddings", []),
                threshold=float(m.get("threshold", 0.35)),
            )
            for m in members
        ]
        self._rebuild_index()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "members": [asdict(m) for m in self.members],
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def upsert_member(
        self,
        member_id: str,
        name: str,
        embeddings: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> None:
        serialized = [emb.astype(np.float32).tolist() for emb in embeddings]

        existing = None
        for member in self.members:
            if member.member_id == member_id:
                existing = member
                break

        if existing is None:
            self.members.append(
                FamilyMember(
                    member_id=member_id,
                    name=name,
                    embeddings=serialized,
                    threshold=0.35 if threshold is None else float(threshold),
                )
            )
        else:
            existing.name = name
            existing.embeddings = serialized
            if threshold is not None:
                existing.threshold = float(threshold)

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        flat_embeddings: List[np.ndarray] = []
        member_ids: List[str] = []
        member_names: List[str] = []
        member_thresholds: List[float] = []

        for member in self.members:
            threshold = float(member.threshold if member.threshold is not None else 0.35)
            for emb_list in member.embeddings:
                emb = np.asarray(emb_list, dtype=np.float32)
                if emb.ndim != 1:
                    continue
                norm = float(np.linalg.norm(emb))
                if norm <= 0.0:
                    continue
                flat_embeddings.append(emb / norm)
                member_ids.append(member.member_id)
                member_names.append(member.name)
                member_thresholds.append(threshold)

        if not flat_embeddings:
            self._emb_matrix = np.empty((0, 0), dtype=np.float32)
            self._member_ids = []
            self._member_names = []
            self._member_thresholds = np.empty((0,), dtype=np.float32)
            return

        self._emb_matrix = np.stack(flat_embeddings).astype(np.float32)
        self._member_ids = member_ids
        self._member_names = member_names
        self._member_thresholds = np.asarray(member_thresholds, dtype=np.float32)

    def get_member_by_id(self, member_id: str) -> Optional[FamilyMember]:
        for member in self.members:
            if member.member_id == member_id:
                return member
        return None

    def match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.35,
    ) -> Optional[Tuple[str, str, float]]:
        if self._emb_matrix.size == 0:
            return None

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1:
            return None

        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0.0:
            return None

        q = q / q_norm
        scores = self._emb_matrix @ q
        thresholds = np.maximum(self._member_thresholds, float(threshold))
        valid = scores >= thresholds
        if not np.any(valid):
            return None

        masked_scores = np.where(valid, scores, -np.inf)
        best_idx = int(np.argmax(masked_scores))
        best_score = float(masked_scores[best_idx])
        if not np.isfinite(best_score):
            return None

        return self._member_ids[best_idx], self._member_names[best_idx], best_score
