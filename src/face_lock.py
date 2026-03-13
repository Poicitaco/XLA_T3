from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


@dataclass
class EncryptedFaceRegion:
    box: List[int]
    nonce_b64: str
    ciphertext_b64: str
    shape: List[int]


class FaceRegionLocker:
    """Reversible face lock with passphrase-based AES-GCM encryption."""

    def __init__(self, passphrase: str, salt: bytes | None = None) -> None:
        if not passphrase:
            raise ValueError("Passphrase must not be empty.")
        self._salt = salt if salt is not None else b"xla_demo_face_lock_v1"
        self._key = self._derive_key(passphrase)

    def _derive_key(self, passphrase: str) -> bytes:
        kdf = Scrypt(
            salt=self._salt,
            length=32,
            n=2**14,
            r=8,
            p=1,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    def _aad(self, box: List[int], shape: Tuple[int, int, int]) -> bytes:
        return f"{box[0]}:{box[1]}:{box[2]}:{box[3]}:{shape[0]}:{shape[1]}:{shape[2]}".encode("utf-8")

    def lock_faces(
        self,
        frame: np.ndarray,
        boxes: List[List[int]],
        overlay_mode: str = "pixelate",
    ) -> Tuple[np.ndarray, List[EncryptedFaceRegion]]:
        if frame.size == 0 or not boxes:
            return frame, []

        locked = frame.copy()
        payloads: List[EncryptedFaceRegion] = []
        h, w = frame.shape[:2]

        for box in boxes:
            x, y, bw, bh = box
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            bw = max(0, min(w - x, bw))
            bh = max(0, min(h - y, bh))
            if bw <= 1 or bh <= 1:
                continue

            roi = frame[y : y + bh, x : x + bw]
            if roi.size == 0:
                continue

            nonce = os.urandom(12)
            aad = self._aad([x, y, bw, bh], roi.shape)
            ciphertext = AESGCM(self._key).encrypt(nonce, roi.tobytes(), aad)

            payloads.append(
                EncryptedFaceRegion(
                    box=[x, y, bw, bh],
                    nonce_b64=base64.b64encode(nonce).decode("utf-8"),
                    ciphertext_b64=base64.b64encode(ciphertext).decode("utf-8"),
                    shape=[int(roi.shape[0]), int(roi.shape[1]), int(roi.shape[2])],
                )
            )

            if overlay_mode == "solid":
                locked[y : y + bh, x : x + bw] = (35, 35, 35)
            elif overlay_mode == "noise":
                digest = sha256(ciphertext).digest()
                seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
                rng = np.random.default_rng(seed)
                noise = rng.integers(0, 256, size=roi.shape, dtype=np.uint8)
                locked[y : y + bh, x : x + bw] = noise
            else:
                # Fast pixelated lock overlay for realtime preview.
                down_w = max(1, bw // 12)
                down_h = max(1, bh // 12)
                small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
                locked[y : y + bh, x : x + bw] = cv2.resize(
                    small,
                    (bw, bh),
                    interpolation=cv2.INTER_NEAREST,
                )

        return locked, payloads

    def unlock_faces(self, frame: np.ndarray, payloads: List[EncryptedFaceRegion]) -> np.ndarray:
        if frame.size == 0 or not payloads:
            return frame

        restored = frame.copy()
        h, w = frame.shape[:2]

        for item in payloads:
            x, y, bw, bh = item.box
            if x < 0 or y < 0 or x + bw > w or y + bh > h:
                continue

            nonce = base64.b64decode(item.nonce_b64.encode("utf-8"))
            ciphertext = base64.b64decode(item.ciphertext_b64.encode("utf-8"))
            shape = (item.shape[0], item.shape[1], item.shape[2])
            aad = self._aad(item.box, shape)

            plaintext = AESGCM(self._key).decrypt(nonce, ciphertext, aad)
            roi = np.frombuffer(plaintext, dtype=np.uint8).reshape(shape)
            restored[y : y + bh, x : x + bw] = roi

        return restored

    @staticmethod
    def payloads_to_jsonable(payloads: List[EncryptedFaceRegion]) -> List[Dict[str, Any]]:
        return [
            {
                "box": item.box,
                "nonce_b64": item.nonce_b64,
                "ciphertext_b64": item.ciphertext_b64,
                "shape": item.shape,
            }
            for item in payloads
        ]

    @staticmethod
    def payloads_from_jsonable(data: List[Dict[str, Any]]) -> List[EncryptedFaceRegion]:
        return [
            EncryptedFaceRegion(
                box=list(item["box"]),
                nonce_b64=str(item["nonce_b64"]),
                ciphertext_b64=str(item["ciphertext_b64"]),
                shape=list(item["shape"]),
            )
            for item in data
        ]

    @staticmethod
    def dumps_payloads(payloads: List[EncryptedFaceRegion]) -> str:
        return json.dumps(FaceRegionLocker.payloads_to_jsonable(payloads), ensure_ascii=True)
