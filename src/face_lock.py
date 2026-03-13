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

    @staticmethod
    def _expand_head_box(box: List[int], width: int, height: int, head_ratio: float) -> List[int]:
        """Expand detected face box to include full head region (hair + ears + jaw)."""
        x, y, bw, bh = box
        r = max(0.0, head_ratio)
        # Build a portrait-style head box centered on the detected face.
        center_x = x + (bw / 2.0)
        center_y = y + (bh / 2.0)

        head_w = int(round(bw * (1.20 + 0.28 * r)))
        head_h = int(round(bh * (1.45 + 0.35 * r)))

        # Shift slightly upward to include hair while avoiding chest.
        x2 = int(round(center_x - (head_w / 2.0)))
        y2 = int(round(center_y - (head_h * 0.58)))

        x2 = max(0, min(width - 2, x2))
        y2 = max(0, min(height - 2, y2))
        w2 = max(2, min(width - x2, head_w))
        h2 = max(2, min(height - y2, head_h))

        return [x2, y2, w2, h2]

    def _crypto_seed(self, nonce: bytes, aad: bytes, ciphertext: bytes) -> int:
        digest = sha256(self._key + nonce + aad + ciphertext[:64]).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _reversible_pixel_shuffle(
        roi: np.ndarray,
        seed: int,
        tile_size: int,
        rounds: int,
    ) -> np.ndarray:
        """Vectorized reversible pixel shuffling with no Python for-loops."""
        h, w = roi.shape[:2]
        ts = max(2, int(tile_size))
        safe_rounds = max(1, int(rounds))

        tiles_h = h // ts
        tiles_w = w // ts
        if tiles_h < 2 or tiles_w < 2:
            return roi.copy()

        crop_h = tiles_h * ts
        crop_w = tiles_w * ts
        out = roi.copy()
        region = out[:crop_h, :crop_w].copy()

        blocks = region.reshape(tiles_h, ts, tiles_w, ts, 3).transpose(0, 2, 1, 3, 4)
        n_blocks = tiles_h * tiles_w
        flat = blocks.reshape(n_blocks, ts * ts, 3)

        rng = np.random.default_rng(seed + 8191 * safe_rounds)
        perm_blocks = rng.permutation(n_blocks)
        perm_pixels = rng.permutation(ts * ts)

        shuffled = flat[perm_blocks][:, perm_pixels, :]
        restored = shuffled.reshape(tiles_h, tiles_w, ts, ts, 3).transpose(0, 2, 1, 3, 4)
        out[:crop_h, :crop_w] = restored.reshape(crop_h, crop_w, 3)
        return out

    @staticmethod
    def _pixelate_overlay(roi: np.ndarray, block: int) -> np.ndarray:
        h, w = roi.shape[:2]
        block_size = max(2, block)
        down_w = max(1, w // block_size)
        down_h = max(1, h // block_size)
        small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _noise_overlay(roi: np.ndarray, seed: int, intensity: int, mix: float) -> np.ndarray:
        alpha = min(max(mix, 0.0), 1.0)
        gain = max(5, min(120, intensity))
        rng = np.random.default_rng(seed)
        noise = rng.integers(0, 256, size=roi.shape, dtype=np.uint8)
        gray = np.empty_like(roi)
        gray[:] = (40, 40, 40)
        mixed = cv2.addWeighted(gray, 1.0 - alpha, noise, alpha, 0.0)
        return cv2.GaussianBlur(mixed, (max(3, (gain // 2) * 2 + 1), max(3, (gain // 2) * 2 + 1)), 0)

    def lock_faces(
        self,
        frame: np.ndarray,
        boxes: List[List[int]],
        overlay_mode: str = "rps",
        overlay_pixel_block: int = 12,
        overlay_noise_intensity: int = 70,
        overlay_noise_mix: float = 0.9,
        head_ratio: float = 0.7,
        rps_tile_size: int = 8,
        rps_rounds: int = 2,
    ) -> Tuple[np.ndarray, List[EncryptedFaceRegion]]:
        if frame.size == 0 or not boxes:
            return frame, []

        locked = frame.copy()
        payloads: List[EncryptedFaceRegion] = []
        h, w = frame.shape[:2]

        for box in boxes:
            x, y, bw, bh = self._expand_head_box(box, w, h, head_ratio=head_ratio)
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

            seed = self._crypto_seed(nonce, aad, ciphertext)

            if overlay_mode == "solid":
                locked[y : y + bh, x : x + bw] = (35, 35, 35)
            elif overlay_mode == "noise":
                locked[y : y + bh, x : x + bw] = self._noise_overlay(
                    roi,
                    seed=seed,
                    intensity=overlay_noise_intensity,
                    mix=overlay_noise_mix,
                )
            elif overlay_mode == "ciphernoise":
                pix = self._reversible_pixel_shuffle(
                    roi,
                    seed=seed,
                    tile_size=rps_tile_size,
                    rounds=rps_rounds,
                )
                noise = self._noise_overlay(
                    roi,
                    seed=seed,
                    intensity=overlay_noise_intensity,
                    mix=overlay_noise_mix,
                )
                locked[y : y + bh, x : x + bw] = cv2.addWeighted(pix, 0.35, noise, 0.65, 0.0)
            elif overlay_mode == "rps":
                locked[y : y + bh, x : x + bw] = self._reversible_pixel_shuffle(
                    roi,
                    seed=seed,
                    tile_size=rps_tile_size,
                    rounds=rps_rounds,
                )
            else:
                locked[y : y + bh, x : x + bw] = self._reversible_pixel_shuffle(
                    roi,
                    seed=seed,
                    tile_size=rps_tile_size,
                    rounds=rps_rounds,
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
