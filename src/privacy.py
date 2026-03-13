from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _clamp_box(box: List[int], width: int, height: int) -> List[int]:
    x, y, w, h = box
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(0, min(width - x, w))
    h = max(0, min(height - y, h))
    return [x, y, w, h]


def _expand_box(box: List[int], width: int, height: int, padding_ratio: float) -> List[int]:
    x, y, w, h = box
    pad_x = int(round(w * max(padding_ratio, 0.0)))
    pad_y = int(round(h * max(padding_ratio, 0.0)))
    expanded = [x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y]
    return _clamp_box(expanded, width, height)


def _expand_head_box(box: List[int], width: int, height: int, head_ratio: float) -> List[int]:
    """Expand face box to include head contour cues (hair, ears, jawline)."""
    x, y, w, h = box
    r = max(0.0, head_ratio)
    pad_left = int(round(w * (0.35 + 0.35 * r)))
    pad_right = int(round(w * (0.35 + 0.35 * r)))
    pad_top = int(round(h * (0.70 + 0.50 * r)))
    pad_bottom = int(round(h * (0.25 + 0.25 * r)))
    expanded = [x - pad_left, y - pad_top, w + pad_left + pad_right, h + pad_top + pad_bottom]
    return _clamp_box(expanded, width, height)


def _expand_silhouette_box(box: List[int], width: int, height: int, silhouette_ratio: float) -> List[int]:
    """Expand to full head-and-shoulders region to hide global identity cues."""
    x, y, w, h = box
    r = max(0.0, silhouette_ratio)
    pad_left = int(round(w * (0.75 + 0.50 * r)))
    pad_right = int(round(w * (0.75 + 0.50 * r)))
    pad_top = int(round(h * (0.95 + 0.65 * r)))
    pad_bottom = int(round(h * (1.25 + 0.95 * r)))
    expanded = [x - pad_left, y - pad_top, w + pad_left + pad_right, h + pad_top + pad_bottom]
    return _clamp_box(expanded, width, height)


def _odd(value: int) -> int:
    v = max(1, value)
    return v if v % 2 == 1 else v + 1


def _fast_gaussian_blur(roi: np.ndarray, blur_scale: float, blur_kernel: int) -> np.ndarray:
    h, w = roi.shape[:2]
    scale = min(max(blur_scale, 0.05), 1.0)
    down_w = max(1, int(round(w * scale)))
    down_h = max(1, int(round(h * scale)))

    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (_odd(blur_kernel), _odd(blur_kernel)), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _pixelate(roi: np.ndarray, pixel_block: int) -> np.ndarray:
    h, w = roi.shape[:2]
    block = max(2, pixel_block)
    down_w = max(1, w // block)
    down_h = max(1, h // block)

    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _solid_mask(roi: np.ndarray) -> np.ndarray:
    masked = np.empty_like(roi)
    masked[:] = (40, 40, 40)
    return masked


def _noise_mask(roi: np.ndarray, seed: int, intensity: int = 28) -> np.ndarray:
    """Create low-detail noisy mask to remove person-specific appearance cues."""
    h, w = roi.shape[:2]
    base = np.empty_like(roi)
    base[:] = (65, 65, 65)

    rng = np.random.default_rng(seed)
    noise = rng.integers(-intensity, intensity + 1, size=(h, w, 3), dtype=np.int16)
    mixed = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(mixed, (_odd(21), _odd(21)), 0)


def _head_cloak(roi: np.ndarray, seed: int) -> np.ndarray:
    """Strong anonymization for close-range shots: flat + noisy + blurred cloak."""
    h, w = roi.shape[:2]
    if h < 4 or w < 4:
        return _solid_mask(roi)

    # Mix a very low-detail version with noise to suppress all biometric textures.
    tiny = cv2.resize(roi, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    noise_layer = _noise_mask(roi, seed=seed, intensity=34)
    cloaked = cv2.addWeighted(coarse, 0.15, noise_layer, 0.85, 0.0)
    return cv2.GaussianBlur(cloaked, (_odd(35), _odd(35)), 0)


def _silhouette_cloak(roi: np.ndarray, seed: int) -> np.ndarray:
    """Strongest anonymization: remove face and upper-body signatures."""
    h, w = roi.shape[:2]
    if h < 8 or w < 8:
        return _solid_mask(roi)

    tiny = cv2.resize(roi, (max(1, w // 40), max(1, h // 40)), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)

    seed_map = np.full((h, w, 3), seed % 255, dtype=np.uint8)
    base_tone = cv2.addWeighted(coarse, 0.08, seed_map, 0.92, 0.0)
    noise_layer = _noise_mask(roi, seed=seed + 7919, intensity=38)
    cloaked = cv2.addWeighted(base_tone, 0.35, noise_layer, 0.65, 0.0)
    cloaked = cv2.GaussianBlur(cloaked, (_odd(45), _odd(45)), 0)
    return _quantize_colors(cloaked, levels=6)


def _quantize_colors(roi: np.ndarray, levels: int) -> np.ndarray:
    lv = max(2, levels)
    step = max(1, 256 // lv)
    quantized = (roi // step) * step
    return quantized.astype(np.uint8)


def _scramble_blocks(roi: np.ndarray, block_size: int, seed: int) -> np.ndarray:
    h, w = roi.shape[:2]
    bs = max(4, block_size)

    nbh = h // bs
    nbw = w // bs
    if nbh < 2 or nbw < 2:
        return roi

    cropped_h = nbh * bs
    cropped_w = nbw * bs

    work = roi[:cropped_h, :cropped_w].copy()
    blocks = work.reshape(nbh, bs, nbw, bs, 3).transpose(0, 2, 1, 3, 4)
    flat = blocks.reshape(nbh * nbw, bs, bs, 3)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(flat.shape[0])
    shuffled = flat[perm]

    restored = shuffled.reshape(nbh, nbw, bs, bs, 3).transpose(0, 2, 1, 3, 4)
    work = restored.reshape(cropped_h, cropped_w, 3)

    output = roi.copy()
    output[:cropped_h, :cropped_w] = work
    return output


def _obliterate_features(
    roi: np.ndarray,
    obliterate_scale: float,
    blur_kernel: int,
    scramble_block: int,
    palette_levels: int,
    seed: int,
) -> np.ndarray:
    """Irreversible anonymization: destroy facial textures and geometry cues."""
    h, w = roi.shape[:2]
    scale = min(max(obliterate_scale, 0.03), 0.25)
    down_w = max(1, int(round(w * scale)))
    down_h = max(1, int(round(h * scale)))

    # Step 1: aggressively compress facial details.
    work = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_AREA)
    work = cv2.resize(work, (w, h), interpolation=cv2.INTER_NEAREST)

    # Step 2: shuffle local blocks to break geometric relationships.
    work = _scramble_blocks(work, block_size=scramble_block, seed=seed)

    # Step 3: smooth remaining high-frequency edges.
    work = cv2.GaussianBlur(work, (_odd(max(blur_kernel, 35)), _odd(max(blur_kernel, 35))), 0)

    # Step 4: reduce color information so subtle textures are removed.
    work = _quantize_colors(work, levels=palette_levels)
    return work


def anonymize_faces(
    frame: np.ndarray,
    boxes: List[List[int]],
    mode: str = "blur",
    blur_scale: float = 0.2,
    blur_kernel: int = 31,
    pixel_block: int = 16,
    padding_ratio: float = 0.12,
    obliterate_scale: float = 0.08,
    scramble_block: int = 12,
    palette_levels: int = 10,
    scramble_seed: int = 1337,
    head_ratio: float = 0.6,
    silhouette_ratio: float = 0.8,
) -> np.ndarray:
    """Apply anonymization to detected face boxes in-place and return frame."""
    if frame.size == 0 or not boxes or mode == "none":
        return frame

    height, width = frame.shape[:2]

    for box in boxes:
        if mode == "headcloak":
            x, y, w, h = _expand_head_box(box, width, height, head_ratio=head_ratio)
        elif mode == "silhouette":
            x, y, w, h = _expand_silhouette_box(box, width, height, silhouette_ratio=silhouette_ratio)
        else:
            x, y, w, h = _expand_box(box, width, height, padding_ratio)
        if w <= 1 or h <= 1:
            continue

        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        if mode == "blur":
            processed = _fast_gaussian_blur(roi, blur_scale=blur_scale, blur_kernel=blur_kernel)
        elif mode == "pixelate":
            processed = _pixelate(roi, pixel_block=pixel_block)
        elif mode == "solid":
            processed = _solid_mask(roi)
        elif mode == "obliterate":
            processed = _obliterate_features(
                roi,
                obliterate_scale=obliterate_scale,
                blur_kernel=blur_kernel,
                scramble_block=scramble_block,
                palette_levels=palette_levels,
                seed=scramble_seed + x * 31 + y * 17 + w * 13 + h * 7,
            )
        elif mode == "headcloak":
            processed = _head_cloak(roi, seed=scramble_seed + x * 19 + y * 23 + w * 29 + h * 31)
        elif mode == "silhouette":
            processed = _silhouette_cloak(roi, seed=scramble_seed + x * 37 + y * 41 + w * 43 + h * 47)
        else:
            processed = roi

        frame[y : y + h, x : x + w] = processed

    return frame
