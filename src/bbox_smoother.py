"""
BBox Smoothing Module
Provides temporal smoothing for bounding boxes to reduce jitter in video streams.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TrackedBox:
    """A tracked bounding box with history."""
    box: List[int]  # [x, y, w, h]
    confidence: float
    last_seen_frame: int
    history: deque  # History of boxes for smoothing


class BBoxSmoother:
    """
    Temporal smoothing for bounding boxes using exponential moving average (EMA).
    
    Reduces jitter and provides stable bounding boxes across video frames.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        history_size: int = 5,
        iou_threshold: float = 0.3,
        max_frames_missing: int = 10,
    ):
        """
        Initialize BBox smoother.
        
        Args:
            alpha: EMA smoothing factor (0-1). Higher = more responsive, lower = smoother
            history_size: Number of frames to keep in history
            iou_threshold: IoU threshold for matching boxes across frames
            max_frames_missing: Max frames a track can be missing before deletion
        """
        self.alpha = max(0.1, min(1.0, alpha))
        self.history_size = max(1, history_size)
        self.iou_threshold = max(0.1, min(0.9, iou_threshold))
        self.max_frames_missing = max_frames_missing
        
        self.tracked_boxes: Dict[int, TrackedBox] = {}
        self.next_id = 0
        self.frame_count = 0

    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute Intersection over Union (IoU) between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Compute intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _match_boxes(
        self,
        detections: List[Tuple[List[int], float]],
    ) -> Dict[int, Tuple[List[int], float]]:
        """Match detected boxes to tracked boxes using IoU."""
        matched: Dict[int, Tuple[List[int], float]] = {}
        unmatched_detections = list(range(len(detections)))
        
        # Try to match each tracked box to a detection
        for track_id, tracked in self.tracked_boxes.items():
            best_iou = 0.0
            best_idx = -1
            
            for det_idx in unmatched_detections:
                det_box, _ = detections[det_idx]
                iou = self._compute_iou(tracked.box, det_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = det_idx
            
            if best_iou >= self.iou_threshold:
                matched[track_id] = detections[best_idx]
                unmatched_detections.remove(best_idx)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det_box, conf = detections[det_idx]
            matched[self.next_id] = (det_box, conf)
            self.next_id += 1
        
        return matched

    def _ema_smooth(self, current: List[int], history: deque) -> List[int]:
        """Apply exponential moving average smoothing."""
        if not history:
            return current
        
        # Convert to numpy for easier computation
        current_arr = np.array(current, dtype=np.float32)
        
        # Compute EMA from history
        smoothed = current_arr
        weight = self.alpha
        
        for past_box in reversed(list(history)):
            past_arr = np.array(past_box, dtype=np.float32)
            smoothed = weight * smoothed + (1 - weight) * past_arr
            weight *= self.alpha  # Decay weight for older frames
        
        return [int(round(v)) for v in smoothed]

    def update(
        self,
        detections: List[Tuple[List[int], float]],
    ) -> List[List[int]]:
        """
        Update tracker with new detections and return smoothed boxes.
        
        Args:
            detections: List of (box, confidence) tuples where box is [x, y, w, h]
        
        Returns:
            List of smoothed boxes [x, y, w, h]
        """
        self.frame_count += 1
        
        # Match detections to tracked boxes
        matched = self._match_boxes(detections)
        
        # Update tracked boxes
        for track_id, (det_box, conf) in matched.items():
            if track_id in self.tracked_boxes:
                # Update existing track
                tracked = self.tracked_boxes[track_id]
                
                # Add current box to history
                tracked.history.append(tracked.box)
                
                # Apply smoothing
                smoothed_box = self._ema_smooth(det_box, tracked.history)
                
                # Update track
                tracked.box = smoothed_box
                tracked.confidence = conf
                tracked.last_seen_frame = self.frame_count
            else:
                # Create new track
                self.tracked_boxes[track_id] = TrackedBox(
                    box=det_box,
                    confidence=conf,
                    last_seen_frame=self.frame_count,
                    history=deque(maxlen=self.history_size),
                )
        
        # Remove old tracks
        to_remove = []
        for track_id, tracked in self.tracked_boxes.items():
            if self.frame_count - tracked.last_seen_frame > self.max_frames_missing:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracked_boxes[track_id]
        
        # Return smoothed boxes
        return [tracked.box for tracked in self.tracked_boxes.values()]

    def reset(self):
        """Reset tracker state."""
        self.tracked_boxes.clear()
        self.next_id = 0
        self.frame_count = 0


class KalmanBBoxSmoother:
    """
    Advanced BBox smoothing using Kalman filter.
    
    More sophisticated than EMA, handles velocity prediction and occlusions better.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        iou_threshold: float = 0.3,
        max_frames_missing: int = 10,
    ):
        """
        Initialize Kalman filter based smoother.
        
        Args:
            process_noise: Process noise covariance (lower = smoother)
            measurement_noise: Measurement noise covariance (lower = trust measurements more)
            iou_threshold: IoU threshold for matching
            max_frames_missing: Max frames before deleting track
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        
        self.kalman_filters: Dict[int, Dict] = {}
        self.next_id = 0
        self.frame_count = 0

    def _init_kalman(self, box: List[int]) -> Dict:
        """Initialize Kalman filter state for a box."""
        x, y, w, h = box
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        state = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # Covariance matrix
        P = np.eye(8, dtype=np.float32) * 1000
        
        return {
            'state': state,
            'P': P,
            'last_measurement': box,
            'last_seen_frame': self.frame_count,
        }

    def _predict(self, kf: Dict) -> np.ndarray:
        """Kalman prediction step."""
        # State transition matrix (constant velocity model)
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = 1  # x += vx
        F[1, 5] = 1  # y += vy
        F[2, 6] = 1  # w += vw
        F[3, 7] = 1  # h += vh
        
        # Predict state
        kf['state'] = F @ kf['state']
        
        # Predict covariance
        Q = np.eye(8, dtype=np.float32) * self.process_noise
        kf['P'] = F @ kf['P'] @ F.T + Q
        
        return kf['state'][:4]  # Return [x, y, w, h]

    def _update(self, kf: Dict, measurement: List[int]):
        """Kalman update step."""
        # Measurement matrix (observe [x, y, w, h])
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1
        
        # Measurement noise
        R = np.eye(4, dtype=np.float32) * self.measurement_noise
        
        # Innovation
        z = np.array(measurement, dtype=np.float32)
        y = z - H @ kf['state']
        
        # Innovation covariance
        S = H @ kf['P'] @ H.T + R
        
        # Kalman gain
        K = kf['P'] @ H.T @ np.linalg.inv(S)
        
        # Update state
        kf['state'] = kf['state'] + K @ y
        
        # Update covariance
        kf['P'] = (np.eye(8) - K @ H) @ kf['P']
        
        kf['last_measurement'] = measurement
        kf['last_seen_frame'] = self.frame_count

    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def update(self, detections: List[Tuple[List[int], float]]) -> List[List[int]]:
        """Update with new detections and return smoothed boxes."""
        self.frame_count += 1
        
        # Predict all tracks
        predictions = {}
        for track_id, kf in self.kalman_filters.items():
            pred = self._predict(kf)
            predictions[track_id] = [int(round(v)) for v in pred]
        
        # Match detections to predictions
        matched = {}
        unmatched_detections = list(range(len(detections)))
        
        for track_id, pred_box in predictions.items():
            best_iou = 0.0
            best_idx = -1
            
            for det_idx in unmatched_detections:
                det_box, _ = detections[det_idx]
                iou = self._compute_iou(pred_box, det_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = det_idx
            
            if best_iou >= self.iou_threshold:
                matched[track_id] = detections[best_idx]
                unmatched_detections.remove(best_idx)
        
        # Update matched tracks
        for track_id, (det_box, _) in matched.items():
            self._update(self.kalman_filters[track_id], det_box)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det_box, _ = detections[det_idx]
            self.kalman_filters[self.next_id] = self._init_kalman(det_box)
            self.next_id += 1
        
        # Remove old tracks
        to_remove = []
        for track_id, kf in self.kalman_filters.items():
            if self.frame_count - kf['last_seen_frame'] > self.max_frames_missing:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.kalman_filters[track_id]
        
        # Return smoothed boxes
        result = []
        for kf in self.kalman_filters.values():
            state = kf['state'][:4]
            result.append([int(round(v)) for v in state])
        
        return result

    def reset(self):
        """Reset tracker state."""
        self.kalman_filters.clear()
        self.next_id = 0
        self.frame_count = 0
