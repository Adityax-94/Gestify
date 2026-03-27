"""
features.py
Converts raw MediaPipe 21-landmark output into a flat feature vector.

Feature vector (63 values):
  - 21 landmarks × 3 (x, y, z), normalized relative to wrist
  - All coordinates are scale-invariant (divided by hand bounding box size)
"""

import numpy as np


LANDMARK_COUNT = 21
FEATURE_DIM = LANDMARK_COUNT * 3  # 63


def extract(hand_landmarks) -> np.ndarray:
    """
    hand_landmarks: mediapipe NormalizedLandmarkList

    Returns a (63,) float32 array, ready to feed into the classifier.
    """
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )  # shape (21, 3)

    # Translate so wrist (landmark 0) is the origin
    pts -= pts[0]

    # Scale by the bounding-box diagonal so features are size-invariant
    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) + 1e-6
    pts /= bbox_diag

    return pts.flatten()  # (63,)


def finger_states(hand_landmarks) -> list[bool]:
    """
    Returns [thumb_up, index_up, middle_up, ring_up, pinky_up].
    "Up" means the fingertip is above its MCP joint (lower y = higher on screen).

    Useful for quick rule-based sanity checks alongside the ML model.
    """
    lm = hand_landmarks.landmark
    # Tip ids:   4  8  12  16  20
    # MCP ids:   2  5   9  13  17
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5,  9, 13, 17]
    return [lm[t].y < lm[m].y for t, m in zip(tips, mcps)]