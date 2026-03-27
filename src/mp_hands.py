"""
mp_hands.py  —  Drop-in MediaPipe HandLandmarker wrapper
─────────────────────────────────────────────────────────
MediaPipe 0.10.x+ removed mp.solutions.hands entirely.
This module wraps the new HandLandmarker task-based API and exposes
the same interface that run.py, collect_data.py, and train.py expect:

    from mp_hands import HandDetector, draw_landmarks, HAND_CONNECTIONS

Usage (replaces the old mp.solutions.hands block):

    detector = HandDetector(max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.6)

    with detector:                          # or call detector.close() manually
        while True:
            ret, frame = cap.read()
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)  # returns a HandResult

            if result.multi_hand_landmarks:
                hl = result.multi_hand_landmarks[0]
                draw_landmarks(frame, hl, HAND_CONNECTIONS)
                # hl.landmark[i].x / .y / .z  — identical to the old API

Model file (~9 MB) is auto-downloaded to  models/hand_landmarker.task
on the very first run.  No manual steps needed.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

# ── Model path ────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_HERE, "..", "models")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(_MODEL_DIR, "hand_landmarker.task")


def _ensure_model() -> str:
    """Download the .task model file once if it doesn't exist yet."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.isfile(_MODEL_PATH):
        print(f"[mp_hands] Downloading HandLandmarker model (~9 MB) …")
        print(f"           → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[mp_hands] Download complete.")
    return _MODEL_PATH


# ── Landmark / connection types ───────────────────────────────────────────────

@dataclass
class Landmark:
    """Mirrors mediapipe.framework.formats.landmark_pb2.NormalizedLandmark."""
    x: float
    y: float
    z: float


@dataclass
class NormalizedLandmarkList:
    """Mirrors the old NormalizedLandmarkList so features.py stays unchanged."""
    landmark: List[Landmark] = field(default_factory=list)


@dataclass
class HandResult:
    """
    Mirrors the object returned by the old Hands.process().
    multi_hand_landmarks is None when no hand is detected, or a list of
    NormalizedLandmarkList (one per detected hand).
    """
    multi_hand_landmarks: Optional[List[NormalizedLandmarkList]] = None


# ── Hand connections (same 21 pairs as the old API) ───────────────────────────
HAND_CONNECTIONS: frozenset = frozenset({
    # Palm
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle bar
    (5, 9), (9, 13), (13, 17),
})


# ── Drawing helper ────────────────────────────────────────────────────────────

def draw_landmarks(
    image: np.ndarray,
    hand_landmarks: NormalizedLandmarkList,
    connections: frozenset | None = None,
    landmark_spec: dict | None = None,
    connection_spec: dict | None = None,
) -> None:
    """
    Draw hand skeleton on *image* (BGR, in-place).

    landmark_spec / connection_spec accept the same keyword-style dicts as
    the old mp.solutions.drawing_utils.DrawingSpec:
        {"color": (B, G, R), "thickness": 2, "circle_radius": 3}
    """
    h, w = image.shape[:2]

    lm_color       = (80, 80, 220)
    lm_radius      = 3
    conn_color     = (160, 160, 255)
    conn_thickness = 1

    if landmark_spec:
        lm_color   = landmark_spec.get("color",         lm_color)
        lm_radius  = landmark_spec.get("circle_radius", lm_radius)
    if connection_spec:
        conn_color     = connection_spec.get("color",     conn_color)
        conn_thickness = connection_spec.get("thickness", conn_thickness)

    if connections:
        for (a, b) in connections:
            lm_a = hand_landmarks.landmark[a]
            lm_b = hand_landmarks.landmark[b]
            pt_a = (int(lm_a.x * w), int(lm_a.y * h))
            pt_b = (int(lm_b.x * w), int(lm_b.y * h))
            cv2.line(image, pt_a, pt_b, conn_color, conn_thickness)

    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), lm_radius, lm_color, -1)


# ── Main detector class ───────────────────────────────────────────────────────

class HandDetector:
    """
    Wraps mediapipe.tasks.python.vision.HandLandmarker with a
    mp.solutions.hands-compatible interface.

    Parameters
    ----------
    max_num_hands            : int   (default 1)
    min_detection_confidence : float (default 0.5)
    min_tracking_confidence  : float (default 0.5)
    static_image_mode        : bool  (default False)
        When True, detection runs on every frame (no tracking).
        When False, tracking is used after the first detection.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model()

        RunningMode = mp_vision.RunningMode
        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._running_mode = running_mode
        self._RunningMode = RunningMode
        self._frame_ts_ms = 0  # monotonically increasing timestamp for VIDEO mode

    # ── Context-manager support ───────────────────────────────────────────
    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    # ── Core method ───────────────────────────────────────────────────────
    def process(self, rgb_frame: np.ndarray) -> HandResult:
        """
        Detect/track hands in *rgb_frame* (H×W×3, uint8, RGB).

        Returns a HandResult whose .multi_hand_landmarks mirrors the old API.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._running_mode == self._RunningMode.IMAGE:
            detection = self._landmarker.detect(mp_image)
        else:
            self._frame_ts_ms += 1          # simple incrementing timestamp
            detection = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not detection.hand_landmarks:
            return HandResult(multi_hand_landmarks=None)

        hand_list: List[NormalizedLandmarkList] = []
        for hand in detection.hand_landmarks:
            nll = NormalizedLandmarkList(
                landmark=[Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand]
            )
            hand_list.append(nll)

        return HandResult(multi_hand_landmarks=hand_list)
