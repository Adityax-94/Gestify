"""
run.py  —  Phase 4: real-time gesture controller
──────────────────────────────────────────────────
Run:
    python src/run.py

Keys:
    Q  →  quit
    P  →  pause / resume action dispatch (detection still shown)
"""

import os
import sys
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import extract
from actions import fire, DESCRIPTIONS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLF_PATH  = os.path.join(MODEL_DIR, "classifier.joblib")
ENC_PATH  = os.path.join(MODEL_DIR, "label_encoder.joblib")

CONFIDENCE_THRESHOLD = 0.75   # ignore predictions below this
SMOOTHING_WINDOW     = 7      # frames — majority vote over this window


# ── Color palette ─────────────────────────────────────────────────────────
COLORS = {
    "open_palm":   (60, 200, 80),
    "fist":        (60, 80, 220),
    "index_up":    (200, 160, 40),
    "two_fingers": (200, 120, 40),
    "pinch":       (180, 60, 200),
    "cursor_mode": (40, 200, 200),
    "none":        (120, 120, 120),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, pos, scale=0.7, color=(240, 240, 240), thickness=2):
    cv2.putText(img, text, pos, FONT, scale, (20, 20, 20), thickness + 2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)


def draw_overlay(frame, gesture, confidence, fps, paused, history):
    h, w = frame.shape[:2]
    color = COLORS.get(gesture, COLORS["none"])

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 72), (18, 18, 18), -1)

    # Gesture name + description
    label = DESCRIPTIONS.get(gesture, gesture) if gesture else "No hand"
    put(frame, label, (14, 32), scale=0.85, color=color, thickness=2)
    if gesture:
        conf_text = f"{confidence*100:.0f}% confidence"
        put(frame, conf_text, (14, 58), scale=0.55, color=(160, 160, 160), thickness=1)

    # FPS + pause indicator
    fps_text = f"FPS {fps:.0f}"
    put(frame, fps_text, (w - 90, 32), scale=0.6, color=(160, 160, 160), thickness=1)
    if paused:
        put(frame, "PAUSED", (w - 90, 58), scale=0.55, color=(80, 80, 220), thickness=1)

    # Confidence bar
    if gesture and confidence > 0:
        bar_w = int((w - 28) * min(confidence, 1.0))
        cv2.rectangle(frame, (14, 66), (14 + bar_w, 70), color, -1)
        cv2.rectangle(frame, (14, 66), (w - 14, 70), (60, 60, 60), 1)

    # History mini-chart (last N gestures as colored dots)
    dot_r = 6
    dot_y = h - 20
    for i, g in enumerate(list(history)[-20:]):
        cx = 14 + i * (dot_r * 2 + 4)
        c  = COLORS.get(g, COLORS["none"])
        cv2.circle(frame, (cx, dot_y), dot_r, c, -1)

    # Bottom hint
    put(frame, "Q: quit    P: pause", (14, h - 8),
        scale=0.45, color=(100, 100, 100), thickness=1)

    # Border flash on detection
    if gesture:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 2)

    return frame


def majority_vote(window: deque) -> str | None:
    if not window:
        return None
    counts: dict[str, int] = {}
    for g in window:
        counts[g] = counts.get(g, 0) + 1
    winner = max(counts, key=counts.__getitem__)
    return winner if counts[winner] >= len(window) // 2 + 1 else None


def palm_center(hand_landmarks) -> tuple[float, float]:
    """Return (x, y) of the palm center (average of wrist + MCP joints)."""
    palm_ids = [0, 1, 5, 9, 13, 17]
    xs = [hand_landmarks.landmark[i].x for i in palm_ids]
    ys = [hand_landmarks.landmark[i].y for i in palm_ids]
    return float(np.mean(xs)), float(np.mean(ys))


def main():
    # ── Load model ──────────────────────────────────────────────────────
    if not os.path.exists(CLF_PATH):
        print(f"ERROR: Model not found at {CLF_PATH}")
        print("Run  python src/train.py  first.")
        sys.exit(1)

    print("Loading model …")
    pipeline = joblib.load(CLF_PATH)
    le       = joblib.load(ENC_PATH)
    print(f"  Classes: {list(le.classes_)}\n")

    # ── MediaPipe ────────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # ── Webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth_window = deque(maxlen=SMOOTHING_WINDOW)
    history       = deque(maxlen=60)
    paused        = False
    prev_time     = time.time()

    print("Gesture controller running. Press Q to quit, P to pause.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture    = None
        confidence = 0.0
        hand_x     = None
        hand_y     = None

        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(80, 80, 220), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(160, 160, 255), thickness=1),
            )

            feat   = extract(hl).reshape(1, -1)
            proba  = pipeline.predict_proba(feat)[0]
            idx    = int(np.argmax(proba))
            conf   = float(proba[idx])
            label  = le.classes_[idx]

            hand_x, hand_y = palm_center(hl)

            if conf >= CONFIDENCE_THRESHOLD:
                smooth_window.append(label)
                voted = majority_vote(smooth_window)
                if voted:
                    gesture    = voted
                    confidence = conf
                    history.append(gesture)

                    if not paused:
                        fire(gesture, hand_x=hand_x, hand_y=hand_y)
            else:
                smooth_window.clear()
        else:
            smooth_window.clear()

        # FPS
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = draw_overlay(frame, gesture, confidence, fps, paused, history)
        cv2.imshow("Hand Gesture Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("  Paused" if paused else "  Resumed")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()
