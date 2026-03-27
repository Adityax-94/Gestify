"""
collect_data.py  —  Phase 1 + 2: gather training samples
─────────────────────────────────────────────────────────
Run:
    python src/collect_data.py

Controls (shown on screen):
    SPACE  →  start / stop recording for the current gesture
    N      →  next gesture
    Q      →  quit and save

Saves data/gestures.csv  (label, f0, f1, … f62)
"""

import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import extract

# ── Gesture labels (order matters — used as class indices) ──────────────────
GESTURES = [
    "open_palm",     # play / volume up
    "fist",          # mute / pause
    "index_up",      # scroll up
    "two_fingers",   # scroll down
    "pinch",         # volume down
    "cursor_mode",   # index+middle spread → mouse control
]

SAMPLES_PER_GESTURE = 200   # aim for at least 150 good frames per class
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gestures.csv")

# ── MediaPipe setup ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── UI helpers ───────────────────────────────────────────────────────────────
FONT      = cv2.FONT_HERSHEY_SIMPLEX
GREEN     = (60, 200, 80)
RED       = (60, 60, 220)
WHITE     = (240, 240, 240)
DARK      = (30, 30, 30)


def put(img, text, pos, scale=0.7, color=WHITE, thickness=2):
    cv2.putText(img, text, pos, FONT, scale, DARK, thickness + 2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)


def draw_hud(frame, gesture_idx, count, recording):
    h, w = frame.shape[:2]
    gesture = GESTURES[gesture_idx]
    remaining = SAMPLES_PER_GESTURE - count

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
    put(frame, f"Gesture: {gesture}  ({gesture_idx+1}/{len(GESTURES)})",
        (12, 28), scale=0.75, color=WHITE)
    put(frame, f"Collected: {count}/{SAMPLES_PER_GESTURE}",
        (12, 56), scale=0.65, color=GREEN if count >= SAMPLES_PER_GESTURE else WHITE)

    # Recording pill
    if recording:
        cv2.rectangle(frame, (w-160, 8), (w-8, 40), (0, 0, 180), -1)
        put(frame, "● REC", (w-148, 30), scale=0.65, color=(80, 80, 255))
    else:
        cv2.rectangle(frame, (w-160, 8), (w-8, 40), (40, 40, 40), -1)
        put(frame, "SPACE to record", (w-158, 30), scale=0.5, color=(160, 160, 160))

    # Bottom instructions
    cv2.rectangle(frame, (0, h-50), (w, h), (20, 20, 20), -1)
    put(frame, "SPACE: record    N: next gesture    Q: save & quit",
        (12, h-16), scale=0.55, color=(180, 180, 180))

    return frame


# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    # Load existing data so we can append
    rows = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            rows = list(csv.reader(f))
        print(f"Loaded {len(rows)} existing rows from {DATA_PATH}")

    gesture_idx = 0
    recording   = False
    counts      = {g: 0 for g in GESTURES}

    # Count pre-existing samples per class
    for row in rows:
        if row and row[0] in counts:
            counts[row[0]] += 1

    print("\nInstructions:")
    print("  • Hold each gesture steadily in front of the camera")
    print("  • Press SPACE to start/stop recording that gesture")
    print("  • Aim for 200 varied samples (move hand slightly while recording)")
    print("  • Press N when done with the current gesture\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            if recording:
                gesture = GESTURES[gesture_idx]
                feat    = extract(hl).tolist()
                rows.append([gesture] + feat)
                counts[gesture] = counts.get(gesture, 0) + 1

                if counts[gesture] >= SAMPLES_PER_GESTURE:
                    recording = False
                    print(f"  Reached {SAMPLES_PER_GESTURE} samples for '{gesture}'")

        frame = draw_hud(frame, gesture_idx, counts.get(GESTURES[gesture_idx], 0), recording)
        cv2.imshow("Gesture Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
            if recording:
                print(f"  Recording '{GESTURES[gesture_idx]}' …")
        elif key == ord("n"):
            recording = False
            gesture_idx = (gesture_idx + 1) % len(GESTURES)
            print(f"  Switched to gesture: {GESTURES[gesture_idx]}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Save
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    total = len(rows)
    print(f"\nSaved {total} rows to {DATA_PATH}")
    for g in GESTURES:
        n = counts.get(g, 0)
        bar = "█" * (n // 10) + "░" * ((SAMPLES_PER_GESTURE - n) // 10)
        print(f"  {g:<16} {n:>3}/{SAMPLES_PER_GESTURE}  {bar}")


if __name__ == "__main__":
    main()
