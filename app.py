"""
app.py  --  Hugging Face Spaces entry point (Gradio)
-----------------------------------------------------
Deploys the gesture classifier as a live webcam demo.
Visitors open the Space, allow webcam access, and see
gesture predictions + confidence in real time.

Local test:
    python app.py
"""

import os
import sys

import cv2
import gradio as gr
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from mp_hands import HandTracker, draw_landmarks
from features import extract

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CLF_PATH  = os.path.join(MODEL_DIR, "classifier.joblib")
ENC_PATH  = os.path.join(MODEL_DIR, "label_encoder.joblib")

CONFIDENCE_THRESHOLD = 0.70

# Gesture descriptions shown in the UI
DESCRIPTIONS = {
    "open_palm":   "Open palm — Play / Volume up",
    "fist":        "Fist — Mute / Pause",
    "index_up":    "Index up — Scroll up",
    "two_fingers": "Two fingers — Scroll down",
    "pinch":       "Pinch — Volume down",
    "cursor_mode": "Index + middle — Move cursor",
}

GESTURE_COLORS_BGR = {
    "open_palm":   (60,  200, 80),
    "fist":        (60,  80,  220),
    "index_up":    (40,  160, 200),
    "two_fingers": (40,  120, 200),
    "pinch":       (200, 60,  180),
    "cursor_mode": (200, 200, 40),
}

# ── Load model once at startup ───────────────────────────────────────────────
print("Loading model ...")
pipeline = joblib.load(CLF_PATH)
le       = joblib.load(ENC_PATH)
tracker  = HandTracker()
print(f"Ready. Classes: {list(le.classes_)}")


# ── Inference function (called per webcam frame by Gradio) ───────────────────
def predict(frame):
    """
    frame: H×W×3 uint8 numpy array in RGB (Gradio sends RGB).
    Returns: (annotated_frame_rgb, gesture_label, confidence_pct)
    """
    if frame is None:
        return None, "No frame", "—"

    # HandTracker expects RGB — Gradio already sends RGB
    result = tracker.detect(frame)

    # Work on a BGR copy for OpenCV drawing
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gesture    = None
    confidence = 0.0

    if result:
        landmarks = result.hand_landmarks[0]
        draw_landmarks(canvas, landmarks)

        feat  = extract(landmarks).reshape(1, -1)
        proba = pipeline.predict_proba(feat)[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        label = le.classes_[idx]

        if conf >= CONFIDENCE_THRESHOLD:
            gesture    = label
            confidence = conf

            # Draw gesture label on frame
            color = GESTURE_COLORS_BGR.get(label, (200, 200, 200))
            cv2.putText(canvas, DESCRIPTIONS.get(label, label),
                        (14, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (20, 20, 20), 4)
            cv2.putText(canvas, DESCRIPTIONS.get(label, label),
                        (14, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)

            # Confidence bar
            h, w = canvas.shape[:2]
            bar_w = int((w - 28) * conf)
            cv2.rectangle(canvas, (14, 50), (14 + bar_w, 56), color, -1)
            cv2.rectangle(canvas, (14, 50), (w - 14,    56), (80, 80, 80), 1)

            # Border
            cv2.rectangle(canvas, (0, 0), (w-1, h-1), color, 2)

    # Convert back to RGB for Gradio
    out_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    label_text = DESCRIPTIONS.get(gesture, "No gesture detected") if gesture else "No gesture detected"
    conf_text  = f"{confidence*100:.1f}%" if gesture else "—"

    return out_rgb, label_text, conf_text


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Gestify — Hand Gesture Controller") as demo:
    gr.Markdown("""
# Gestify — Real-Time Hand Gesture Recognition
Trained MLP classifier on MediaPipe 21-point hand landmarks.
Allow webcam access, then try one of the gestures below.
""")

    with gr.Row():
        with gr.Column(scale=2):
            webcam = gr.Image(sources=["webcam"], streaming=True,
                              label="Webcam", mirror_webcam=True)
        with gr.Column(scale=1):
            out_frame   = gr.Image(label="Annotated output")
            out_gesture = gr.Textbox(label="Detected gesture", interactive=False)
            out_conf    = gr.Textbox(label="Confidence",        interactive=False)

    webcam.stream(predict,
                  inputs=[webcam],
                  outputs=[out_frame, out_gesture, out_conf])

    gr.Markdown("""
## Gesture reference
| Gesture | Action |
|---|---|
| Open palm (all 5 fingers) | Play / Volume up |
| Fist (all fingers curled) | Mute / Pause |
| Index finger only | Scroll up |
| Index + middle finger | Scroll down |
| Pinch (thumb + index) | Volume down |
| Index + middle spread wide | Move cursor |

> **Note:** This demo shows predictions only — system actions (volume, scroll) run locally via `run.py`.
""")


if __name__ == "__main__":
    demo.launch()
