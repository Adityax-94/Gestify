
---
title: Gestify
emoji: 🖐
colorFrom: purple
colorTo: teal
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
python_version: "3.10"
---

# Hand Gesture Controller

Real-time hand gesture recognition that controls your system — volume, media playback, scroll, and mouse cursor — using a webcam and a trained MLP classifier on MediaPipe hand landmarks.

---

## Gestures

| Gesture | Action | How to make it |
|---|---|---|
| Open palm | Play / Volume up | All 5 fingers extended |
| Fist | Mute / Pause | All fingers curled |
| Index up | Scroll up | Only index finger extended |
| Two fingers | Scroll down | Index + middle extended |
| Pinch | Volume down | Thumb + index touching |
| Cursor mode | Move mouse | Index + middle spread wide |

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage — 3 steps

### Step 1: Collect training data

```bash
python src/collect_data.py
```

- The app opens your webcam
- Hold each gesture in front of the camera
- Press **SPACE** to start/stop recording
- Press **N** to move to the next gesture
- Aim for **200 samples per gesture** (move your hand slightly while recording for variety)
- Press **Q** to save and quit

Data is saved to `data/gestures.csv`.

### Step 2: Train the model

```bash
python src/train.py
```

Trains an MLP (128→64 hidden layers) with 5-fold cross-validation. Typical accuracy: **96–99%**.

Model saved to `models/classifier.joblib`.

### Step 3: Run the controller

```bash
python src/run.py
```

- **Q** — quit
- **P** — pause/resume action dispatch (landmark detection still shown)
- Move mouse to top-left corner as an emergency stop (PyAutoGUI failsafe)

---

## Project structure

```
gesture_controller/
├── data/
│   └── gestures.csv          # collected training samples
├── models/
│   ├── classifier.joblib     # trained MLP pipeline
│   └── label_encoder.joblib  # gesture label encoder
├── src/
│   ├── features.py           # landmark → feature vector (63-dim)
│   ├── collect_data.py       # Phase 1: data collection UI
│   ├── train.py              # Phase 2: model training + evaluation
│   ├── actions.py            # Phase 3: gesture → system action mapping
│   └── run.py                # Phase 4: real-time inference + overlay
└── requirements.txt
```

---

## How it works

```
Webcam frame
    └─▶ MediaPipe Hands (21 landmarks × x,y,z)
            └─▶ Feature extraction (normalized, scale-invariant, 63-dim)
                    └─▶ MLP classifier (StandardScaler + 128-64 MLP)
                            └─▶ Smoothing (7-frame majority vote)
                                    └─▶ Action dispatch (cooldown-gated)
                                            └─▶ pyautogui system call
```

### Why MLP over rules?

Rule-based systems (e.g. "index up if fingertip.y < MCP.y") break when hands tilt, rotate, or vary in size. The MLP learns from your own hand's geometry, making it robust to your natural holding style. Scale-invariant normalization handles different hand sizes and distances from the camera.

### Tuning tips

- **More samples = better accuracy.** If a gesture misclassifies, re-run `collect_data.py` to add more samples for just that gesture (existing data is preserved).
- **Confidence threshold** (`CONFIDENCE_THRESHOLD` in `run.py`) — raise to 0.85 to reduce false positives; lower to 0.65 if detection is too sluggish.
- **Smoothing window** (`SMOOTHING_WINDOW`) — increase to 10 for fewer accidental triggers; decrease to 5 for faster response.
- **Cooldowns** (`COOLDOWNS` in `actions.py`) — adjust per-gesture repeat rates to taste.

---

## Portfolio notes

- **Inference speed**: ~30 fps on CPU (MediaPipe is optimized for real-time)
- **Model size**: ~50 KB (MLP with 63→128→64→6 weights)
- **No GPU required**: runs entirely on CPU
- **Extensible**: add new gestures by recording samples and retraining — no code changes needed
