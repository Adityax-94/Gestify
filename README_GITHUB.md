# Gestify — Hand Gesture Controller

A real-time hand gesture recognition system that controls your computer using just a webcam. Built with MediaPipe, scikit-learn, and OpenCV — no special hardware required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## Demo

> Record a short screen capture and paste the YouTube/GIF link here

---

## What it does

Gestify detects hand gestures from your webcam in real time and maps them to system actions — no mouse or keyboard needed.

| Gesture | Action | How to make it |
|---|---|---|
| Open palm | Play / Volume up | All 5 fingers extended |
| Fist | Mute / Pause | All fingers curled |
| Index finger up | Scroll up | Only index finger extended |
| Index + middle up | Scroll down | Index and middle extended |
| Pinch | Volume down | Thumb and index touching |
| Index + middle spread | Move cursor | Both fingers spread wide apart |

---

## How it works

```
Webcam frame
    └─▶ MediaPipe HandLandmarker
            21 landmarks (x, y, z) per hand
        └─▶ Feature extraction
                63-dim vector, scale-invariant, wrist-normalized
            └─▶ MLP Classifier
                    StandardScaler → 128 → 64 → 6 classes
                └─▶ 7-frame majority vote smoothing
                        └─▶ System action dispatch
                                volume / scroll / cursor / media
```

### Why MLP over rule-based?

Rule-based systems (e.g. "index up if fingertip.y < MCP.y") break when hands tilt, rotate, or vary in size across users. The MLP learns from your own hand's geometry, making it robust to your natural holding style. Scale-invariant normalization handles different hand sizes and camera distances.

---

## Project structure

```
gestify/
├── app.py                     ← Hugging Face Spaces entry point (Gradio)
├── requirements.txt
├── README.md
├── data/
│   └── gestures.csv           ← collected training samples (gitignored)
├── models/
│   ├── classifier.joblib      ← trained MLP pipeline (gitignored)
│   ├── label_encoder.joblib   ← gesture label encoder (gitignored)
│   └── hand_landmarker.task   ← MediaPipe model (gitignored)
└── src/
    ├── mp_hands.py            ← MediaPipe Tasks API wrapper
    ├── features.py            ← landmark → 63-dim feature vector
    ├── collect_data.py        ← data collection UI
    ├── train.py               ← model training + evaluation
    ├── actions.py             ← gesture → system action mapping
    └── run.py                 ← real-time inference + overlay
```

---

## Setup

### Requirements

- Python 3.10+
- Webcam
- Windows / macOS / Linux

### Install

```bash
# Clone the repo
git clone https://github.com/Adityax-94/gestify
cd gestify

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pyautogui pandas    # local-only deps
```

---

## Usage

### Step 1 — Collect training data

```bash
python src/collect_data.py
```

The app opens your webcam with a guided UI:

- Hold each gesture steadily in front of the camera
- Press **SPACE** to start / stop recording
- Press **N** to move to the next gesture
- Aim for **200 samples per gesture** — move your hand slightly while recording for variety
- Press **Q** to save and quit

Data is saved to `data/gestures.csv`.

### Step 2 — Train the model

```bash
python src/train.py
```

Runs 5-fold cross-validation and prints a full classification report. Typical accuracy: **96–99%** with 200 samples per class. Model is saved to `models/`.

### Step 3 — Run the controller

```bash
python src/run.py
```

| Key | Action |
|---|---|
| `Q` | Quit |
| `P` | Pause / resume action dispatch |
| Mouse to top-left corner | Emergency stop (PyAutoGUI failsafe) |

---

## Tuning tips

- **More samples = better accuracy.** If a gesture misclassifies, re-run `collect_data.py` to add more samples — existing data is preserved and appended.
- **Confidence threshold** — raise `CONFIDENCE_THRESHOLD` in `run.py` to `0.85` to reduce false positives, lower to `0.65` if detection feels sluggish.
- **Smoothing window** — increase `SMOOTHING_WINDOW` to `10` for fewer accidental triggers, decrease to `5` for faster response.
- **Cooldowns** — adjust per-gesture repeat rates in `actions.py` to taste.

---

## Live demo

A Gradio-based web demo is hosted on Hugging Face Spaces — visitors can test gesture predictions live in their browser using their own webcam (system actions are disabled in the hosted version).

👉 [huggingface.co/spaces/Adityax-94/gestify](https://huggingface.co/spaces/Adityax-94/gestify)

---

## Tech stack

| Component | Library |
|---|---|
| Hand landmark detection | MediaPipe Tasks API |
| ML classifier | scikit-learn MLPClassifier |
| Feature extraction | NumPy |
| Webcam + overlay | OpenCV |
| System actions | PyAutoGUI |
| Web demo | Gradio |

---

## Performance

- **Inference speed:** ~30 fps on CPU (no GPU required)
- **Model size:** ~50 KB
- **Latency:** <50ms per frame end-to-end
- **Accuracy:** 96–99% (5-fold CV, 200 samples/class)

---

## Roadmap

- [ ] Two-hand gesture support
- [ ] Custom gesture recording without code changes
- [ ] Gesture macro sequences (chain multiple gestures)
- [ ] Browser extension for tab/bookmark control

---

## License

MIT — free to use, modify, and distribute.
