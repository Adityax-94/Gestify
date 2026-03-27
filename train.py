"""
train.py  —  Phase 3: train the gesture classifier
────────────────────────────────────────────────────
Run:
    python src/train.py

Reads   data/gestures.csv
Saves   models/classifier.joblib
        models/label_encoder.joblib

Architecture: MLPClassifier (2 hidden layers, 128-64)
Typical accuracy on 200 samples/class: 96–99%
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "gestures.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
CLF_PATH   = os.path.join(MODEL_DIR, "classifier.joblib")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoder.joblib")
FEAT_DIM   = 63   # 21 landmarks × 3


def load_data():
    print(f"Loading data from {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH, header=None)
    print(f"  {len(df)} total rows, {df[0].nunique()} classes")
    print(df[0].value_counts().to_string())
    print()

    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values

    if X.shape[1] != FEAT_DIM:
        raise ValueError(f"Expected {FEAT_DIM} features, got {X.shape[1]}")

    return X, y


def train(X, y_raw):
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Pipeline: StandardScaler → MLP
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False,
        )),
    ])

    # 5-fold cross-validation on training set
    print("Running 5-fold cross-validation …")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print()

    # Final fit on full training set
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    acc    = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion matrix (rows=true, cols=pred):")
    labels = le.classes_
    cm = confusion_matrix(y_test, y_pred)
    header = "       " + "  ".join(f"{l[:6]:>6}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {labels[i][:6]:>6} " + "  ".join(f"{v:>6}" for v in row))
    print()

    return pipeline, le


def save(pipeline, le):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, CLF_PATH)
    joblib.dump(le,       ENC_PATH)
    print(f"Saved classifier  → {CLF_PATH}")
    print(f"Saved label encoder → {ENC_PATH}")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run  python src/collect_data.py  first.")
        sys.exit(1)

    X, y = load_data()
    pipeline, le = train(X, y)
    save(pipeline, le)
    print("\nTraining complete. Run  python src/run.py  to start the controller.")


if __name__ == "__main__":
    main()
