
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

# 1. Paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "output" / "epochs" / "motormind_all_participants_epochs.npz"
OUTPUT_DIR = ROOT / "output" / "model_results" / "csp_lda_random_80_20"

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output folder already exists. Choose a new folder: {OUTPUT_DIR}"
    )

# 2. Load existing EEG data
with np.load(DATA_PATH, allow_pickle=False) as data:
    X = data["X"].astype(np.float64)
    y = data["y"].astype(int)

print("Dataset shape:", X.shape)
print("Class counts [0, 1]:", np.bincount(y))

# 3. Random 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2024,
    stratify=y,
)

print("Training trials:", len(y_train))
print("Testing trials:", len(y_test))
print("Train class counts:", np.bincount(y_train))
print("Test class counts:", np.bincount(y_test))

# 4. CSP + LDA pipeline
csp = mne.decoding.CSP(
    n_components=2,
    reg=None,
    log=True,
    norm_trace=False,
)

model = Pipeline([
    ("csp", csp),
    ("lda", LinearDiscriminantAnalysis(solver="svd")),
])

# 5. Train and evaluate
start = time.perf_counter()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

runtime = time.perf_counter() - start

# 6. Calculate metrics
metrics = {
    "split": "random stratified 80/20 trial-level",
    "random_state": 2024,
    "train_trials": int(len(y_train)),
    "test_trials": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, y_pred)
    ),
    "macro_f1": float(
        f1_score(y_test, y_pred, average="macro")
    ),
    "roc_auc": float(roc_auc_score(y_test, y_score)),
    "runtime_seconds": round(runtime, 2),
    "confusion_matrix": confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).tolist(),
}

print("\nClassification report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Left", "Right"],
    zero_division=0,
))

print("\nFinal metrics:")
print(json.dumps(metrics, indent=2))

# 7. Save results
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

pd.DataFrame({
    "true_label": y_test,
    "predicted_label": y_pred,
    "prob_class1": y_score,
}).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nResults saved to:", OUTPUT_DIR)
