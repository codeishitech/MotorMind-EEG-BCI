
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from run_eegnet_fusion_v2_loso import (
    EEGNetFusionV2,
    load_and_validate_dataset,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "output" / "epochs" / "motormind_all_participants_epochs.npz"
OUTPUT_DIR = (
    ROOT / "output" / "model_results"
    / "eegnet_fusion_v2_3ch_random_split"
)

SEED = 2024
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 5

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists. Choose a new path: {OUTPUT_DIR}"
    )

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)
device = torch.device("cpu")

# --------------------------------------------------
# 2. Load existing processed EEG data
# --------------------------------------------------
dataset = load_and_validate_dataset(str(DATA_PATH))
X = np.asarray(dataset["X"], dtype=np.float32)
y = np.asarray(dataset["y"], dtype=np.int64)

if X.ndim != 3 or X.shape[1:] != (3, 640):
    raise ValueError(f"Unexpected EEG shape: {X.shape}")

if set(np.unique(y)) != {0, 1}:
    raise ValueError(f"Expected binary labels 0 and 1; got {np.unique(y)}")

print(f"Dataset shape: {X.shape}")
print(f"Class counts: {np.bincount(y)}")

# --------------------------------------------------
# 3. Random trial-level split: 70% / 15% / 15%
# --------------------------------------------------
all_indices = np.arange(len(y))

train_idx, remaining_idx = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

val_idx, test_idx = train_test_split(
    remaining_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=y[remaining_idx],
)

print(
    f"Trials: train={len(train_idx)}, "
    f"validation={len(val_idx)}, test={len(test_idx)}"
)

# Verify that no trial index appears in multiple splits.
assert not set(train_idx) & set(val_idx)
assert not set(train_idx) & set(test_idx)
assert not set(val_idx) & set(test_idx)

def make_loader(indices, shuffle):
    x_tensor = torch.from_numpy(X[indices])
    y_tensor = torch.from_numpy(y[indices])
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )

train_loader = make_loader(train_idx, True)
val_loader = make_loader(val_idx, False)
test_loader = make_loader(test_idx, False)

# --------------------------------------------------
# 4. Initialize the existing EEGNet architecture
# --------------------------------------------------
model = EEGNetFusionV2(
    n_channels=3,
    n_samples=640,
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

# --------------------------------------------------
# 5. Train and select the best validation checkpoint
# --------------------------------------------------
started = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(yb)
        train_count += len(yb)

    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_sum += loss.item() * len(yb)
            val_count += len(yb)

    train_loss = train_loss_sum / train_count
    val_loss = val_loss_sum / val_count

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f}",
        flush=True,
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation loss stopped improving.")
        break

if best_state is None:
    raise RuntimeError("No valid model checkpoint was selected.")

model.load_state_dict(best_state)
model.eval()

# --------------------------------------------------
# 6. Evaluate the test set only once
# --------------------------------------------------
true_labels = []
predicted_labels = []
probabilities = []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        probs = torch.softmax(logits, dim=1)

        true_labels.extend(yb.numpy().tolist())
        predicted_labels.extend(
            probs.argmax(dim=1).cpu().numpy().tolist()
        )
        probabilities.extend(probs[:, 1].cpu().numpy().tolist())

true_labels = np.asarray(true_labels)
predicted_labels = np.asarray(predicted_labels)
probabilities = np.asarray(probabilities)

metrics = {
    "accuracy": float(accuracy_score(true_labels, predicted_labels)),
    "balanced_accuracy": float(
        balanced_accuracy_score(true_labels, predicted_labels)
    ),
    "macro_f1": float(
        f1_score(true_labels, predicted_labels, average="macro")
    ),
    "roc_auc": float(
        roc_auc_score(true_labels, probabilities)
    ),
    "test_trials": int(len(true_labels)),
    "best_epoch": int(best_epoch),
    "training_seconds": round(time.perf_counter() - started, 2),
    "seed": SEED,
    "split": "random trial-level 70/15/15; participants may overlap",
}

# --------------------------------------------------
# 7. Save results without touching LOSO outputs
# --------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

pd.DataFrame({
    "true_label": true_labels,
    "predicted_label": predicted_labels,
    "prob_class1": probabilities,
}).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nRandom-split test results:")
print(json.dumps(metrics, indent=2))
print(f"\nSaved results to: {OUTPUT_DIR}")
