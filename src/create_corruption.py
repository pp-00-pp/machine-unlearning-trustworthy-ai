import json
import random
from pathlib import Path

import numpy as np
from torchvision import datasets


# ============================================================
# Configuration
# ============================================================

SEED = 42
CORRUPTION_RATE = 0.10
NUM_CLASSES = 10

DATA_DIR = Path("./data")
RESULT_DIR = Path("./results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

rng = np.random.default_rng(SEED)


# ============================================================
# Load CIFAR-10 training labels
# ============================================================

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False
)

true_labels = np.array(train_dataset.targets, dtype=np.int64)

num_samples = len(true_labels)
num_corrupted = int(num_samples * CORRUPTION_RATE)


print("=" * 65)
print("CIFAR-10 CONTROLLED LABEL CORRUPTION")
print("=" * 65)
print(f"Random seed          : {SEED}")
print(f"Training samples     : {num_samples}")
print(f"Corruption rate      : {CORRUPTION_RATE * 100:.1f}%")
print(f"Samples to corrupt   : {num_corrupted}")
print("=" * 65)


# ============================================================
# Select samples globally at random
# ============================================================

corrupted_indices = rng.choice(
    num_samples,
    size=num_corrupted,
    replace=False
)

corrupted_indices = np.sort(corrupted_indices)


# ============================================================
# Generate incorrect labels
# ============================================================

corrupted_labels = true_labels.copy()

corruption_records = []

for index in corrupted_indices:

    true_label = int(true_labels[index])

    # Select uniformly from the other nine CIFAR-10 classes.
    possible_labels = [
        label
        for label in range(NUM_CLASSES)
        if label != true_label
    ]

    wrong_label = int(rng.choice(possible_labels))

    corrupted_labels[index] = wrong_label

    corruption_records.append({
        "index": int(index),
        "true_label": true_label,
        "corrupted_label": wrong_label
    })


# ============================================================
# Verification
# ============================================================

changed_count = np.sum(true_labels != corrupted_labels)

assert len(corrupted_indices) == num_corrupted
assert len(np.unique(corrupted_indices)) == num_corrupted
assert changed_count == num_corrupted

for record in corruption_records:
    assert record["true_label"] != record["corrupted_label"]


print(f"Verified corruptions : {changed_count}")
print(
    "All corrupted labels differ from true labels: "
    "YES"
)


# ============================================================
# Class-distribution information
# ============================================================

class_names = train_dataset.classes

selected_true_distribution = np.bincount(
    true_labels[corrupted_indices],
    minlength=NUM_CLASSES
)

selected_wrong_distribution = np.bincount(
    corrupted_labels[corrupted_indices],
    minlength=NUM_CLASSES
)


print("\nForget-set distribution:")
print("-" * 65)

for class_id, class_name in enumerate(class_names):

    print(
        f"{class_id:2d} {class_name:12s} | "
        f"True: {selected_true_distribution[class_id]:4d} | "
        f"Corrupted-to: {selected_wrong_distribution[class_id]:4d}"
    )


# ============================================================
# Save machine-readable arrays
# ============================================================

np.save(
    RESULT_DIR / "corrupted_indices.npy",
    corrupted_indices
)

np.save(
    RESULT_DIR / "corrupted_labels.npy",
    corrupted_labels
)


# ============================================================
# Save auditable metadata
# ============================================================

metadata = {
    "dataset": "CIFAR-10",
    "seed": SEED,
    "corruption_type": "symmetric_random_label_noise",
    "corruption_rate": CORRUPTION_RATE,
    "total_training_samples": num_samples,
    "number_corrupted": num_corrupted,
    "number_retain": num_samples - num_corrupted,
    "class_names": class_names,
    "forget_true_class_distribution":
        selected_true_distribution.tolist(),
    "forget_corrupted_class_distribution":
        selected_wrong_distribution.tolist(),
    "records": corruption_records
}

metadata_path = RESULT_DIR / "corruption_metadata.json"

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)


print("\n" + "=" * 65)
print("CORRUPTION SET CREATED SUCCESSFULLY")
print("=" * 65)
print(
    f"Forget samples       : {num_corrupted}"
)
print(
    f"Retain samples       : {num_samples - num_corrupted}"
)
print(
    f"Indices saved        : "
    f"{RESULT_DIR / 'corrupted_indices.npy'}"
)
print(
    f"Labels saved         : "
    f"{RESULT_DIR / 'corrupted_labels.npy'}"
)
print(
    f"Metadata saved       : {metadata_path}"
)
print("=" * 65)