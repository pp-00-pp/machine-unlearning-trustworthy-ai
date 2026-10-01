import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 128
NUM_WORKERS = 0

DATA_DIR = Path("./data")
RESULT_DIR = Path("./results")
MODEL_DIR = Path("./models")

MODEL_PATH = MODEL_DIR / "retrained_reference_cnn.pth"
CORRUPTED_INDICES_PATH = RESULT_DIR / "corrupted_indices.npy"
CORRUPTED_LABELS_PATH = RESULT_DIR / "corrupted_labels.npy"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CIFAR-10 transform
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# Load datasets and labels
# ============================================================

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transform
)

true_labels = np.array(
    train_dataset.targets,
    dtype=np.int64
)

corrupted_indices = np.load(
    CORRUPTED_INDICES_PATH
)

corrupted_labels = np.load(
    CORRUPTED_LABELS_PATH
)

all_indices = np.arange(
    len(train_dataset)
)

retain_indices = np.setdiff1d(
    all_indices,
    corrupted_indices
)


# ============================================================
# Compact CNN
# Identical to Original and Retrained models
# ============================================================

class CompactCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 8 * 8,
                256
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                10
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Load Retrained Reference Model
# ============================================================

model = CompactCNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()


# ============================================================
# Prediction function
# ============================================================

def get_predictions(dataset, indices=None):

    if indices is None:
        evaluation_dataset = dataset
    else:
        evaluation_dataset = Subset(
            dataset,
            indices.tolist()
        )

    loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    predictions = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            outputs = model(images)

            predicted = outputs.argmax(
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

    return np.array(
        predictions,
        dtype=np.int64
    )


# ============================================================
# Forget-set evaluation
# ============================================================

forget_predictions = get_predictions(
    train_dataset,
    corrupted_indices
)

forget_true_labels = true_labels[
    corrupted_indices
]

forget_wrong_labels = corrupted_labels[
    corrupted_indices
]

forget_true_accuracy = (
    100.0
    * np.mean(
        forget_predictions
        == forget_true_labels
    )
)

forget_corrupted_accuracy = (
    100.0
    * np.mean(
        forget_predictions
        == forget_wrong_labels
    )
)

forget_other_rate = (
    100.0
    - forget_true_accuracy
    - forget_corrupted_accuracy
)


# ============================================================
# Retain-set evaluation
# ============================================================

retain_predictions = get_predictions(
    train_dataset,
    retain_indices
)

retain_true_labels = true_labels[
    retain_indices
]

retain_accuracy = (
    100.0
    * np.mean(
        retain_predictions
        == retain_true_labels
    )
)


# ============================================================
# Clean test-set evaluation
# ============================================================

test_predictions = get_predictions(
    test_dataset
)

test_true_labels = np.array(
    test_dataset.targets,
    dtype=np.int64
)

test_accuracy = (
    100.0
    * np.mean(
        test_predictions
        == test_true_labels
    )
)


# ============================================================
# Display results
# ============================================================

print("=" * 70)
print("RETRAINED REFERENCE MODEL EVALUATION")
print("=" * 70)

print(f"Device                         : {device}")
print(f"Forget samples                 : {len(corrupted_indices)}")
print(f"Retain samples                 : {len(retain_indices)}")
print(f"Clean test samples             : {len(test_dataset)}")

print("-" * 70)

print(
    f"Forget -> TRUE label accuracy  : "
    f"{forget_true_accuracy:.2f}%"
)

print(
    f"Forget -> WRONG label accuracy : "
    f"{forget_corrupted_accuracy:.2f}%"
)

print(
    f"Forget -> OTHER label rate     : "
    f"{forget_other_rate:.2f}%"
)

print(
    f"Retain-set accuracy            : "
    f"{retain_accuracy:.2f}%"
)

print(
    f"Clean test accuracy            : "
    f"{test_accuracy:.2f}%"
)

print("=" * 70)


# ============================================================
# Save results
# ============================================================

results = {
    "model": "retrained_reference_cnn",
    "stage": "retrained_reference",
    "forget_samples": int(
        len(corrupted_indices)
    ),
    "retain_samples": int(
        len(retain_indices)
    ),
    "test_samples": int(
        len(test_dataset)
    ),
    "forget_true_label_accuracy": float(
        forget_true_accuracy
    ),
    "forget_corrupted_label_accuracy": float(
        forget_corrupted_accuracy
    ),
    "forget_other_label_rate": float(
        forget_other_rate
    ),
    "retain_accuracy": float(
        retain_accuracy
    ),
    "clean_test_accuracy": float(
        test_accuracy
    )
}

output_path = (
    RESULT_DIR
    / "retrained_reference_evaluation.json"
)

with open(
    output_path,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print(
    f"Results saved: {output_path}"
)