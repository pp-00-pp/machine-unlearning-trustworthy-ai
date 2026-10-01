import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 0

DATA_DIR = Path("./data")
RESULT_DIR = Path("./results")
MODEL_DIR = Path("./models")

CORRUPTED_INDICES_PATH = RESULT_DIR / "corrupted_indices.npy"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CIFAR-10 transform
# Same normalization as Original Model
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# Load CIFAR-10
# ============================================================

full_train_dataset = datasets.CIFAR10(
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


# ============================================================
# Construct retain set
# ============================================================

corrupted_indices = np.load(
    CORRUPTED_INDICES_PATH
)

all_indices = np.arange(
    len(full_train_dataset)
)

retain_indices = np.setdiff1d(
    all_indices,
    corrupted_indices
)

if len(corrupted_indices) != 5000:
    raise ValueError(
        f"Expected 5000 forget samples, "
        f"found {len(corrupted_indices)}."
    )

if len(retain_indices) != 45000:
    raise ValueError(
        f"Expected 45000 retain samples, "
        f"found {len(retain_indices)}."
    )

train_dataset = Subset(
    full_train_dataset,
    retain_indices.tolist()
)


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# Compact CNN
# Identical to Original Model architecture
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
# Fresh model initialization
# ============================================================

model = CompactCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Model information
# ============================================================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print("=" * 70)
print("CIFAR-10 RETRAINED REFERENCE MODEL")
print("=" * 70)

print(f"Device               : {device}")

if torch.cuda.is_available():
    print(
        f"GPU                  : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Full training set    : {len(full_train_dataset)}")
print(f"Excluded forget set  : {len(corrupted_indices)}")
print(f"Retain training set  : {len(train_dataset)}")
print(f"Test samples         : {len(test_dataset)}")
print(f"Model parameters     : {parameter_count:,}")
print(f"Batch size           : {BATCH_SIZE}")
print(f"Epochs               : {EPOCHS}")
print(f"Learning rate        : {LEARNING_RATE}")

print("=" * 70)


# ============================================================
# Evaluation function
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        running_loss / total
    )

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# Training
# ============================================================

history = []

training_start = time.perf_counter()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.perf_counter()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100.0 * correct / total
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "epoch_time_seconds": epoch_time
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )


training_time = (
    time.perf_counter()
    - training_start
)


# ============================================================
# Final evaluation
# ============================================================

final_test_loss, final_test_accuracy = evaluate(
    model,
    test_loader
)


# ============================================================
# Save model
# ============================================================

model_path = (
    MODEL_DIR
    / "retrained_reference_cnn.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# Save results
# ============================================================

results = {
    "experiment": "retrained_reference_model",
    "dataset": "CIFAR-10",
    "seed": SEED,
    "architecture": "CompactCNN",
    "model_parameters": parameter_count,
    "full_training_samples": len(
        full_train_dataset
    ),
    "excluded_forget_samples": len(
        corrupted_indices
    ),
    "retain_training_samples": len(
        train_dataset
    ),
    "test_samples": len(
        test_dataset
    ),
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "training_time_seconds": training_time,
    "final_test_loss": final_test_loss,
    "final_test_accuracy": final_test_accuracy,
    "history": history
}

result_path = (
    RESULT_DIR
    / "retrained_reference_results.json"
)

with open(
    result_path,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("RETRAINING COMPLETE")
print("=" * 70)

print(
    f"Final clean test loss : "
    f"{final_test_loss:.4f}"
)

print(
    f"Final clean test acc  : "
    f"{final_test_accuracy:.2f}%"
)

print(
    f"Total training time   : "
    f"{training_time:.2f} seconds"
)

print(
    f"Model saved           : "
    f"{model_path}"
)

print(
    f"Results saved         : "
    f"{result_path}"
)

print("=" * 70)