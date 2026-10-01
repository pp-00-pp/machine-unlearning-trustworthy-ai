import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
MODEL_DIR = Path("./models")
RESULT_DIR = Path("./results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("CIFAR-10 ORIGINAL MODEL TRAINING")
print("=" * 65)
print(f"Device        : {device}")

if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")

print(f"PyTorch       : {torch.__version__}")
print(f"Seed          : {SEED}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Epochs        : {EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")
print("=" * 65)


# ============================================================
# CIFAR-10 preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")


# ============================================================
# Compact CNN
# ============================================================

class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CompactCNN().to(device)

total_parameters = sum(
    p.numel() for p in model.parameters()
)

print(f"Model parameters : {total_parameters:,}")


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ============================================================
# Training
# ============================================================

history = []

if torch.cuda.is_available():
    torch.cuda.synchronize()

training_start = time.perf_counter()

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.perf_counter()

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    epoch_time = time.perf_counter() - epoch_start

    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total

    test_loss, test_accuracy = evaluate(
        model,
        test_loader
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
        f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )


if torch.cuda.is_available():
    torch.cuda.synchronize()

training_time = time.perf_counter() - training_start


# ============================================================
# Final evaluation
# ============================================================

final_test_loss, final_test_accuracy = evaluate(
    model,
    test_loader
)

print("=" * 65)
print("ORIGINAL MODEL RESULTS")
print("=" * 65)
print(f"Final Test Loss     : {final_test_loss:.4f}")
print(f"Final Test Accuracy : {final_test_accuracy:.2f}%")
print(f"Training Time       : {training_time:.2f} seconds")
print(f"Model Parameters    : {total_parameters:,}")
print("=" * 65)


# ============================================================
# Save trained model
# ============================================================

model_path = MODEL_DIR / "original_cnn.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "architecture": "CompactCNN",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "parameters": total_parameters
    },
    model_path
)


# ============================================================
# Save experimental results
# ============================================================

results = {
    "model": "CompactCNN",
    "model_role": "Original Model",
    "dataset": "CIFAR-10",
    "seed": SEED,
    "training_samples": len(train_dataset),
    "test_samples": len(test_dataset),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "parameters": total_parameters,
    "training_time_seconds": training_time,
    "final_test_loss": final_test_loss,
    "final_test_accuracy": final_test_accuracy,
    "history": history
}

results_path = RESULT_DIR / "original_results.json"

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


print(f"Model saved   : {model_path}")
print(f"Results saved : {results_path}")