import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 0

UNLEARNING_STEPS = 150
LEARNING_RATE = 0.0001
GRAD_CLIP = 1.0
ALPHA = 0.9

DATA_DIR = Path("./data")
RESULT_DIR = Path("./results")
MODEL_DIR = Path("./models")

ORIGINAL_MODEL_PATH = MODEL_DIR / "original_corrupted_cnn.pth"
CORRUPTED_INDICES_PATH = RESULT_DIR / "corrupted_indices.npy"
CORRUPTED_LABELS_PATH = RESULT_DIR / "corrupted_labels.npy"

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
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# Load CIFAR-10 and corruption definition
# ============================================================

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
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
# Verify experimental split
# ============================================================

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

if not np.all(
    true_labels[corrupted_indices]
    != corrupted_labels[corrupted_indices]
):
    raise ValueError(
        "At least one forget sample does not have "
        "a corrupted label."
    )


# ============================================================
# Dataset used for unlearning
# ============================================================

# The Original Model learned using the corrupted label vector.
# Keep those labels available so that the forget loss targets
# the erroneous associations actually learned by that model.

corrupted_train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform
)

corrupted_train_dataset.targets = (
    corrupted_labels.tolist()
)

forget_dataset = Subset(
    corrupted_train_dataset,
    corrupted_indices.tolist()
)

# Retain samples were never corrupted, so their labels remain
# the original correct CIFAR-10 labels.
retain_dataset = Subset(
    train_dataset,
    retain_indices.tolist()
)


print("=" * 70)
print("NEGGRAD+ MACHINE UNLEARNING")
print("=" * 70)
print(f"Device               : {device}")
print(f"Forget samples       : {len(forget_dataset)}")
print(f"Retain samples       : {len(retain_dataset)}")
print(f"Unlearning steps     : {UNLEARNING_STEPS}")
print(f"Learning rate        : {LEARNING_RATE}")
print("=" * 70)

# ============================================================
# Compact CNN
# Must exactly match the Original Model
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
# Load the trained corrupted Original Model
# ============================================================

model = CompactCNN().to(device)

state_dict = torch.load(
    ORIGINAL_MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Loaded Original Model : {ORIGINAL_MODEL_PATH}"
)
print(
    f"Model parameters      : {parameter_count:,}"
)

# ============================================================
# NegGrad+ data loaders
# ============================================================

forget_loader = DataLoader(
    forget_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

retain_loader = DataLoader(
    retain_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


# ============================================================
# NegGrad+ optimization
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print(f"NegGrad+ alpha        : {ALPHA}")
print(f"Gradient clipping     : {GRAD_CLIP}")

# ============================================================
# NegGrad+ unlearning loop
# ============================================================

model.train()

retain_iterator = iter(retain_loader)

start_time = time.perf_counter()

forget_iterator = iter(forget_loader)

for step in range(UNLEARNING_STEPS):

    try:
        forget_inputs, forget_targets = next(forget_iterator)
    except StopIteration:
        forget_iterator = iter(forget_loader)
        forget_inputs, forget_targets = next(forget_iterator)

    # Get one retain batch.
    try:
        retain_inputs, retain_targets = next(retain_iterator)
    except StopIteration:
        retain_iterator = iter(retain_loader)
        retain_inputs, retain_targets = next(retain_iterator)

    forget_inputs = forget_inputs.to(device)
    forget_targets = forget_targets.to(device)

    retain_inputs = retain_inputs.to(device)
    retain_targets = retain_targets.to(device)

    optimizer.zero_grad()

    # Forget loss:
    # maximize CE for the corrupted labels.
    forget_outputs = model(forget_inputs)
    forget_loss = criterion(
        forget_outputs,
        forget_targets
    )

    # Retain loss:
    # preserve performance on retained clean samples.
    retain_outputs = model(retain_inputs)
    retain_loss = criterion(
        retain_outputs,
        retain_targets
    )

    # NegGrad+ objective:
    # minimize retain loss while maximizing forget loss.
    loss = (
        ALPHA * retain_loss
        - (1.0 - ALPHA) * forget_loss
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )

    optimizer.step()

    if (
        step == 0
        or (step + 1) % 10 == 0
    ):
        print(
            f"Step {step + 1:03d} | "
            f"Forget Loss: {forget_loss.item():.4f} | "
            f"Retain Loss: {retain_loss.item():.4f} | "
            f"Combined Loss: {loss.item():.4f}"
        )

unlearning_time = (
    time.perf_counter() - start_time
)

print("=" * 70)
print(
    f"Unlearning completed in "
    f"{unlearning_time:.2f} seconds"
)
print("=" * 70)

# ============================================================
# Save unlearned model and experiment metadata
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

unlearned_model_path = (
    MODEL_DIR / "unlearned_neggrad_plus_cnn.pth"
)

torch.save(
    model.state_dict(),
    unlearned_model_path
)

results = {
    "method": "NegGrad+",
    "seed": SEED,
    "forget_samples": len(forget_dataset),
    "retain_samples": len(retain_dataset),
    "batch_size": BATCH_SIZE,
    "unlearning_steps": UNLEARNING_STEPS,
    "learning_rate": LEARNING_RATE,
    "alpha": ALPHA,
    "gradient_clip": GRAD_CLIP,
    "unlearning_time_seconds": unlearning_time,
    "model_parameters": parameter_count
}

results_path = (
    RESULT_DIR / "neggrad_plus_unlearning_results.json"
)

with open(
    results_path,
    "w"
) as f:
    json.dump(
        results,
        f,
        indent=4
    )

print(f"Model saved           : {unlearned_model_path}")
print(f"Results saved         : {results_path}")